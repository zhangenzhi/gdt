import torch, time 
import torch.optim 
import torch.utils.data 
import torch.distributed as dist 
from torch.nn.parallel.distributed import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 

# modify batch size according to GPU memory 
batch_size = 64 

from timm.models.vision_transformer import VisionTransformer 

from torch.utils.data import Dataset 

import transformer_engine.pytorch as te 
from transformer_engine.common import recipe 

# use random data 
class FakeDataset(Dataset): 
    def __len__(self): 
        return 1000000 

    def __getitem__(self, index): 
        rand_image = torch.randn([3, 224, 224], dtype=torch.float32) 
        label = torch.tensor(data=[index % 1000], dtype=torch.int64) 
        return rand_image, label 

class TE_Block(te.transformer.TransformerLayer): 
    def __init__( 
            self, 
            dim, 
            num_heads, 
            mlp_ratio=4., 
            qkv_bias=False, 
            qk_norm=False, 
            proj_drop=0., 
            attn_drop=0., 
            init_values=None, 
            drop_path=0., 
            act_layer=None, 
            norm_layer=None, 
            mlp_layer=None 
    ): 
        super().__init__( 
            hidden_size=dim, 
            ffn_hidden_size=int(dim * mlp_ratio), 
            num_attention_heads=num_heads, 
            hidden_dropout=proj_drop, 
            attention_dropout=attn_drop 
            )
        

def mp_fn(local_rank, *args): 
    # configure process 
    dist.init_process_group("nccl", 
                            rank=local_rank, 
                            world_size=torch.cuda.device_count()) 
    torch.cuda.set_device(local_rank) 
    device = torch.cuda.current_device() 
 
    # create dataset and dataloader 
    train_set = FakeDataset() 
    train_loader = torch.utils.data.DataLoader( 
        train_set, batch_size=batch_size, 
        num_workers=12, pin_memory=True) 

    # define ViT-Huge model 
    model = VisionTransformer( 
            embed_dim=1280, 
            depth=32, 
            num_heads=16,
            block_fn=TE_Block  
        ).cuda(device) 
    model = DDP(model, device_ids=[local_rank]) 

    # define loss and optimizer 
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

    model.train() 

    t0 = time.perf_counter() 
    summ = 0 
    count = 0 

    for step, data in enumerate(train_loader): 
        # copy data to GPU 
        inputs = data[0].to(device=device, non_blocking=True) 
        label = data[1].squeeze(-1).to(device=device, non_blocking=True) 
 
        # use mixed precision to take advantage of bfloat16 support 
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
            outputs = model(inputs) 
            loss = criterion(outputs, label) 
        optimizer.zero_grad(set_to_none=True) 
        loss.backward() 
        optimizer.step() 
 
        # capture step time 
        batch_time = time.perf_counter() - t0 
        if step > 10:  # skip first steps 
            summ += batch_time 
            count += 1 
        t0 = time.perf_counter() 
        if step > 50: 
            break 
    print(f'average step time: {summ/count}') 


if __name__ == '__main__': 
    mp.spawn(mp_fn, 
             args=(), 
             nprocs=torch.cuda.device_count(), 
             join=True)