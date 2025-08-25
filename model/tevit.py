import torch, time 
import torch.optim 
import torch.utils.data 
import torch.distributed as dist 
from torch.nn.parallel.distributed import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 
import os

# modify batch size according to GPU memory 
batch_size = 1024 

from timm.models.vision_transformer import VisionTransformer 

from torch.utils.data import Dataset 

import transformer_engine.pytorch as te 
from transformer_engine.common import recipe 

# use random data 
class FakeDataset(Dataset): 
    def __len__(self): 
        return 1024000 

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
            mlp_layer=None,
            **kwargs
    ): 
        super().__init__( 
            hidden_size=dim, 
            ffn_hidden_size=int(dim * mlp_ratio), 
            num_attention_heads=num_heads, 
            hidden_dropout=proj_drop, 
            attention_dropout=attn_drop 
            )
        
def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # create dataset and dataloader
    train_set = FakeDataset()
    # Use DistributedSampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        num_workers=32, pin_memory=True, sampler=train_sampler, drop_last=True)

    # define ViT-Huge model
    model = VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
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

    # Set epoch for sampler to ensure data shuffling
    train_loader.sampler.set_epoch(0)

    for step, data in enumerate(train_loader):
        # copy data to GPU
        inputs = data[0].to(device=device, non_blocking=True)
        label = data[1].squeeze(-1).to(device=device, non_blocking=True)

        # use mixed precision to take advantage of bfloat16 support
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True): 
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
        if step > 312:
            break
    
    # Only print from the main process to avoid cluttered logs
    if local_rank == 0:
        if count > 0:
            print(f'average step time: {summ/count}, total: {summ}')
        else:
            print('Not enough steps to measure average time.')

if __name__ == '__main__':
    # FIX 2: Remove mp.spawn and call main function directly
    main()