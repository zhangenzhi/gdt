import torch, time 
import torch.optim 
import torch.utils.data 
import torch.distributed as dist 
from torch.nn.parallel.distributed import DistributedDataParallel as DDP 
import torch.multiprocessing as mp 
import os

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer 

from torch.utils.data import Dataset 

import transformer_engine.pytorch as te 
from transformer_engine.common import recipe 


# modify batch size according to GPU memory
batch_size = 1024 

# use random data 
class FakeDataset(Dataset): 
    def __len__(self): 
        return 1281167 

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


# -----------------------------
# TE MLP
# -----------------------------
class TE_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = te.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = te.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------------------
# TE Norm (LayerNorm)
# -----------------------------
class TE_Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = te.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)




# =============================================
# Add-ons: TE replacements for timm.VisionTransformer
# =============================================
class TEPatchEmbedLinear(nn.Module):
    """Patch embedding via unfold + TE Linear so the projection participates in FP8.
    API-compatible with timm's PatchEmbed where needed (num_patches, grid_size).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_vec = in_chans * patch_size[0] * patch_size[1]
        Linear = te.Linear 
        self.proj = Linear(patch_vec, embed_dim, bias=True)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        patches = self.unfold(x)          # (B, patch_vec, L)
        patches = patches.transpose(1, 2) # (B, L, patch_vec)
        x = self.proj(patches)            # (B, L, embed_dim)
        return x


class TE_MLP(nn.Module):
    """timm-compatible MLP using TE Linear."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = te.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def te_norm_layer(norm_dim, eps: float = 1e-6, **kwargs):
    """Factory callable to pass into timm as norm_layer."""
    return te.LayerNorm(norm_dim, eps=eps)



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
            img_size = 224,
            patch_size = 16,
            in_chans = 3,
            num_classes = 1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            embed_layer=TEPatchEmbedLinear,
            block_fn=TE_Block,
            mlp_layer=TE_MLP,
            norm_layer=te_norm_layer
        ).cuda(device)
    
    # model = VisionTransformer(
    # img_size=224, patch_size=16, in_chans=3, num_classes=1000,
    # embed_dim=768, depth=12, num_heads=12,
    # block_fn=TE_Block,
    # embed_layer=TEPatchEmbedLinear,
    # mlp_layer=TE_MLP,
    # norm_layer=te_norm_layer)
    # model.head = te.Linear(768, 1000, bias=True)
    model = model.cuda(device)
    
    if dist.get_rank() == 0: 
        print("正在应用 torch.compile()...")
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, fused=True)

    model.train()
    # 为 FP8 训练创建 recipe
    fp8_recipe = recipe.DelayedScaling(
        margin=0, interval=16, fp8_format=recipe.Format.HYBRID
    )
    
    t0 = time.perf_counter()
    summ = 0
    count = 0

    # Set epoch for sampler to ensure data shuffling
    train_loader.sampler.set_epoch(0)
        
    # use mixed precision to take advantage of bfloat16 support
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for step, data in enumerate(train_loader):
            # copy data to GPU
            inputs = data[0].to(device=device, non_blocking=True)
            label = data[1].squeeze(-1).to(device=device, non_blocking=True)
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe): 
                outputs = model(inputs)
            # outputs = model(inputs)
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
            
    dist.destroy_process_group()
    
if __name__ == '__main__':
    # FIX 2: Remove mp.spawn and call main function directly
    main()