import os
import sys
import yaml
import logging
import argparse
import contextlib
from typing import Dict, Any
from functools import partial

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP


import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup
from timm.models.vision_transformer import Block


# 导入 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8


from timm.models.vision_transformer import Block as TimmBlock
from timm import create_model as timm_create_model


# -------------------------------
# Utilities
# -------------------------------

def _get_rank0() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0


def _maybe_log(msg: str):
    if _get_rank0():
        logging.info(msg)


# -------------------------------
# TE-based ViT Block (drop-in for timm Block)
# -------------------------------
class TEViTBlock(nn.Module):
    """Transformer block using Transformer Engine fused ops.

    Structure mimics timm's Block: LN1 -> Attn -> DropPath -> residual -> LN2 -> MLP -> DropPath -> residual.
    We explicitly use TE's LayerNormLinear for (LN + Linear) fusions and TE's FusedAttention.

    This keeps the outer interface consistent so it can replace timm blocks one-by-one.
    """

    def __init__(self, timm_block: TimmBlock):
        super().__init__()
        # Extract shapes from the reference timm block
        d_model = timm_block.attn.proj.in_features
        nheads = timm_block.attn.num_heads
        mlp_hidden = timm_block.mlp.fc1.out_features
        attn_drop = float(timm_block.attn.attn_drop.p)
        proj_drop = float(timm_block.attn.proj_drop.p)
        ln_eps = float(timm_block.norm1.eps)
        self.drop_path = getattr(timm_block, 'drop_path', nn.Identity())

        # --- QKV (LN + Linear) fused ---
        self.qkv = te.LayerNormLinear(
            d_model,
            3 * d_model,
            bias=True,
            eps=ln_eps,
        )
        # Attention (fused)
        self.attn = te.MultiheadAttention(
            hidden_size=d_model,
            num_attention_heads=nheads,
            attention_dropout=attn_drop,
            fuse_qkv_params=False,  # we supply qkv explicitly via LayerNormLinear
        )
        # Output projection
        self.proj = te.Linear(d_model, d_model, bias=True)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # --- MLP with fused LN+Linear on the first layer ---
        self.mlp_fc1 = te.LayerNormLinear(d_model, mlp_hidden, bias=True, eps=ln_eps)
        self.act = nn.GELU()
        self.mlp_fc2 = te.Linear(mlp_hidden, d_model, bias=True)
        self.mlp_drop = getattr(timm_block.mlp, 'drop', nn.Identity())

        # Keep for weight mapping
        self._timm_block = timm_block
        self._map_weights_from_timm()

    def _split_qkv(self, qkv_weight: torch.Tensor, qkv_bias: torch.Tensor, nheads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split timm's fused qkv [3*d, d] into separate q,k,v for TE MHA shape [d, d].
        We load into LayerNormLinear which expects a single combined linear of size 3*d, so we keep it fused.
        Return values are kept only if you'd like to debug individual projections.
        """
        d3, d = qkv_weight.shape
        assert d3 % 3 == 0
        d_model = d
        qw, kw, vw = torch.split(qkv_weight, d_model, dim=0)
        qb, kb, vb = torch.split(qkv_bias, d_model, dim=0)
        return qw, kw, vw, qb, kb, vb

    @torch.no_grad()
    def _map_weights_from_timm(self):
        tb = self._timm_block
        # --- Map LN1 + QKV ---
        # LayerNormLinear has separate (layernorm_weight/bias) and (weight/bias) for the linear.
        self.qkv.layernorm_weight.copy_(tb.norm1.weight)
        self.qkv.layernorm_bias.copy_(tb.norm1.bias)
        self.qkv.weight.copy_(tb.attn.qkv.weight)
        if tb.attn.qkv.bias is not None:
            self.qkv.bias.copy_(tb.attn.qkv.bias)

        # --- Attention output proj ---
        self.proj.weight.copy_(tb.attn.proj.weight)
        if tb.attn.proj.bias is not None:
            self.proj.bias.copy_(tb.attn.proj.bias)

        # --- Map LN2 + MLP ---
        self.mlp_fc1.layernorm_weight.copy_(tb.norm2.weight)
        self.mlp_fc1.layernorm_bias.copy_(tb.norm2.bias)
        self.mlp_fc1.weight.copy_(tb.mlp.fc1.weight)
        if tb.mlp.fc1.bias is not None:
            self.mlp_fc1.bias.copy_(tb.mlp.fc1.bias)

        self.mlp_fc2.weight.copy_(tb.mlp.fc2.weight)
        if tb.mlp.fc2.bias is not None:
            self.mlp_fc2.bias.copy_(tb.mlp.fc2.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Residual 1: Attention
        residual = x
        # qkv: fused LN + linear
        qkv = self.qkv(x)
        # TE MHA expects [S, B, D] by default; our tensors are [B, S, D]
        B, S, D = x.shape
        qkv = qkv.view(B, S, 3, D).permute(1, 0, 2, 3).contiguous()  # [S,B,3,D]
        # Fused MHA returns [S, B, D]
        out = self.attn(qkv, attn_mask=attn_mask)
        out = out.permute(1, 0, 2).contiguous()  # [B,S,D]
        out = self.proj_drop(self.proj(out))
        x = residual + self.drop_path(out)

        # Residual 2: MLP
        residual = x
        y = self.mlp_fc2(self.act(self.mlp_fc1(x)))
        y = self.mlp_drop(y)
        x = residual + self.drop_path(y)
        return x


# -------------------------------
# Patch Embed wrapper (use timm's)
# -------------------------------
class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class TEViT(nn.Module):
    """A ViT that uses timm's stem/head, with TE blocks in the encoder."""
    def __init__(self, timm_model: nn.Module, use_cls_token: bool = True):
        super().__init__()
        self.patch_embed = timm_model.patch_embed
        self.pos_drop = timm_model.pos_drop
        self.cls_token = getattr(timm_model, 'cls_token', None) if use_cls_token else None
        self.pos_embed = timm_model.pos_embed
        self.norm = timm_model.norm
        self.pre_logits = getattr(timm_model, 'pre_logits', IdentityLayer())
        self.head = timm_model.head

        # Replace blocks with TE blocks
        self.blocks = nn.ModuleList([TEViTBlock(b) for b in timm_model.blocks])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # TE MHA expects sequence-first; we handle layout inside the block
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cls_token is not None:
            return self.pre_logits(x[:, 0])
        else:
            return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -------------------------------
# FP8 wrapper
# -------------------------------
class FP8AutocastWrapper(nn.Module):
    def __init__(self, module: nn.Module, enabled: bool = True, autocast: bool = True, margin: int = 0):
        super().__init__()
        self.module = module
        self.enabled = enabled and fp8.is_fp8_available()
        self.autocast = autocast
        self.margin = margin

    def forward(self, *args, **kwargs):
        if self.enabled and self.autocast:
            with fp8.autocast(enabled=True, fp8_margin=self.margin):
                return self.module(*args, **kwargs)
        return self.module(*args, **kwargs)


# -------------------------------
# Factory
# -------------------------------

def create_vit_with_te(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create a timm ViT and replace blocks with TE fused blocks.

    config keys:
      - timm_model: str, e.g., 'vit_base_patch16_224'
      - pretrained: bool
      - drop_path: float (optional, will be respected via copied drop_path)
      - fp8: dict(enabled: bool, autocast: bool, margin: int)
    """
    timm_model_name = config.get('timm_model', 'vit_base_patch16_224')
    pretrained = bool(config.get('pretrained', True))

    _maybe_log(f"Creating timm model: {timm_model_name}, pretrained={pretrained}")
    tm = timm_create_model(timm_model_name, pretrained=pretrained)

    # Build TE ViT
    _maybe_log("Replacing timm blocks with Transformer Engine blocks...")
    model = TEViT(tm).to(device)

    # Optional FP8
    fp8_cfg = config.get('fp8', {"enabled": False})
    if fp8_cfg and fp8_cfg.get('enabled', False):
        model = FP8AutocastWrapper(
            model,
            enabled=True,
            autocast=fp8_cfg.get('autocast', True),
            margin=int(fp8_cfg.get('margin', 0)),
        )
        _maybe_log("Wrapped model with FP8 autocast.")

    _maybe_log("Done. TE blocks in place.")
    return model

# --- 核心代码 (与之前版本类似，但更加通用) ---

def setup_logging(args):
    """配置日志记录器。"""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "out.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, args, start_epoch, best_val_acc):
    """主训练和评估循环。"""
    num_epochs = config['training']['num_epochs']
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = dist.get_rank() == 0

    fp8_recipe_obj = te_recipe.fp8_recipe(fp8_format=te_recipe.Format.E4M3, amax_history_len=16, amax_compute_algo="max")
    
    if is_main_process:
        logging.info(f"启动训练，使用 {'Transformer Engine FP8' if args.use_fp8 else 'BF16'}...")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
        
    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            sync_context = model.no_sync() if is_accumulation_step else contextlib.nullcontext()
            
            with sync_context:
                autocast_ctx = te.fp8_autocast(enabled=args.use_fp8, fp8_recipe=fp8_recipe_obj) if args.use_fp8 \
                    else torch.cuda.amp.autocast(dtype=torch.bfloat16)

                with autocast_ctx:
                    outputs = model(images)
                
                loss = criterion(outputs.float(), labels)
                loss = loss / accumulation_steps
            
            loss.backward()

            if not is_accumulation_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
                
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 50 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 50
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Acc: {train_acc:.2f}%, LR: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        val_acc = evaluate_model(model, val_loader, device, args)
            
        if is_main_process:
            # ... (检查点保存逻辑与之前相同) ...
            pass

    if is_main_process:
        logging.info(f'训练完成。最佳验证精度: {best_val_acc:.4f}')

def evaluate_model(model, val_loader, device, args):
    """评估函数。"""
    model.eval()
    correct, total = 0, 0
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if args.use_fp8:
        fp8_recipe_obj = te_recipe.fp8_recipe(fp8_format=te_recipe.Format.E4M3)
        autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_obj)

    with torch.no_grad(), autocast_ctx:
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_tensor = torch.tensor(total, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    dist.all_reduce(total_tensor)
    dist.all_reduce(correct_tensor)
    total, correct = total_tensor.item(), correct_tensor.item()
    return 100 * correct / total if total > 0 else 0

def main_worker(args, config):
    """主工作函数，由每个 DDP 进程执行。"""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if rank == 0:
        setup_logging(args)
        logging.info(f"开始训练，共 {world_size} 个进程...")

    # --- 数据加载 (使用timm) ---
    # ... (此处应添加真实的timm数据加载逻辑) ...
    # 为了演示，我们暂时保留模拟数据
    train_dataset = torch.utils.data.TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
    val_dataset = torch.utils.data.TensorDataset(torch.randn(256, 3, 224, 224), torch.randint(0, 1000, (256,)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], sampler=val_sampler)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # --- 模型创建和转换 ---
    model = create_vit_with_te(config, device)
    model = DDP(model, device_ids=[local_rank])

    # --- 优化器和调度器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloaders['train']) * config['training']['num_epochs'])
    
    # --- 启动训练循环 ---
    train_loop(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=0, best_val_acc=0.0)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT FP8/BF16 Training Script with timm and TE")
    parser.add_argument('--config', type=str, default='./configs/vit-b16_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='vit-fp8-timm-run', help='Subdirectory for logs/models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--reload', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--use_fp8', action='store_true', help='Use Transformer Engine FP8 for training.')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output = os.path.join(args.output, "imagenet")
    os.makedirs(args.output, exist_ok=True)
    
    main_worker(args, config)