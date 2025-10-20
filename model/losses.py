import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    一个结合了Dice Loss和二元交叉熵（BCE）的复合损失函数。
    这种组合通常比单独使用任何一种都更稳定和有效。
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-6):
        """
        Args:
            logits (torch.Tensor): 模型的原始输出 (在激活函数之前)。
            targets (torch.Tensor): 真实标签 (0或1)。
        
        Returns:
            torch.Tensor: 计算出的总损失。
        """
        # 使用sigmoid将logits转换为[0, 1]范围内的概率
        probs = torch.sigmoid(logits)
        
        # 将张量展平以便计算
        probs = probs.view(-1)
        targets = targets.view(-1)
        print(f"probs:{probs.mean()}, targets:{targets.mean()}")
        
        # --- 1. BCE Loss component ---
        # 直接在logits上使用BCEWithLogitsLoss以保证数值稳定性
        bce_loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='mean')
        
        # --- 2. Dice Loss component ---
        intersection = (probs * targets).sum()                            
        dice_coeff = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        
        # Dice Loss的正确形式是 1 - Dice系数
        # 它的值域是 [0, 1]，永远不会是负数。
        dice_loss = 1 - dice_coeff
        print(f"dice_loss:{dice_loss}, bce_loss:{bce_loss}")
        # 最终损失是两者的和
        loss = bce_loss + dice_loss
        return loss