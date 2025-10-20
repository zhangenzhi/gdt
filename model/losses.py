import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """计算Dice损失，用于图像分割"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 使用sigmoid将logits转换为概率
        probs = torch.sigmoid(logits)
        
        # 展平logits和targets
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_coeff

class DiceBCELoss(nn.Module):
    """
    将Dice损失和二元交叉熵损失（BCE）结合。
    这在分割任务中是一种非常常见且有效的损失函数。
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        
        # 加权组合两种损失
        total_loss = (self.dice_weight * dice_loss) + (self.bce_weight * bce_loss)
        
        return total_loss
