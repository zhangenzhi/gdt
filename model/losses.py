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
        # print(f"probs:{probs.mean()}, targets:{targets.mean(), targets.max(), targets.min()}")
        
        # --- 1. BCE Loss component ---
        # 直接在logits上使用BCEWithLogitsLoss以保证数值稳定性
        bce_loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='mean')
        
        # --- 2. Dice Loss component ---
        intersection = (probs * targets).sum()                            
        dice_coeff = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        
        # Dice Loss的正确形式是 1 - Dice系数
        # 它的值域是 [0, 1]，永远不会是负数。
        dice_loss = 1 - dice_coeff
        # print(f"dice_loss:{dice_loss}, bce_loss:{bce_loss}")
        # 最终损失是两者的和
        loss = bce_loss + dice_loss
        return loss

# --- Multi-class Dice Loss Component ---

def dice_coeff_multi(logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6, exclude_background: bool = True) -> torch.Tensor:
    """
    计算多类别 Dice 系数 (不计算损失，只计算系数)。
    Args:
        logits: 模型输出的原始 logits, shape (N, C, H, W)
        target: 真实标签, shape (N, H, W), 值域 [0, C-1]
        num_classes: 类别总数 C
        smooth: 防止除以零的平滑项
        exclude_background: 是否排除背景类别 (通常是类别 0) 的计算
    Returns:
        每个类别的 Dice 系数, shape (C,) 或 (C-1,)
    """
    # 1. 获取预测类别
    pred = logits.argmax(dim=1) # (N, H, W)

    # 2. 将 target 转换为 one-hot 编码
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float() # (N, C, H, W)
    pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()   # (N, C, H, W)

    # 3. 计算每个类别的交集和总和
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=(0, 2, 3)) # (C,)
    sum_pred = torch.sum(pred_one_hot, dim=(0, 2, 3)) # (C,)
    sum_target = torch.sum(target_one_hot, dim=(0, 2, 3)) # (C,)

    # 4. 计算 Dice 系数
    dice = (2. * intersection + smooth) / (sum_pred + sum_target + smooth) # (C,)

    if exclude_background:
        return dice[1:] # 排除类别 0
    else:
        return dice

# --- Combined Dice + Cross Entropy Loss ---

class DiceCELossMulti(nn.Module):
    """
    结合了多类别 Dice Loss 和 Cross Entropy Loss 的复合损失函数。
    """
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6, exclude_background: bool = True, use_softmax: bool = False):
        """
        Args:
            num_classes: 类别总数 (包括背景)
            ce_weight: Cross Entropy Loss 的权重
            dice_weight: Dice Loss 的权重
            smooth: Dice Loss 的平滑项
            exclude_background: 计算 Dice Loss 时是否排除背景类别
            use_softmax: 是否在 Dice Loss 计算前应用 Softmax (CE Loss 已内置)
        """
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.exclude_background = exclude_background
        self.use_softmax = use_softmax

        # Cross Entropy Loss (内置了 Softmax)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算多类别 Dice Loss (1 - mean(Dice Coeff))"""
        # 如果需要，应用 Softmax
        if self.use_softmax:
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1) # (N, H, W)
        else:
            # 直接从 logits 获取预测类别，避免重复 softmax
            pred = logits.argmax(dim=1) # (N, H, W)

        # 将 target 转换为 one-hot 编码
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float() # (N, C, H, W)
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes).permute(0, 3, 1, 2).float()   # (N, C, H, W)

        # 计算每个类别的交集和总和
        intersection = torch.sum(pred_one_hot * target_one_hot, dim=(0, 2, 3)) # (C,)
        sum_pred = torch.sum(pred_one_hot, dim=(0, 2, 3)) # (C,)
        sum_target = torch.sum(target_one_hot, dim=(0, 2, 3)) # (C,)

        # 计算 Dice 系数
        dice_coeffs = (2. * intersection + self.smooth) / (sum_pred + sum_target + self.smooth) # (C,)

        # 计算 Dice Loss (1 - mean Dice Coeff)
        if self.exclude_background:
            dice_loss = 1. - torch.mean(dice_coeffs[1:]) # 平均前景类别的 Dice
        else:
            dice_loss = 1. - torch.mean(dice_coeffs) # 平均所有类别的 Dice

        return dice_loss


    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): 模型输出的原始 logits, shape (N, C, H, W)
            target (torch.Tensor): 真实标签, shape (N, H, W), 值域 [0, C-1]
        Returns:
            torch.Tensor: 计算出的总损失。
        """
        # 确保 target 是 LongTensor
        target = target.long()

        # 1. Cross Entropy Loss
        # CrossEntropyLoss 期望 logits (N, C, H, W) 和 target (N, H, W)
        ce_loss = self.cross_entropy(logits, target)

        # 2. Dice Loss
        dice_loss = self._dice_loss(logits, target)

        # 3. 组合损失
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss

# --- Helper function to calculate mean Dice score for evaluation ---
def mean_dice_score_multi(logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6, exclude_background: bool = True) -> float:
    """计算平均 Dice 分数 (用于评估)。"""
    dice_per_class = dice_coeff_multi(logits, target.long(), num_classes, smooth, exclude_background)
    return torch.mean(dice_per_class).item()