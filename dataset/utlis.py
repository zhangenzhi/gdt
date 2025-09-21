import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

import torch
import torch.nn.functional as F
import numpy as np

class SHFMixup:
    """
    A unified class to apply Mixup or CutMix to batches from the SHF Dataloader,
    mirroring the functionality of timm.data.Mixup.

    It correctly handles the dictionary-based batch format of the SHF dataloader,
    mixing patches and metadata for CutMix while only mixing patches for Mixup.
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                 switch_prob=0.5, label_smoothing=0.1, num_classes=1000, img_size=224):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.img_size = img_size

    def __call__(self, batch_dict, labels):
        """
        Applies the augmentation to a batch.
        Args:
            batch_dict (dict): The dictionary of tensors from the SHF dataloader.
            labels (Tensor): The corresponding labels for the batch.
        Returns:
            Tuple[dict, Tensor]: The augmented batch dictionary and the new soft labels.
        """
        # Decide whether to apply any augmentation based on probability
        if np.random.rand() > self.prob:
            # If not applying, return original data but with label smoothing
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
            smoothed_labels = labels_onehot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            return batch_dict, smoothed_labels
        
        # Decide whether to use Mixup or CutMix based on the switch probability
        if np.random.rand() < self.switch_prob:
            return self._mixup(batch_dict, labels)
        else:
            return self._cutmix(batch_dict, labels)

    def _mixup(self, batch_dict, labels):
        """Applies Mixup."""
        # 1. Generate shuffled batch indices
        shuffle_indices = torch.randperm(labels.shape[0]).to(labels.device)
        
        # 2. Get shuffled tensors
        patches_shuffled = batch_dict['patches'][shuffle_indices]
        labels_shuffled = labels[shuffle_indices]

        # 3. Generate mixing coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # 4. Interpolate patch tensors
        batch_dict['patches'] = lam * batch_dict['patches'] + (1 - lam) * patches_shuffled

        # 5. Interpolate labels
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        labels_shuffled_onehot = F.one_hot(labels_shuffled, num_classes=self.num_classes).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled_onehot
        
        return batch_dict, mixed_labels

    def _cutmix(self, batch_dict, labels):
        """Applies CutMix, correctly handling serialized metadata."""
        B, N = batch_dict['patches'].shape[0], batch_dict['patches'].shape[1]
        
        # 1. Generate shuffled batch indices
        shuffle_indices = torch.randperm(B).to(labels.device)
        
        # 2. Generate random bounding box coordinates
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(self.img_size * cut_ratio)
        cut_h = int(self.img_size * cut_ratio)
        
        cx = np.random.randint(self.img_size)
        cy = np.random.randint(self.img_size)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, self.img_size)
        bby1 = np.clip(cy - cut_h // 2, 0, self.img_size)
        bbx2 = np.clip(cx + cut_w // 2, 0, self.img_size)
        bby2 = np.clip(cy + cut_h // 2, 0, self.img_size)

        # 3. Create a mask based on patch centers falling inside the bounding box
        patch_centers = batch_dict['positions']
        in_box_mask = (patch_centers[..., 0] >= bbx1) & (patch_centers[..., 0] < bbx2) & \
                      (patch_centers[..., 1] >= bby1) & (patch_centers[..., 1] < bby2)

        # 4. Use the mask to combine data from the original and shuffled batches
        # Expand mask for broadcasting to different tensor shapes
        in_box_mask_patches = in_box_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        in_box_mask_meta = in_box_mask.unsqueeze(-1)
        
        # Replace patches that are inside the box with patches from the shuffled batch
        batch_dict['patches'] = torch.where(
            in_box_mask_patches, batch_dict['patches'][shuffle_indices], batch_dict['patches']
        )
        
        # Crucially, also replace the corresponding metadata
        batch_dict['positions'] = torch.where(
            in_box_mask_meta, batch_dict['positions'][shuffle_indices], batch_dict['positions']
        )
        batch_dict['sizes'] = torch.where(
            in_box_mask, batch_dict['sizes'][shuffle_indices], batch_dict['sizes']
        )

        # 5. Mix labels based on the area of the bounding box
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.img_size * self.img_size))
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        labels_shuffled_onehot = F.one_hot(labels[shuffle_indices], num_classes=self.num_classes).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled_onehot

        return batch_dict, mixed_labels