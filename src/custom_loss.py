# src/custom_loss.py

import torch
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean", ignore_index=-100):
    """
    Compute the Focal Loss with support for class imbalance and ignoring a specific class.
    Args:
        inputs: Logits (before softmax) from the model's prediction
        targets: Ground truth class labels
        alpha: Balancing factor for class imbalance
        gamma: Focusing parameter for difficult examples
        reduction: Specifies the reduction to apply to the output
        ignore_index: Class index to ignore
    """
    # Ensure the inputs and targets are on the same device
    device = inputs.device
    targets = targets.to(device)

    # Compute the standard cross-entropy loss
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=ignore_index)

    # Compute the probability of the correct class
    pt = torch.exp(-ce_loss)  # This is the probability for the true class

    # Apply the focal loss formula: (1 - pt)^gamma * log(pt)
    focal_loss_value = alpha * ((1 - pt) ** gamma) * ce_loss

    # Apply reduction (mean or sum)
    if reduction == "mean":
        return focal_loss_value.mean()
    elif reduction == "sum":
        return focal_loss_value.sum()
    else:
        return focal_loss_value
