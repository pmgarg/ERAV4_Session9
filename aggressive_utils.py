"""
Aggressive Training Utilities - Mixup, EMA, and helpers
For rapid convergence to 80%+ accuracy
"""

import torch
import torch.nn as nn
import numpy as np


def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Mixup data augmentation

    Args:
        x: Input batch
        y: Target labels
        alpha: Mixup interpolation strength
        device: Device for computation

    Returns:
        mixed_x, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss calculation

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup interpolation parameter

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EMA:
    """
    Exponential Moving Average of model weights
    Helps stabilize training and improve final accuracy
    """

    def __init__(self, model, decay=0.9999):
        """
        Initialize EMA

        Args:
            model: PyTorch model
            decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights after optimizer step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model (for validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original weights (after validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()


def validate_with_tta(model, loader, criterion, device, num_augments=2):
    """
    Validation with Test-Time Augmentation

    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: Device for computation
        num_augments: Number of TTA augmentations (default: 2)

    Returns:
        avg_loss, top1_acc, top5_acc
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Original prediction
            outputs = model(inputs)

            # TTA: Horizontal flip
            if num_augments > 1:
                outputs_flip = model(torch.flip(inputs, [3]))
                outputs = (outputs + outputs_flip) / 2

            # Calculate loss (only on original)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    avg_loss = total_loss / len(loader)
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total

    return avg_loss, top1_acc, top5_acc


if __name__ == "__main__":
    print("=" * 60)
    print("Aggressive Training Utilities Test")
    print("=" * 60)

    # Test mixup
    print("\n✓ Mixup utilities loaded")
    print("✓ EMA class loaded")
    print("✓ TTA validation loaded")
    print("\nReady for aggressive training!")
