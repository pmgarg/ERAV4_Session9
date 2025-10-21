"""
ResNet50 ImageNet Training Package
Modular implementation for training ResNet50 on ImageNet-1K
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .model import create_resnet50, ResNet50, get_model_stats
from .data_loader_small import get_small_dataloaders
from .data_loader_full import get_full_dataloaders
from .data_normalisation import (
    get_train_transforms,
    get_val_transforms,
    get_imagenet_normalization,
    denormalize_image
)
from .train import Trainer
from .lr_finder import LRFinder, find_lr_quick

__all__ = [
    'create_resnet50',
    'ResNet50',
    'get_model_stats',
    'get_small_dataloaders',
    'get_full_dataloaders',
    'get_train_transforms',
    'get_val_transforms',
    'get_imagenet_normalization',
    'denormalize_image',
    'Trainer',
    'LRFinder',
    'find_lr_quick',
]
