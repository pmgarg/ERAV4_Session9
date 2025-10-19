"""
ImageNet Full Dataset Loader
For training on the complete ImageNet-1K dataset (~1.28M images)
Optimized for multi-GPU training on AWS
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.distributed as dist

from data_normalisation import get_train_transforms, get_val_transforms


def get_full_dataloaders(data_dir,
                         batch_size=256,
                         num_workers=8,
                         advanced_augmentation=True,
                         augmentation_strength='medium',
                         pin_memory=True,
                         distributed=False):
    """
    Get data loaders for full ImageNet-1K dataset

    Args:
        data_dir (str): Directory containing full ImageNet data
        batch_size (int): Batch size for training (scale with GPUs)
        num_workers (int): Number of worker processes (scale with CPU cores)
        advanced_augmentation (bool): Use advanced data augmentation
        augmentation_strength (str): 'light', 'medium', or 'heavy' augmentation strength
        pin_memory (bool): Pin memory for faster data transfer to GPU
        distributed (bool): Use distributed data parallel training

    Returns:
        tuple: (train_loader, val_loader, num_classes, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # Check if data exists
    if not (train_dir.is_dir() and val_dir.is_dir()):
        raise FileNotFoundError(
            f"ImageNet directories not found: {train_dir}, {val_dir}\n"
            f"Please ensure the full ImageNet-1K dataset is downloaded to: {data_dir}\n"
            "Expected structure:\n"
            "  imagenet/\n"
            "    train/\n"
            "      n01440764/\n"
            "      n01443537/\n"
            "      ...\n"
            "    val/\n"
            "      n01440764/\n"
            "      n01443537/\n"
            "      ...\n"
        )

    # Get transforms
    train_transform = get_train_transforms(
        advanced_augmentation=advanced_augmentation,
        augmentation_strength=augmentation_strength
    )
    val_transform = get_val_transforms()

    # Create datasets
    print(f"\nLoading full ImageNet dataset from {data_dir}...")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # Setup distributed training if enabled
    if distributed:
        try:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )

            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False
            )

            if rank == 0:
                print(f"Distributed training enabled with {world_size} GPUs")
                print(f"Per-GPU batch size: {batch_size}")
                print(f"Effective batch size: {batch_size * world_size}")

        except Exception as e:
            print(f"Warning: Could not setup distributed training: {e}")
            print("Falling back to single-GPU training")
            distributed = False

    if not distributed:
        # Standard single-GPU or CPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

    print(f"Full dataset loaded:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader):,} train, {len(val_loader):,} val")
    print(f"  Distributed: {distributed}\n")

    return train_loader, val_loader, num_classes, class_names


def setup_distributed_training():
    """
    Setup distributed training for multi-GPU training

    Returns:
        tuple: (local_rank, world_size) or (None, 1) if not distributed
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        print(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        return local_rank, world_size
    else:
        print("Not using distributed training")
        return None, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import os

    print("="*60)
    print("Full Dataset Loader Test")
    print("="*60)

    # Test with mock directory (you'll need real ImageNet data)
    try:
        # This will fail unless you have full ImageNet
        data_dir = os.environ.get('IMAGENET_DIR', './imagenet_full')

        print(f"\nAttempting to load full ImageNet from: {data_dir}")
        print("Note: This requires the complete ImageNet-1K dataset\n")

        train_loader, val_loader, num_classes, class_names = get_full_dataloaders(
            data_dir=data_dir,
            batch_size=256,
            num_workers=8,
            advanced_augmentation=True,
            distributed=False
        )

        print(f"\n✓ Data loaders created successfully!")
        print(f"  Train batches: {len(train_loader):,}")
        print(f"  Val batches: {len(val_loader):,}")
        print(f"  Number of classes: {num_classes}")

        # Test loading a batch
        print("\nTesting batch loading...")
        for inputs, targets in train_loader:
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            break

        print("\n✓ Full dataset loader test passed!")

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nThis is expected if you don't have the full ImageNet dataset.")
        print("For testing and development, use data_loader_small.py instead.")
        print("\nTo use full dataset:")
        print("  1. Download full ImageNet-1K from official source")
        print("  2. Set IMAGENET_DIR environment variable")
        print("  3. Run this test again")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
