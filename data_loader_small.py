"""
ImageNet Small Dataset Loader
For training on a subset of ImageNet (50k-100k samples)
Includes automatic download from Hugging Face
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np

from data_normalisation import get_train_transforms, get_val_transforms

# Check if Hugging Face datasets is available
try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("Warning: Hugging Face datasets not available. Install with: pip install datasets huggingface-hub")


def download_imagenet_subset(data_dir, num_samples=80000):
    """
    Download ImageNet subset from Hugging Face

    Args:
        data_dir (str or Path): Directory to save dataset
        num_samples (int): Number of training samples to download (50k-100k recommended)

    Returns:
        bool: True if successful, False otherwise
    """
    if not HAS_HF_DATASETS:
        raise ImportError(
            "Hugging Face datasets is required for auto-download. "
            "Install it with: pip install datasets huggingface-hub"
        )

    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    print(f"\n{'='*60}")
    print(f"Downloading ImageNet-1K subset ({num_samples:,} samples)...")
    print(f"This will be saved to: {data_dir}")
    print(f"{'='*60}\n")

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image

        # Load ImageNet-1K dataset from Hugging Face
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
        val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

        print(f"Dataset loaded successfully!")
        print(f"Extracting {num_samples:,} training samples...")

        # Counter for samples
        train_samples = 0
        val_samples = 0

        # Process training data
        print("\nProcessing training samples...")
        for idx, sample in enumerate(dataset):
            if train_samples >= num_samples:
                break

            try:
                img = sample['image']
                label = sample['label']

                # Create class directory
                class_dir = train_dir / f"class_{label:04d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"img_{idx:08d}.jpg"
                if isinstance(img, Image.Image):
                    img.save(img_path)
                train_samples += 1

                if train_samples % 1000 == 0:
                    print(f"  Processed {train_samples:,}/{num_samples:,} training samples")

            except Exception as e:
                print(f"  Warning: Skipped sample {idx} due to error: {e}")
                continue

        # Process validation data (10% of training samples)
        val_target = num_samples // 10
        print(f"\nProcessing validation samples (target: {val_target:,})...")
        for idx, sample in enumerate(val_dataset):
            if val_samples >= val_target:
                break

            try:
                img = sample['image']
                label = sample['label']

                # Create class directory
                class_dir = val_dir / f"class_{label:04d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"img_{idx:08d}.jpg"
                if isinstance(img, Image.Image):
                    img.save(img_path)
                val_samples += 1

                if val_samples % 500 == 0:
                    print(f"  Processed {val_samples:,}/{val_target:,} validation samples")

            except Exception as e:
                print(f"  Warning: Skipped sample {idx} due to error: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Dataset download complete!")
        print(f"  Training samples: {train_samples:,}")
        print(f"  Validation samples: {val_samples:,}")
        print(f"  Location: {data_dir}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error downloading dataset from Hugging Face: {e}")
        print("\nAlternative: Download ImageNet manually or use Tiny ImageNet")
        print("Visit: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("\nFor Tiny ImageNet (200 classes, 100k images):")
        print("  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print("  unzip tiny-imagenet-200.zip")
        return False


def get_small_dataloaders(data_dir='./imagenet_data',
                          batch_size=128,
                          num_workers=4,
                          num_samples=80000,
                          auto_download=True,
                          subset_percentage=None,
                          advanced_augmentation=True,
                          augmentation_strength='light',
                          pin_memory=True):
    """
    Get data loaders for ImageNet small subset

    Args:
        data_dir (str): Directory containing ImageNet data
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        num_samples (int): Number of samples to download (if auto_download=True)
        auto_download (bool): Automatically download data if not found
        subset_percentage (float): Use a percentage of available data (overrides num_samples)
        advanced_augmentation (bool): Use advanced data augmentation
        augmentation_strength (str): 'light', 'medium', or 'heavy' augmentation strength
        pin_memory (bool): Pin memory for faster data transfer to GPU

    Returns:
        tuple: (train_loader, val_loader, num_classes, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # Check if data exists
    if not (train_dir.is_dir() and val_dir.is_dir()):
    #if (train_dir.is_dir() and val_dir.is_dir()):
        if auto_download:
            print(f"\n⚠️  ImageNet directories not found at: {data_dir}")
            print("Attempting to download ImageNet subset from Hugging Face...")

            # Ensure num_samples is in reasonable range
            num_samples = max(50_000, min(num_samples, 1000000))
            print(f"Will download approximately {num_samples:,} training samples")
            print("Note: This may take 30-90 minutes depending on your internet connection.\n")

            success = download_imagenet_subset(data_dir, num_samples)

            if not success:
                raise FileNotFoundError(
                    f"Failed to auto-download ImageNet data.\n"
                    f"Please manually download ImageNet to: {data_dir}\n"
                    "Or set auto_download=False and provide a valid data_dir path."
                )
        else:
            raise FileNotFoundError(
                f"ImageNet directories not found: {train_dir}, {val_dir}\n"
                f"Set auto_download=True to automatically download a subset."
            )

    # Get transforms
    train_transform = get_train_transforms(
        advanced_augmentation=advanced_augmentation,
        augmentation_strength=augmentation_strength
    )
    val_transform = get_val_transforms()

    # Create datasets
    print(f"\nLoading datasets from {data_dir}...")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # Create subset if specified
    if subset_percentage is not None:
        num_train = int(len(train_dataset) * subset_percentage)
        num_val = int(len(val_dataset) * subset_percentage)

        train_indices = np.random.choice(len(train_dataset), num_train, replace=False)
        val_indices = np.random.choice(len(val_dataset), num_val, replace=False)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

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

        print(f"Using subset: {num_train:,} training, {num_val:,} validation samples")
    else:
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

        print(f"Using full dataset: {len(train_dataset):,} training, {len(val_dataset):,} validation samples")

    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(train_loader)} train, {len(val_loader)} val\n")

    return train_loader, val_loader, num_classes, class_names


if __name__ == "__main__":
    print("="*60)
    print("Small Dataset Loader Test")
    print("="*60)

    # Test with very small subset
    try:
        train_loader, val_loader, num_classes, class_names = get_small_dataloaders(
            data_dir='./imagenet_data',
            batch_size=32,
            num_workers=2,
            num_samples=80000,  # Will download if not exists
            auto_download=True,
            subset_percentage=0.01,  # Use 1% for quick test
            advanced_augmentation=True
        )

        print(f"\n✓ Data loaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Number of classes: {num_classes}")

        # Test loading a batch
        print("\nTesting batch loading...")
        for inputs, targets in train_loader:
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            break

        print("\n✓ Small dataset loader test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nThis is expected if you haven't downloaded the dataset yet.")
        print("Run with auto_download=True to download the dataset first.")
