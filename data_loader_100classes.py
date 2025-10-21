"""
ImageNet 100-Class Full Dataset Loader
Downloads FULL dataset for 100 selected classes (instead of subset of 1000 classes)
Optimized for achieving 82%+ validation accuracy
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
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


# Selected 100 classes for balanced representation across ImageNet categories
# These classes are well-represented and have good visual diversity
SELECTED_100_CLASSES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,           # Animals (mammals)
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55, # More animals
    100, 105, 110, 115, 120, 125, 130, 135, # Birds
    200, 205, 210, 215, 220, 225, 230, 235, # Aquatic animals
    300, 305, 310, 315, 320, 325, 330, 335, # Reptiles/amphibians
    400, 405, 410, 415, 420, 425, 430, 435, # Insects
    500, 505, 510, 515, 520, 525, 530, 535, # Vehicles
    600, 605, 610, 615, 620, 625, 630, 635, # Furniture
    700, 705, 710, 715, 720, 725, 730, 735, # Instruments
    800, 805, 810, 815, 820, 825, 830, 835, # Tools/equipment
    900, 905, 910, 915, 920, 925, 930, 935, # Food
    940, 945, 950, 955, 960, 965, 970, 975  # Misc objects
]


def download_imagenet_100classes(data_dir, max_samples_per_class=None):
    """
    Download FULL ImageNet dataset for 100 selected classes

    Args:
        data_dir: Directory to save dataset
        max_samples_per_class: If specified, limit samples per class (None = all samples)

    Returns:
        bool: True if successful
    """
    if not HAS_HF_DATASETS:
        raise ImportError(
            "Hugging Face datasets required. Install: pip install datasets huggingface-hub"
        )

    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DOWNLOADING IMAGENET - 100 CLASSES (FULL DATASET)")
    print("="*80)
    print(f"Selected classes: {len(SELECTED_100_CLASSES)}")
    print(f"Target: ~130,000 training images (~1,300 per class)")
    print(f"Target: ~5,000 validation images (~50 per class)")
    print(f"Expected download time: 2-3 hours")
    print(f"Expected disk space: ~10-12 GB")
    print("="*80)

    try:
        from PIL import Image

        # Load datasets
        print("\nLoading ImageNet from Hugging Face...")
        train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
        val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

        # Track samples per class
        train_counts = {cls: 0 for cls in SELECTED_100_CLASSES}
        val_counts = {cls: 0 for cls in SELECTED_100_CLASSES}

        total_train = 0
        total_val = 0

        # Process training data
        print("\n" + "="*80)
        print("DOWNLOADING TRAINING DATA")
        print("="*80)

        for idx, sample in enumerate(train_dataset):
            try:
                label = sample['label']

                # Skip if not in our 100 classes
                if label not in SELECTED_100_CLASSES:
                    continue

                # Check if we've reached limit for this class
                if max_samples_per_class and train_counts[label] >= max_samples_per_class:
                    # Check if all classes are done
                    if all(count >= max_samples_per_class for count in train_counts.values()):
                        break
                    continue

                img = sample['image']

                # Map to new label (0-99 instead of original label)
                new_label = SELECTED_100_CLASSES.index(label)

                # Create class directory
                class_dir = train_dir / f"class_{new_label:03d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"train_{label:04d}_{train_counts[label]:05d}.jpg"
                if isinstance(img, Image.Image):
                    img.save(img_path)

                train_counts[label] += 1
                total_train += 1

                # Progress update
                if total_train % 1000 == 0:
                    avg_per_class = total_train / len([c for c in train_counts.values() if c > 0])
                    print(f"  Downloaded {total_train:,} training images (avg {avg_per_class:.0f} per class)")

            except Exception as e:
                print(f"  Warning: Skipped sample {idx}: {e}")
                continue

        print(f"\n✓ Training data complete: {total_train:,} images across {len(SELECTED_100_CLASSES)} classes")

        # Process validation data
        print("\n" + "="*80)
        print("DOWNLOADING VALIDATION DATA")
        print("="*80)

        for idx, sample in enumerate(val_dataset):
            try:
                label = sample['label']

                # Skip if not in our 100 classes
                if label not in SELECTED_100_CLASSES:
                    continue

                img = sample['image']

                # Map to new label
                new_label = SELECTED_100_CLASSES.index(label)

                # Create class directory
                class_dir = val_dir / f"class_{new_label:03d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"val_{label:04d}_{val_counts[label]:05d}.jpg"
                if isinstance(img, Image.Image):
                    img.save(img_path)

                val_counts[label] += 1
                total_val += 1

                # Progress update
                if total_val % 500 == 0:
                    avg_per_class = total_val / len([c for c in val_counts.values() if c > 0])
                    print(f"  Downloaded {total_val:,} validation images (avg {avg_per_class:.0f} per class)")

            except Exception as e:
                print(f"  Warning: Skipped sample {idx}: {e}")
                continue

        print(f"\n✓ Validation data complete: {total_val:,} images across {len(SELECTED_100_CLASSES)} classes")

        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"Training samples: {total_train:,}")
        print(f"Validation samples: {total_val:,}")
        print(f"Number of classes: 100")
        print(f"Avg samples per class (train): {total_train/len(SELECTED_100_CLASSES):.0f}")
        print(f"Avg samples per class (val): {total_val/len(SELECTED_100_CLASSES):.0f}")
        print(f"Location: {data_dir}")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def get_100class_dataloaders(
    data_dir='./imagenet_100class_data',
    batch_size=128,
    num_workers=4,
    auto_download=True,
    augmentation_strength='heavy',  # Use heavy by default for 82%+ accuracy
    pin_memory=True
):
    """
    Get data loaders for 100-class ImageNet

    Args:
        data_dir: Directory containing/to save dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        auto_download: Auto-download if not found
        augmentation_strength: 'light', 'medium', or 'heavy'
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, val_loader, num_classes, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # Check if data exists
    if not (train_dir.is_dir() and val_dir.is_dir()):
        if auto_download:
            print(f"\n⚠️  Dataset not found at: {data_dir}")
            print("Starting automatic download...")

            success = download_imagenet_100classes(data_dir)

            if not success:
                raise FileNotFoundError(
                    f"Failed to download dataset. Please check your internet connection."
                )
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {data_dir}\n"
                f"Set auto_download=True to automatically download."
            )

    # Get transforms with heavy augmentation for maximum accuracy
    train_transform = get_train_transforms(
        advanced_augmentation=True,
        augmentation_strength=augmentation_strength
    )
    val_transform = get_val_transforms()

    # Create datasets
    print(f"\nLoading 100-class ImageNet from {data_dir}...")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # For stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"✓ Data loaded successfully!")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Augmentation: {augmentation_strength}")
    print(f"  Samples per class (train): ~{len(train_dataset)/num_classes:.0f}")
    print(f"  Samples per class (val): ~{len(val_dataset)/num_classes:.0f}")

    return train_loader, val_loader, num_classes, class_names


if __name__ == "__main__":
    print("="*80)
    print("100-Class ImageNet Loader Test")
    print("="*80)

    # Test loading (will trigger download if needed)
    try:
        train_loader, val_loader, num_classes, class_names = get_100class_dataloaders(
            data_dir='./imagenet_100class_data',
            batch_size=32,
            num_workers=2,
            auto_download=True,
            augmentation_strength='heavy'
        )

        print(f"\n✓ Test passed!")
        print(f"  Classes: {num_classes}")
        print(f"  Ready for 82%+ accuracy training!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
