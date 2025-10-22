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

# Check if Hugging Face datasets is available
try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("Warning: Hugging Face datasets not available. Install with: pip install datasets huggingface-hub")


def download_imagenet_full(data_dir):
    """
    Download complete ImageNet-1K dataset (1000 classes)

    Args:
        data_dir: Directory to save dataset

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
    print("DOWNLOADING IMAGENET-1K - FULL DATASET (1000 CLASSES)")
    print("="*80)
    print(f"Total classes: 1000")
    print(f"Target: ~1,281,167 training images (~1,281 per class)")
    print(f"Target: ~50,000 validation images (~50 per class)")
    print(f"Expected download time: 4-6 hours (depends on internet speed)")
    print(f"Expected disk space: ~150-160 GB")
    print("="*80)
    print("\n⚠️  WARNING: This is a VERY large download!")
    print("  - Ensure you have stable internet connection")
    print("  - Ensure you have >200 GB free disk space")
    print("  - Consider running on AWS/cloud with fast network")
    print("="*80)

    try:
        from PIL import Image

        # Load datasets from Hugging Face
        print("\n[1/4] Loading ImageNet metadata from Hugging Face...")
        train_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        val_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True
        )
        print("✓ Metadata loaded")

        # Track progress
        train_counts = {}
        val_counts = {}
        total_train = 0
        total_val = 0

        # Process training data
        print("\n[2/4] Downloading training data (~1.28M images)...")
        print("="*80)

        for idx, sample in enumerate(train_dataset):
            try:
                label = sample['label']
                img = sample['image']

                # Track counts
                if label not in train_counts:
                    train_counts[label] = 0

                # Create class directory (using ImageNet synset format)
                class_dir = train_dir / f"n{label:08d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"train_{label:04d}_{train_counts[label]:05d}.JPEG"
                if isinstance(img, Image.Image):
                    img.save(img_path, 'JPEG')

                train_counts[label] += 1
                total_train += 1

                # Progress update every 5000 images
                if total_train % 5000 == 0:
                    num_classes_seen = len(train_counts)
                    avg_per_class = total_train / num_classes_seen if num_classes_seen > 0 else 0
                    progress_pct = (total_train / 1281167) * 100
                    print(f"  Progress: {total_train:,}/1,281,167 ({progress_pct:.1f}%) | "
                          f"Classes: {num_classes_seen}/1000 | "
                          f"Avg/class: {avg_per_class:.0f}")

            except Exception as e:
                if total_train % 1000 == 0:  # Only print errors occasionally
                    print(f"  Warning: Skipped sample {idx}: {e}")
                continue

        print(f"\n✓ Training data complete: {total_train:,} images across {len(train_counts)} classes")

        # Process validation data
        print("\n[3/4] Downloading validation data (~50K images)...")
        print("="*80)

        for idx, sample in enumerate(val_dataset):
            try:
                label = sample['label']
                img = sample['image']

                # Track counts
                if label not in val_counts:
                    val_counts[label] = 0

                # Create class directory
                class_dir = val_dir / f"n{label:08d}"
                class_dir.mkdir(exist_ok=True)

                # Save image
                img_path = class_dir / f"val_{label:04d}_{val_counts[label]:05d}.JPEG"
                if isinstance(img, Image.Image):
                    img.save(img_path, 'JPEG')

                val_counts[label] += 1
                total_val += 1

                # Progress update every 2000 images
                if total_val % 2000 == 0:
                    num_classes_seen = len(val_counts)
                    avg_per_class = total_val / num_classes_seen if num_classes_seen > 0 else 0
                    progress_pct = (total_val / 50000) * 100
                    print(f"  Progress: {total_val:,}/50,000 ({progress_pct:.1f}%) | "
                          f"Classes: {num_classes_seen}/1000 | "
                          f"Avg/class: {avg_per_class:.0f}")

            except Exception as e:
                if total_val % 500 == 0:  # Only print errors occasionally
                    print(f"  Warning: Skipped sample {idx}: {e}")
                continue

        print(f"\n✓ Validation data complete: {total_val:,} images across {len(val_counts)} classes")

        # Summary
        print("\n[4/4] Download complete!")
        print("="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Training samples: {total_train:,}")
        print(f"Validation samples: {total_val:,}")
        print(f"Training classes: {len(train_counts)}")
        print(f"Validation classes: {len(val_counts)}")
        print(f"Avg samples per class (train): {total_train/len(train_counts):.0f}")
        print(f"Avg samples per class (val): {total_val/len(val_counts):.0f}")
        print(f"Location: {data_dir}")
        print(f"Disk space used: ~{((total_train + total_val) * 120) / (1024**3):.1f} GB")
        print("="*80)
        print("✓ Ready for training!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_full_dataloaders(data_dir,
                         batch_size=256,
                         num_workers=8,
                         advanced_augmentation=True,
                         augmentation_strength='medium',
                         pin_memory=True,
                         distributed=False,
                         auto_download=False):
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
        auto_download (bool): Automatically download dataset if not found (WARNING: ~150GB download!)

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
            print("Starting automatic download of FULL ImageNet-1K dataset...")
            print("This will download ~150GB of data (1.28M images)")

            success = download_imagenet_full(data_dir)

            if not success:
                raise FileNotFoundError(
                    f"Failed to download dataset. Please check your internet connection.\n"
                    f"Alternatively, manually download ImageNet-1K and place it at: {data_dir}"
                )
        else:
            raise FileNotFoundError(
                f"ImageNet directories not found: {train_dir}, {val_dir}\n"
                f"Please ensure the full ImageNet-1K dataset is downloaded to: {data_dir}\n"
                "Expected structure:\n"
                "  imagenet_1000class_data/\n"
                "    train/\n"
                "      n00000001/\n"
                "      n00000002/\n"
                "      ...\n"
                "    val/\n"
                "      n00000001/\n"
                "      n00000002/\n"
                "      ...\n\n"
                "To automatically download (WARNING: ~150GB):\n"
                "  Set auto_download=True in get_full_dataloaders()\n"
                "  Or run: python data_loader_full.py --download\n"
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
    import sys

    # Check for download flag
    if '--download' in sys.argv:
        print("="*80)
        print("ImageNet-1K Download Script")
        print("="*80)

        # Get data directory from command line or use default
        data_dir = './imagenet_1000class_data'
        if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
            data_dir = sys.argv[2]

        print(f"\nDownload directory: {data_dir}")
        print("\n⚠️  WARNING: This will download ~150GB of data!")
        response = input("Do you want to continue? (yes/no): ")

        if response.lower() in ['yes', 'y']:
            success = download_imagenet_full(data_dir)
            if success:
                print("\n✓ Download complete! Ready for training.")
                sys.exit(0)
            else:
                print("\n❌ Download failed!")
                sys.exit(1)
        else:
            print("\nDownload cancelled.")
            sys.exit(0)

    # Test mode
    print("="*80)
    print("Full Dataset Loader Test")
    print("="*80)

    try:
        # Get data directory
        data_dir = os.environ.get('IMAGENET_DIR', './imagenet_1000class_data')

        print(f"\nAttempting to load full ImageNet from: {data_dir}")
        print("Note: This requires the complete ImageNet-1K dataset\n")

        # Try to load without auto-download (test only)
        train_loader, val_loader, num_classes, class_names = get_full_dataloaders(
            data_dir=data_dir,
            batch_size=256,
            num_workers=8,
            advanced_augmentation=True,
            augmentation_strength='medium',
            distributed=False,
            auto_download=False
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
        print(f"\n⚠️  Dataset not found")
        print(f"\n{e}")
        print("\n" + "="*80)
        print("TO DOWNLOAD IMAGENET-1K:")
        print("="*80)
        print("Run: python data_loader_full.py --download")
        print("\nOr in your notebook/script:")
        print("  train_loader, val_loader, num_classes, class_names = get_full_dataloaders(")
        print("      data_dir='./imagenet_1000class_data',")
        print("      auto_download=True  # Enable automatic download")
        print("  )")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
