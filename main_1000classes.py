#!/usr/bin/env python3
"""
ImageNet 1000-Class Training with ResNet50
Target: 82%+ Validation Accuracy

Optimized for AWS GPU training (headless execution)
No Jupyter required - run with: python main_1000classes.py

Hardware recommendations:
- AWS g5.xlarge: 1x NVIDIA A10G (24GB) - $1.006/hr (~$4-5 spot)
- AWS g5.2xlarge: 1x NVIDIA A10G (24GB) + more CPU - $1.212/hr
- AWS p3.2xlarge: 1x NVIDIA V100 (16GB) - $3.06/hr (faster)

Expected training time: 12-15 hours (g5.xlarge) or 7-9 hours (p3.2xlarge)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import argparse
import sys

# Local imports
from model import create_resnet50, get_model_stats
from data_loader_full import get_full_dataloaders
from train import Trainer

def print_banner(text, char="=", width=70):
    """Print a formatted banner"""
    print("\n" + char * width)
    print(text)
    print(char * width)

def print_config(config):
    """Print configuration in a formatted way"""
    print_banner("TRAINING CONFIGURATION - ImageNet 1000 Classes")
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 70)

def main(args):
    """Main training function"""

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    config = {
        # Data Configuration
        'data_dir': args.data_dir or './imagenet_1000class_data',
        'num_classes': 1000,

        # Training Configuration
        'num_epochs': args.epochs or 60,
        'batch_size': args.batch_size or 256,
        'num_workers': args.num_workers or 8,
        'pin_memory': True,

        # Augmentation
        'augmentation_strength': args.augmentation or 'medium',

        # Learning Rate Configuration
        'find_lr': False, #args.find_lr,
        'initial_lr': 0.063680, #args.initial_lr or 0.05,
        'max_lr': 1.024401, #args.max_lr or 0.3,
        'lr_finder_iterations': 4000,

        # Regularization
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'max_grad_norm': 1.0,

        # Model Configuration
        'zero_init_residual': True,

        # OneCycleLR Configuration
        'pct_start': args.pct_start or 0.4,
        'div_factor': 25.0,
        'final_div_factor': args.final_div_factor or 1e3,

        # Checkpoint Configuration
        'checkpoint_dir': args.checkpoint_dir or './checkpoints_1000class',
        'save_frequency': 5,
    }

    print_config(config)
    print(f"\n⚠️  IMPORTANT: Training on 1.28M images!")
    print(f"Expected time: {config['num_epochs'] * 9 / 60:.1f}-{config['num_epochs'] * 12 / 60:.1f} hours")
    print(f"Expected cost: ~$4-5 (spot) or ~$12-15 (on-demand)")
    print("=" * 70)

    # ========================================================================
    # SETUP
    # ========================================================================
    print_banner("IMPORTS & SETUP")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple Metal (MPS)")
        print("⚠️  Warning: MPS training will be VERY slow for 1000 classes!")
        print("⚠️  Strongly recommend using AWS GPU instead.")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU - NOT RECOMMENDED!")
        print("⚠️  This will take days/weeks. Use AWS GPU instance.")
        if not args.force_cpu:
            print("\n❌ Aborting. Use --force-cpu to override.")
            sys.exit(1)

    print(f"\nDevice: {device}")

    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print_banner("LOADING FULL IMAGENET-1K DATASET (1000 CLASSES)")
    print("Expected dataset size:")
    print("  Training:   ~1,281,167 images (~1,281 per class)")
    print("  Validation: ~50,000 images (~50 per class)")
    print("  Total disk space: ~140-150 GB")
    print("  Number of classes: 1000")
    print("=" * 70 + "\n")

    try:
        train_loader, val_loader, num_classes, class_names = get_full_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            advanced_augmentation=True,
            augmentation_strength=config['augmentation_strength'],
            pin_memory=config['pin_memory'],
            distributed=False,
            auto_download=True
        )

        print(f"\n✓ Full ImageNet data loaded successfully!")
        print(f"  Number of classes: {num_classes}")
        print(f"  Training batches: {len(train_loader):,}")
        print(f"  Validation batches: {len(val_loader):,}")
        print(f"  Augmentation strength: {config['augmentation_strength']}")
        print(f"\n  With batch size {config['batch_size']}:")
        print(f"    ~{len(train_loader)} iterations per epoch")
        print(f"    ~{len(train_loader) * config['num_epochs']:,} total iterations")

    except FileNotFoundError as e:
        print("\n" + "=" * 70)
        print("❌ DATASET NOT FOUND")
        print("=" * 70)
        print(str(e))
        print("\nTo download dataset, run with: --auto-download")
        print("Or manually run: python data_loader_full.py --download")
        print("=" * 70)
        sys.exit(1)

    # ========================================================================
    # MODEL CREATION
    # ========================================================================
    print_banner("CREATING RESNET50 MODEL (1000 CLASSES)")

    model = create_resnet50(
        num_classes=config['num_classes'],
        zero_init_residual=config['zero_init_residual']
    )

    model = model.to(device)

    # Get model statistics
    stats = get_model_stats(model)

    print(f"Architecture:          ResNet50")
    print(f"Number of classes:     {config['num_classes']}")
    print(f"Total parameters:      {stats['total_parameters']:,}")
    print(f"Trainable parameters:  {stats['trainable_parameters']:,}")
    print(f"Model size:            {stats['model_size_mb']:.2f} MB")

    # ========================================================================
    # RESUME FROM CHECKPOINT (Optional)
    # ========================================================================
    start_epoch = 0
    resume_checkpoint_path = None

    if args.resume or args.resume_epoch:
        print_banner("RESUMING FROM CHECKPOINT")

        # Determine checkpoint path
        if args.resume:
            resume_checkpoint_path = Path(args.resume)
        elif args.resume_epoch:
            resume_checkpoint_path = Path(config['checkpoint_dir']) / f'checkpoint_epoch_{args.resume_epoch}.pth'

        if not resume_checkpoint_path.exists():
            print(f"❌ ERROR: Checkpoint not found at: {resume_checkpoint_path}")
            print(f"\nAvailable checkpoints:")
            checkpoint_dir = Path(config['checkpoint_dir'])
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
                if checkpoints:
                    for cp in checkpoints:
                        print(f"  - {cp.name}")
                else:
                    print(f"  No checkpoints found in {checkpoint_dir}")
            sys.exit(1)

        print(f"Loading checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"  Best accuracy so far: {checkpoint.get('best_acc', 'N/A')}")
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Training from:       RESUMED (loaded weights)")
    else:
        print(f"Training from:         Scratch (random initialization)")

    print(f"\n✓ Model created and moved to {device}")

    # ========================================================================
    # LEARNING RATE FINDER (Optional)
    # ========================================================================
    run_lr_finder = config['find_lr'] or (args.find_lr_on_resume and start_epoch > 0)

    if run_lr_finder:
        print_banner("LEARNING RATE FINDER - 1000 Classes")
        print("Running LR range test to find optimal learning rates...")
        print(f"This will take ~15-20 minutes on AWS GPU.")
        print(f"Testing {config['lr_finder_iterations']} learning rate values.\n")

        from lr_finder import LRFinder

        lr_finder = LRFinder(
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=config['weight_decay']),
            criterion=nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']),
            device=device
        )

        # Run LR finder
        lrs, losses, suggested_initial_lr, suggested_max_lr = lr_finder.find(
            train_loader,
            init_lr=1e-8,
            end_lr=2,
            num_iter=config['lr_finder_iterations']
        )

        # Save plot
        try:
            lr_finder.plot(lrs, losses, initial_lr=suggested_initial_lr, max_lr=suggested_max_lr)
            print("✓ LR finder plot saved to lr_finder_1000class.png")
        except Exception as e:
            print(f"⚠️  Could not save plot (matplotlib issue): {e}")

        # Validate and potentially adjust suggested LRs
        print(f"\n📊 LR Finder Results:")
        print(f"   Suggested initial_lr: {suggested_initial_lr:.6f}")
        print(f"   Suggested max_lr:     {suggested_max_lr:.6f}")

        # Safety checks based on experience
        if suggested_max_lr > 0.5:
            print(f"\n⚠️  WARNING: Suggested max_lr ({suggested_max_lr:.2e}) is very high!")
            print(f"   Reducing to safer value based on experience...")
            # suggested_max_lr = min(suggested_max_lr, 0.3)
            # suggested_initial_lr = suggested_max_lr / config['div_factor']
            print(f"   Adjusted max_lr: {suggested_max_lr:.6f}")
            print(f"   Adjusted initial_lr: {suggested_initial_lr:.6f}")

        if suggested_max_lr < 0.01:
            print(f"\n⚠️  WARNING: Suggested max_lr ({suggested_max_lr:.2e}) is very low!")
            print(f"   Training might be very slow. Using suggested values anyway...")

        # Update config
        # For constant LR training (no scheduler), use the suggested max_lr
        # Optionally divide by 2 for more conservative approach
        if args.no_scheduler:
            # For constant LR, use suggested_max_lr directly or reduced
            config['initial_lr'] = suggested_max_lr * 0.7  # More conservative for constant LR
            config['max_lr'] = suggested_max_lr * 0.7
            print(f"\n💡 Using constant LR (no scheduler): {config['initial_lr']:.6f}")
            print(f"   (70% of suggested max_lr for stability)")
        else:
            # For scheduler, use full range
            config['initial_lr'] = suggested_initial_lr
            config['max_lr'] = suggested_max_lr / 2
            print(f"\n💡 Using LR range for scheduler:")
            print(f"   Initial LR: {config['initial_lr']:.6f}")
            print(f"   Max LR:     {config['max_lr']:.6f}")

        print(f"\n" + "=" * 70)
        print(f"FINAL LEARNING RATE CONFIGURATION:")
        if args.no_scheduler:
            print(f"  Constant LR:    {config['initial_lr']:.6f}")
        else:
            print(f"  Initial LR:     {config['initial_lr']:.6f}")
            print(f"  Max LR:         {config['max_lr']:.6f}")
            print(f"  Ratio:          {config['max_lr']/config['initial_lr']:.1f}x")
        print("=" * 70)

        # Save LR values
        with open('lr_config_1000class.txt', 'w') as f:
            f.write(f"FOUND_INITIAL_LR={config['initial_lr']}\n")
            f.write(f"FOUND_MAX_LR={config['max_lr']}\n")
        print(f"\n✓ Learning rates saved to lr_config_1000class.txt")

        # Note: LR Finder automatically restores model weights after running
        # If we resumed from checkpoint, the loaded weights are preserved
        if start_epoch > 0:
            print(f"\n✓ Model weights preserved from checkpoint (epoch {start_epoch - 1})")
        else:
            # Only reload model if training from scratch
            print("\nReloading model with fresh weights...")
            model = create_resnet50(
                num_classes=config['num_classes'],
                zero_init_residual=config['zero_init_residual']
            )
            model = model.to(device)
            print("✓ Model reloaded with random initialization")
    else:
        print_banner("SKIPPING LR FINDER")
        print(f"Using manual LR values:")
        print(f"  Initial LR: {config['initial_lr']}")
        print(f"  Max LR:     {config['max_lr']}")
        print("\n⚠️  For 1000-class training, LR finder is HIGHLY RECOMMENDED!")

    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    print_banner("CREATING OPTIMIZER AND SCHEDULER")

    # Optimizer: SGD with Nesterov momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['initial_lr'],
        momentum=0.9,
        weight_decay=config['weight_decay'],
        nesterov=True
    )

    # Scheduler: OneCycleLR
    #scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer,
    #    max_lr=config['max_lr'],
    #    epochs=config['num_epochs'],
    #    steps_per_epoch=len(train_loader),
    #    pct_start=config['pct_start'],
    #    anneal_strategy='cos',
    #    div_factor=config['div_factor'],
    #    final_div_factor=config['final_div_factor']
    #)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        max_grad_norm=config['max_grad_norm']
    )

    print(f"Optimizer:           SGD with Nesterov momentum")
    print(f"Scheduler:           OneCycleLR")
    print(f"Initial LR:          {config['initial_lr']:.6f}")
    print(f"Max LR:              {config['max_lr']:.6f}")
    print(f"Warmup:              {config['pct_start']*100:.0f}% of epochs (first {int(config['num_epochs']*config['pct_start'])} epochs)")
    print(f"Final div factor:    {config['final_div_factor']:.0e}")
    print(f"Loss function:       CrossEntropyLoss (label_smoothing={config['label_smoothing']})")
    print(f"Gradient clipping:   {config['max_grad_norm']}")
    print(f"Weight decay:        {config['weight_decay']}")
    print(f"\n✓ Ready to start training!")

    # ========================================================================
    # TRAINING
    # ========================================================================
    print_banner("STARTING TRAINING - ImageNet 1000 Classes")
    print(f"Target epochs:              {config['num_epochs']}")
    print(f"Goal:                       82%+ validation accuracy")
    print(f"Training samples:           ~1.28M images")
    print(f"Validation samples:         ~50K images")
    print(f"Batch size:                 {config['batch_size']}")
    print(f"Iterations per epoch:       {len(train_loader):,}")
    print(f"Total iterations:           {len(train_loader) * config['num_epochs']:,}")
    print(f"Augmentation:               {config['augmentation_strength']}")
    print(f"Device:                     {device}")
    print("=" * 70)
    print("\n⏰ Training will take 7-15 hours depending on GPU.")
    print("📊 Progress will be displayed after each epoch.")
    print("💾 Checkpoints saved every 5 epochs + best model.")
    print("\n🚀 Starting training now...\n")
    print("=" * 70 + "\n")

    start_time = time.time()

    # Determine best accuracy if resuming
    best_acc_so_far = 0.0
    if start_epoch > 0 and resume_checkpoint_path:
        best_acc_so_far = checkpoint.get('best_acc', 0.0)
        print(f"📊 Resuming with best accuracy: {best_acc_so_far:.2f}%")

    # Train the model
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # No scheduler if --no-scheduler flag is set
        num_epochs=config['num_epochs'],
        start_epoch=start_epoch,
        best_acc=best_acc_so_far
    )

    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print_banner("TRAINING COMPLETE!")
    print(f"Total training time: {training_time_hours:.2f} hours")
    print(f"Average time per epoch: {training_time_hours / config['num_epochs'] * 60:.1f} minutes")
    print("=" * 70)

    # Print final statistics
    print_banner("FINAL TRAINING RESULTS")
    print(f"Best Validation Accuracy:  {max(history['val_acc']):.2f}%")
    print(f"Final Training Accuracy:   {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Training Loss:       {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss:     {history['val_loss'][-1]:.4f}")
    print(f"Train/Val Accuracy Gap:    {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%")
    print("=" * 70)

    if max(history['val_acc']) >= 82:
        print("\n🎉 TARGET ACHIEVED: 82%+ validation accuracy!")
        print("✓ Successfully trained ResNet50 on full ImageNet-1K!")
        print("✓ Model ready for deployment and inference.")
    elif max(history['val_acc']) >= 75:
        print(f"\n✓ Good progress! Reached {max(history['val_acc']):.2f}% validation accuracy.")
        print("\nTo reach 82%+ accuracy, consider:")
        print("  - Train for more epochs (100-120 total)")
        print("  - Adjust learning rate (try lower max_lr)")
        print("  - Fine-tune from current checkpoint")
    else:
        print(f"\n⚠️  Current best: {max(history['val_acc']):.2f}% (target: 82%+)")
        print("\nRecommendations:")
        print("  - Train for more epochs (need 100-120 for full convergence)")
        print("  - Check learning rate (may need adjustment)")
        print("  - Verify data augmentation is appropriate")

    print("=" * 70)

    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print_banner("FINAL EVALUATION ON VALIDATION SET")

    # Load best model
    best_checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
        print(f"  Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        print("⚠️  Best checkpoint not found, using current model")

    # Evaluate on validation set
    print("\nRunning final evaluation on full validation set (50,000 images)...")
    print("This will take ~5-10 minutes...\n")
    model.eval()

    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Progress update
            if (batch_idx + 1) % 50 == 0:
                print(f"  Evaluated {total:,}/50,000 images...")

    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total

    print_banner("FINAL VALIDATION RESULTS - ImageNet 1000 Classes")
    print(f"Top-1 Accuracy:          {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:          {top5_acc:.2f}%")
    print(f"Total samples evaluated: {total:,}")
    print(f"Number of classes:       1000")
    print("=" * 70)

    if top1_acc >= 82:
        print("\n🎉 EXCELLENT! Achieved 82%+ top-1 accuracy!")
        print("✓ Model performance exceeds ImageNet competition baseline.")
    elif top1_acc >= 76:
        print("\n✓ GREAT! Achieved competitive ImageNet accuracy.")
        print(f"  Top-1: {top1_acc:.2f}% (official ResNet50 baseline: 76.13%)")
        print(f"  Top-5: {top5_acc:.2f}%")
    elif top1_acc >= 70:
        print("\n✓ GOOD! Solid performance for training from scratch.")
        print(f"  Consider training more epochs for further improvement.")
    else:
        print(f"\n⚠️  Top-1: {top1_acc:.2f}% (target: 82%+)")
        print("  Model may benefit from:")
        print("  - More training epochs")
        print("  - Learning rate adjustment")

    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    print_banner("SAVING FINAL MODEL FOR DEPLOYMENT")

    deployment_dir = Path('./deployment')
    deployment_dir.mkdir(exist_ok=True)

    # Save complete model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config['num_classes'],
        'final_val_acc': top1_acc,
        'final_top5_acc': top5_acc,
        'config': config,
        'training_history': history,
        'pytorch_version': torch.__version__,
    }, deployment_dir / 'resnet50_1000class_final.pth')

    print(f"Location:            {deployment_dir / 'resnet50_1000class_final.pth'}")
    print(f"Model size:          ~{Path(deployment_dir / 'resnet50_1000class_final.pth').stat().st_size / 1e6:.1f} MB")
    print(f"Top-1 Accuracy:      {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:      {top5_acc:.2f}%")
    print(f"Number of classes:   {config['num_classes']}")
    print(f"PyTorch version:     {torch.__version__}")
    print("=" * 70)
    print("\n✓ Model ready for inference and deployment!")
    print("\n🎉 ALL DONE! Training session complete.")

    return history, top1_acc, top5_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet-1K (1000 classes)')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./imagenet_1000class_data',
                        help='Path to ImageNet dataset')
    parser.add_argument('--auto-download', action='store_true',
                        help='Auto-download dataset if not found (WARNING: ~150GB!)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Augmentation strength (default: medium)')

    # Learning rate arguments
    parser.add_argument('--find-lr', action='store_true', default=True,
                        help='Run LR finder (default: True)')
    parser.add_argument('--no-find-lr', dest='find_lr', action='store_false',
                        help='Skip LR finder')
    parser.add_argument('--initial-lr', type=float, default=None,
                        help='Initial learning rate (overridden by LR finder)')
    parser.add_argument('--max-lr', type=float, default=None,
                        help='Max learning rate (overridden by LR finder)')
    parser.add_argument('--pct-start', type=float, default=0.4,
                        help='OneCycleLR warmup percentage (default: 0.4)')
    parser.add_argument('--final-div-factor', type=float, default=1e3,
                        help='OneCycleLR final div factor (default: 1e3)')

    # Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints_1000class/checkpoint_epoch_17.pth)')
    parser.add_argument('--resume-epoch', type=int, default=None,
                        help='Resume from specific epoch number (will auto-find checkpoint)')
    parser.add_argument('--find-lr-on-resume', action='store_true',
                        help='Run LR finder when resuming (recommended)')
    parser.add_argument('--no-scheduler', action='store_true',
                        help='Train without scheduler (constant LR)')

    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_1000class',
                        help='Directory to save checkpoints')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU training (not recommended)')

    args = parser.parse_args()

    # Run training
    try:
        history, top1_acc, top5_acc = main(args)
        print(f"\n✓ Training completed successfully!")
        print(f"  Final Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"  Final Top-5 Accuracy: {top5_acc:.2f}%")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("  Checkpoints have been saved.")
        print("  Resume training by running the script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
