"""
Resume Aggressive Training - Multi-phase LR finder strategy
Goal: Reach 80%+ validation accuracy in 15 epochs (3 phases × 5 epochs)
Resume from epoch 15 checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Import existing modules
from model import create_resnet50
from data_loader_full import get_full_dataloaders
from lr_finder import LRFinder
from aggressive_utils import mixup_data, mixup_criterion, EMA, validate_with_tta


def setup_logging(log_dir='./logs_aggressive'):
    """Setup logging for aggressive training"""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f'aggressive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class AggressiveTrainer:
    """Multi-phase aggressive training strategy"""

    def __init__(self, model, train_loader, val_loader, device,
                 checkpoint_dir='./checkpoints_aggressive', logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

        # Training phases - REDUCED TO 5 EPOCHS EACH
        self.phases = [
            {
                'name': 'Phase 1: Fast Climb',
                'epochs': 5,
                'find_lr': True,
                'lr_range': (1e-6, 1.0),
                'num_iter': 1000,
                'mixup_alpha': 0.4,
                'use_tta': False,
                'use_ema': True,
                'target_acc': 68.0,
                'current_lr': 0.06368  # Known from training_log_3.txt
            },
            {
                'name': 'Phase 2: Consolidation',
                'epochs': 5,
                'find_lr': True,
                'lr_range': (1e-7, 0.5),
                'num_iter': 1000,
                'mixup_alpha': 0.3,
                'use_tta': True,
                'use_ema': True,
                'target_acc': 76.0,
                'current_lr': None  # Will be found
            },
            {
                'name': 'Phase 3: Fine-tuning',
                'epochs': 5,
                'find_lr': True,
                'lr_range': (1e-8, 0.2),
                'num_iter': 1000,
                'mixup_alpha': 0.2,
                'use_tta': True,
                'use_ema': True,
                'target_acc': 80.0,
                'current_lr': None  # Will be found
            }
        ]

    def find_optimal_lr(self, lr_range, num_iter):
        """Run LR finder and return optimal max_lr"""
        self.logger.info("🔍 Running LR Finder...")
        self.logger.info(f"   Range: {lr_range[0]:.2e} to {lr_range[1]:.2e}")
        self.logger.info(f"   Iterations: {num_iter}")

        # Create temporary optimizer
        temp_optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr_range[0],
            momentum=0.9,
            weight_decay=0.0001
        )

        criterion = nn.CrossEntropyLoss()

        lr_finder = LRFinder(
            self.model,
            temp_optimizer,
            criterion,
            self.device
        )

        # Run LR finder
        lrs, losses, initial_lr, max_lr = lr_finder.find(
            self.train_loader,
            init_lr=lr_range[0],
            end_lr=lr_range[1],
            num_iter=num_iter,
            smooth_factor=0.05,
            diverge_threshold=4.0
        )

        # Conservative adjustment: reduce by 20%
        max_lr_adjusted = max_lr * 0.8

        self.logger.info(f"✓ LR Finder Results:")
        self.logger.info(f"   Suggested max_lr: {max_lr:.6f}")
        self.logger.info(f"   Adjusted max_lr:  {max_lr_adjusted:.6f} (80% of suggested)")

        return max_lr_adjusted

    def train_epoch_with_mixup(self, optimizer, scheduler, criterion, ema,
                               mixup_alpha=0.4, epoch_num=0):
        """Train one epoch with mixup augmentation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} [Train]')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            # Apply mixup with 50% probability
            use_mixup = mixup_alpha > 0 and np.random.rand() < 0.5

            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, mixup_alpha, self.device
                )
                outputs = self.model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if use_mixup:
                correct += (lam * predicted.eq(targets_a).sum().item() +
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate_epoch(self, criterion, ema=None, use_tta=False, epoch_num=0):
        """Validate one epoch with optional EMA and TTA"""
        # Apply EMA weights if available
        if ema is not None:
            ema.apply_shadow()

        if use_tta:
            val_loss, val_acc, val_top5 = validate_with_tta(
                self.model, self.val_loader, criterion, self.device
            )
        else:
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            top5_correct = 0

            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f'Epoch {epoch_num} [Val]')

                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Top-5
                    _, top5_pred = outputs.topk(5, 1, True, True)
                    top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

                    pbar.set_postfix({
                        'loss': total_loss / (len(pbar) + 1),
                        'top1': 100. * correct / total,
                        'top5': 100. * top5_correct / total
                    })

            val_loss = total_loss / len(self.val_loader)
            val_acc = 100. * correct / total
            val_top5 = 100. * top5_correct / total

        # Restore original weights if EMA was used
        if ema is not None:
            ema.restore()

        return val_loss, val_acc, val_top5

    def run_phase(self, phase_config, start_epoch, phase_num):
        """Run a single training phase"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🚀 {phase_config['name']}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Epochs: {start_epoch} - {start_epoch + phase_config['epochs'] - 1}")
        self.logger.info(f"Target Accuracy: {phase_config['target_acc']}%")
        self.logger.info(f"Mixup Alpha: {phase_config['mixup_alpha']}")
        self.logger.info(f"Use TTA: {phase_config['use_tta']}")
        self.logger.info(f"Use EMA: {phase_config['use_ema']}")

        # Step 1: Find optimal LR (or use current if known)
        if phase_config['find_lr'] and phase_config['current_lr'] is None:
            max_lr = self.find_optimal_lr(
                phase_config['lr_range'],
                phase_config['num_iter']
            )
        else:
            # Use known LR or increase from current
            if phase_config['current_lr']:
                max_lr = phase_config['current_lr'] * 1.5  # Boost by 50%
                self.logger.info(f"📊 Using boosted LR: {max_lr:.6f} (1.5x current)")
            else:
                max_lr = 0.1  # Default fallback

        # Step 2: Setup optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=max_lr / 25,  # Start from max_lr / div_factor
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True
        )

        # Step 3: Setup OneCycleLR scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=phase_config['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # 30% warmup
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )

        # Step 4: Setup criterion
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Step 5: Setup EMA
        ema = EMA(self.model, decay=0.9999) if phase_config['use_ema'] else None

        # Step 6: Train for this phase
        best_acc = 0.0
        best_epoch = start_epoch

        for epoch_idx in range(phase_config['epochs']):
            current_epoch = start_epoch + epoch_idx
            current_lr = optimizer.param_groups[0]['lr']

            self.logger.info(f"\n📍 Epoch {current_epoch}/{start_epoch + phase_config['epochs'] - 1}")
            self.logger.info(f"   Learning Rate: {current_lr:.6f}")

            # Train
            train_loss, train_acc = self.train_epoch_with_mixup(
                optimizer, scheduler, criterion, ema,
                mixup_alpha=phase_config['mixup_alpha'],
                epoch_num=current_epoch
            )

            # Validate
            val_loss, val_acc, val_top5 = self.validate_epoch(
                criterion, ema=ema, use_tta=phase_config['use_tta'],
                epoch_num=current_epoch
            )

            # Log results
            self.logger.info(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            self.logger.info(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Top-5={val_top5:.2f}%")

            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = current_epoch
                self.save_checkpoint(current_epoch, val_acc, optimizer, scheduler, phase_num)

            # Early stopping if target reached
            if val_acc >= phase_config['target_acc']:
                self.logger.info(f"\n🎉 Target accuracy {phase_config['target_acc']}% REACHED!")
                self.logger.info(f"   Stopping phase early at epoch {current_epoch}")
                break

        self.logger.info(f"\n✓ Phase {phase_num} Complete!")
        self.logger.info(f"   Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")

        return best_acc, best_epoch

    def save_checkpoint(self, epoch, val_acc, optimizer, scheduler, phase_num):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'phase': phase_num
        }

        # Save latest
        latest_path = self.checkpoint_dir / f'phase{phase_num}_latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        best_path = self.checkpoint_dir / f'phase{phase_num}_best_acc{val_acc:.2f}.pth'
        torch.save(checkpoint, best_path)

        self.logger.info(f"   💾 Saved checkpoint: {best_path.name}")

    def run_all_phases(self, start_epoch=16):
        """Run all training phases"""
        current_epoch = start_epoch
        overall_best = 0.0

        self.logger.info("\n" + "="*70)
        self.logger.info("🔥 AGGRESSIVE TRAINING STRATEGY - MULTI-PHASE")
        self.logger.info("="*70)
        self.logger.info(f"Goal: 80%+ validation accuracy in 15 epochs")
        self.logger.info(f"Strategy: 3 phases × 5 epochs with iterative LR finding")
        self.logger.info(f"Start from: Epoch {start_epoch}")
        self.logger.info("="*70)

        for phase_num, phase_config in enumerate(self.phases, start=1):
            best_acc, best_epoch = self.run_phase(phase_config, current_epoch, phase_num)
            current_epoch += phase_config['epochs']

            if best_acc > overall_best:
                overall_best = best_acc

            # Update next phase's current_lr based on this phase's best
            if phase_num < len(self.phases):
                self.phases[phase_num]['current_lr'] = None  # Force LR finder

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ AGGRESSIVE TRAINING COMPLETE!")
        self.logger.info("="*70)
        self.logger.info(f"Overall Best Accuracy: {overall_best:.2f}%")
        self.logger.info(f"Final Epoch: {current_epoch - 1}")

        if overall_best >= 80.0:
            self.logger.info("🎯 TARGET ACHIEVED: 80%+ accuracy!")
        else:
            self.logger.info(f"Progress: {overall_best:.2f}% (Target: 80%)")

        return overall_best


def main():
    """Main execution"""
    # Setup
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    checkpoint_path = './checkpoints_1000class/best_model.pth'
    logger.info(f"\n📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_resnet50(num_classes=1000, zero_init_residual=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    logger.info(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"   Previous best accuracy: {checkpoint.get('best_acc', 'N/A')}")

    # Load data
    logger.info("\n📦 Loading ImageNet-1K dataset...")
    train_loader, val_loader, num_classes, class_names = get_full_dataloaders(
        data_dir='./imagenet_1000class_data',
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        augmentation_strength='medium',
        advanced_augmentation=True,
        distributed=False,
        auto_download=False
    )

    logger.info(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create aggressive trainer
    trainer = AggressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir='./checkpoints_aggressive',
        logger=logger
    )

    # Run aggressive training
    final_acc = trainer.run_all_phases(start_epoch=17)

    logger.info(f"\n🏁 Training finished with {final_acc:.2f}% validation accuracy")


if __name__ == "__main__":
    main()
