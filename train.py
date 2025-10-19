"""
Training Module for ResNet50 on ImageNet
Includes Trainer class with all training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import logging
from typing import Optional

# Import autocast with version compatibility
try:
    from torch.amp import autocast as amp_autocast, GradScaler
    HAS_UNIFIED_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as amp_autocast, GradScaler
    HAS_UNIFIED_AMP = False


class Trainer:
    """Main training class with all features"""

    def __init__(self, model, device, checkpoint_dir='checkpoints', max_grad_norm: Optional[float] = None):
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_grad_norm = max_grad_norm

        # Setup logging
        self.setup_logging()

        # Training history
        self.history = defaultdict(list)
        self.problem_images = defaultdict(list)

        # Determine device type for autocast and AMP
        if torch.cuda.is_available():
            self.amp_device = 'cuda'
            self.use_amp = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.amp_device = 'cpu'
            self.use_amp = False  # Disable AMP for MPS
        else:
            self.amp_device = 'cpu'
            self.use_amp = False

        # Mixed precision training scaler (only for CUDA)
        if self.use_amp:
            if HAS_UNIFIED_AMP:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        self.logger.info(f"Trainer initialized on device: {device}")
        self.logger.info(f"Mixed precision (AMP) enabled: {self.use_amp}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.checkpoint_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, model, loader, criterion, optimizer, epoch, scheduler=None):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            # Mixed precision training (only on CUDA)
            if self.use_amp and HAS_UNIFIED_AMP:
                with amp_autocast(device_type=self.amp_device):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            elif self.use_amp:
                with amp_autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with gradient scaling (only on CUDA)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()

            # Learning rate scheduler (per-batch stepping when supported)
            if scheduler is not None:
                scheduler_name = scheduler.__class__.__name__
                if scheduler_name == "OneCycleLR":
                    scheduler.step()
                elif scheduler_name == "CosineAnnealingWarmRestarts":
                    scheduler.step(epoch + batch_idx / len(loader))

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Track problematic images (high loss)
            if loss.item() > 5.0:
                for i in range(len(targets)):
                    single_loss = F.cross_entropy(outputs[i:i+1], targets[i:i+1]).item()
                    if single_loss > 5.0:
                        self.problem_images[epoch].append({
                            'batch_idx': batch_idx,
                            'image_idx': i,
                            'true_label': targets[i].item(),
                            'predicted_label': predicted[i].item(),
                            'loss': single_loss
                        })

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return running_loss / len(loader), 100. * correct / total

    def validate(self, model, loader, criterion, epoch):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        top5_correct = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'top1_acc': 100. * correct / total,
                    'top5_acc': 100. * top5_correct / total
                })

        return running_loss / len(loader), 100. * correct / total, 100. * top5_correct / total

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_acc, is_best=False,
                       dataset_info=None, training_phase=None):
        """Save training checkpoint with phase tracking"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'history': dict(self.history),
            'problem_images': dict(self.problem_images),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            # New fields for incremental training
            'dataset_info': dataset_info or {},
            'training_phase': training_phase or 'unknown',
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

            model_only_path = self.checkpoint_dir / 'best_model_weights.pth'
            torch.save(model.state_dict(), model_only_path)

            self.logger.info(f"Saved best model with accuracy: {best_acc:.2f}%")

            # Save metrics
            metrics = {
                'best_accuracy': best_acc,
                'best_epoch': epoch,
                'total_epochs_trained': epoch + 1,
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
                'best_top5_accuracy': max(self.history.get('val_top5_acc', [0])),
                'timestamp': datetime.now().isoformat()
            }

            with open(self.checkpoint_dir / 'best_model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        # Keep only last 3 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None,
                       load_optimizer_state=True):
        """
        Load training checkpoint

        Args:
            load_optimizer_state: If False, skip loading optimizer/scheduler (use for data size changes)
        """
        if checkpoint_path is None:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if not checkpoints:
                return None, 0, 0
            checkpoint_path = checkpoints[-1]

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optionally load optimizer/scheduler state
        if load_optimizer_state:
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✓ Loaded optimizer state from checkpoint")

            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("✓ Loaded scheduler state from checkpoint")
        else:
            self.logger.info("⚠️  Skipped optimizer/scheduler loading (fresh LR schedule)")

        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.problem_images = defaultdict(list, checkpoint.get('problem_images', {}))

        # Log checkpoint info
        self.logger.info(f"📂 Loaded checkpoint from epoch {checkpoint['epoch']} with best acc {checkpoint['best_acc']:.2f}%")

        # Log dataset info if available
        if 'dataset_info' in checkpoint:
            dataset_info = checkpoint['dataset_info']
            self.logger.info(f"📊 Previous dataset: {dataset_info}")

        # Log training phase if available
        if 'training_phase' in checkpoint:
            self.logger.info(f"🔄 Previous phase: {checkpoint['training_phase']}")

        return checkpoint, checkpoint['epoch'], checkpoint['best_acc']

    def train(self, model, train_loader, val_loader, criterion, optimizer, scheduler,
              num_epochs, start_epoch=0, best_acc=0.0):
        """
        Main training loop

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            best_acc: Best accuracy so far (for resuming)

        Returns:
            tuple: (model, best_acc, history)
        """
        self.logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")

        epoch_times = []

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            self.logger.info(f"\nEpoch {epoch}/{num_epochs-1}, LR: {current_lr:.2e}")

            per_batch_scheduler = scheduler is not None and scheduler.__class__.__name__ in {
                "OneCycleLR",
                "CosineAnnealingWarmRestarts"
            }

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch,
                scheduler if per_batch_scheduler else None
            )

            # Validate
            val_loss, val_acc, val_top5_acc = self.validate(
                model, val_loader, criterion, epoch
            )

            # Track epoch time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            self.history['epoch_time'] = epoch_times

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_top5_acc'].append(val_top5_acc)

            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Top-5 Acc: {val_top5_acc:.2f}%")
            self.logger.info(f"Epoch time: {epoch_time:.2f}s")

            # Save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc

            self.save_checkpoint(model, optimizer, scheduler, epoch, best_acc, is_best)

            # Update learning rate
            if scheduler and not per_batch_scheduler:
                scheduler.step()

            # Record final LR for this epoch
            self.history['lr'][-1] = optimizer.param_groups[0]['lr']

        self.logger.info(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
        return model, best_acc, dict(self.history)
    
    @torch.no_grad()
    def _num_batches(self, loader, max_batches):
        return min(len(loader), max_batches) if max_batches else len(loader)

    def lr_find(self, model, loader, criterion, optimizer, 
                start_lr=1e-6, end_lr=1.0, num_iter=200, 
                beta=0.98, max_loss_increase=4.0):
        """
        Returns: dict with 'suggested', 'lr', 'losses'
        """
        was_training = model.training
        model.train()
        if self.use_amp:
            raise RuntimeError("Run LR finder without AMP on CUDA; on MPS keep AMP disabled.")

        # Save initial states
        init_state = {
            'model': {k: v.clone() for k, v in model.state_dict().items()},
            'optim': optimizer.state_dict()
        }

        # Exponential LR schedule over num_iter steps
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))
        for pg in optimizer.param_groups:
            pg['lr'] = start_lr

        avg_loss, best_loss = 0.0, float('inf')
        losses, lrs = [], []
        iter_count = 0

        data_iter = iter(loader)
        while iter_count < num_iter:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Smooth loss for stability
            iter_count += 1
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed = avg_loss / (1 - beta ** iter_count)

            # Track
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr); losses.append(smoothed)

            # Book-keeping for early stop
            if smoothed < best_loss:
                best_loss = smoothed
            if smoothed > max_loss_increase * best_loss:
                break

            # Increase LR
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_mult

        # Restore model/optimizer
        model.load_state_dict(init_state['model'])
        optimizer.load_state_dict(init_state['optim'])
        if not was_training: model.eval()

        # Pick LR at steepest negative gradient in log-space
        # (simple heuristic: loss[i-1]-loss[i+1] max)
        import numpy as np
        log_lrs = np.log10(np.array(lrs))
        losses_np = np.array(losses)
        grad = np.gradient(losses_np, log_lrs)
        idx = np.argmin(grad[5:-5]) + 5  # avoid edges
        suggested = float(lrs[idx])

        return {
            'suggested': suggested,
            'lrs': lrs,
            'losses': losses,
            'best_loss': best_loss
        }

if __name__ == "__main__":
    print("="*60)
    print("Trainer Module Test")
    print("="*60)

    # Simple test with dummy model and data
    from model import create_resnet50

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create dummy data loader
    dummy_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,))) for _ in range(5)]
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(
        torch.randn(20, 3, 224, 224),
        torch.randint(0, 10, (20,))
    )
    dummy_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

    # Test trainer
    trainer = Trainer(model, device, checkpoint_dir='test_checkpoints')

    print("\nRunning 2 test epochs...")
    model, best_acc, history = trainer.train(
        model, dummy_loader, dummy_loader,
        criterion, optimizer, None,
        num_epochs=2, start_epoch=0
    )

    print(f"\n✓ Training test passed!")
    print(f"  Best accuracy: {best_acc:.2f}%")
    print(f"  History keys: {list(history.keys())}")

    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoints')
    
     
