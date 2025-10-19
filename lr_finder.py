"""
Learning Rate Finder for Optimal LR Detection
Implements the LR range test to find the best learning rate
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


class LRFinder:
    """Learning rate finder using exponential increase method"""

    def __init__(self, model, optimizer, criterion, device):
        """
        Initialize LR Finder

        Args:
            model: PyTorch model
            optimizer: Optimizer (will be reset after finding)
            criterion: Loss function
            device: Device to run on (cuda/mps/cpu)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store original state
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def find(self, data_loader, init_lr=1e-8, end_lr=10, num_iter=100,
             smooth_factor=0.05, diverge_threshold=5):
        """
        Find optimal learning rate using exponential increase

        Args:
            data_loader: Training data loader
            init_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            smooth_factor: Smoothing factor for loss (0-1)
            diverge_threshold: Stop if loss > diverge_threshold * best_loss

        Returns:
            tuple: (learning_rates, losses, suggested_lr)
        """
        print(f"\n{'='*60}")
        print("Learning Rate Finder")
        print(f"{'='*60}")
        print(f"Testing range: {init_lr:.2e} to {end_lr:.2e}")
        print(f"Number of iterations: {num_iter}")
        print(f"{'='*60}\n")

        # Set model to training mode
        self.model.train()

        # Calculate learning rate multiplier
        lr_mult = (end_lr / init_lr) ** (1 / num_iter)

        # Initialize tracking variables
        lr = init_lr
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        lrs = []

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Create iterator
        iterator = iter(data_loader)

        # Progress bar
        pbar = tqdm(range(num_iter), desc='Finding LR')

        for iteration in pbar:
            batch_num += 1

            # Get batch
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                inputs, targets = next(iterator)

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Compute smoothed loss
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
            smoothed_loss = avg_loss / (1 - (1 - smooth_factor) ** batch_num)

            # Stop if loss is exploding
            if batch_num > 1 and smoothed_loss > diverge_threshold * best_loss:
                print(f"\nStopping early - loss is diverging")
                print(f"Smoothed loss: {smoothed_loss:.4f}, Best loss: {best_loss:.4f}")
                break

            # Record best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store values
            losses.append(smoothed_loss)
            lrs.append(lr)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Update progress bar
            pbar.set_postfix({'lr': f'{lrs[-1]:.2e}', 'loss': f'{losses[-1]:.4f}'})

        print(f"\n✓ LR finding complete!")
        print(f"  Tested {len(lrs)} learning rates")
        print(f"  Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

        # Find suggested LR (steepest gradient point)
        suggested_lr = self._suggest_lr(lrs, losses)

        # Restore original model and optimizer state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        print(f"\n✓ Model and optimizer restored to original state\n")

        return lrs, losses, suggested_lr

    def _suggest_lr(self, lrs, losses):
        """
        Suggest optimal learning rate based on gradient

        Args:
            lrs: List of learning rates
            losses: List of losses

        Returns:
            float: Suggested learning rate
        """
        # Find steepest gradient (maximum negative gradient)
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]

        # Alternative: Use point where loss is minimum
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]

        # Use the steepest gradient method (more conservative)
        return suggested_lr

    def plot(self, lrs, losses, suggested_lr=None, skip_start=10, skip_end=5,
             save_path=None, show=True):
        """
        Plot learning rate vs loss

        Args:
            lrs: List of learning rates
            losses: List of losses
            suggested_lr: Suggested learning rate to mark on plot
            skip_start: Number of initial points to skip
            skip_end: Number of final points to skip
            save_path: Path to save plot (optional)
            show: Whether to show plot
        """
        # Skip some points for better visualization
        if skip_start > 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        if skip_end > 0:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Log scale
        ax1.plot(lrs, losses, linewidth=2, color='blue')
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate (log scale)', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Learning Rate Finder - Log Scale', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        if suggested_lr:
            ax1.axvline(x=suggested_lr, color='red', linestyle='--',
                       label=f'Suggested LR: {suggested_lr:.2e}', linewidth=2)
            ax1.legend(fontsize=10)

        # Plot 2: Linear scale (zoomed in)
        ax2.plot(lrs, losses, linewidth=2, color='green')
        ax2.set_xlabel('Learning Rate', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Learning Rate Finder - Linear Scale', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        if suggested_lr:
            ax2.axvline(x=suggested_lr, color='red', linestyle='--',
                       label=f'Suggested LR: {suggested_lr:.2e}', linewidth=2)
            ax2.legend(fontsize=10)

            # Add text annotation
            suggested_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - suggested_lr))
            ax1.annotate(f'{suggested_lr:.2e}',
                        xy=(suggested_lr, losses[suggested_idx]),
                        xytext=(10, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def get_summary(self, lrs, losses, suggested_lr):
        """
        Get summary statistics

        Args:
            lrs: List of learning rates
            losses: List of losses
            suggested_lr: Suggested learning rate

        Returns:
            dict: Summary statistics
        """
        min_loss_idx = np.argmin(losses)

        summary = {
            'suggested_lr': suggested_lr,
            'min_loss': losses[min_loss_idx],
            'min_loss_lr': lrs[min_loss_idx],
            'lr_range_tested': (lrs[0], lrs[-1]),
            'num_iterations': len(lrs),
            'recommendation': self._get_recommendation(suggested_lr, lrs[min_loss_idx])
        }

        return summary

    def _get_recommendation(self, suggested_lr, min_loss_lr):
        """Get recommendation message"""
        recommendation = []

        recommendation.append(f"Suggested LR: {suggested_lr:.2e}")
        recommendation.append(f"Minimum loss LR: {min_loss_lr:.2e}")
        recommendation.append("")
        recommendation.append("Recommendations:")
        recommendation.append(f"  - Start with: {suggested_lr:.2e}")
        recommendation.append(f"  - Max LR (OneCycleLR): {min_loss_lr:.2e}")
        recommendation.append(f"  - Conservative: {suggested_lr/10:.2e}")
        recommendation.append("")
        recommendation.append("Typical usage:")
        recommendation.append("  optimizer = SGD(model.parameters(), lr=suggested_lr)")
        recommendation.append("  scheduler = OneCycleLR(optimizer, max_lr=min_loss_lr, ...)")

        return "\n".join(recommendation)


def find_lr_quick(model, train_loader, criterion, device,
                  init_lr=1e-8, end_lr=10, num_iter=100):
    """
    Quick helper function to find learning rate

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        device: Device to run on
        init_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations

    Returns:
        tuple: (lrs, losses, suggested_lr, fig)
    """
    # Create temporary optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device)

    # Find LR
    lrs, losses, suggested_lr = lr_finder.find(
        train_loader, init_lr, end_lr, num_iter
    )

    # Plot
    fig = lr_finder.plot(lrs, losses, suggested_lr)

    # Print summary
    summary = lr_finder.get_summary(lrs, losses, suggested_lr)
    print(f"\n{'='*60}")
    print("LR Finder Summary")
    print(f"{'='*60}")
    print(summary['recommendation'])
    print(f"{'='*60}\n")

    return lrs, losses, suggested_lr, fig


if __name__ == "__main__":
    print("="*60)
    print("LR Finder Module Test")
    print("="*60)

    # Create dummy model and data
    from model import create_resnet50
    import torch.nn as nn

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Create dummy data loader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 10, (100,))
    )
    dummy_loader = DataLoader(dummy_dataset, batch_size=10, shuffle=True)

    print("\nRunning LR finder test...")

    # Test LR finder
    lrs, losses, suggested_lr, fig = find_lr_quick(
        model, dummy_loader, criterion, device,
        init_lr=1e-7, end_lr=1, num_iter=50
    )

    print(f"✓ LR Finder test passed!")
    print(f"  Suggested LR: {suggested_lr:.2e}")
    print(f"  Tested {len(lrs)} learning rates")
