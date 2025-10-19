"""
Data Normalization and Transforms for ImageNet
Includes training and validation transforms with augmentation
"""

import torchvision.transforms as transforms


def get_imagenet_normalization():
    """
    Get ImageNet standard normalization

    Returns:
        transforms.Normalize: Normalization transform
    """
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def get_train_transforms(advanced_augmentation=True, augmentation_strength='medium'):
    """
    Get training transforms with augmentation

    Args:
        advanced_augmentation (bool): Use advanced augmentation techniques
        augmentation_strength (str): 'light', 'medium', or 'heavy' augmentation

    Returns:
        transforms.Compose: Training transforms
    """
    normalize = get_imagenet_normalization()

    if advanced_augmentation:
        if augmentation_strength == 'light':
            # Light augmentation - better for small datasets
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # Less aggressive crop
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Subtle color changes
                transforms.ToTensor(),
                normalize,
            ])
        elif augmentation_strength == 'medium':
            # Medium augmentation - balanced
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))  # Reduced erasing
            ])
        else:  # heavy
            # Heavy augmentation - for large datasets only
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
            ])
    else:
        # Basic augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    return train_transform


def get_val_transforms():
    """
    Get validation transforms (no augmentation)

    Returns:
        transforms.Compose: Validation transforms
    """
    normalize = get_imagenet_normalization()

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return val_transform


def get_test_transforms():
    """
    Get test transforms (same as validation)

    Returns:
        transforms.Compose: Test transforms
    """
    return get_val_transforms()


def denormalize_image(tensor):
    """
    Denormalize an ImageNet normalized tensor for visualization

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]

    Returns:
        Denormalized tensor
    """
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    if tensor.dim() == 4:
        # Batch of images
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denorm_tensor = tensor * std + mean
    return denorm_tensor.clamp(0, 1)


if __name__ == "__main__":
    import torch
    from PIL import Image
    import numpy as np

    print("="*60)
    print("Data Normalization Test")
    print("="*60)

    # Test transforms
    train_transform = get_train_transforms(advanced_augmentation=True)
    val_transform = get_val_transforms()

    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    # Apply transforms
    train_tensor = train_transform(dummy_img)
    val_tensor = val_transform(dummy_img)

    print(f"\nTrain transform output shape: {train_tensor.shape}")
    print(f"Val transform output shape: {val_tensor.shape}")

    print(f"\nTrain tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    print(f"Val tensor range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")

    # Test denormalization
    denorm_tensor = denormalize_image(train_tensor)
    print(f"\nDenormalized tensor range: [{denorm_tensor.min():.3f}, {denorm_tensor.max():.3f}]")

    print("\n✓ Normalization test passed!")
