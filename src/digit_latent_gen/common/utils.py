import torch
from torchvision import datasets, transforms


def get_mnist_dataset(train=True, root_dir="data"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    return datasets.MNIST(
        root=root_dir,
        train=train,
        download=True,
        transform=transform
    )


def get_transform(normalize_mean=(0.1307,), normalize_std=(0.3081,)):
    """Get a standard transform pipeline with configurable normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])


def compute_dataset_stats(dataset, batch_size=128, num_workers=4):
    """Compute mean and std of a dataset for normalization."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    
    mean = None
    std = None
    total_images = 0
    
    with torch.no_grad():
        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.reshape(batch_samples, data.size(1), -1)
            batch_mean = data.mean(2).sum(0)
            batch_std = data.std(2).sum(0)

            if mean is None or std is None:
                mean = torch.zeros_like(batch_mean)
                std = torch.zeros_like(batch_std)

            mean += batch_mean
            std += batch_std
            total_images += batch_samples

    if mean is None or std is None:
        raise ValueError("Dataset is empty, cannot compute mean and std.")
    
    mean /= total_images
    std /= total_images
    
    return mean.tolist(), std.tolist()
