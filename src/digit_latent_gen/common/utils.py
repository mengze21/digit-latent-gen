import torch
from torchvision import datasets, transforms


def get_mnist_dataset(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return datasets.MNIST(
        root='./data',
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
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    with torch.no_grad():
        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    return mean.tolist(), std.tolist()
