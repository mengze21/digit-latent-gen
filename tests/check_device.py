#!/usr/bin/env python3
"""Simple script to check current PyTorch device support."""

import torch


def get_available_devices():
    """Return a dictionary of available devices."""
    devices = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "mps": False,
        "cuda_count": 0,
        "cuda_names": [],
    }

    # Check CUDA
    if torch.cuda.is_available():
        devices["cuda"] = True
        devices["cuda_count"] = torch.cuda.device_count()
        devices["cuda_names"] = [
            torch.cuda.get_device_name(i) for i in range(devices["cuda_count"])
        ]

    # Check MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices["mps"] = True

    return devices


def print_device_info():
    """Print information about available devices."""
    print("PyTorch Device Information")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")

    devices = get_available_devices()

    print("\nAvailable devices:")
    print(f"  CPU: Available")

    if devices["cuda"]:
        print(f"  CUDA: Available ({devices['cuda_count']} GPU(s))")
        for i, name in enumerate(devices["cuda_names"]):
            print(f"    GPU {i}: {name}")
    else:
        print(f"  CUDA: Not available")

    if devices["mps"]:
        print(f"  MPS: Available (Apple Silicon)")
    else:
        print(f"  MPS: Not available")

    # Determine best device
    print("\nBest device for training:")
    if devices["cuda"]:
        print("  CUDA (NVIDIA GPU) - Best for deep learning")
    elif devices["mps"]:
        print("  MPS (Apple Silicon) - Good for Apple devices")
    else:
        print("  CPU - Slowest option, use for small models")

    # Current device
    print(
        f"\nCurrent default device: {torch.device('cuda' if devices['cuda'] else 'mps' if devices['mps'] else 'cpu')}"
    )


if __name__ == "__main__":
    print_device_info()
