#!/usr/bin/env python3
"""Test script to detect available devices (MPS, CUDA, CPU)."""

import torch
import sys


def detect_device():
    """Detect and return the best available device."""
    devices = []

    # Check CUDA
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        devices.append(("CUDA", f"cuda:0 (GPUs: {cuda_count})"))
        for i in range(cuda_count):
            devices.append(("CUDA", f"  GPU {i}: {torch.cuda.get_device_name(i)}"))

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(("MPS", "mps"))

    # CPU is always available
    devices.append(("CPU", "cpu"))

    return devices


def test_device_performance(device_str):
    """Test basic tensor operations on the specified device."""
    print(f"\nTesting performance on {device_str}...")

    try:
        device = torch.device(device_str)

        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Test matrix multiplication
        import time

        start_time = time.time()
        z = torch.matmul(x, y)
        elapsed = time.time() - start_time

        print(f"  Matrix multiplication (1000x1000): {elapsed:.4f} seconds")
        print(f"  Result shape: {z.shape}")
        print(f"  Result mean: {z.mean().item():.4f}")

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Device Detection Test")
    print("=" * 60)

    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Detect available devices
    print("\nAvailable devices:")
    devices = detect_device()

    for device_type, device_info in devices:
        print(f"  {device_type}: {device_info}")

    # Determine best device
    print("\nBest available device:")
    if torch.cuda.is_available():
        print("  CUDA is available and will be used by default")
        best_device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  MPS (Apple Silicon) is available")
        best_device = "mps"
    else:
        print("  Only CPU is available")
        best_device = "cpu"

    print(f"\nDefault device would be: {best_device}")

    # Test performance on each available device
    print("\n" + "=" * 60)
    print("Performance Tests")
    print("=" * 60)

    test_devices = []
    if torch.cuda.is_available():
        test_devices.append("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        test_devices.append("mps")
    test_devices.append("cpu")

    for device_str in test_devices:
        test_device_performance(device_str)

    # Test current device from config
    print("\n" + "=" * 60)
    print("Testing with config file settings")
    print("=" * 60)

    try:
        import yaml

        config_path = "configs/vae_config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        device_from_config = config["training"].get("device", "auto")
        print(f"Device from config: '{device_from_config}'")

        if device_from_config == "auto":
            if torch.cuda.is_available():
                actual_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                actual_device = "mps"
            else:
                actual_device = "cpu"
            print(f"  Auto-detected device: {actual_device}")
        else:
            actual_device = device_from_config
            print(f"  Using configured device: {actual_device}")

            # Test if configured device is available
            if actual_device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print(
                        f"  WARNING: {actual_device} is configured but not available!"
                    )
            elif actual_device == "mps":
                if not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    print(
                        f"  WARNING: {actual_device} is configured but not available!"
                    )

    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
    except Exception as e:
        print(f"Error reading config: {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("To change device in training:")
    print("1. Edit configs/vae_config.yaml")
    print("2. Set 'device' under 'training' section to:")
    print("   - 'cuda' for NVIDIA GPU")
    print("   - 'mps' for Apple Silicon")
    print("   - 'cpu' for CPU only")
    print("   - 'auto' for automatic detection (default)")


if __name__ == "__main__":
    main()
