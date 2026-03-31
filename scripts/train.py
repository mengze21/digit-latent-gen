"""Backward-compatible entrypoint for VAE training.

Prefer running `python scripts/train_vae.py` directly.
"""

from train_vae import main

if __name__ == "__main__":
    main()
