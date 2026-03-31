"""Backward-compatible entrypoint for VAE generation.

Prefer running `python scripts/generate_vae.py` directly.
"""

from generate_vae import main

if __name__ == "__main__":
    main()
