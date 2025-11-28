#!/usr/bin/env python3
"""
Add 'model.' prefix to all tensor names in a safetensors file.

This is needed for Candle compatibility, as Candle's Gemma implementation
expects tensors with 'model.' prefix (e.g., 'model.embed_tokens.weight')
while PyTorch checkpoints don't have this prefix (e.g., 'embed_tokens.weight').
"""

import argparse
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def add_model_prefix(input_path: str, output_path: str):
    """Add 'model.' prefix to all tensor names."""
    tensors = {}

    print(f"Loading tensors from {input_path}...")
    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            new_key = f"model.{key}"
            tensors[new_key] = tensor
            print(f"  {key} -> {new_key}")

    print(f"\nSaving {len(tensors)} tensors to {output_path}...")
    save_file(tensors, output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Add 'model.' prefix to safetensors file")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("output", help="Output safetensors file")

    args = parser.parse_args()
    add_model_prefix(args.input, args.output)

if __name__ == "__main__":
    main()
