#!/usr/bin/env python3
"""
Download and cache embedding models from HuggingFace
"""

import argparse
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer


def download_model(model_id: str, output_dir: Path):
    """Download model from HuggingFace"""
    print(f"Downloading model: {model_id}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download using sentence-transformers (caches automatically)
    model = SentenceTransformer(model_id)

    # Save to specific directory
    model.save(str(output_dir))

    print(f"✓ Model downloaded and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download embedding models')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model key from models.yaml'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/models.yaml',
        help='Path to models.yaml'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Load model config
    with open(args.config) as f:
        models_config = yaml.safe_load(f)

    if args.model not in models_config['models']:
        print(f"Error: Model '{args.model}' not found in config")
        return 1

    model_config = models_config['models'][args.model]
    model_id = model_config['huggingface_id']

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"models/{args.model}/pytorch")

    download_model(model_id, output_dir)


if __name__ == '__main__':
    main()
