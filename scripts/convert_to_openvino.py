#!/usr/bin/env python3
"""
Convert ONNX models to OpenVINO IR format
"""

import argparse
import yaml
from pathlib import Path


def convert_to_openvino(
    onnx_path: Path,
    openvino_output_dir: Path,
    precision: str = "FP32"
):
    """
    Convert ONNX model to OpenVINO IR format

    Args:
        onnx_path: Path to ONNX model
        openvino_output_dir: Output directory for OpenVINO IR
        precision: Model precision (FP32, FP16, INT8)
    """
    try:
        from openvino.tools import mo
    except ImportError:
        print("Error: OpenVINO tools not installed")
        print("Install with: pip install openvino-dev")
        return 1

    print(f"Converting ONNX to OpenVINO IR")
    print(f"  Input: {onnx_path}")
    print(f"  Output: {openvino_output_dir}")
    print(f"  Precision: {precision}")

    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        return 1

    # Create output directory
    openvino_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert using Model Optimizer
    output_model = openvino_output_dir / "model"

    # Run conversion
    mo_args = {
        "input_model": str(onnx_path),
        "output_dir": str(openvino_output_dir),
        "model_name": "model",
        "compress_to_fp16": precision == "FP16"
    }

    print("\nRunning OpenVINO Model Optimizer...")
    mo.convert_model(**mo_args)

    # Copy tokenizer from ONNX directory
    onnx_dir = onnx_path.parent
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt"
    ]

    print("\nCopying tokenizer files...")
    for filename in tokenizer_files:
        src = onnx_dir / filename
        if src.exists():
            dst = openvino_output_dir / filename
            import shutil
            shutil.copy2(src, dst)
            print(f"  ✓ {filename}")

    print(f"\n✓ OpenVINO conversion successful!")
    print(f"  Model: {openvino_output_dir / 'model.xml'}")
    print(f"  Weights: {openvino_output_dir / 'model.bin'}")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to OpenVINO')
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
        '--precision',
        type=str,
        default='FP32',
        choices=['FP32', 'FP16'],
        help='Model precision (default: FP32)'
    )

    args = parser.parse_args()

    # Load model config
    with open(args.config) as f:
        models_config = yaml.safe_load(f)

    if args.model not in models_config['models']:
        print(f"Error: Model '{args.model}' not found in config")
        return 1

    model_config = models_config['models'][args.model]

    # Paths - handle both container paths (/models/...) and local paths (models/...)
    onnx_path = Path(model_config['paths']['onnx'])
    openvino_dir = Path(model_config['paths']['openvino'])

    # If container path doesn't exist, try local path
    if not onnx_path.exists():
        local_onnx_path = Path(f"models/{args.model}/onnx/model.onnx")
        if local_onnx_path.exists():
            onnx_path = local_onnx_path
            openvino_dir = Path(f"models/{args.model}/openvino")

    convert_to_openvino(
        onnx_path=onnx_path,
        openvino_output_dir=openvino_dir,
        precision=args.precision
    )


if __name__ == '__main__':
    main()
