#!/usr/bin/env python3
"""
Convert PyTorch embedding models to ONNX format
"""

import argparse
import yaml
import torch
import onnx
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Try to import optimum for better ONNX export support
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from optimum.exporters.onnx import main_export
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False
    print("Warning: optimum not installed. Install with: pip install optimum[onnxruntime]")


def export_with_optimum(
    model_id: str,
    onnx_output_path: Path,
    opset_version: int = 14
):
    """
    Export using Hugging Face Optimum (handles complex models like Gemma better)
    """
    print(f"Exporting with Optimum: {model_id}")

    output_dir = onnx_output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use optimum's export which handles model-specific quirks
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="feature-extraction",
        opset=opset_version,
        device="cpu",
        fp16=False,
    )

    # Optimum exports to model.onnx by default
    exported_path = output_dir / "model.onnx"
    if exported_path.exists() and exported_path != onnx_output_path:
        exported_path.rename(onnx_output_path)

    print(f"Export successful: {onnx_output_path}")
    print(f"Model size: {onnx_output_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_with_torch(
    model_id: str,
    onnx_output_path: Path,
    max_seq_length: int = 512,
    opset_version: int = 14
):
    """
    Export using torch.onnx.export (fallback for simpler models)
    """
    print(f"Loading model from HuggingFace: {model_id}")

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.eval()

    # Create dummy input
    dummy_text = "This is a sample sentence for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )

    # Prepare input names and dynamic axes
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "last_hidden_state": {0: "batch_size", 1: "sequence"}
    }

    # Create output directory
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {onnx_output_path}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Opset version: {opset_version}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    # Verify the ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)

    # Save tokenizer to the same directory
    tokenizer_path = onnx_output_path.parent
    tokenizer.save_pretrained(str(tokenizer_path))

    print(f"ONNX export successful!")
    print(f"  Model: {onnx_output_path}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Model size: {onnx_output_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_to_onnx(
    model_id: str,
    pytorch_model_path: Path,
    onnx_output_path: Path,
    max_seq_length: int = 512,
    opset_version: int = 14,
    use_optimum: bool = True
):
    """
    Export a model to ONNX format

    Args:
        model_id: HuggingFace model ID
        pytorch_model_path: Path to saved PyTorch model (unused with optimum)
        onnx_output_path: Output path for ONNX model
        max_seq_length: Maximum sequence length for input
        opset_version: ONNX opset version
        use_optimum: Use Hugging Face Optimum for export (recommended)
    """
    if use_optimum and HAS_OPTIMUM:
        export_with_optimum(model_id, onnx_output_path, opset_version)
    else:
        if use_optimum and not HAS_OPTIMUM:
            print("Optimum not available, falling back to torch.onnx.export")
        export_with_torch(model_id, onnx_output_path, max_seq_length, opset_version)


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
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
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--no-optimum',
        action='store_true',
        help='Disable Optimum export, use torch.onnx.export instead'
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
    max_seq_length = model_config['max_seq_length']

    # Paths
    pytorch_path = Path(f"models/{args.model}/pytorch")
    onnx_path = Path(f"models/{args.model}/onnx/model.onnx")

    export_to_onnx(
        model_id=model_id,
        pytorch_model_path=pytorch_path,
        onnx_output_path=onnx_path,
        max_seq_length=max_seq_length,
        opset_version=args.opset_version,
        use_optimum=not args.no_optimum
    )


if __name__ == '__main__':
    main()
