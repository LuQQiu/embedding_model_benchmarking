# Benchmark Model List

## Overview
Benchmarking 5 embedding models across 6 frameworks (PyTorch, Candle, ONNX Runtime Native/Rust/Python, OpenVINO)

**Starting with: EmbeddingGemma 300M** ⭐

---

## 1. EmbeddingGemma 300M ⭐ (START HERE)

**HuggingFace ID**: `google/embeddinggemma-300m`

### Key Specs
- **Parameters**: 308M
- **Type**: Text embedding only
- **Embedding Dimension**: 256 (supports MRL - can truncate to lower dims)
- **Max Sequence Length**: 8192 tokens
- **Multilingual**: Yes (100+ languages)
- **Memory**: <200MB with quantization

### Why This Model?
- **Smallest model** (300M) - great for speed/accuracy tradeoff
- **Fast inference** - optimized for on-device AI
- **Best-in-class** for models under 500M on MTEB
- Released September 2025 (very recent)
- Built on Gemma3 with bi-directional attention

### Framework Compatibility
- ✅ PyTorch (via transformers, sentence-transformers)
- ✅ ONNX export supported
- ✅ Candle support available
- ✅ OpenVINO conversion supported
- ✅ GGUF quantization available

### Expected Performance
- **Fastest inference** among the 5 models
- **Lowest memory usage**
- **Good accuracy** despite smaller size

---

## 2. Qwen3-Embedding 0.6B

**HuggingFace ID**: `Qwen/Qwen3-Embedding-0.6B`

### Key Specs
- **Parameters**: 600M
- **Type**: Text embedding only
- **Embedding Dimension**: 768
- **Max Sequence Length**: 8192 tokens
- **Multilingual**: Yes (100+ languages)
- **Code Retrieval**: Strong support for programming languages

### Why This Model?
- **Medium-sized** (600M) - balanced performance
- **Top performer** on MTEB multilingual (sibling 8B model is #1)
- Strong code retrieval capabilities
- Support for diverse tasks (retrieval, classification, clustering)

### Framework Compatibility
- ✅ PyTorch (transformers>=4.51.0, sentence-transformers>=2.7.0)
- ✅ ONNX export supported
- ✅ GGUF version available
- ✅ OpenVINO conversion supported

### Expected Performance
- **Moderate speed** (2x slower than EmbeddingGemma)
- **Better accuracy** than EmbeddingGemma
- **Balanced** speed/accuracy

---

## 3. BGE-M3

**HuggingFace ID**: `BAAI/bge-m3`

### Key Specs
- **Parameters**: 600M
- **Type**: Text embedding only
- **Embedding Dimension**: 1024
- **Max Sequence Length**: 8192 tokens
- **Multilingual**: Yes (100+ languages)
- **Multi-functionality**: Dense, sparse, and multi-vector retrieval

### Why This Model?
- **Very popular** model in production
- **Versatile**: Supports multiple retrieval methods
- Strong performance on benchmarks
- Well-tested and widely deployed

### Framework Compatibility
- ✅ PyTorch (sentence-transformers, FlagEmbedding)
- ✅ ONNX export supported
- ✅ OpenVINO conversion supported
- ✅ Well-documented

### Expected Performance
- **Similar speed** to Qwen3-600M
- **High accuracy** with larger embedding dim (1024)
- **Production-ready**

---

## 4. SigLIP Base

**HuggingFace ID**: `google/siglip-base-patch16-224`

### Key Specs
- **Parameters**: ~400M
- **Type**: Multimodal (text + image)
- **Embedding Dimension**: 768
- **Max Text Length**: 64 tokens
- **Image Size**: 224x224
- **Training**: WebLI dataset with sigmoid loss

### Why This Model?
- **Multimodal** capabilities (text + image embeddings)
- **Improved over CLIP** with better loss function
- Good for image-text retrieval and zero-shot classification
- Google's latest vision-language model

### Framework Compatibility
- ✅ PyTorch (transformers library)
- ✅ ONNX export (separate text/image encoders)
- ⚠️ OpenVINO conversion (requires separate model handling)
- ⚠️ Candle support (may require custom implementation)

### Expected Performance
- **Slower than text-only** models (vision encoder overhead)
- **Unique use case** (multimodal embeddings)
- Need **image dataset** for benchmarking

### Benchmark Considerations
- Requires **image inputs** for full testing
- Can test **text-only** encoder separately
- May need **different test harness** for multimodal

---

## 5. OpenAI CLIP ViT-B/16

**HuggingFace ID**: `openai/clip-vit-base-patch16`

### Key Specs
- **Parameters**: ~150M
- **Type**: Multimodal (text + image)
- **Embedding Dimension**: 512
- **Max Text Length**: 77 tokens
- **Image Size**: 224x224
- **Training**: 400M image-text pairs

### Why This Model?
- **Most popular** multimodal embedding model
- **Industry standard** for vision-language tasks
- Widely used in production (Stable Diffusion, etc.)
- Strong zero-shot capabilities

### Framework Compatibility
- ✅ PyTorch (transformers library)
- ✅ ONNX export (well-documented)
- ✅ OpenVINO conversion supported
- ⚠️ Candle support (community implementations available)

### Expected Performance
- **Smallest multimodal** model (150M)
- **Faster than SigLIP** (smaller size)
- **Good baseline** for multimodal benchmarks

### Benchmark Considerations
- Requires **image inputs** for full testing
- Can test **text-only** encoder separately
- **Well-established baseline** for comparison

---

## Model Comparison Summary

| Model | Type | Params | Embed Dim | Use Case | Expected Speed |
|-------|------|--------|-----------|----------|----------------|
| **EmbeddingGemma** ⭐ | Text | 308M | 256 | Fast on-device | ⚡⚡⚡ Fastest |
| Qwen3-Embedding | Text | 600M | 768 | Balanced | ⚡⚡ Fast |
| BGE-M3 | Text | 600M | 1024 | Production | ⚡⚡ Fast |
| CLIP | Multimodal | 150M | 512 | Vision+Text | ⚡ Moderate |
| SigLIP | Multimodal | 400M | 768 | Vision+Text | ⚡ Moderate |

---

## Benchmark Strategy

### Phase 1: Start with EmbeddingGemma ⭐
1. Implement all 6 framework benchmarks
2. Validate methodology and metrics
3. Establish baseline performance numbers
4. Debug any infrastructure issues

### Phase 2: Add Text-Only Models
5. Benchmark Qwen3-Embedding
6. Benchmark BGE-M3
7. Compare text-only models

### Phase 3: Add Multimodal Models
8. Extend harness for image inputs
9. Benchmark CLIP
10. Benchmark SigLIP
11. Full comparison report

---

## Next Steps

**Immediate Action**: Begin EmbeddingGemma setup
1. ✅ Create infrastructure (Terraform)
2. ✅ Download and convert EmbeddingGemma models
3. ✅ Implement PyTorch benchmark (baseline)
4. ✅ Implement ONNX Runtime benchmarks
5. ✅ Implement Candle benchmark
6. ✅ Implement OpenVINO benchmark
7. ✅ Run benchmarks and collect results
8. ✅ Generate initial performance report

Then expand to other models following the same pattern.
