# alpakaNN Chat CLI

A chatbot implementation using the alpakaNN library with Llama models.

## Quick Start

```bash
# Build
cmake -S . -B build -DalpakaNN_BUILD_TESTS=OFF
cmake --build build --target ChatCli

# Run with default model (tiny random Llama for testing)
./build/example/cli/ChatCli
```

## Current Model Support

### Working Models (Standard Architecture)

The alpakaNN loader expects models with:
- **Standard Multi-Head Attention (MHA)**: `num_key_value_heads == num_attention_heads`
- **Simple MLP**: `gate_proj -> up_proj` (2 projection layers)

**Currently working:**
- `tiny_llama` - Tiny random Llama (untrained, for testing only)

### Models NOT YET Supported

Models using these architectures will NOT produce correct output:
- **Grouped Query Attention (GQA)**: `num_key_value_heads < num_attention_heads`
  - TinyLlama-1.1B (4 KV heads, 32 Q heads)
  - Llama-3, Llama-2 (8 KV heads, 32+ Q heads)
  - Mistral-7B-Instruct
- **SwiGLU MLP**: Uses 3 projection layers (gate, up, down)
  - Most modern Llama models

## Usage

### Basic Chat
```bash
./build/example/cli/ChatCli --model tiny_llama
```

**Expected output**: Garbled/random characters. The tiny_llama model is untrained and produces random token predictions.

### List Available Models
```bash
./build/example/cli/ChatCli --list
```

## Downloading Models

The download script supports multiple model presets:

```bash
# Download TinyLlama-1.1B (note: GQA - not yet compatible)
python3 tools/download_tiny_llama.py tinyllama-1.1b-chat models/tinyllama-1.1b-chat

# List all available presets
python3 tools/download_tiny_llama.py --help
```

### Model Format

After download, each model directory contains:
- `config.json` - Model architecture configuration
- `vocab.json` - Token vocabulary (list format)
- `tokenizer.json` - Full tokenizer configuration
- `*.bin` - Binary model weights for alpakaNN
- `model.safetensors` - Original format (backup)

## Architecture Requirements for Human-Readable Output

To get human-readable chat responses, a model needs:

1. **Trained weights** - Must be a real model, not random initialization
2. **Matching architecture** - Currently only MHA + simple MLP

The `tiny_llama` model is **untrained** - it produces random tokens regardless of input.

## Troubleshooting

### Garbled/Unreadable Output

1. **tiny_llama model**: Expected - this model is untrained
2. **GQA model**: Architecture mismatch - loader doesn't support grouped query attention
3. **Wrong model**: Ensure you're using a supported model format

### Inference Hangs

If the CLI hangs during inference:
- Check GPU/backend availability
- Verify model file is complete and uncorrupted
- Try with smaller models first

## Known Limitations

1. **No GQA Support**: Models with grouped query attention will not decode correctly
2. **No SwiGLU Support**: Models with SwiGLU activation need loader updates
3. **Untrained Default**: tiny_llama is random, not a trained chat model
4. **Greedy Decoding Only**: No temperature sampling for varied output
5. **No Chat Template**: Responses are not formatted for conversation

## Building

```bash
cmake -S . -B build -DalpakaNN_BUILD_TESTS=OFF
cmake --build build --target ChatCli
```

## License

ISC License - See LICENSE file for details.