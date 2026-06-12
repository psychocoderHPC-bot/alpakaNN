# alpakaNN Chat CLI

Minimal CLI example for loading a Llama-family checkpoint with `alpaka::nn`.

## Build

```bash
cmake -S . -B build -DalpakaNN_BUILD_TESTS=OFF
cmake --build build --target ChatCli
```

The CLI consumes the public umbrella header `<alpaka/nn/nn.hpp>` and links against the exported target `alpaka::nn`.

## Usage

Run without arguments to do a quick self-test with `tiny_llama`:

```bash
./build/example/cli/ChatCli
```

Start interactive chat mode explicitly:

```bash
./build/example/cli/ChatCli -i --model /path/to/model
./build/example/cli/ChatCli --interactive --model /path/to/model
./build/example/cli/ChatCli -i --model /path/to/model --system-prompt "You are a concise assistant."
./build/example/cli/ChatCli -i --model /path/to/model --max-new-tokens 64
```

List known model presets:

```bash
./build/example/cli/ChatCli --list
```

## Model Notes

- `tiny_llama` is only for CI and smoke testing.
- `tiny_llama` is untrained and is not expected to produce useful English chat output.
- Real English output requires a trained, supported Llama-family checkpoint.

## Supported vs Unsupported

Supported checkpoints must match the loader's currently supported architecture:

- Llama-family
- trained weights
- standard attention or GQA where `num_attention_heads % num_key_value_heads == 0`

Unsupported architectures should be treated as incompatible, not "low quality":

- architectures outside the supported Llama-family loader path
- other unsupported attention or MLP variants

If you load an unsupported architecture, incorrect or unreadable output is expected.

## Downloading Models

```bash
python3 tools/download_tiny_llama.py --help
```

Each model directory is expected to contain the tokenizer/config files plus converted binary weights.

TinyLlama chat command:

```bash
python3 tools/download_tiny_llama.py tinyllama-1.1b-chat models/tinyllama-1.1b-chat
./build/example/cli/ChatCli -i --model models/tinyllama-1.1b-chat --system-prompt "You are a concise assistant."
```
