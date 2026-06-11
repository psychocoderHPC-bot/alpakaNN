# alpakaNN Chat CLI

Minimal CLI example for loading a Llama-family checkpoint with alpakaNN.

## Build

```bash
cmake -S . -B build -DalpakaNN_BUILD_TESTS=OFF
cmake --build build --target ChatCli
```

## Usage

Run without arguments to do a quick self-test with `tiny_llama`:

```bash
./build/example/cli/ChatCli
```

Start interactive chat mode explicitly:

```bash
./build/example/cli/ChatCli -i --model /path/to/model
./build/example/cli/ChatCli --interactive --model /path/to/model
```

List known model presets:

```bash
./build/example/cli/ChatCli --list
```

## Model Notes

- `tiny_llama` is only for CI and smoke testing.
- `tiny_llama` is untrained and is not expected to produce useful English chat output.
- Real English output requires a trained, supported, Llama-family non-GQA checkpoint.

## Supported vs Unsupported

Supported checkpoints must match the loader's currently supported architecture:

- Llama-family
- trained weights
- non-GQA attention: `num_key_value_heads == num_attention_heads`

Unsupported architectures should be treated as incompatible, not "low quality":

- GQA models: `num_key_value_heads < num_attention_heads`
- architectures outside the supported Llama-family loader path
- other unsupported attention or MLP variants

If you load an unsupported architecture, incorrect or unreadable output is expected.

## Downloading Models

```bash
python3 tools/download_tiny_llama.py --help
```

Each model directory is expected to contain the tokenizer/config files plus converted binary weights.
