#!/usr/bin/env python3

import json
import pathlib
import struct
import sys
import urllib.request
import urllib.error

MODEL_MAP = {
    "tiny_llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

MAGIC = b"ANN1"


def get_hf_url(model_name: str, filename: str) -> str:
    if model_name in MODEL_MAP:
        hf_model = MODEL_MAP[model_name]
    else:
        hf_model = model_name
    
    # Use huggingface-cli or direct URL
    return f"https://huggingface.co/{hf_model}/resolve/main/{filename}"


def get_download_url(model_name: str, filename: str) -> str:
    return get_hf_url(model_name, filename)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_safetensors(blob: bytes):
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    data = memoryview(blob)[8 + header_len :]
    return header, data


def bf16_to_f32(bf16_bytes):
    """Convert BF16 to F32"""
    val = struct.unpack("<H", bf16_bytes)[0]
    f32_bits = val << 16
    return struct.unpack("<f", struct.pack("<I", f32_bits))[0]

def tensor_f32(header, data, name, convert_f16=True):
    info = header[name]
    dtype = info["dtype"]
    if dtype not in ("F32", "F16", "BF16"):
        raise RuntimeError(f"{name} expected F32/F16/BF16, got {dtype}")
    start, end = info["data_offsets"]
    shape = info["shape"]
    
    if dtype == "F32":
        values = struct.unpack_from(f"<{(end - start) // 4}f", data, start)
    elif dtype == "F16":
        f16_data = data[start:end]
        values = []
        for i in range(0, len(f16_data), 2):
            f16_val = struct.unpack_from("<e", f16_data, i)[0]
            values.append(f16_val)
    else:  # BF16
        bf16_data = data[start:end]
        values = []
        for i in range(0, len(bf16_data), 2):
            bf16_val = bf16_to_f32(bf16_data[i:i+2])
            values.append(bf16_val)
    
    return shape, list(values)


def transpose_2d(shape, values):
    rows, cols = shape
    out = [0.0] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            out[col * rows + row] = values[row * cols + col]
    return out


def write_tensor(handle, values, name=""):
    import sys
    size_bytes = len(values) * 4
    print(f"  Writing {name}: {len(values)} values = {size_bytes} bytes", file=sys.stderr)
    handle.write(struct.pack(f"<{len(values)}f", *values))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "tiny_llama"
    
    # Determine output directory
    if len(sys.argv) > 2:
        out_dir = pathlib.Path(sys.argv[2])
    else:
        # Use model name as directory name
        out_dir = pathlib.Path("models") / model_name
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    if model_name in MODEL_MAP:
        print(f"  HuggingFace: {MODEL_MAP[model_name]}")
    else:
        print(f"  HuggingFace: {model_name}")
    
    config_url = get_download_url(model_name, "config.json")
    try:
        config = json.loads(download(config_url).decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Error downloading config: {e}")
        print(f"Model '{model_name}' may require authentication or not exist.")
        print(f"Available presets: {list(MODEL_MAP.keys())}")
        sys.exit(1)
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Download the actual tokenizer from HuggingFace
    print("  Downloading tokenizer.json...")
    tokenizer_url = get_download_url(model_name, "tokenizer.json")
    tokenizer_data = download(tokenizer_url)
    with open(out_dir / "tokenizer.json", "wb") as f:
        f.write(tokenizer_data)
    
    # Parse tokenizer.json to get vocab as list for compatibility
    import re
    tokenizer = json.loads(tokenizer_data.decode())
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if vocab:
        # Convert to array format
        vocab_list = [""] * (max(vocab.values()) + 1)
        for token, id in vocab.items():
            vocab_list[id] = token
        vocab_json = json.dumps(vocab_list, indent=2)
        with open(out_dir / "vocab.json", "w") as f:
            f.write(vocab_json)
    
    print("  Downloading model.safetensors...")
    
    # Try single-file first, then handle shards
    try:
        header, data = parse_safetensors(download(get_download_url(model_name, "model.safetensors")))
    except urllib.error.HTTPError:
        print("  Model uses sharded safetensors, downloading shards...")
        
        # Find total number of shards by trying to download
        num_shards = 1
        while True:
            shard_url = get_download_url(model_name, f"model-0000{num_shards + 1}-of-0000X.safetensors")
            shard_url = shard_url.replace("0000X", f"000{num_shards + 1}")
            try:
                download(shard_url)
                num_shards += 1
            except urllib.error.HTTPError:
                break
        
        print(f"  Found {num_shards} shards")
        
        all_header = {}
        all_data_parts = []
        
        for i in range(num_shards):
            shard_filename = f"model-0000{i + 1}-of-0000{num_shards}.safetensors"
            shard_url = get_download_url(model_name, shard_filename)
            print(f"  Downloading shard {i + 1}/{num_shards}...")
            blob = download(shard_url)
            
            header_len = struct.unpack_from("<Q", blob, 0)[0]
            header_part = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
            
            for key, val in header_part.items():
                all_header[key] = val
            
            data_part = blob[8 + header_len:]
            all_data_parts.append(data_part)
        
        combined_data = b"".join(all_data_parts)
        header = all_header
        data = memoryview(combined_data)
    
    out_path = out_dir / f"{model_name}.bin"

    with out_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<9I2f",
                config["hidden_size"],
                config["intermediate_size"],
                config["num_hidden_layers"],
                config["num_attention_heads"],
                config["num_attention_heads"],  # num_key_value_heads (phi uses regular attention)
                config["vocab_size"],
                config["bos_token_id"] if config.get("bos_token_id") else 1,
                config["eos_token_id"] if config.get("eos_token_id") else 2,
                config["max_position_embeddings"],
                float(config.get("layer_norm_eps", config.get("rms_norm_eps", 1e-5))),
                float(config["rope_theta"]),
            )
        )

        shape, values = tensor_f32(header, data, "model.embed_tokens.weight")
        # Convert F16 to F32 for alpakaNN
        values_f32 = [float(v) for v in values]
        write_tensor(handle, values_f32)

        # Check model architecture to determine layer names
        model_type = config.get("model_type", "")
        
        # Find the actual layer names from the safetensors header
        layer_names = set()
        for name in header.keys():
            if 'layers' in name:
                # Extract layer index
                import re
                m = re.search(r'layers\.(\d+)', name)
                if m:
                    layer_names.add(int(m.group(1)))
        
        num_layers = len(layer_names)
        print(f"  Detected {num_layers} layers")
        
        for layer in range(num_layers):
            # Try to find layernorm weights
            ln_weights = []
            ln_biases = []
            for name in header.keys():
                if f'layers.{layer}' in name and 'layernorm' in name.lower():
                    if 'weight' in name:
                        ln_weights.append(name)
                    if 'bias' in name:
                        ln_biases.append(name)
            
            for ln_weight in ln_weights:
                shape, values = tensor_f32(header, data, ln_weight)
                values_f32 = [float(v) for v in values]
                write_tensor(handle, values_f32)
            for ln_bias in ln_biases:
                shape, values = tensor_f32(header, data, ln_bias)
                values_f32 = [float(v) for v in values]
                write_tensor(handle, values_f32)

            # Find attention weights
            attn_qkv = []
            for name in header.keys():
                if f'layers.{layer}' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                    attn_qkv.append(name)
            
            # Handle QKV projection weights
            for name in sorted(attn_qkv):
                shape, values = tensor_f32(header, data, name)
                write_tensor(handle, transpose_2d(shape, values))
            
            # Handle O projection
            o_name = f'model.layers.{layer}.self_attn.o_proj.weight'
            if o_name in header:
                shape, values = tensor_f32(header, data, o_name)
                write_tensor(handle, transpose_2d(shape, values))
            else:
                # Try alternate name
                o_name = f'model.layers.{layer}.self_attn.dense.weight'
                if o_name in header:
                    shape, values = tensor_f32(header, data, o_name)
                    write_tensor(handle, transpose_2d(shape, values))
            
            # Handle MLP weights (try different naming conventions)
            mlp_names = [
                (f'model.layers.{layer}.mlp.gate_proj.weight', f'model.layers.{layer}.mlp.up_proj.weight'),
                (f'model.layers.{layer}.mlp.fc1.weight', f'model.layers.{layer}.mlp.fc2.weight'),
            ]
            for gate_name, up_name in mlp_names:
                if gate_name in header and up_name in header:
                    for name in [gate_name, up_name]:
                        shape, values = tensor_f32(header, data, name)
                        write_tensor(handle, transpose_2d(shape, values))
                    # Also write down_proj
                    down_name = f'model.layers.{layer}.mlp.down_proj.weight'
                    if down_name in header:
                        shape, values = tensor_f32(header, data, down_name)
                        write_tensor(handle, transpose_2d(shape, values))
                    break

        # Final layer norm (try different names)
        norm_names = ["model.norm.weight", "model.final_layernorm.weight"]
        norm_weight = None
        for name in norm_names:
            if name in header:
                norm_weight = name
                break
        if norm_weight:
            _, values = tensor_f32(header, data, norm_weight)
            write_tensor(handle, values)
        
        # Language model head (may not exist if sharing embeddings)
        lm_names = ["lm_head.weight", "model.embed_tokens.weight"]
        lm_weight = None
        lm_shape = None
        for name in lm_names:
            if name in header:
                lm_weight = name
                lm_shape = header[name]['shape']
                break
        if lm_weight:
            _, values = tensor_f32(header, data, lm_weight)
            write_tensor(handle, transpose_2d(lm_shape, values))

    print(out_path)


if __name__ == "__main__":
    main()
