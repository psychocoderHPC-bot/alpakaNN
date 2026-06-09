#!/usr/bin/env python3

import json
import pathlib
import struct
import sys
import urllib.request

BASE = "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/"
MAGIC = b"ANN1"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_safetensors(blob: bytes):
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    data = memoryview(blob)[8 + header_len :]
    return header, data


def tensor_f32(header, data, name):
    info = header[name]
    if info["dtype"] != "F32":
        raise RuntimeError(f"{name} expected F32, got {info['dtype']}")
    start, end = info["data_offsets"]
    shape = info["shape"]
    values = struct.unpack_from(f"<{(end - start) // 4}f", data, start)
    return shape, list(values)


def transpose_2d(shape, values):
    rows, cols = shape
    out = [0.0] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            out[col * rows + row] = values[row * cols + col]
    return out


def write_tensor(handle, values):
    handle.write(struct.pack(f"<{len(values)}f", *values))


def main():
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("testdata/tiny_llama")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(download(BASE + "config.json").decode("utf-8"))
    header, data = parse_safetensors(download(BASE + "model.safetensors"))
    out_path = out_dir / "tiny_llama.bin"

    with out_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<9I2f",
                config["hidden_size"],
                config["intermediate_size"],
                config["num_hidden_layers"],
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["vocab_size"],
                config["bos_token_id"],
                config["eos_token_id"],
                config["max_position_embeddings"],
                float(config["rms_norm_eps"]),
                float(config["rope_theta"]),
            )
        )

        _, values = tensor_f32(header, data, "model.embed_tokens.weight")
        write_tensor(handle, values)

        for layer in range(config["num_hidden_layers"]):
            for name in [
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
            ]:
                _, values = tensor_f32(header, data, name)
                write_tensor(handle, values)

            for name in [
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            ]:
                shape, values = tensor_f32(header, data, name)
                write_tensor(handle, transpose_2d(shape, values))

        _, values = tensor_f32(header, data, "model.norm.weight")
        write_tensor(handle, values)
        shape, values = tensor_f32(header, data, "lm_head.weight")
        write_tensor(handle, transpose_2d(shape, values))

    print(out_path)


if __name__ == "__main__":
    main()
