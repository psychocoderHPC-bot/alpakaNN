#!/usr/bin/env python3

import argparse
import json
import pathlib
import struct
import sys
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Tuple

MODEL_MAP = {
    "tiny_llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

MAGIC = b"ANN1"
MANIFEST_NAME = "alpaka_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model plus tokenizer assets and convert supported Llama checkpoints into alpakaNN format."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="tiny_llama",
        help="Preset name from --list or a HuggingFace repo id. Legacy one-arg path mode is also supported.",
    )
    parser.add_argument("out_dir", nargs="?", help="Output directory. Defaults to models/<model>.")
    parser.add_argument("--list", action="store_true", help="List known presets and exit.")
    return parser.parse_args()


def resolve_invocation(args: argparse.Namespace) -> Tuple[str, pathlib.Path]:
    if args.list:
        print("Known presets:")
        for name, repo_id in MODEL_MAP.items():
            print(f"  {name}: {repo_id}")
        raise SystemExit(0)

    model = args.model
    out_dir = args.out_dir

    if args.out_dir is None and (model.startswith("/") or model.startswith("./") or model.startswith("../")):
        return "tiny_llama", pathlib.Path(model)

    if out_dir is None:
        out_dir = pathlib.Path("models") / model
    else:
        out_dir = pathlib.Path(out_dir)
    return model, out_dir


def get_hf_repo_id(model_name: str) -> str:
    return MODEL_MAP.get(model_name, model_name)


def get_download_url(model_name: str, filename: str) -> str:
    return f"https://huggingface.co/{get_hf_repo_id(model_name)}/resolve/main/{filename}"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download_optional(model_name: str, filename: str) -> bytes | None:
    try:
        return download(get_download_url(model_name, filename))
    except urllib.error.HTTPError:
        return None


def parse_safetensors(blob: bytes) -> Tuple[dict, memoryview]:
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    data = memoryview(blob)[8 + header_len :]
    return header, data


def bf16_to_f32(bf16_bytes: bytes) -> float:
    value = struct.unpack("<H", bf16_bytes)[0]
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def tensor_f32(header: dict, data: memoryview, name: str) -> Tuple[List[int], List[float]]:
    info = header[name]
    dtype = info["dtype"]
    if dtype not in {"F32", "F16", "BF16"}:
        raise RuntimeError(f"{name} expected F32/F16/BF16, got {dtype}")

    start, end = info["data_offsets"]
    shape = info["shape"]
    if dtype == "F32":
        values = struct.unpack_from(f"<{(end - start) // 4}f", data, start)
        return shape, list(values)
    if dtype == "F16":
        return shape, [struct.unpack_from("<e", data[start:end], offset)[0] for offset in range(0, end - start, 2)]
    return shape, [bf16_to_f32(bytes(data[start + offset : start + offset + 2])) for offset in range(0, end - start, 2)]


def transpose_2d(shape: Iterable[int], values: List[float]) -> List[float]:
    rows, cols = shape
    out = [0.0] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            out[col * rows + row] = values[row * cols + col]
    return out


def write_tensor(handle, values: List[float], name: str) -> None:
    print(f"  Writing {name}: {len(values)} values", file=sys.stderr)
    handle.write(struct.pack(f"<{len(values)}f", *values))


def load_tokenizer_assets(model_name: str, out_dir: pathlib.Path) -> None:
    tokenizer_data = download(get_download_url(model_name, "tokenizer.json"))
    (out_dir / "tokenizer.json").write_bytes(tokenizer_data)

    tokenizer = json.loads(tokenizer_data.decode("utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if not vocab:
        raise RuntimeError("tokenizer.json does not contain a BPE vocab")
    vocab_list = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        vocab_list[token_id] = token
    (out_dir / "vocab.json").write_text(json.dumps(vocab_list, indent=2), encoding="utf-8")

    tokenizer_config = download_optional(model_name, "tokenizer_config.json")
    if tokenizer_config is not None:
        (out_dir / "tokenizer_config.json").write_bytes(tokenizer_config)


def load_safetensor_bundle(model_name: str) -> Tuple[dict, memoryview]:
    try:
        return parse_safetensors(download(get_download_url(model_name, "model.safetensors")))
    except urllib.error.HTTPError:
        index_blob = download(get_download_url(model_name, "model.safetensors.index.json"))
        index = json.loads(index_blob.decode("utf-8"))
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        if not shard_names:
            raise RuntimeError("Sharded model index is present but contains no shard names")

        combined_header: Dict[str, dict] = {}
        combined_data = bytearray()
        for shard_name in shard_names:
            blob = download(get_download_url(model_name, shard_name))
            shard_header, shard_data = parse_safetensors(blob)
            shard_offset = len(combined_data)
            combined_data.extend(shard_data)
            for tensor_name, tensor_info in shard_header.items():
                if tensor_name == "__metadata__":
                    continue
                start, end = tensor_info["data_offsets"]
                combined_header[tensor_name] = {
                    **tensor_info,
                    "data_offsets": [start + shard_offset, end + shard_offset],
                }
        return combined_header, memoryview(combined_data)


def detect_supported_llama(config: dict) -> Tuple[bool, str]:
    if config.get("model_type") != "llama":
        return False, f"Unsupported model_type '{config.get('model_type')}', only llama is supported"
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    if num_kv_heads != num_heads:
        return False, (
            f"Unsupported grouped-query attention: num_key_value_heads={num_kv_heads}, "
            f"num_attention_heads={num_heads}"
        )
    return True, "supported"


def required_tensor_names(config: dict) -> List[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer in range(int(config["num_hidden_layers"])):
        names.extend(
            [
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            ]
        )
    return names


def validate_required_tensors(header: dict, config: dict) -> None:
    missing = [name for name in required_tensor_names(config) if name not in header]
    if missing:
        raise RuntimeError("Missing required tensors: " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else ""))


def write_manifest(
    out_dir: pathlib.Path,
    *,
    model_name: str,
    repo_id: str,
    config: dict,
    status: str,
    reason: str,
    binary_name: str | None,
) -> None:
    manifest = {
        "model_name": model_name,
        "source_repo": repo_id,
        "model_type": config.get("model_type"),
        "binary_name": binary_name,
        "config_name": "config.json",
        "tokenizer_name": "tokenizer.json",
        "status": status,
        "reason": reason,
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads", config.get("num_attention_heads")),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def convert_supported_llama(model_name: str, out_dir: pathlib.Path, config: dict) -> pathlib.Path:
    header, data = load_safetensor_bundle(model_name)
    validate_required_tensors(header, config)

    out_path = out_dir / f"{pathlib.Path(model_name).name}.bin"
    with out_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<9I2f",
                int(config["hidden_size"]),
                int(config["intermediate_size"]),
                int(config["num_hidden_layers"]),
                int(config["num_attention_heads"]),
                int(config.get("num_key_value_heads", config["num_attention_heads"])),
                int(config["vocab_size"]),
                int(config.get("bos_token_id", 1)),
                int(config.get("eos_token_id", 2)),
                int(config["max_position_embeddings"]),
                float(config.get("layer_norm_eps", config.get("rms_norm_eps", 1e-5))),
                float(config.get("rope_theta", 10000.0)),
            )
        )

        _, embedding = tensor_f32(header, data, "model.embed_tokens.weight")
        write_tensor(handle, embedding, "model.embed_tokens.weight")

        for layer in range(int(config["num_hidden_layers"])):
            norm_names = [
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
            ]
            for name in norm_names:
                _, values = tensor_f32(header, data, name)
                write_tensor(handle, values, name)

            projection_names = [
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            ]
            for name in projection_names:
                shape, values = tensor_f32(header, data, name)
                write_tensor(handle, transpose_2d(shape, values), name)

        _, final_norm = tensor_f32(header, data, "model.norm.weight")
        write_tensor(handle, final_norm, "model.norm.weight")

        lm_head_name = "lm_head.weight" if "lm_head.weight" in header else "model.embed_tokens.weight"
        lm_shape, lm_head = tensor_f32(header, data, lm_head_name)
        write_tensor(handle, transpose_2d(lm_shape, lm_head), lm_head_name)

    return out_path


def main() -> int:
    args = parse_args()
    model_name, out_dir = resolve_invocation(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = get_hf_repo_id(model_name)
    print(f"Downloading model: {model_name}")
    print(f"  HuggingFace: {repo_id}")

    try:
        config = json.loads(download(get_download_url(model_name, "config.json")).decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"Error downloading config: {error}", file=sys.stderr)
        print(f"Available presets: {list(MODEL_MAP.keys())}", file=sys.stderr)
        return 1

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    load_tokenizer_assets(model_name, out_dir)

    supported, reason = detect_supported_llama(config)
    if not supported:
        write_manifest(
            out_dir,
            model_name=model_name,
            repo_id=repo_id,
            config=config,
            status="unsupported",
            reason=reason,
            binary_name=None,
        )
        raise RuntimeError(reason)

    out_path = convert_supported_llama(model_name, out_dir, config)
    write_manifest(
        out_dir,
        model_name=model_name,
        repo_id=repo_id,
        config=config,
        status="supported",
        reason="supported",
        binary_name=out_path.name,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
