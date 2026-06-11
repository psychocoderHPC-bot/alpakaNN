#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Tuple


class LlamaTokenizer:
    def __init__(self, model_dir: pathlib.Path):
        tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "config.json"
        if not tokenizer_path.exists():
            raise RuntimeError(f"Missing tokenizer.json in {model_dir}")
        if not config_path.exists():
            raise RuntimeError(f"Missing config.json in {model_dir}")

        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = tokenizer.get("model", {})
        if model.get("type") != "BPE":
            raise RuntimeError(f"Unsupported tokenizer model type: {model.get('type')}")

        self.vocab: Dict[str, int] = model["vocab"]
        self.id_to_token: Dict[int, str] = {token_id: token for token, token_id in self.vocab.items()}
        self.merges: Dict[Tuple[str, str], int] = {
            tuple(merge.split(" ", 1)): rank for rank, merge in enumerate(model["merges"])
        }
        self.byte_fallback = bool(model.get("byte_fallback", False))
        self.unk_token = model.get("unk_token", "<unk>")
        self.unk_id = self.vocab.get(self.unk_token, 0)
        self.bos_id = int(config.get("bos_token_id", 1))
        self.eos_id = int(config.get("eos_token_id", 2))
        self.special_ids = {token["id"] for token in tokenizer.get("added_tokens", []) if token.get("special")}

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> List[int]:
        normalized = "▁" + text.replace(" ", "▁")
        pieces = [self._encode_piece(piece) for piece in normalized.split("▁")]
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        for index, piece_tokens in enumerate(pieces):
            prefix = "▁" if index > 0 else ""
            tokens.extend(self._encode_normalized_piece(prefix + piece_tokens))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        text_parts: List[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            nonlocal pending_bytes
            if pending_bytes:
                text_parts.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes = bytearray()

        for token_id in token_ids:
            if token_id in self.special_ids:
                flush_bytes()
                continue
            token = self.id_to_token.get(token_id)
            if token is None:
                flush_bytes()
                continue
            byte_match = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", token)
            if byte_match:
                pending_bytes.append(int(byte_match.group(1), 16))
                continue
            flush_bytes()
            text_parts.append(token)

        flush_bytes()
        return "".join(text_parts).replace("▁", " ").lstrip(" ")

    def _encode_piece(self, piece: str) -> str:
        return piece

    def _encode_normalized_piece(self, text: str) -> List[int]:
        if not text:
            return []
        symbols = list(text)
        while len(symbols) > 1:
            best_index = -1
            best_rank = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = self.merges.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_index = index
                    best_rank = rank
            if best_index < 0:
                break
            symbols = (
                symbols[:best_index]
                + [symbols[best_index] + symbols[best_index + 1]]
                + symbols[best_index + 2 :]
            )

        encoded: List[int] = []
        for symbol in symbols:
            token_id = self.vocab.get(symbol)
            if token_id is not None:
                encoded.append(token_id)
                continue
            if self.byte_fallback:
                encoded.extend(self._encode_bytes(symbol.encode("utf-8")))
                continue
            encoded.append(self.unk_id)
        return encoded

    def _encode_bytes(self, data: bytes) -> List[int]:
        tokens: List[int] = []
        for value in data:
            token = f"<0x{value:02X}>"
            token_id = self.vocab.get(token)
            if token_id is None:
                tokens.append(self.unk_id)
            else:
                tokens.append(token_id)
        return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode or decode Llama tokenizer.json assets.")
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("model_dir")
    parser.add_argument("--bos", action="store_true", help="Prepend bos_token_id during encode.")
    parser.add_argument("--eos", action="store_true", help="Append eos_token_id during encode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = LlamaTokenizer(pathlib.Path(args.model_dir))
    payload = sys.stdin.read()
    if args.mode == "encode":
        token_ids = tokenizer.encode(payload, add_bos=args.bos, add_eos=args.eos)
        sys.stdout.write(json.dumps(token_ids))
        return 0

    token_ids = json.loads(payload)
    if not isinstance(token_ids, list):
        raise RuntimeError("decode expects a JSON list of token ids on stdin")
    sys.stdout.write(tokenizer.decode([int(token_id) for token_id in token_ids]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"tokenizer error: {exc}", file=sys.stderr)
        raise
