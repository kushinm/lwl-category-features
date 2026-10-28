"""Pinned public model assets; prefer existing caches and never download on inference."""

import hashlib
import os
from pathlib import Path
from .study import ROOT

SIGLIP_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
TOKENIZER_HASHES = {
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    "tokenizer_config.json": "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
}
ICNET_REVISION = "3891e25ad1605763995801c1ddbba218a03f8720"
ICNET_SOURCE_SHA256 = "0fa784ed72582bbf4b396863f25711c3f6a7eae932e39264c605e92a8943ab1a"
ICNET_WEIGHTS_SHA256 = "2d4098b4adb3b299f5422a6201d0caed9ff9fff2a4eb8c25a0b67fe85485fdaa"
ICNET_SOURCE_URL = f"https://raw.githubusercontent.com/tinglyfeng/IC9600/{ICNET_REVISION}/ICNet.py"
ICNET_WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1KZGpPRduAsi8ra3gORO78xjWLL8M-vNx&export=download&confirm=t"
ICNET_DIR = ROOT / ".model-cache" / "ic9600"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {path}; see python -m lwl_models.setup_assets --help")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Unexpected asset checksum for {path}: {actual}")
    return actual


def cache_roots():
    candidates = [Path(value) for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
                  if (value := os.environ.get(key))]
    candidates.append(Path.home() / ".cache" / "huggingface")
    result = []
    for root in candidates:
        for candidate in (root, root / "hub"):
            if candidate not in result:
                result.append(candidate)
    return result


def resolve_siglip(model_dir=None, tokenizer_dir=None):
    model_candidates = ([Path(model_dir)] if model_dir else
                        [ROOT / ".model-cache" / "siglip2-base"] + [
                            root / "models--google--siglip2-base-patch16-224" / "snapshots" / SIGLIP_REVISION
                            for root in cache_roots()])
    model_path = next((p for p in model_candidates
                       if all((p / name).is_file() for name in
                              ("config.json", "model.safetensors", "preprocessor_config.json"))), None)
    if model_path is None:
        raise FileNotFoundError("Cached SigLIP2 base weights not found. Set --siglip-dir or explicitly run "
                                "python -m lwl_models.setup_assets --download-siglip.")
    verify_hash(model_path / "model.safetensors", SIGLIP_WEIGHTS_SHA256)
    tokenizer_candidates = [Path(tokenizer_dir)] if tokenizer_dir else [model_path]
    if not tokenizer_dir:
        for root in cache_roots():
            tokenizer_candidates.extend(sorted(root.glob("models--google--siglip2*/snapshots/*")))
    for candidate in tokenizer_candidates:
        if all((candidate / name).is_file() for name in TOKENIZER_HASHES):
            if all(sha256(candidate / name) == expected for name, expected in TOKENIZER_HASHES.items()):
                return model_path, candidate
    raise FileNotFoundError("No tokenizer with the pinned SigLIP2 base hashes was found. "
                            "Set --tokenizer-dir or run setup_assets --download-siglip.")
