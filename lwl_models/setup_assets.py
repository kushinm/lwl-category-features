"""Download ICNet assets; SigLIP2 download is explicitly opt-in."""

import argparse
from pathlib import Path
import requests
from .assets import (
    ROOT, ICNET_DIR, ICNET_SOURCE_URL, ICNET_SOURCE_SHA256,
    ICNET_WEIGHTS_URL, ICNET_WEIGHTS_SHA256, SIGLIP_ID, SIGLIP_REVISION,
    verify_hash, resolve_siglip,
)


def fetch(url, destination, expected):
    destination = Path(destination)
    if destination.exists():
        verify_hash(destination, expected)
        print(f"Verified existing {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}", flush=True)
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    verify_hash(temporary, expected)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-siglip", action="store_true",
                        help="Only if needed: download pinned SigLIP2 base assets into the ignored project cache.")
    args = parser.parse_args()
    fetch(ICNET_SOURCE_URL, ICNET_DIR / "ICNet.py", ICNET_SOURCE_SHA256)
    fetch(ICNET_WEIGHTS_URL, ICNET_DIR / "ck.pth", ICNET_WEIGHTS_SHA256)
    if args.download_siglip:
        from huggingface_hub import snapshot_download
        snapshot_download(SIGLIP_ID, revision=SIGLIP_REVISION,
                          local_dir=ROOT / ".model-cache" / "siglip2-base",
                          allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json",
                                          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
    try:
        model, tokenizer = resolve_siglip()
        print(f"SigLIP2 weights: {model}\nVerified tokenizer: {tokenizer}")
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
