"""Inference-only adapters for released ICNet and cached SigLIP2 weights."""

import importlib.util
import itertools
import os
import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from .assets import (
    ICNET_DIR, ICNET_REVISION, ICNET_SOURCE_SHA256, ICNET_WEIGHTS_SHA256,
    SIGLIP_ID, SIGLIP_REVISION, SIGLIP_WEIGHTS_SHA256, TOKENIZER_HASHES,
    verify_hash, resolve_siglip,
)


def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no working CUDA device is available.")
    return requested


class ICNetPredictor:
    def __init__(self, device="cpu"):
        from torchvision import transforms
        verify_hash(ICNET_DIR / "ICNet.py", ICNET_SOURCE_SHA256)
        verify_hash(ICNET_DIR / "ck.pth", ICNET_WEIGHTS_SHA256)
        spec = importlib.util.spec_from_file_location("lwl_official_icnet", ICNET_DIR / "ICNet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The full released checkpoint supplies the backbones too: no ImageNet download.
        self.model = module.ICNet(is_pretrain=False)
        state = torch.load(ICNET_DIR / "ck.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.device = device
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.provenance = {"source_revision": ICNET_REVISION,
                           "source_sha256": ICNET_SOURCE_SHA256,
                           "checkpoint_sha256": ICNET_WEIGHTS_SHA256,
                           "kind": "complexity, not fixation probability"}

    @torch.inference_mode()
    def predict(self, image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        score, native_map = self.model(tensor)
        full_map = torch.nn.functional.interpolate(
            native_map, size=(image.height, image.width), mode="bilinear", align_corners=False)
        return float(score.item()), full_map[0, 0].cpu().numpy(), native_map[0, 0].cpu().numpy()


class SiglipScorer:
    def __init__(self, labels, prompt_template="a photo of {label}.", device="cpu",
                 model_dir=None, tokenizer_dir=None):
        from transformers import AutoTokenizer, AutoImageProcessor, SiglipModel
        model_path, tokenizer_path = resolve_siglip(model_dir, tokenizer_dir)
        self.labels = tuple(labels)
        self.prompts = [prompt_template.format(label=label).lower() for label in self.labels]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
        # The pinned fixed-resolution SigLIP2 release uses the SiglipModel architecture.
        self.model = SiglipModel.from_pretrained(model_path, local_files_only=True,
                                                use_safetensors=True).to(device).eval()
        self.text_inputs = self.tokenizer(self.prompts, padding="max_length", max_length=64,
                                          truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            self.text_features = torch.nn.functional.normalize(
                self.model.get_text_features(**self.text_inputs), dim=-1)
        self.provenance = {
            "model_id": SIGLIP_ID, "revision": SIGLIP_REVISION,
            "weights_sha256": SIGLIP_WEIGHTS_SHA256,
            "weights_directory": str(model_path), "tokenizer_directory": str(tokenizer_path),
            "tokenizer_sha256": TOKENIZER_HASHES, "prompts": self.prompts,
            "score": "pre-sigmoid image-text match logit", "max_text_length": 64,
        }

    @torch.inference_mode()
    def score(self, images, batch_size=8):
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        iterator = iter(images)
        results = []
        while batch := list(itertools.islice(iterator, batch_size)):
            pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
            features = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
            logits = features @ self.text_features.T
            logits = logits * self.model.logit_scale.exp() + self.model.logit_bias
            results.append(logits.cpu().numpy())
        if not results:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        return np.concatenate(results, axis=0)

    @torch.inference_mode()
    def verify_forward(self, image):
        """Check cached-text scoring against the public full model forward pass."""
        pixels = self.processor(images=image, return_tensors="pt").to(self.device)
        expected = self.model(**pixels, **self.text_inputs).logits_per_image.cpu().numpy()
        actual = self.score([image])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        return float(np.max(np.abs(actual - expected)))
