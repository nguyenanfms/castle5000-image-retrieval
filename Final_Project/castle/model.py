import threading
import numpy as np


class ClipEncoder:
    def __init__(self, cfg):
        # Lazy import: the UI, doctor and unit tests do not download weights.
        import torch
        import open_clip
        self.torch = torch
        self.lock = threading.RLock()
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Da chon CUDA nhung torch.cuda.is_available() la False. Chay doctor truoc.")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained, device=self.device,
            cache_dir=str(cfg.cache_dir / "open_clip"))
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

    def _numpy(self, feats):
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().numpy().astype("float32")

    def images(self, images):
        t = self.torch
        with self.lock, t.inference_mode():
            batch = t.stack([self.preprocess(im) for im in images]).to(self.device)
            with t.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                features = self.model.encode_image(batch)
            return self._numpy(features)

    def text(self, query):
        t = self.torch
        with self.lock, t.inference_mode():
            # The standard CLIP tokenizer truncates at its context length.
            tokens = self.tokenizer([query]).to(self.device)
            return self._numpy(self.model.encode_text(tokens))[0]

    def clear_oom(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
