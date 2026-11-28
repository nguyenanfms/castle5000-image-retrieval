"""Run a small real CLIP check before building the corpus. No DRES access."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

from castle.catalog import scan
from castle.config import load_config
from castle.index import atomic_json
from castle.model import ClipEncoder


def main():
    cfg = load_config()
    rows = [r for r in scan(cfg) if not r["excluded"]]
    samples = [rows[i] for i in (0, len(rows)//2, len(rows)-1)]
    started = time.perf_counter()
    encoder = ClipEncoder(cfg)
    load_s = time.perf_counter() - started
    torch = encoder.torch
    if encoder.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    images = []
    try:
        for row in samples:
            with Image.open(cfg.data_dir / row["path"]) as im:
                images.append(ImageOps.exif_transpose(im).convert("RGB"))
        started = time.perf_counter()
        features = encoder.images(images)
        query = encoder.text("people preparing food around a kitchen island")
        inference_s = time.perf_counter() - started
    finally:
        for im in images:
            im.close()
    assert features.shape == (len(samples), 512), features.shape
    assert query.shape == (512,), query.shape
    assert np.isfinite(features).all() and np.isfinite(query).all()
    assert np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    assert np.isclose(np.linalg.norm(query), 1, atol=1e-5)
    report = {
        "model": cfg.model_id, "device": encoder.device,
        "samples": [r["path"] for r in samples], "shape": list(features.shape),
        "load_s": round(load_s, 3), "inference_s": round(inference_s, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1)
        if encoder.device.startswith("cuda") else None,
        "cosine": (features @ query).tolist(),
        "notice": "Real inference sanity check; these scores are not retrieval quality metrics.",
    }
    atomic_json(cfg.artifacts_dir / "clip_smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
