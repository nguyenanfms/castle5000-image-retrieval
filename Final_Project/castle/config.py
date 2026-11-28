from dataclasses import dataclass
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    metadata_csv: Path | None
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    batch_size: int = 16
    dres_url: str = "https://dres.2hbread.com/api/v2"
    collection: str = "castle5000"
    item_prefix: str = "day1_Kitchen_"
    image_metadata_dir: Path | None = None
    partner_repo_dir: Path | None = None
    metadata_weight: float = 0.5

    @property
    def model_id(self):
        return {"model_name": self.model_name, "pretrained": self.pretrained}


def load_config(path=None):
    path = Path(path or ROOT / "config.json").resolve()
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    root = path.parent

    def resolve(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    def optional_path(name):
        value = values.pop(name, "")
        return resolve(value) if value else None

    cfg = Config(root=root, data_dir=resolve(values.pop("data_dir", "data")),
                 artifacts_dir=resolve(values.pop("artifacts_dir", "artifacts")),
                 cache_dir=resolve(values.pop("cache_dir", ".cache")),
                 metadata_csv=optional_path("metadata_csv"),
                 image_metadata_dir=optional_path("image_metadata_dir"),
                 partner_repo_dir=optional_path("partner_repo_dir"),
                 **values)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep model downloads on the project drive, before torch/open_clip is imported.
    os.environ.setdefault("HF_HOME", str(cfg.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cfg.cache_dir / "torch"))
    return cfg
