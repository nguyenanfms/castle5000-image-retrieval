from collections import Counter
from hashlib import sha256
from importlib import import_module
from math import isfinite
import sys
from types import ModuleType

from .catalog import EXTENSIONS


def load_metadata_matcher(cfg, records):
    if not cfg.image_metadata_dir and not cfg.partner_repo_dir:
        return None
    if not cfg.image_metadata_dir or not cfg.partner_repo_dir:
        raise ValueError("Cần cấu hình cả image_metadata_dir và partner_repo_dir.")
    if not isfinite(cfg.metadata_weight) or cfg.metadata_weight < 0:
        raise ValueError("metadata_weight phải là một số hữu hạn không âm.")
    if cfg.metadata_weight == 0:
        return None

    # Load the two partner modules under their own namespace without changing sys.path.
    source = cfg.partner_repo_dir / "backend" / "app"
    if not source.is_dir():
        raise FileNotFoundError("Không tìm thấy backend/app của repo partner.")
    package = "_castle_partner_metadata_" + sha256(str(source).encode()).hexdigest()[:12]
    namespace = ModuleType(package)
    namespace.__path__ = [str(source)]
    sys.modules.setdefault(package, namespace)
    store_type = import_module(f"{package}.image_metadata").MetadataStore
    matcher_type = import_module(f"{package}.metadata_search").MetadataMatcher

    # The package covers raw filenames, including frames excluded from the CLIP index.
    filenames = [record["filename"] for record in records]
    raw_counts = Counter(p.name for p in cfg.data_dir.rglob("*")
                         if p.suffix.lower() in EXTENSIONS and p.is_file())
    known = set(filenames) | raw_counts.keys()
    store = store_type.load(cfg.image_metadata_dir, known)
    counts = Counter(filenames)
    aligned = [name if counts[name] == 1 and raw_counts[name] <= 1 else "" for name in filenames]
    matcher = matcher_type(store, aligned)
    if matcher.coverage == 0:
        raise ValueError("Metadata không khớp tên ảnh nào trong chỉ mục.")
    return matcher
