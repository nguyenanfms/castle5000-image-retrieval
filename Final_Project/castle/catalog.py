import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FRAME = re.compile(r"^(?:(day\d+_.+)_)?(\d{2})_(\d+)$")
# Exact SHA-256 values of the two calibration images provided with this project.
# More images can be excluded explicitly using data/exclude.txt.
KNOWN_CALIBRATION = {
    "beed8ffc1b0b0dc9175339939a788e6e57f9c8e1a605621a386824b7533e4431",
    "5853be58e0eb1f4a379bd7a8d6497d2ea3be6c4ac36ca111010837c9ccdf5560",
}


def to_item_name(filename, prefix="day1_Kitchen_"):
    stem = Path(filename.replace("\\", "/")).stem
    match = FRAME.fullmatch(stem)
    if not match or int(match[2]) > 23:
        return None
    return stem if match[1] else prefix + stem


def frame_parts(filename):
    m = FRAME.fullmatch(Path(filename).stem)
    if not m:
        return None, None, "unknown"
    return int(m[2]), int(m[3]), m[1] or "default"


def scan(cfg):
    if not cfg.data_dir.is_dir():
        raise ValueError(f"Khong tim thay data: {cfg.data_dir}")
    excluded = set()
    exclude_file = cfg.data_dir / "exclude.txt"
    if exclude_file.exists():
        excluded = {s.strip().replace("\\", "/") for s in exclude_file.read_text(encoding="utf-8-sig").splitlines()
                    if s.strip() and not s.lstrip().startswith("#")}
    metadata = {}
    if cfg.metadata_csv:
        with cfg.metadata_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                item = row.get("imageID", "").strip()
                if not item:
                    continue
                rel = row.get("relative_path", "").strip().replace("\\", "/")
                filename = row.get("filename", "").strip()
                if not filename and row.get("hour") and row.get("frame"):
                    filename = f"{int(row['hour']):02d}_{row['frame']}.jpg"
                join = rel or filename
                if not join:
                    continue
                if join in metadata and metadata[join]["imageID"] != item:
                    raise ValueError(f"Metadata trung khoa {join}; them cot relative_path cho tung Batch.")
                metadata[join] = dict(row, imageID=item)
    records = []
    for path in sorted(cfg.data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if not path.resolve().is_relative_to(cfg.data_dir.resolve()):
            continue
        rel = path.relative_to(cfg.data_dir).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        stat = path.stat()
        hour, frame, source = frame_parts(path.name)
        meta = metadata.get(rel, metadata.get(path.name, {}))
        item = meta.get("imageID") or to_item_name(path.name, cfg.item_prefix)
        if item:
            _, _, source = frame_parts(item)
        key = hashlib.sha256(rel.encode()).hexdigest()[:20]
        signature = hashlib.sha256(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
        placeholder = hashlib.sha256(path.read_bytes()).hexdigest() in KNOWN_CALIBRATION
        records.append({"key": key, "path": rel, "filename": path.name,
                        "batch": Path(rel).parts[0] if len(Path(rel).parts) > 1 else "root",
                        "hour": hour, "frame": frame, "source": source,
                        "item_name": item, "fingerprint": signature,
                        "excluded": placeholder or rel in excluded or path.name in excluded,
                        "description": meta.get("description", "")})
    if not records:
        raise ValueError("Khong tim thay anh. Dat anh trong data/Batch1, Batch2, Batch3.")
    names = Counter(r["item_name"] for r in records if r["item_name"])
    # Never silently submit an ambiguous mapping, even if internal paths are unique.
    for r in records:
        r["ambiguous_item"] = bool(r["item_name"] and names[r["item_name"]] > 1)
    return records


def chronological(record):
    return (record["source"], record["hour"] if record["hour"] is not None else 99,
            record["frame"] if record["frame"] is not None else 10**20, record["path"])
