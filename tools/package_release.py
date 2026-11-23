"""Pack the current SigLIP 2 indexes and the image metadata into runtime/castle5000-siglip2.zip.

tools/start_local.py unpacks this bundle after a fresh clone (Git LFS). Run it after rebuilding an index
or merging new metadata:  python tools/package_release.py
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from start_local import SIGLIP, SIGLIP_FILES  # noqa: E402

DATA = ROOT / "data"
SOURCES = {
    "ids.json": DATA / "ids.json",
    "index.faiss": DATA / "index.faiss",
    **{f"regions-siglip2/{n}": DATA / "regions-siglip2" / n
       for n in ("manifest.json", "regions.json", "regions.npy", "context.npy", "text_parity.json")},
    **{f"image-metadata/{n}": DATA / "image-metadata" / n for n in ("metadata.jsonl", "manifest.json", "missing.json")},
}


def main():
    assert set(SOURCES) == SIGLIP_FILES, "package_release.py and start_local.py list different files"
    missing = [str(p) for p in SOURCES.values() if not p.is_file()]
    if missing:
        sys.exit("Missing: " + ", ".join(missing))
    ids = json.loads(SOURCES["ids.json"].read_text(encoding="utf-8"))
    regions = json.loads(SOURCES["regions-siglip2/manifest.json"].read_text(encoding="utf-8"))
    metadata = json.loads(SOURCES["image-metadata/manifest.json"].read_text(encoding="utf-8"))
    if len(ids) != 5000 or regions.get("num_images") != 5000 or not regions.get("complete"):
        sys.exit("The SigLIP 2 indexes do not cover the 5,000 images")
    if not metadata.get("complete"):
        print(f"warning: metadata is incomplete ({metadata.get('missing')} images missing)")
    target = ROOT / "runtime" / SIGLIP
    partial = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(SOURCES.items()):
            print("adding", name, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)
            archive.write(path, name)
    partial.replace(target)
    print(f"wrote {target} ({target.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
