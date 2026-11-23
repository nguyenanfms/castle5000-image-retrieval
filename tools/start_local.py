"""Unpack the Castle5000 release and start the local search application."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
DATA = ROOT / "runtime-data"
MEDIA = "castle5000-media.zip"
# Current release: SigLIP 2 whole-image and region indexes plus the Gemini image metadata. The SigLIP 2
# weights (about 4.3 GB) come from Hugging Face on the first start and are cached in .cache/huggingface.
SIGLIP = "castle5000-siglip2.zip"
SIGLIP_FILES = {
    "ids.json", "index.faiss",
    "regions-siglip2/manifest.json", "regions-siglip2/regions.json", "regions-siglip2/regions.npy",
    "regions-siglip2/context.npy", "regions-siglip2/text_parity.json",
    "image-metadata/metadata.jsonl", "image-metadata/manifest.json", "image-metadata/missing.json",
}
# Previous release (ViT-B/32 + RegionCLIP), used only when the SigLIP 2 bundle is absent.
LEGACY = "castle5000-indexes.zip"
WEIGHTS = RUNTIME / "open_clip_model.safetensors"
INDEX_FILES = {
    "ids.json", "index.faiss", "regions/index.faiss",
    "regions/manifest.json", "regions/regions.json", "regions/text_encoder.onnx",
    "regions/text_parity.json", "regions/tokenizer_bpe.txt.gz",
}
MEDIA_PATH = re.compile(r"(?:images|thumbs)/[0-9]+_[0-9]+\.jpg")


def lfs_ready(path):
    return path.is_file() and path.stat().st_size >= 1024


def release():
    """'siglip2' when its bundle is in runtime/, else 'legacy'."""
    return "siglip2" if lfs_ready(RUNTIME / SIGLIP) else "legacy"


def prepare_data():
    profile = release()
    if profile == "siglip2":
        archives = [RUNTIME / MEDIA, RUNTIME / SIGLIP]
        required = archives
    else:
        archives = [RUNTIME / MEDIA, RUNTIME / LEGACY]
        required = (*archives, WEIGHTS)
    for artifact in required:
        if not lfs_ready(artifact):
            raise ValueError(f"Missing Git LFS data for {artifact.name}. Run: git lfs pull")
    print(f"Release: {profile}", flush=True)

    DATA.mkdir(exist_ok=True)
    for archive_path in archives:
        print(f"Checking {archive_path.name}...", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if len(names) != len(members):
                raise ValueError(f"Duplicate filenames in {archive_path.name}")
            if archive_path.name == MEDIA:
                if len(names) != 10000 or not all(MEDIA_PATH.fullmatch(name) for name in names):
                    raise ValueError("The media bundle must contain 5,000 photos and 5,000 thumbnails")
            elif archive_path.name == SIGLIP:
                if names != SIGLIP_FILES:
                    raise ValueError("SigLIP 2 bundle contents do not match the release")
            elif names not in (INDEX_FILES, INDEX_FILES | {"search-detections.json"}):
                raise ValueError("Index bundle contents do not match the release")
            digest = json.dumps({member.filename: member.CRC for member in members}, sort_keys=True)
            marker = DATA / f".{archive_path.stem}.json"
            reusable = marker.is_file() and marker.read_text(encoding="utf-8") == digest
            for member in members:
                # Historical offline detections are not used or extracted.
                if member.filename == "search-detections.json":
                    continue
                target = DATA.joinpath(*PurePosixPath(member.filename).parts)
                if reusable and target.is_file() and target.stat().st_size == member.file_size:
                    continue
                if not target.resolve().is_relative_to(DATA.resolve()):
                    raise ValueError(f"Refusing to extract an external path: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                with archive.open(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                temporary.replace(target)
            marker.write_text(digest, encoding="utf-8")

    ids = json.loads((DATA / "ids.json").read_text(encoding="utf-8"))
    if len(ids) != 5000 or len(set(ids)) != len(ids):
        raise ValueError("The index must cover 5,000 distinct image IDs")
    if profile == "legacy":
        manifest = json.loads((DATA / "regions/manifest.json").read_text(encoding="utf-8"))
        if ids != manifest["ids"]:
            raise ValueError("The CLIP and RegionCLIP indexes must cover the same 5,000 image IDs")
    else:
        manifest = json.loads((DATA / "regions-siglip2/manifest.json").read_text(encoding="utf-8"))
        if manifest.get("num_images") != len(ids) or not manifest.get("complete"):
            raise ValueError("The SigLIP 2 region index is incomplete")
    for image in ids:
        if not MEDIA_PATH.fullmatch(f"images/{image}") or not all(
            (DATA / directory / image).is_file() for directory in ("images", "thumbs")
        ):
            raise ValueError(f"Missing media for {image}")


def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(f"Port {port} is busy. Choose a different port.") from error


def start_service(module, port, env):
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env,
    )


def wait_for_web(web, mock, port):
    # The first SigLIP 2 start also downloads its weights, so allow up to an hour.
    for _ in range(3600 if release() == "siglip2" else 180):
        if web.poll() is not None or (mock is not None and mock.poll() is not None):
            raise ValueError("The search app or the DRES demo stopped during startup")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                health = json.load(response)
        except (URLError, TimeoutError):
            try:
                web.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            continue
        if health.get("images") != 5000 or not health.get("region_ready"):
            raise ValueError("Expected 5,000 images and a ready region index. See the startup log.")
        return
    raise ValueError("The search app did not finish loading its indexes in time")


def stop_service(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run(args):
    env = os.environ.copy()
    env.update(
        IMAGES_DIR=str(DATA / "images"), THUMBS_DIR=str(DATA / "thumbs"),
        INDEX_PATH=str(DATA / "index.faiss"), IDS_PATH=str(DATA / "ids.json"),
        OMP_NUM_THREADS="1", SESSION_COOKIE_SECURE="false",
        SESSION_COOKIE_NAME=f"castle5000_{args.port}",
        HF_HOME=env.get("HF_HOME", str(ROOT / ".cache" / "huggingface")),
    )
    if release() == "siglip2":
        # Thresholds belong to the model: 0.1088 for SigLIP 2 regions, 0.86 for merging people across frames.
        env.update(
            EMBED_MODEL_NAME="ViT-SO400M-16-SigLIP2-384", EMBED_PRETRAINED="webli",
            REGION_INDEX_DIR=str(DATA / "regions-siglip2"), SEARCH_REGION_THRESHOLD="0.1088",
            QA_DISTINCT_SIMILARITY_THRESHOLD="0.86", IMAGE_METADATA_DIR=str(DATA / "image-metadata"),
        )
        print("First start downloads the SigLIP 2 weights (about 4.3 GB) from Hugging Face.", flush=True)
    else:
        env.update(
            EMBED_MODEL_NAME="ViT-B-32", EMBED_PRETRAINED=str(WEIGHTS),
            REGION_INDEX_DIR=str(DATA / "regions"), SEARCH_REGION_THRESHOLD="0.21761782467365265",
            QA_DISTINCT_SIMILARITY_THRESHOLD="0.76", IMAGE_METADATA_DIR=str(DATA / "image-metadata-none"),
        )
    port_available(args.port)
    mock = None
    web = None
    try:
        if not args.real_dres:
            port_available(args.dres_port)
            env.update(
                DRES_BASE_URL=f"http://127.0.0.1:{args.dres_port}",
                DRES_EVALUATION_ID="mock-eval", SESSION_SECRET=secrets.token_urlsafe(32),
                QA_LOCAL_MODE="false",
            )
            mock = start_service("tools.mock_dres:app", args.dres_port, env)
        web = start_service("backend.app.main:app", args.port, env)
        wait_for_web(web, mock, args.port)
        print(f"Open http://127.0.0.1:{args.port}/search-v2/", flush=True)
        if mock is not None:
            print("Demo login: student / student", flush=True)
        return web.wait()
    finally:
        stop_service(web)
        stop_service(mock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Search web port")
    parser.add_argument("--dres-port", type=int, default=8090, help="Local DRES demo port")
    parser.add_argument("--real-dres", action="store_true", help="Use the DRES configured in .env")
    parser.add_argument("--prepare", action="store_true", help="Extract data and exit")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.dres_port <= 65535:
        parser.error("Ports must be between 1 and 65535")
    if not args.real_dres and args.port == args.dres_port:
        parser.error("Web and DRES demo ports must differ")
    try:
        prepare_data()
        return 0 if args.prepare else run(args)
    except KeyboardInterrupt:
        return 0
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(f"Castle5000: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
