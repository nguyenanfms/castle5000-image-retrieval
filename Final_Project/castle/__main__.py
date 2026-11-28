import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def doctor(cfg):
    import importlib.metadata
    healthy = True
    print("Python:", sys.version.split()[0], "|", sys.executable)
    print("Project:", cfg.root)
    print("Data:", cfg.data_dir)
    print("Model cache:", cfg.cache_dir)
    for name in ["torch", "torchvision", "open_clip_torch", "fastapi", "uvicorn", "numpy", "Pillow", "httpx"]:
        try:
            print(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name, "MISSING")
            healthy = False
    try:
        import torch
        print("CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
        else:
            print("CPU mode van chay duoc; khong can cai lai torch ngay.")
        import torchvision
        import open_clip
        print("torch/torchvision/open_clip imports: OK")
    except Exception as exc:
        print("MODEL IMPORT ERROR:", str(exc))
        healthy = False
    try:
        from .catalog import scan
        rows = scan(cfg)
        from collections import Counter
        print("Images:", len(rows), "| Batches:", dict(Counter(r["batch"] for r in rows)))
        print("Excluded calibration/manual:", sum(r["excluded"] for r in rows))
        print("Ambiguous DRES IDs:", sum(r["ambiguous_item"] for r in rows))
        print("Unknown DRES IDs:", sum(r["item_name"] is None for r in rows))
        if any(r["ambiguous_item"] or r["item_name"] is None for r in rows):
            healthy = False
    except Exception as exc:
        print("DATA ERROR:", str(exc))
        healthy = False
    return healthy


def evaluate(cfg, query_file):
    from .index import SearchIndex
    idx = SearchIndex(cfg)
    queries = [json.loads(line) for line in Path(query_file).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not queries:
        raise ValueError("File query rong.")
    hits = {1: 0, 5: 0, 10: 0, 50: 0}
    rr, details = [], []
    for q in queries:
        relevant = set(q["relevant"])
        if not relevant:
            raise ValueError("Moi query can it nhat 1 ma anh relevant.")
        known_ids = {r["item_name"] for r in idx.records}
        if not relevant <= known_ids:
            raise ValueError(f"Ma anh relevant khong co trong index: {sorted(relevant-known_ids)}")
        results = idx.search(q["query"], k=200)
        rank = next((i+1 for i, r in enumerate(results) if r["item_name"] in relevant), None)
        for k in hits:
            hits[k] += int(rank is not None and rank <= k)
        rr.append(1/rank if rank else 0)
        details.append({"query": q["query"], "first_relevant_rank": rank})
    report = {"n_queries": len(queries), **{f"Hit@{k}": v/len(queries) for k, v in hits.items()},
              "MRR@200": sum(rr)/len(rr), "details": details}
    from .index import atomic_json
    atomic_json(cfg.artifacts_dir / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description="Castle Local · DS314")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("scan")
    b = sub.add_parser("build")
    b.add_argument("--device", choices=["cpu", "cuda", "auto"])
    b.add_argument("--batch-size", type=int)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("evaluate")
    e.add_argument("queries")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.command == "doctor":
        raise SystemExit(0 if doctor(cfg) else 1)
    elif args.command == "scan":
        from .catalog import scan
        from .index import atomic_json
        rows = scan(cfg)
        atomic_json(cfg.artifacts_dir / "catalog_preview.json", rows)
        print(f"{len(rows)} anh. Xem artifacts/catalog_preview.json")
    elif args.command == "build":
        if args.device:
            cfg.device = args.device
        if args.batch_size:
            cfg.batch_size = args.batch_size
        from .index import build
        build(cfg)
    elif args.command == "serve":
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(cfg), host="127.0.0.1", port=args.port, access_log=False)
    elif args.command == "evaluate":
        evaluate(cfg, args.queries)


if __name__ == "__main__":
    main()
