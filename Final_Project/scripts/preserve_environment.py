"""Create constraints without changing the existing GPU stack."""
import importlib.metadata as metadata
from pathlib import Path

pins = []
for name in ("torch", "torchvision", "torchaudio", "numpy", "Pillow"):
    try:
        pins.append(name + "==" + metadata.version(name))
    except metadata.PackageNotFoundError:
        pass
if not any(pin.startswith("torch==") for pin in pins):
    raise SystemExit("Venv chua co torch. Dung setup; kiem tra dung ml-gpu truoc.")
Path(".runtime").mkdir(exist_ok=True)
Path(".runtime/gpu_constraints.txt").write_text("\n".join(pins)+"\n", encoding="utf-8")
