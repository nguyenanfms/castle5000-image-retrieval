# Validation — 2026-11-20, v0.1

## Verified on the user's Windows laptop, 20/11/2026

- Interpreter: `D:\UIT\.venvs\ml-gpu\Scripts\python.exe`, Python 3.14.3; torch 2.14.1+cu130, torchvision 0.29.2, OpenCLIP 3.3.0; RTX 3050 Laptop 6 GiB, CUDA True and actual CUDA CLIP image/text inference passed. Existing installed package versions were preserved; no driver change.
- All 5000 source JPGs fully decoded, 1280×720, no corrupt images, duplicate filenames or imageIDs. No related authoritative metadata found locally. 12 files match the two known calibration hashes; another 17 exact paths were visually verified and excluded manually. Final index: **4971×512**, finite/unit norm, unique keys/IDs, correct source coverage, zero errors. Original images untouched.
- First CUDA build: 218.63 s, batch 16, no OOM. Final curated rebuild: 44.13 s, all 4971 features reused from cache. Original baseline generations retained.
- `python -m unittest discover -s tests -v`: **26 tests passed** (1.521 s). Includes model-load/encoding failure recovery and doctor exit status; fixture encoders are still only logic tests.
- Windows PowerShell setup/build/doctor executed successfully. Doctor/setup now fail on diagnostics errors instead of reporting success; CPU fallback is allowed.
- Real installed Chrome 158.0.8127.36 via Playwright: desktop/mobile, gallery count, real CLIP text search, filters, thumbnails/full image, neighbors, image search, local mock connection/evaluation, canceled/confirmed submission, history/refresh/logout, empty results, injected model/network errors and recovery. Separate real browser test of an app without an index also passed. No uncaught page errors or external browser requests.
- Final server loaded model successfully with `HF_HUB_OFFLINE=1`; model/cache stored on D. Warm localhost search, k=200, 9 uncached query strings: wall p50 32.164 ms / p95 58.2604 ms; repeated query-vector cache p50 10.262 ms. Small sample, excludes cold loading/rendering.
- Internal sparse labels: dev 9 queries, Hit@50=1/9, MRR@200=0.00652135; holdout 3 queries all metrics zero. Other Hit@K on dev also zero. Metrics unchanged after calibration cleanup; these are incomplete internal labels, not official contest ground truth. See `evaluation/README.md` and `notes/EXPERIMENTS.md` before interpreting quality.
- No live DRES login/submission or authoritative collection check performed. The user must operate the provided account and confirm live mapping/task/evaluation. Other calibration variants are not guaranteed detected.

Reports/screenshots are under `artifacts/`; current handover and commands are in `notes/PROGRESS.md`.

The sections below preserve the original package's Linux validation history; their environment limitations describe that earlier workspace, not this Windows run.

## Completed

- Python 3.12.14, FastAPI 0.141.1, httpx 0.28.1, NumPy 2.3.5, Pillow 13.0.0 in the build workspace.
- `python -m unittest discover -s tests -v`: **21 tests passed**.
- Python compilation of castle/ and tests/ passed.
- `node --check web/app.js` passed.
- DOM integration checks with jsdom and a real local HTTP API, using a fixture encoder and the three uploaded kitchen images: gallery/count, text search form, image dialog logic, numeric neighbor selection, local mock login, evaluation selection, confirmed mock submit, history, hour filter and empty state passed; no captured JavaScript errors.
- DRES tests use httpx.MockTransport: correct/wrong/pending, expired session, exact imageID + collection payload, ambiguous ID block, timeout without automatic retry. No requests were sent to the contest server.
- Index tests cover all Batch subdirectories, numeric ordering, metadata imageID, changed-source reencoding, feature cache reuse, simulated CUDA OOM batch reduction and atomic generation publication.

## Not verified here

- This workspace has no torch/open_clip checkpoint or NVIDIA GPU. Actual CLIP inference, CUDA availability, GPU memory use and embedding time must be verified in the user's ml-gpu environment.
- The complete 5,000-image corpus and metadata CSV were not provided. Full-corpus coverage, duplicates, all calibration images and retrieval quality remain to be measured.
- The Windows PowerShell scripts were reviewed but not executed on Windows. The setup resolver intentionally preserves installed torch/vision/audio/numpy versions and may stop on an incompatible environment.
- Full browser rendering and responsive layout were not visually verified: the workspace browser could not connect to the isolated local test server. The jsdom check validates DOM behavior, not pixels, native dialog rendering or mobile appearance.
- DRES credentials/evaluation were not provided; live login/submission has not been tested.

Model inference and scoring in the tests are explicitly fixtures, never a substitute for real CLIP quality evaluation. The distributed application uses ClipEncoder by default; it has no fake production search mode. Local mock submission is visibly labeled and never claims an image is correct.

The package contains no image corpus, model weights, credentials or generated fixture index.
