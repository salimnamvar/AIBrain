# AIBrain — onboarding for Copilot (actionable)

Summary

- What this repo is: A medium-sized, idiomatic Python package for computer vision and ML pipelines. Core packages: `aib.sys`, `aib.cnt`, `aib.cv`, `aib.ml`, `aib.cfg`, `aib.perf`, plus `aib.legacy` for older model adapters. Typical flow: images (as `Image2D`/`Frame2D`) -> model `preproc` -> `infer`/`infer_async` -> `postproc` -> output (often via `QueueIO`).
- Languages / runtimes: Python (>=3.13). Primary tooling managed through `pyproject.toml` and `requirements.txt`.

High-level guidance (short)

- Match typing & style: target Python 3.13 features (use `|` unions), add type hints for public APIs and Google-style docstrings.
- Constructors use `a_*` arguments and set `_*` attributes. Reuse `self.logger`, `self.cfg`, and `self.profiler` available from `BaseObject`.
- Extend `BaseModel` / `BaseMLModel` for new models and implement `load`, `preproc`, `postproc`, `infer` and `infer_async` where appropriate. See `aib/ml/det/yolo.py` for a complete example.
- Use `Image2D` (`aib/cv/img/image.py`) as the canonical image container; when receiving `Image2D` use `.data` to obtain the `numpy` array.
- Use `QueueIO` (`aib/cnt/io.py`) for cross-component queues; respect `mode` (`sync`/`async`/`mp`) and model `io_mode` (`args`/`queue`/`ipc`).
- Avoid import-time side-effects and lazy-import heavy libraries (OpenVINO, TensorFlow, torch, ultralytics, decord).

Bootstrap / build / test / lint — validated notes

The repository does not include CI workflows by default. The safe, reproducible steps below were validated (outputs included):

1) Configure a Python environment (recommended):
	- Conda (recommended for heavy ML deps):
	  - `conda create -n aibrain python=3.13 -y`
	  - `conda activate aibrain`
	- Or virtualenv: `python -m venv .venv && source .venv/bin/activate`

2) Install dependencies:
	- Full: `pip install -r requirements.txt` — WARNING: this installs heavy packages (tensorflow, torch, openvino, ultralytics) that can take minutes and may require system packages (CUDA, drivers) or special wheel URLs. Run only in CI or on developer machines prepared for ML workloads.
	- Faster local smoke checks: install minimal set for quick checks: `pip install numpy opencv-python pytest`.

3) Quick validations performed (you can re-run these):
	- Import package from project root (no install required):
		 - `python -c "import aib; print(aib.__file__)"` -> verified imports from local `aib/` (no editable install needed).
			Observed: `aib import ok, /home/.../AIBrain/aib/__init__.py` (project root imports work).
	- Minimal smoke test for `Image2D` (only requires NumPy):
		 - `python -c "from aib.cv.img.image import Image2D; import numpy as np; print(Image2D(data=np.zeros((10,20))).width)"` -> printed `20` during validation.
	- Running unit tests without installing requirements will fail on heavy deps. Example:
		 - `python -m unittest -v tests/test_image2d_numpy_interop.py`
		 - Observed error: `ModuleNotFoundError: No module named 'tensorflow'` (test imports TF, Torch, SciPy).
	- Lightweight imports present in workspace: `python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"` -> printed `2.2.x 4.12.0` on the validated environment. Attempting to import `scipy` failed in that environment.

4) Lint & format
	- Formatting is enforced via `pyproject.toml`: `black` (line-length 120), `isort` (google profile), `docformatter`.
	- Run locally: `black --line-length 120 . && isort . && docformatter -r --in-place .`
	- There is a `.pylintrc` file — run `pylint aib` if configured in CI.

Key caveats discovered during validation

- Tests and many examples require heavy ML runtimes. Do not attempt automated `pip install -r requirements.txt` in ephemeral CI runners without caching — it can take a long time and fail for platform-specific binaries.
- Some test files import `tensorflow` and `torch` directly; for quick CI feedback either run a minimal subset (see smoke checks) or provide a separate `test-requirements.txt` that excludes heavy runtimes.
- Static analysis (Pylance/pylint) may flag members from binary-only extensions (e.g., `cv2`) as missing. That's an expected false-positive in some IDEs.

Project layout (quick map)

- Root files: `pyproject.toml` (packaging & formatter configs), `requirements.txt` (runtime deps), `README.md`, `.pylintrc`, `COPILOT_INSTRUCTIONS.md` (this file), `tests/`.
- Important dirs:
  - `aib/` — main package
	 - `aib/cv/` — geometry, `img/Image2D`, `vid` utilities
	 - `aib/cnt/` — containers and `QueueIO` (`io.py`)
	 - `aib/ml/` — models (see `aib/ml/det/yolo.py` for canonical example)
	 - `aib/sys/` — base classes: `BaseObject`, `BaseModel`
	 - `aib/cfg/` — `Configuration` singleton
	 - `aib/perf/` — `Profiler` utilities
	 - `aib/legacy/` — many legacy adapters (contains many `TODO` comments)
  - `sln/` — example scripts (e.g., `sln/pose_denoising/main.py` demonstrates usage patterns and model loading paths).

Checks & PR expectations (what to run before opening a PR)

1. Run local smoke tests (see minimal `Image2D` import snippet) to validate your change does not break core imports.
2. Add unit tests that cover new behavior. If tests require heavy runtimes either:
	- add lightweight unit tests that mock heavy runtimes, or
	- document how to run the heavier test and mark them opt-in for CI.
3. Run formatters: `black`, `isort`, `docformatter`.
4. If adding dependencies, add them to `pyproject.toml` and a corresponding license file under `licenses/`.
5. Avoid import-time initialization of hardware or runtimes (do GPU init in `load()` or factories).

Search & exploration guidance for the agent

- Trust this document for the first pass. Only search the codebase if:
  - a public API, constructor, or lifecycle method (`load`, `infer`, `run`) is missing from the above guidance, or
  - tests or local smoke checks fail in a way not covered here.
- Useful quick lookups: `aib/ml/det/yolo.py` (full model lifecycle), `aib/cnt/io.py` (queue modes), `aib/cv/img/image.py` (`Image2D` behavior).

If something in these instructions does not match runtime behavior, run the minimal smoke checks first (they are fast) and then search the repository for the failing symbol.

---
If you want I can also:
- Add a small `test-requirements.txt` for CI that splits heavy vs light tests, or
- Add a GitHub Actions workflow template that runs fast checks and optionally triggers heavy jobs on larger runners.