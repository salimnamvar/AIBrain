# AIBrain Repository - Coding Agent Instructions

## Overview
Python 3.12+ computer vision framework optimized for OpenVINO. 175 Python files (~2.4MB), modular architecture for object detection, pose estimation, tracking, and re-identification. Core dependencies: OpenCV, OpenVINO, Ultralytics, NumPy, TensorFlow, PyTorch.

## Project Structure
```
aib/                    # Core package - use absolute imports (aib.*)
├── sys/               # Base classes: b_obj.py (BaseObject), b_mdl.py (BaseModel), b_pipe.py, b_job.py
├── ml/                # Production ML models
│   ├── det/          # yolo.py, rfdetr.py - OV/OVMS backends
│   └── trk/          # OCSORT tracking with re-ID
├── cv/                # Computer vision: geom/ (boxes, poses), img/ (Frame2D), vid/, plot/
├── cfg/               # Configuration singleton (config.py, type_parser.py)
├── cnt/               # Data containers, I/O queues
├── ds/                # Dataset loaders/recorders
├── legacy/            # Older implementations (det/, pos/, reid/, seg/, trk/)
├── misc/              # Utilities, logging, type checking
└── perf/              # Profiling

sln/pose_denoising/    # Example (requires local models/data)
licenses/              # Dependency licenses
```

**Config Files**: pyproject.toml, .pylintrc (120 char lines), requirements.txt, .gitignore

## Installation & Build

**CRITICAL**: `pip install -e .` fails with setuptools error: "Multiple top-level packages discovered in a flat-layout: ['sln', 'aib', 'licenses']". Package is NOT installable.

**Setup (REQUIRED before changes)**:
```bash
# 1. Install dependencies (5-10 min)
pip install -r requirements.txt
# If torch==2.8.0+cpu fails, install: pip install opencv-python numpy openvino ultralytics scipy filterpy

# 2. Verify import (dependencies must be installed first)
python -c "import sys; sys.path.insert(0, '.'); import aib"

# 3. Use package via PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Linting & Testing

**Lint with Pylint** (install: `pip install pylint`):
```bash
pylint --rcfile=.pylintrc aib/sys/b_obj.py        # Single file
pylint --rcfile=.pylintrc aib/                     # Full package
```
Style: 120 char lines, Black formatter, isort (Google profile). Core modules score 9.0+/10.0. Legacy has TODO markers (acceptable).

**NO AUTOMATED TESTS EXIST**: No tests/ dir, no pytest/tox config, no CI/CD pipelines. Validate by: (1) lint, (2) verify imports, (3) manually run examples if needed.

## Development Workflow

**Before**: Verify `python -c "import sys; sys.path.insert(0, '.'); import aib"` succeeds (install deps if fails).
**During**: Inherit from base classes (BaseObject, BaseModel), use type hints, 120-char limit, absolute imports (aib.*).
**After**: `pylint --rcfile=.pylintrc file.py`, verify imports work.

**Common Issues**: (1) Circular deps - test imports after changes, (2) Missing deps cause import errors, (3) Must use absolute aib.* imports (not installed), (4) aib/ml/ = production, aib/legacy/ = older code. Naming: b_*.py = base classes.

## Key Files

**Root**: pyproject.toml, requirements.txt, .pylintrc, .gitignore, README.md, LICENSE
**aib/sys/**: b_obj.py (BaseObject), b_mdl.py (BaseModel), b_pipe.py, b_job.py  
**aib/ml/**: det/yolo.py, det/rfdetr.py, trk/ocsort/
**aib/cfg/**: config.py (singleton), type_parser.py

## Critical Notes

1. **Trust these instructions** - setup validated, don't re-explore unless encountering undocumented errors
2. **NEVER use `pip install -e .`** - it fails; use PYTHONPATH instead
3. **Always install dependencies first** - most import errors = missing deps
4. **No tests exist** - no pytest/unittest; validate via lint + imports only
5. **Async patterns**: Look for `call_mode="async"` and `io_mode="queue"` in ML models
6. **OpenVINO optimized** - maintain this optimization
7. **Work in aib/ml/** for new features, not aib/legacy/
8. **TODO markers acceptable** in legacy code
9. **Examples need local files** - sln/pose_denoising/ won't run without models/data
10. **Time: deps 5-10min, lint 30-60sec, imports 1-2sec**

## Quick Reference
```bash
# Setup (ALWAYS do this first)
pip install opencv-python numpy openvino ultralytics scipy filterpy
python -c "import sys; sys.path.insert(0, '.'); import aib"  # Verify

# Validate changes
pylint --rcfile=.pylintrc file.py
python -c "import sys; sys.path.insert(0, '.'); import aib"

# Use in scripts
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
