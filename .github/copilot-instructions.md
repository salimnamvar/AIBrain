# AIBrain - Comprehensive Project Instructions for AI Assistants

**Version:** 2.0.0  
**Last Updated:** October 18, 2025  
**Python Version:** >= 3.13  
**License:** Apache-2.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Design Principles](#2-architecture--design-principles)
3. [Project Structure](#3-project-structure)
4. [Core Components](#4-core-components)
5. [Development Environment](#5-development-environment)
6. [Coding Standards & Conventions](#6-coding-standards--conventions)
7. [Testing Strategy](#7-testing-strategy)
8. [Build & Deployment](#8-build--deployment)
9. [Common Patterns & Workflows](#9-common-patterns--workflows)
10. [Troubleshooting & Known Issues](#10-troubleshooting--known-issues)
11. [Contributing Guidelines](#11-contributing-guidelines)
12. [Critical Rules for AI Agents](#12-critical-rules-for-ai-agents)

---

## 1. Project Overview

### 1.1 Purpose

AIBrain is a production-ready Python framework for computer vision and machine learning applications. It provides:

- **Reusable Components**: Modular building blocks for CV/ML pipelines
- **Model Wrappers**: Unified interfaces for detection, tracking, pose estimation, and re-identification
- **Production Pipeline Support**: Sync/async execution, queue-based processing, and concurrent inference
- **Backend Flexibility**: OpenVINO, OVMS, TensorFlow, PyTorch, Ultralytics support

### 1.2 Key Features

- **Idiomatic Python**: Modern Python 3.13+ with type hints, dataclasses, and protocols
- **Frozen Data Structures**: Immutable geometry and image classes for safety and performance
- **NumPy Integration**: Custom array protocol implementation for seamless NumPy interoperability
- **Async-First Design**: Native async/await support with concurrent execution options
- **Comprehensive Profiling**: Built-in performance monitoring and profiling utilities
- **Configuration Management**: Centralized configuration system with type parsing

### 1.3 Primary Use Cases

1. **Real-time Object Detection**: YOLO, RT-DETR with OpenVINO acceleration
2. **Multi-Object Tracking**: OCSORT with optional re-identification
3. **Pose Estimation**: RTMPose, MoveNet integration
4. **Video Processing**: Camera capture, video I/O with OpenCV and Decord backends
5. **Pipeline Orchestration**: Queue-based processing with multiple workers

---

## 2. Architecture & Design Principles

### 2.1 Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│              (sln/, custom applications)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML Model Layer (aib.ml)                   │
│        Detection, Tracking, Pose, Re-ID Models               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               CV Processing Layer (aib.cv)                   │
│    Geometry, Image, Video, Visualization Utilities           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Infrastructure Layer (aib.sys, aib.cnt)            │
│      Base Classes, I/O, Queues, Lifecycle Management         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│        Cross-Cutting Concerns (aib.cfg, aib.perf, aib.misc) │
│       Configuration, Profiling, Logging, Type Checking       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Design Principles

#### 2.2.1 Single Responsibility Principle
- Each class has one clear responsibility
- Separate concerns: preprocessing, inference, postprocessing
- Dedicated base classes for different model types

#### 2.2.2 Open/Closed Principle
- Base classes define extensible interfaces
- Subclasses override specific methods (`preproc`, `infer`, `postproc`)
- Configuration-driven behavior when possible

#### 2.2.3 Dependency Inversion
- Depend on abstractions (`BaseIO`, `BaseModel`, `BaseObject`)
- Heavy dependencies (TensorFlow, PyTorch, OpenVINO) are lazy-loaded
- Optional dependencies handled gracefully

#### 2.2.4 Composition Over Inheritance
- Models compose `Configuration`, `Profiler`, and `Logger` via `BaseObject`
- `QueueIO` composition for async/sync/mp modes
- Executor pattern for concurrency

#### 2.2.5 Immutability by Default
- Frozen dataclasses for geometry (`BBox2D`, `Point2D`, `Pose2D`)
- `Image2D` is immutable with copy-on-modify semantics
- Thread-safe data structures

---

## 3. Project Structure

### 3.1 Top-Level Organization

```
AIBrain/
├── aib/                    # Core library code
│   ├── cfg/                # Configuration management
│   ├── cnt/                # Containers (I/O, queues, lists, dicts)
│   ├── cv/                 # Computer vision utilities
│   ├── ds/                 # Dataset loaders and recorders
│   ├── legacy/             # Backward compatibility (DO NOT MODIFY)
│   ├── misc/               # Utilities (logging, types, singletons)
│   ├── ml/                 # Machine learning models
│   ├── perf/               # Performance profiling
│   └── sys/                # System core (base classes, lifecycle)
├── licenses/               # License information for dependencies
├── sln/                    # Example solutions and applications
├── tests/                  # Unit and integration tests
├── .github/                # CI/CD configuration and instructions
├── .gitignore              # Git ignore rules
├── .pylintrc               # Pylint configuration
├── LICENSE                 # Apache 2.0 license
├── pyproject.toml          # Build configuration and dependencies
├── README.md               # User-facing documentation
└── requirements.txt        # Pinned dependencies
```

### 3.2 Package Hierarchy

#### 3.2.1 `aib.sys` - System Core
- **`b_obj.py`**: `BaseObject` - foundation for all objects (logging, config, profiling)
- **`b_mdl.py`**: `BaseModel` - base for all models (lifecycle, I/O, concurrency)
- **`b_pipe.py`**: `BasePipe` - pipeline components
- **`b_job.py`**: `BaseJob` - asynchronous job execution
- **`b_subsys.py`**: `BaseSubsys` - subsystem management

#### 3.2.2 `aib.ml` - Machine Learning
- **`ml/utils/b_ml_mdl.py`**: `BaseMLModel` - foundation for ML models
- **`ml/det/`**: Object detection (YOLO, RT-DETR)
- **`ml/trk/`**: Object tracking (OCSORT)
- **`ml/det/utils/`**: Detection utilities (NMS, preprocessing)
- **`ml/trk/utils/`**: Tracking utilities (association, state management)

#### 3.2.3 `aib.cv` - Computer Vision
- **`cv/img/image.py`**: `Image2D` - immutable image container with NumPy protocol
- **`cv/img/frame.py`**: `Frame2D` - timestamped image for video processing
- **`cv/geom/`**: Geometry primitives (boxes, points, poses, lines, contours)
- **`cv/plot/`**: Visualization utilities
- **`cv/vid/`**: Video capture and I/O (OpenCV, Decord backends)

#### 3.2.4 `aib.cnt` - Containers & I/O
- **`io.py`**: `BaseIO`, `QueueIO` - input/output abstractions
- **`b_data.py`**: `BaseData` - base for data classes
- **`b_seq_data.py`**: `BaseSeqData` - sequence data base
- **`b_list.py`**, **`b_dict.py`**: Enhanced collections with size limits

#### 3.2.5 `aib.cfg` - Configuration
- **`config.py`**: `Configuration` - singleton configuration loader
- **`type_parser.py`**: Type parsing utilities for config values

#### 3.2.6 `aib.perf` - Performance Profiling
- **`profile.py`**: `Profiler` - singleton profiler with decorators and context managers

#### 3.2.7 `aib.legacy` - Legacy Models
- **DO NOT MODIFY** - backward compatibility only
- Contains: FusionDet, HTD0001, PDA0001, PDR0002, PDR0013, PPYOLOE, PVBDC0078, YOLOv8 (old API)
- Legacy pose: MNetSPTv4, MoveNetSinglePose, RTMPose
- Legacy tracking: OCSORT (old API), OCSORTSeg
- Legacy reid: MobileNetv1, OVReidFeatExt, Reid0001

---

## 4. Core Components

### 4.1 Base Object (`aib.sys.b_obj.BaseObject`)

**Purpose**: Foundation class providing logging, configuration, and profiling.

**Key Features**:
- **Logger**: Automatic logger creation with name-based scoping
- **Configuration**: Access to singleton `Configuration` instance
- **Profiler**: Optional performance profiling support
- **Initialization Time**: Tracks object creation timestamp

**Constructor Pattern**:
```python
def __init__(
    self,
    a_id: Optional[int] = None,
    a_name: str = 'ClassName',
    a_use_prof: bool = False,
    a_use_cfg: bool = True,
    a_use_log: bool = True,
    **kwargs: Any,
) -> None:
    super().__init__(**kwargs)
    self._id = a_id
    self._name = a_name
    self._use_prof = a_use_prof
    self._use_cfg = a_use_cfg
    self._use_log = a_use_log
    # Initialize logger, config, profiler if enabled
```

**Usage**:
```python
class MyComponent(BaseObject):
    def __init__(self, a_param: str, **kwargs):
        super().__init__(a_name="MyComponent", a_use_log=True, **kwargs)
        self._param = a_param
        if self.logger:
            self.logger.info(f"MyComponent initialized with {a_param}")
```

### 4.2 Base Model (`aib.sys.b_mdl.BaseModel`)

**Purpose**: Foundation for all models with lifecycle, I/O, and concurrency management.

**Key Attributes**:
- **`call_mode`**: `"sync"` or `"async"` - execution mode
- **`io_mode`**: `"args"`, `"queue"`, or `"ipc"` - input feeding method
- **`proc_mode`**: `"batch"` or `"online"` - processing mode
- **`backend`**: Backend engine (e.g., `"openvino"`, `"ovms"`, `"ultralytics"`)
- **`conc_mode`**: `"thread"` or `"process"` - concurrency model
- **`io`**: Optional `BaseIO` instance for queue-based I/O
- **`stop_event`**: Graceful shutdown signal

**Abstract Methods** (must implement):
```python
@abstractmethod
def load(self, *args, **kwargs): ...

@abstractmethod
def infer(self, *args, **kwargs): ...

@abstractmethod
async def infer_async(self, *args, **kwargs): ...

@abstractmethod
def run(self, *args, **kwargs): ...

@abstractmethod
async def run_async(self, *args, **kwargs): ...
```

### 4.3 Base ML Model (`aib.ml.utils.b_ml_mdl.BaseMLModel`)

**Purpose**: Specialized base class for machine learning models.

**Additional Attributes**:
- **`model_uri`**: Path or URI to model file
- **`model_size`**: Expected input size `IntSize(width, height)`
- **`model_in_layers`**, **`model_out_layers`**: Layer names
- **`device`**: Target device (`"CPU"`, `"GPU"`, `"AUTO"`)
- **`precision`**: Model precision (`"FP32"`, `"FP16"`, `"INT8"`)
- **`infer_timeout`**: Maximum inference time
- **`backend_core`**: Backend-specific core object (e.g., `ov.Core`)

**Abstract Methods** (in addition to BaseModel):
```python
@abstractmethod
def preproc(self, *args, **kwargs): ...

@abstractmethod
def postproc(self, *args, **kwargs): ...

@abstractmethod
def train(self, *args, **kwargs): ...

@abstractmethod
def test(self, *args, **kwargs): ...

@abstractmethod
def create_infer_request(self, *args, **kwargs): ...
```

### 4.4 Image2D (`aib.cv.img.image.Image2D`)

**Purpose**: Immutable 2D image container with NumPy protocol integration.

**Key Features**:
- **Frozen Dataclass**: Immutable by design (thread-safe)
- **NumPy Protocol**: Implements `__array__`, `__array_ufunc__`, `__array_function__`
- **Metadata Preservation**: Filename tracking across operations
- **Arithmetic Operations**: Overloaded operators return `Image2D` instances
- **Slicing Support**: Preserves type on indexing operations

**Properties**:
```python
@property
def width(self) -> int: ...

@property
def height(self) -> int: ...

@property
def size(self) -> Size[int]: ...

@property
def aspect_ratio(self) -> float: ...

@property
def channels(self) -> int: ...
```

**Usage**:
```python
# Create from numpy array
image = Image2D(data=np.zeros((480, 640, 3), dtype=np.uint8))

# NumPy operations return Image2D
blurred = np.mean(image, axis=2)  # Returns Image2D
normalized = image / 255.0        # Returns Image2D

# Explicit conversion when needed
cv2_image = np.asarray(image)     # Convert to ndarray for OpenCV
```

### 4.5 QueueIO (`aib.cnt.io.QueueIO`)

**Purpose**: Multi-mode queue abstraction for sync/async/multiprocessing I/O.

**Modes**:
- **`sync`**: Standard `queue.Queue` for thread-safe communication
- **`async`**: `asyncio.Queue` for async/await workflows
- **`mp`**: `multiprocessing.Queue` for inter-process communication

**Key Methods**:
```python
# Synchronous
def put_input(self, a_input: IT) -> None: ...
def get_input(self) -> Optional[IT]: ...
def put_output(self, a_output: OT) -> None: ...
def get_output(self) -> Optional[OT]: ...

# Asynchronous
async def put_input_async(self, a_input: IT) -> None: ...
async def get_input_async(self) -> Optional[IT]: ...
async def put_output_async(self, a_output: OT) -> None: ...
async def get_output_async(self) -> Optional[OT]: ...

# Completion signaling
def input_done(self) -> None: ...
async def input_done_async(self) -> None: ...
```

**Sentinel Pattern**:
- Special sentinel object signals queue completion
- Automatically sent `num_consumers` times
- Consumers should check for sentinel in get operations

### 4.6 Configuration (`aib.cfg.config.Configuration`)

**Purpose**: Singleton configuration management from `.properties` files.

**File Format**:
```properties
# Comments start with #
# Nested keys use dot notation
models.detection.conf_thre=0.5
models.detection.nms_thre=0.4
models.tracking.iou_thre=0.3
system.device=AUTO
```

**Usage**:
```python
from aib.cfg import Configuration

# Initialize singleton
config = Configuration.get_instance()
config.load2("config.properties")

# Access nested values
conf_thre = config.models.detection.conf_thre  # 0.5
device = config.system.device                   # "AUTO"
```

### 4.7 Profiler (`aib.perf.profile.Profiler`)

**Purpose**: Singleton profiler for performance monitoring.

**Features**:
- **Decorators**: `@Profiler.profile` for functions/methods/classes
- **Context Managers**: `with Profiler.start("name")` for code blocks
- **Async Support**: `async with Profiler.start_async("name")`
- **Metrics**: Execution time, CPU time, memory usage, I/O wait
- **Export**: CSV export for analysis

**Usage**:
```python
from aib.perf.profile import Profiler

profiler = Profiler.get_instance()

# Decorator
@profiler.profile
def my_function():
    # function code
    pass

# Context manager (preferred for models)
def infer(self, image):
    if self.use_prof:
        self.profiler.start("inference")
    try:
        result = self._do_inference(image)
    finally:
        if self.use_prof:
            self.profiler.end("inference")
    return result

# Export results
profiler.export("profiling_results.csv")
```

---

## 5. Development Environment

### 5.1 System Requirements

- **Python**: >= 3.13
- **Operating System**: Linux (primary), Windows (supported), macOS (untested)
- **Memory**: 8GB minimum, 16GB recommended for heavy models
- **GPU**: Optional (OpenVINO supports CPU, GPU, MYRIAD, FPGA)

### 5.2 Environment Setup

#### 5.2.1 Quick Start (Minimal Dependencies)

```bash
# Clone repository
git clone https://github.com/salimnamvar/AIBrain.git
cd AIBrain

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install minimal dependencies
pip install numpy opencv-python pytest black isort docformatter

# Verify installation
python -c "import aib; print(aib.__file__)"
```

#### 5.2.2 Full Development Setup (All Dependencies)

```bash
# Use Conda for heavy dependencies (recommended)
conda create -n aibrain python=3.13 -y
conda activate aibrain

# Install all dependencies
pip install -r requirements.txt

# Optional: Install in editable mode
pip install -e .
```

#### 5.2.3 Dependency Categories

**Lightweight (Always Required)**:
- `numpy>=2.3.2`
- `opencv-python>=4.12.0`
- `scipy>=1.16.1`
- `pyparsing>=3.2.3`

**Computer Vision**:
- `decord>=0.6.0` - Video I/O backend
- `filterpy>=1.4.5` - Kalman filtering

**Machine Learning (Optional)**:
- `openvino>=2025.3.0` - Inference runtime
- `tensorflow>=2.20.0` - TensorFlow backend
- `torch>=2.8.0+cpu` - PyTorch backend
- `ultralytics>=8.3.195` - YOLO models

**Utilities**:
- `pandas>=2.3.2`, `pyarrow>=21.0.0` - Data handling
- `grpcio>=1.74.0` - OVMS communication
- `psutil` - System monitoring

### 5.3 Verification Steps

#### 5.3.1 Core Import Test (Fast)
```bash
python -c "import aib; print(aib.__file__)"
# Expected: /path/to/AIBrain/aib/__init__.py
```

#### 5.3.2 Image2D Smoke Test (NumPy Only)
```bash
python -c "from aib.cv.img.image import Image2D; import numpy as np; print(Image2D(data=np.zeros((10,20))).width)"
# Expected: 20
```

#### 5.3.3 Dependency Check
```bash
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
# Expected: 2.3.2 4.12.0.88 (or similar)
```

#### 5.3.4 Heavy Runtime Test (Requires Full Install)
```bash
python -c "import openvino, tensorflow, torch; print('All backends available')"
# Expected: All backends available
```

### 5.4 IDE Configuration

#### 5.4.1 VS Code (`settings.json`)
```json
{
  "python.pythonPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "120"],
  "editor.formatOnSave": true,
  "python.sortImports.args": ["--profile", "google"],
  "[python]": {
    "editor.rulers": [120],
    "editor.tabSize": 4
  }
}
```

#### 5.4.2 PyCharm
- **Code Style**: Project settings → Python → Line length: 120
- **Imports**: Use `isort` with Google profile
- **Type Checking**: Enable type hints validation
- **Docstrings**: Google style

---

## 6. Coding Standards & Conventions

### 6.1 Python Style Guide

#### 6.1.1 PEP 8 Compliance
- **Line Length**: 120 characters (configured in `pyproject.toml`)
- **Indentation**: 4 spaces (NO TABS)
- **Naming**:
  - `snake_case` for functions, methods, variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
  - `_leading_underscore` for private/internal members

#### 6.1.2 Formatting Tools (Mandatory)

**Black** (code formatter):
```bash
black --line-length 120 aib/
```

**isort** (import sorter):
```bash
isort --profile google aib/
```

**docformatter** (docstring formatter):
```bash
docformatter --in-place --recursive aib/
```

**Run all formatters before commit**:
```bash
black --line-length 120 . && isort . && docformatter -r --in-place .
```

### 6.2 Naming Conventions

#### 6.2.1 Constructor Parameters (CRITICAL)

**Rule**: Constructor parameters MUST use `a_*` prefix.

```python
def __init__(
    self,
    a_model_uri: str,              # ✅ Correct
    a_conf_thre: float = 0.5,      # ✅ Correct
    a_use_prof: bool = False,      # ✅ Correct
    **kwargs: Any,
) -> None:
    super().__init__(**kwargs)
    self._model_uri = a_model_uri  # Store in _* attribute
    self._conf_thre = a_conf_thre
    self._use_prof = a_use_prof
```

**Rationale**: Distinguishes constructor args from instance attributes, prevents shadowing.

#### 6.2.2 Instance Attributes

- **Private**: `self._attr` (single underscore)
- **Public**: `self.attr` (property access only)
- **Protected**: `self._attr` (used in base classes)
- **Constants**: `UPPER_CASE` at module level

#### 6.2.3 Properties vs Direct Access

**Rule**: Always use properties for public access to internal state.

```python
@property
def conf_thre(self) -> Optional[float]:
    """Optional[float]: Confidence threshold for detections."""
    return self._conf_thre
```

### 6.3 Type Hints (Mandatory)

#### 6.3.1 Function Signatures
```python
from typing import Optional, Tuple, List, Dict, Any

def process_image(
    a_image: Image2D,
    a_conf_thre: float = 0.5,
    a_classes: Optional[Tuple[int, ...]] = None,
) -> Tuple[IntBBox2DList, npt.NDArray[np.float32]]:
    """Process image and return detections."""
    ...
```

#### 6.3.2 Type Aliases
```python
from typing import TypeAlias

IntBBox2DList: TypeAlias = "BBox2DList[BBox2D[Point2D[int]]]"
StopEvent: TypeAlias = Union[threading.Event, multiprocessing.Event, asyncio.Event]
```

#### 6.3.3 Generic Types
```python
from typing import TypeVar, Generic

IT = TypeVar("IT", bound=Any, default=Any)
OT = TypeVar("OT", bound=Any, default=Any)

class QueueIO(Generic[IT, OT], BaseIO[IT, OT]):
    ...
```

### 6.4 Docstrings (Google Style)

#### 6.4.1 Module Docstrings
```python
"""Module Name - Brief Description

Long description explaining the purpose and contents of the module.

Classes:
    ClassName1: Brief description.
    ClassName2: Brief description.

Functions:
    function_name: Brief description.

Type Variables:
    TypeVarName: Description.

Type Aliases:
    AliasName: Description.
"""
```

#### 6.4.2 Class Docstrings
```python
class MyModel(BaseMLModel):
    """Brief one-line description.

    More detailed description of the class, its purpose, and usage.
    Can span multiple paragraphs.

    Attributes:
        attr1 (type): Description.
        attr2 (type): Description.

    Methods:
        method1: Brief description.
        method2: Brief description.

    Example:
        >>> model = MyModel(a_model_uri="model.xml")
        >>> model.load()
        >>> result = model.infer(image)
    """
```

#### 6.4.3 Method/Function Docstrings
```python
def infer(
    self,
    a_images: Union[Image2D, Sequence[Image2D]],
    a_conf_thre: Optional[float] = None,
) -> IntBBox2DList:
    """Perform synchronous inference on input images.

    Args:
        a_images (Union[Image2D, Sequence[Image2D]]):
            Single image or sequence of images to process.
        a_conf_thre (Optional[float], optional):
            Confidence threshold override. If None, uses model default.
            Defaults to None.

    Returns:
        IntBBox2DList: List of detected bounding boxes with scores and labels.

    Raises:
        ValueError: If images are empty or invalid format.
        RuntimeError: If model is not loaded.

    Example:
        >>> detections = model.infer(image, a_conf_thre=0.7)
        >>> print(len(detections))
        5
    """
```

### 6.5 Import Organization

**Order** (enforced by `isort --profile google`):
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

# Third-party
import cv2
import numpy as np
import openvino as ov

# Local
from aib.cnt.io import QueueIO
from aib.cv.img import Image2D
from aib.ml.utils import BaseMLModel
```

### 6.6 Error Handling

#### 6.6.1 Exception Hierarchy
- Use built-in exceptions when appropriate
- Create custom exceptions in `aib.misc.common_errors`
- Always chain exceptions with `from` clause

```python
try:
    result = model.infer(image)
except ImportError as e:
    raise RuntimeError("OpenVINO is required for this backend") from e
except ValueError as e:
    raise ValueError(f"Invalid image format: {e}") from e
```

#### 6.6.2 Validation
- Validate inputs early (fail fast)
- Use assertions for internal consistency checks
- Provide clear error messages

```python
def __init__(self, a_conf_thre: float, **kwargs):
    if not 0.0 <= a_conf_thre <= 1.0:
        raise ValueError(f"conf_thre must be in [0, 1], got {a_conf_thre}")
    super().__init__(**kwargs)
```

### 6.7 Lazy Imports (CRITICAL)

**Rule**: Heavy dependencies MUST be imported inside methods, NOT at module level.

```python
# ❌ WRONG - imports at module level
import tensorflow as tf
import torch

class MyModel(BaseMLModel):
    def load(self):
        self.model = tf.keras.models.load_model(...)

# ✅ CORRECT - lazy import
class MyModel(BaseMLModel):
    def load(self):
        try:
            import tensorflow as tf
        except ImportError as e:
            raise RuntimeError("TensorFlow is required for this backend") from e
        self.model = tf.keras.models.load_model(...)
```

**Rationale**: Allows lightweight usage without forcing heavy dependencies.

---

## 7. Testing Strategy

### 7.1 Testing Pyramid

```
        ┌─────────────────┐
        │   Integration   │  5% - Heavy runtimes, E2E
        ├─────────────────┤
        │  Component      │  20% - Model wrappers, pipelines
        ├─────────────────┤
        │    Unit         │  75% - Core logic, utilities
        └─────────────────┘
```

### 7.2 Test Organization

```
tests/
├── __init__.py
├── README.md                      # Test documentation
├── test_image2d_numpy_interop.py  # Image2D comprehensive tests
├── unit/                          # Unit tests (fast, no heavy deps)
│   ├── test_base_object.py
│   ├── test_configuration.py
│   ├── test_geometry.py
│   └── test_queue_io.py
├── component/                     # Component tests (model wrappers)
│   ├── test_yolo_detector.py
│   ├── test_ocsort_tracker.py
│   └── test_video_capture.py
└── integration/                   # Integration tests (heavy deps)
    ├── test_detection_pipeline.py
    └── test_async_inference.py
```

### 7.3 Testing Guidelines

#### 7.3.1 Unit Tests (Fast, No Heavy Deps)

**Requirements**:
- Run in < 1 second each
- No external dependencies (OpenVINO, TensorFlow, PyTorch)
- Mock heavy imports

```python
import unittest
from unittest.mock import Mock, patch
import numpy as np
from aib.cv.img import Image2D

class TestImage2D(unittest.TestCase):
    def test_image_creation(self):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        image = Image2D(data=data)
        self.assertEqual(image.width, 640)
        self.assertEqual(image.height, 480)
        self.assertEqual(image.channels, 3)

    def test_numpy_interop(self):
        image = Image2D(data=np.ones((10, 20)))
        result = np.mean(image)
        self.assertIsInstance(result, Image2D)
```

#### 7.3.2 Component Tests (Model Wrappers)

**Requirements**:
- Test with mock models or lightweight models
- Use fixtures for test data
- Can have optional heavy runtime tests

```python
import unittest
from unittest.mock import Mock
from aib.ml.det import YOLO

class TestYOLODetector(unittest.TestCase):
    def setUp(self):
        self.detector = YOLO(
            a_model_uri="dummy.xml",
            a_backend="sys",  # No real backend
            a_conf_thre=0.5,
        )

    @patch('aib.ml.det.yolo.ov.Core')
    def test_load_model(self, mock_core):
        # Test model loading logic without real OpenVINO
        self.detector.load()
        mock_core.assert_called_once()
```

#### 7.3.3 Integration Tests (Heavy Deps Required)

**Requirements**:
- Marked with `@pytest.mark.integration` or similar
- Document required models/data
- Skip if dependencies not available

```python
import pytest
import os
from aib.ml.det import YOLO
from aib.cv.img import Image2D

@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists("models/yolo.xml"), reason="Model not available")
class TestYOLOIntegration:
    def test_real_inference(self):
        detector = YOLO(
            a_model_uri="models/yolo.xml",
            a_backend="openvino",
            a_conf_thre=0.5,
        )
        detector.load()
        image = Image2D.from_file("test_data/image.jpg")
        detections = detector.infer(image)
        assert len(detections) > 0
```

### 7.4 Test Execution

#### 7.4.1 Run All Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

#### 7.4.2 Run Specific Test File
```bash
python -m unittest tests.test_image2d_numpy_interop -v
```

#### 7.4.3 Run With Coverage
```bash
pip install coverage
coverage run -m unittest discover -s tests
coverage report -m
coverage html
```

### 7.5 Test Data Management

**Guidelines**:
- Keep test data small (< 1MB per file)
- Use `tests/data/` for fixtures
- Generate synthetic data when possible
- Document data requirements in test docstrings

---

## 8. Build & Deployment

### 8.1 Build System

**Tool**: `setuptools` with `pyproject.toml` configuration.

#### 8.1.1 Build Configuration (`pyproject.toml`)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aib"
version = "2.0.0"
description = "AIBrain core framework for computer vision projects"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.3.2",
    "opencv-python>=4.12.0",
    # ... (see full list in pyproject.toml)
]
```

#### 8.1.2 Build Package
```bash
# Install build tools
pip install build

# Build wheel and source distribution
python -m build

# Output: dist/aib-2.0.0-py3-none-any.whl
#         dist/aib-2.0.0.tar.gz
```

### 8.2 Installation Methods

#### 8.2.1 From Source (Development)
```bash
git clone https://github.com/salimnamvar/AIBrain.git
cd AIBrain
pip install -e .  # Editable install
```

#### 8.2.2 From Wheel (Production)
```bash
pip install dist/aib-2.0.0-py3-none-any.whl
```

#### 8.2.3 From PyPI (Future)
```bash
pip install aib  # When published
```

### 8.3 Dependencies Management

#### 8.3.1 Pin Dependencies for Reproducibility
```bash
pip freeze > requirements.txt
```

#### 8.3.2 Dependency Groups (Future)
```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0", "isort>=5.0", "pylint>=2.0"]
viz = ["matplotlib>=3.0", "seaborn>=0.11"]
all = ["aib[dev,viz]"]
```

### 8.4 Versioning

**Scheme**: Semantic Versioning (SemVer) - `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Current Version**: `2.0.0` (defined in `pyproject.toml`)

### 8.5 Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Run full test suite (`python -m unittest discover`)
- [ ] Run formatters (`black`, `isort`, `docformatter`)
- [ ] Run linters (`pylint aib`)
- [ ] Build package (`python -m build`)
- [ ] Tag release in Git (`git tag v2.0.0`)
- [ ] Push to repository (`git push --tags`)
- [ ] Upload to PyPI (`twine upload dist/*`)

### 8.6 Deployment Environments

#### 8.6.1 Development
- Full dependencies
- Editable install
- Logging level: DEBUG

#### 8.6.2 Staging
- Production-like environment
- Pinned dependencies from `requirements.txt`
- Logging level: INFO

#### 8.6.3 Production
- Minimal dependencies (only required backends)
- Wheel installation
- Logging level: WARNING or ERROR
- Optional: Docker containerization

---

## 9. Common Patterns & Workflows

### 9.1 Creating a New ML Model Wrapper

#### 9.1.1 Detection Model Template
```python
"""Machine Learning - Object Detection - MyDetector

This module implements MyDetector, a custom object detection model.

Classes:
    MyDetector: Wrapper class for MyDetector model.
"""

from typing import Optional, Tuple, Sequence, Union
import numpy as np
import numpy.typing as npt
from aib.ml.det.utils import BaseDetModel
from aib.cv.img import Image2D
from aib.cv.geom.box import IntBBox2DList
from aib.cv.geom.size import IntSize

class MyDetector(BaseDetModel):
    """MyDetector object detection model wrapper.

    Implements preprocessing, inference, and postprocessing for MyDetector.

    Attributes:
        (Inherited from BaseDetModel)
    """

    def __init__(
        self,
        a_model_uri: str,
        a_conf_thre: float = 0.5,
        a_nms_thre: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            a_model_uri=a_model_uri,
            a_conf_thre=a_conf_thre,
            a_nms_thre=a_nms_thre,
            a_name="MyDetector",
            **kwargs,
        )

    def load(self) -> None:
        """Load the model for inference."""
        # Lazy import heavy dependencies
        try:
            import openvino as ov
        except ImportError as e:
            raise RuntimeError("OpenVINO required for MyDetector") from e

        # Load and compile model
        core = ov.Core()
        model = core.read_model(self.model_uri)
        self._compiled_model = core.compile_model(model, self.device)

        if self.logger:
            self.logger.info(f"MyDetector loaded on {self.device}")

    @staticmethod
    def preproc(
        a_images: Union[Image2D, Sequence[Image2D]],
        a_model_size: IntSize,
    ) -> Tuple[npt.NDArray[np.float32], Sequence[IntSize]]:
        """Preprocess images for model input.

        Args:
            a_images: Input image(s).
            a_model_size: Target model input size.

        Returns:
            Preprocessed batch and original sizes.
        """
        # Implement preprocessing logic
        ...

    def infer(
        self,
        a_images: Union[Image2D, Sequence[Image2D]],
    ) -> IntBBox2DList:
        """Perform synchronous inference.

        Args:
            a_images: Input image(s).

        Returns:
            Detected bounding boxes.
        """
        # Preprocess
        batch, orig_sizes = self.preproc(a_images, self.model_size)

        # Inference
        outputs = self._compiled_model(batch)

        # Postprocess
        detections = self.postproc(outputs, orig_sizes, self.conf_thre)
        return detections

    @staticmethod
    def postproc(
        a_outputs: npt.NDArray[Any],
        a_orig_sizes: Sequence[IntSize],
        a_conf_thre: float,
    ) -> IntBBox2DList:
        """Postprocess model outputs to bounding boxes.

        Args:
            a_outputs: Raw model outputs.
            a_orig_sizes: Original image sizes.
            a_conf_thre: Confidence threshold.

        Returns:
            Filtered and scaled bounding boxes.
        """
        # Implement postprocessing logic
        ...
```

### 9.2 Creating a Processing Pipeline

#### 9.2.1 Synchronous Pipeline
```python
from aib.ml.det import YOLO
from aib.ml.trk import OCSORT
from aib.cv.vid import VideoCapture

# Initialize components
detector = YOLO(a_model_uri="yolo.xml", a_conf_thre=0.5)
detector.load()

tracker = OCSORT(a_iou_thre=0.3)

video = VideoCapture("input.mp4")

# Processing loop
for frame in video:
    # Detection
    detections = detector.infer(frame)

    # Tracking
    tracks = tracker.infer(frame, a_boxes=detections)

    # Visualization
    visualize(frame, tracks)

video.release()
```

#### 9.2.2 Asynchronous Pipeline with Queues
```python
import asyncio
from aib.ml.det import YOLO
from aib.cnt.io import QueueIO
from aib.cv.img import Frame2D

async def detection_pipeline():
    # Setup async detector
    io_queue = QueueIO(a_mode="async")
    detector = YOLO(
        a_model_uri="yolo.xml",
        a_call_mode="async",
        a_io_mode="queue",
        a_io=io_queue,
        a_max_workers=4,
    )
    detector.load()

    # Producer: Feed frames
    async def producer():
        video = VideoCapture("input.mp4")
        for frame in video:
            await io_queue.put_input_async(frame)
        await io_queue.input_done_async()

    # Consumer: Get results
    async def consumer():
        while True:
            result = await io_queue.get_output_async()
            if result is None:  # Sentinel
                break
            frame, detections = result
            print(f"Detected {len(detections)} objects")

    # Run pipeline
    await asyncio.gather(
        producer(),
        detector.run_async(),
        consumer(),
    )

asyncio.run(detection_pipeline())
```

### 9.3 Using Configuration System

#### 9.3.1 Configuration File (`config.properties`)
```properties
# Model paths
models.detection.uri=/path/to/yolo.xml
models.detection.device=AUTO
models.detection.precision=FP16

# Detection parameters
models.detection.conf_thre=0.5
models.detection.nms_thre=0.4
models.detection.top_k_thre=100

# Tracking parameters
models.tracking.iou_thre=0.3
models.tracking.ttl=30

# System settings
system.log_level=INFO
system.enable_profiling=true
```

#### 9.3.2 Loading Configuration
```python
from aib.cfg import Configuration
from aib.ml.det import YOLO

# Load configuration
config = Configuration.get_instance()
config.load2("config.properties")

# Use configuration values
detector = YOLO(
    a_model_uri=config.models.detection.uri,
    a_device=config.models.detection.device,
    a_conf_thre=config.models.detection.conf_thre,
    a_nms_thre=config.models.detection.nms_thre,
)
```

### 9.4 Profiling Performance

#### 9.4.1 Method-Level Profiling
```python
from aib.perf.profile import Profiler

class MyModel(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__(a_use_prof=True, **kwargs)

    def infer(self, image):
        # Start profiling
        if self.use_prof:
            self.profiler.start("total_inference")

        # Preprocessing
        if self.use_prof:
            self.profiler.start("preprocessing")
        preprocessed = self.preproc(image)
        if self.use_prof:
            self.profiler.end("preprocessing")

        # Inference
        if self.use_prof:
            self.profiler.start("model_inference")
        outputs = self._model(preprocessed)
        if self.use_prof:
            self.profiler.end("model_inference")

        # Postprocessing
        if self.use_prof:
            self.profiler.start("postprocessing")
        result = self.postproc(outputs)
        if self.use_prof:
            self.profiler.end("postprocessing")

        # End total profiling
        if self.use_prof:
            self.profiler.end("total_inference")

        return result
```

#### 9.4.2 Export Profiling Results
```python
profiler = Profiler.get_instance()

# Run your code...

# Export to CSV
profiler.export("profiling_results.csv")

# Print summary
profiler.summary("total_inference")
```

### 9.5 Custom Geometry Types

#### 9.5.1 Creating a New Geometry Class
```python
from dataclasses import dataclass, field
from aib.cv.geom import BaseGeom
from aib.cv.geom.point import Point2D

@dataclass(frozen=True)
class Circle2D(BaseGeom):
    """2D circle with center and radius.

    Attributes:
        center (Point2D): Center point of the circle.
        radius (float): Radius of the circle.
    """
    center: Point2D = field(compare=True)
    radius: float = field(compare=True)

    def area(self) -> float:
        """Calculate circle area."""
        import math
        return math.pi * self.radius ** 2

    def intersects(self, other: "Circle2D") -> bool:
        """Check if this circle intersects another."""
        distance = ((self.center.x - other.center.x) ** 2 +
                    (self.center.y - other.center.y) ** 2) ** 0.5
        return distance < (self.radius + other.radius)
```

---

## 10. Troubleshooting & Known Issues

### 10.1 Common Issues

#### 10.1.1 Import Errors with Heavy Dependencies

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'` when running tests.

**Solution**:
- Install heavy dependencies: `pip install tensorflow openvino torch`
- OR: Skip heavy tests: `pytest -m "not integration"`
- OR: Mock heavy imports in tests

#### 10.1.2 OpenCV Binary Extensions Warnings

**Problem**: Pylint reports missing members in `cv2` module.

**Solution**: This is a known limitation of static analysis tools with binary extensions.
- Add to `.pylintrc`: `extension-pkg-whitelist=cv2`
- Or use `# pylint: disable=no-member` for specific lines

#### 10.1.3 Image2D Not Working with OpenCV

**Problem**: OpenCV functions don't recognize `Image2D` objects.

**Solution**: OpenCV requires explicit conversion to `numpy.ndarray`.
```python
image = Image2D(data=np.zeros((480, 640, 3)))
# ❌ cv2.GaussianBlur(image, (5, 5), 0)  # Won't work
# ✅ cv2.GaussianBlur(np.asarray(image), (5, 5), 0)  # Correct
```

#### 10.1.4 Queue Sentinel Not Working

**Problem**: Async consumers hang, never receiving sentinel.

**Solution**: Ensure `input_done_async()` is called after all inputs are queued.
```python
# Producer
for item in items:
    await queue.put_input_async(item)
await queue.input_done_async()  # CRITICAL: Signal completion
```

### 10.2 Performance Optimization

#### 10.2.1 Slow Inference

**Checklist**:
- [ ] Model compiled with correct precision (FP16 vs FP32)
- [ ] Using correct device (GPU vs CPU)
- [ ] Batch size optimization
- [ ] Preprocessing bottleneck (profiling can identify)

#### 10.2.2 Memory Leaks

**Checklist**:
- [ ] Explicitly release video capture objects
- [ ] Close queues and executors
- [ ] Check for circular references in custom classes

### 10.3 Debugging Tips

#### 10.3.1 Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

model = YOLO(a_name="MyDetector", a_use_log=True)
# Will log detailed information
```

#### 10.3.2 Profile Performance Bottlenecks
```python
profiler = Profiler.get_instance()
with profiler.start("suspect_code"):
    suspect_function()
profiler.summary("suspect_code")
```

#### 10.3.3 Check Queue States
```python
print(f"Input queue size: {queue._input_queue.qsize()}")
print(f"Output queue size: {queue._output_queue.qsize()}")
print(f"Input done: {queue._input_done}")
```

---

## 11. Contributing Guidelines

### 11.1 Contribution Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AIBrain.git
   cd AIBrain
   git remote add upstream https://github.com/salimnamvar/AIBrain.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow coding standards (section 6)
   - Add tests for new functionality
   - Update docstrings and documentation

4. **Run Quality Checks**
   ```bash
   # Format code
   black --line-length 120 aib/
   isort --profile google aib/
   docformatter -r --in-place aib/

   # Run tests
   python -m unittest discover -s tests -v

   # Run linters
   pylint aib/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: Add new detection model wrapper"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### 11.2 Commit Message Convention

**Format**: `<type>(<scope>): <subject>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring, no behavior change
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(ml/det): Add RT-DETR detection model
fix(cv/img): Fix Image2D slicing behavior
docs(readme): Update installation instructions
test(cv/geom): Add tests for BBox2D intersection
```

### 11.3 Code Review Checklist

#### For Authors:
- [ ] Code follows project style guide
- [ ] All tests pass
- [ ] New functionality has tests
- [ ] Documentation updated (docstrings, README)
- [ ] No unnecessary dependencies added
- [ ] Profiling results included for performance-critical changes

#### For Reviewers:
- [ ] Code is readable and maintainable
- [ ] No obvious bugs or security issues
- [ ] Tests adequately cover new code
- [ ] Documentation is clear and accurate
- [ ] No backward compatibility breakage (without justification)

---

## 12. Critical Rules for AI Agents

### 12.1 Pre-Flight Checklist (ALWAYS RUN FIRST)

Before making ANY changes, verify the environment:

1. **Core Import Smoke Test**:
   ```bash
   python -c "import aib; print(aib.__file__)"
   ```

2. **Image2D Smoke Test**:
   ```bash
   python -c "from aib.cv.img.image import Image2D; import numpy as np; print(Image2D(data=np.zeros((10,20))).width)"
   ```

3. **Check Current Branch**:
   ```bash
   git branch --show-current
   ```

### 12.2 Code Modification Rules (NON-NEGOTIABLE)

#### 12.2.1 Naming Conventions
- ✅ Constructor parameters MUST use `a_*` prefix
- ✅ Instance attributes MUST use `_*` prefix for private members
- ✅ Properties for public access to internal state
- ❌ NEVER use direct attribute access without properties

#### 12.2.2 Lazy Imports
- ✅ Heavy dependencies (TensorFlow, PyTorch, OpenVINO) imported inside methods
- ✅ Wrap imports in try/except with clear error messages
- ❌ NEVER import heavy dependencies at module level

#### 12.2.3 Type Hints
- ✅ All public functions and methods have type hints
- ✅ Use `Optional[T]` for nullable types
- ✅ Use type aliases for complex types
- ❌ NEVER use bare `Any` without justification

#### 12.2.4 Docstrings
- ✅ Google-style docstrings for all public APIs
- ✅ Include Args, Returns, Raises sections
- ✅ Provide usage examples in class docstrings
- ❌ NEVER leave public APIs undocumented

#### 12.2.5 Base Class Inheritance
- ✅ Extend `BaseObject` for logging/config/profiling support
- ✅ Extend `BaseMLModel` for ML models
- ✅ Extend specialized base classes (BaseDetModel, BaseTrkModel)
- ❌ NEVER create standalone model classes without base class

### 12.3 Testing Rules

- ✅ Add unit tests for new functionality
- ✅ Mock heavy dependencies in tests
- ✅ Tests run in < 1 second each (unit tests)
- ❌ NEVER commit code that breaks existing tests
- ❌ NEVER add tests with hard-coded file paths

### 12.4 Formatting Rules (RUN BEFORE COMMIT)

```bash
# ALWAYS run these three commands before committing
black --line-length 120 .
isort --profile google .
docformatter -r --in-place .
```

### 12.5 Legacy Code Rules

- ✅ Read legacy code for understanding patterns
- ❌ NEVER modify code in `aib/legacy/` directory
- ❌ NEVER add new features to legacy modules
- ✅ Create new implementations in `aib/ml/` for new APIs

### 12.6 File Creation Rules

- ✅ Create new files with proper module docstring
- ✅ Add `__all__` export list to `__init__.py`
- ✅ Follow existing directory structure
- ❌ NEVER create files outside established structure
- ❌ NEVER create new top-level packages without approval

### 12.7 Dependency Rules

- ✅ Use existing dependencies when possible
- ✅ Document why new dependencies are needed
- ✅ Add to `pyproject.toml` dependencies list
- ❌ NEVER add dependencies without justification
- ❌ NEVER use deprecated or unmaintained packages

### 12.8 Documentation Rules

- ✅ Update README.md for user-facing changes
- ✅ Update this instruction file for significant architectural changes
- ✅ Add examples to docstrings for complex APIs
- ❌ NEVER leave TODOs in committed code without linked issues

### 12.9 Error Handling Rules

- ✅ Validate inputs early (fail fast)
- ✅ Provide clear, actionable error messages
- ✅ Chain exceptions with `from` clause
- ❌ NEVER catch exceptions without re-raising or logging
- ❌ NEVER use bare `except:` clauses

### 12.10 Profiling Rules

- ✅ Use `Profiler.start()`/`end()` for performance-critical code
- ✅ Include profiling results in PR for perf-impacting changes
- ✅ Conditional profiling based on `use_prof` flag
- ❌ NEVER leave profiling code always-on in production paths

---

## Appendix A: Quick Reference

### A.1 Essential Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Formatting
black --line-length 120 . && isort . && docformatter -r --in-place .

# Testing
python -m unittest discover -s tests -v

# Linting
pylint aib/

# Build
python -m build
```

### A.2 Common Imports

```python
# Core
from aib.sys import BaseObject, BaseModel
from aib.ml.utils import BaseMLModel
from aib.cnt.io import QueueIO

# Computer Vision
from aib.cv.img import Image2D, Frame2D
from aib.cv.geom.box import IntBBox2D, IntBBox2DList
from aib.cv.geom.point import IntPoint2D, FloatPoint2D
from aib.cv.geom.size import IntSize

# ML Models
from aib.ml.det import YOLO, RFDETR
from aib.ml.trk import OCSORT

# Utilities
from aib.cfg import Configuration
from aib.perf.profile import Profiler
```

### A.3 File Locations

| Component | File Path |
|-----------|-----------|
| Base Object | `aib/sys/b_obj.py` |
| Base Model | `aib/sys/b_mdl.py` |
| Base ML Model | `aib/ml/utils/b_ml_mdl.py` |
| Image2D | `aib/cv/img/image.py` |
| QueueIO | `aib/cnt/io.py` |
| Configuration | `aib/cfg/config.py` |
| Profiler | `aib/perf/profile.py` |
| YOLO Detector | `aib/ml/det/yolo.py` |
| OCSORT Tracker | `aib/ml/trk/ocsort/core.py` |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Backend** | Inference runtime (e.g., OpenVINO, OVMS, TensorFlow, PyTorch) |
| **Call Mode** | Execution mode: `"sync"` (synchronous) or `"async"` (asynchronous) |
| **Conf Thre** | Confidence threshold for filtering predictions |
| **Frozen Dataclass** | Immutable dataclass with `frozen=True` |
| **I/O Mode** | Input/output method: `"args"`, `"queue"`, or `"ipc"` |
| **Lazy Import** | Importing modules inside functions rather than at module level |
| **NMS Thre** | Non-Maximum Suppression threshold for removing overlapping detections |
| **Proc Mode** | Processing mode: `"batch"` or `"online"` |
| **Sentinel** | Special object signaling queue completion |
| **Top-K Thre** | Maximum number of top predictions to keep |

---

## Appendix C: Resources

### C.1 Internal Documentation
- **README.md**: User-facing documentation
- **tests/README.md**: Test suite documentation
- **licenses/README.md**: Dependency licenses

### C.2 External Resources
- **OpenVINO Docs**: https://docs.openvino.ai/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **Google Python Style Guide**: https://google.github.io/styleguide/pyguide.html

### C.3 Contact
- **Repository**: https://github.com/salimnamvar/AIBrain
- **Issues**: https://github.com/salimnamvar/AIBrain/issues
- **Email**: salim.namvar@gmail.com

---

**End of Instructions**

This document should be treated as the authoritative reference for all development activities on the AIBrain project. When in doubt, refer to this document first, then examine existing code implementations, and finally consult external resources.
