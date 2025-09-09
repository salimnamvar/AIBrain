# AIBrain

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange.svg)](pyproject.toml)

AIBrain is a comprehensive Python framework designed for computer vision and machine learning applications. It provides a modular, extensible architecture that streamlines AI development with reusable components, pre-built models, and utilities for various computer vision tasks including object detection, pose estimation, tracking, and re-identification.

## ✨ Features

- **🔧 Modular Architecture**: Clean, extensible design with well-defined interfaces
- **👁️ Computer Vision**: Comprehensive geometry utilities, image/video processing, and visualization tools
- **🤖 Machine Learning**: Pre-built models for detection, pose estimation, and tracking
- **🚀 High Performance**: Optimized for OpenVINO Runtime with async inference support
- **📊 Dataset Management**: Tools for loading, recording, and managing computer vision datasets
- **⚙️ Configuration System**: Flexible configuration management with type parsing
- **📈 Performance Monitoring**: Built-in profiling and logging capabilities
- **🔗 Pipeline Support**: Base classes for building complex ML pipelines

## 🚀 Quick Start

### Installation

install from source:

```bash
git clone https://github.com/salimnamvar/AIBrain.git
cd AIBrain
```

### Basic Usage

```python
# Object Detection with YOLO
from aib.ml.det import YOLO
from aib.cv.img import Image2D

# Initialize YOLO detector
detector = YOLO(
    model_uri="path/to/yolo/model.xml",
    backend="openvino",
    conf_thre=0.5,
    nms_thre=0.4
)

# Load and process image
image = Image2D.from_file("image.jpg")
detections = detector.infer(image)

# Visualize results
from aib.cv.plot import GeometryPlotter
plotter = GeometryPlotter()
result_image = plotter.draw_bboxes(image, detections)
```

## 📁 Project Structure

```
aib/
├── cfg/           # Configuration management
├── cnt/           # Data containers and I/O utilities
├── cv/            # Computer vision utilities
│   ├── geom/      # Geometry (boxes, points, poses, etc.)
│   ├── img/       # Image processing
│   ├── plot/      # Visualization tools
│   └── vid/       # Video processing
├── ds/            # Dataset loading and recording
├── ml/            # Machine learning models
│   ├── det/       # Object detection
│   └── trk/       # Object tracking
├── misc/          # Miscellaneous utilities
├── perf/          # Performance profiling
└── sys/           # System core (base classes)
```

## 🎯 Core Components

### Computer Vision (`aib.cv`)

Comprehensive computer vision utilities for real-world applications:

- **Geometry**: 2D/3D points, bounding boxes, poses, lines, and contours
- **Image Processing**: Frame handling, image utilities with timestamp support
- **Video Processing**: Camera capture, video I/O with OpenCV and Decord backends
- **Visualization**: Plotting utilities for annotations and results visualization

### Machine Learning (`aib.ml`)

Production-ready ML models with standardized interfaces:

- **Object Detection**: YOLO and RT-DETR implementations with OpenVINO optimization
- **Object Tracking**: OCSORT-based tracking with re-identification support
- **Base Models**: Abstract classes for building custom ML pipelines

### System Core (`aib.sys`)

Foundation classes for building robust applications:

- **BaseObject**: Common functionality (logging, configuration, profiling)
- **BaseModel**: ML model interface with lifecycle management
- **BasePipe**: Pipeline components for data processing
- **BaseJob**: Asynchronous job execution framework

### Dataset Management (`aib.ds`)

Tools for handling computer vision datasets:

- **Detection Datasets**: Loading and managing object detection annotations
- **Video Datasets**: Video sequence processing and management
- **Recording**: Utilities for capturing and storing inference results

## 📚 Advanced Usage

### Async Pipeline Example

```python
import asyncio
from aib.ml.det import YOLO
from aib.cnt.io import QueueIO
from aib.cv.vid import VideoCapture

async def detection_pipeline():
    # Setup async YOLO detector
    detector = YOLO(
        model_uri="model.xml",
        call_mode="async",
        io_mode="queue"
    )
    
    # Create I/O queues
    input_queue = QueueIO()
    output_queue = QueueIO()
    
    # Setup video capture
    cap = VideoCapture("video.mp4")
    
    # Process frames asynchronously
    detector.io = QueueIO(input_queue, output_queue)
    await detector.run_async()

asyncio.run(detection_pipeline())
```

### Custom Model Integration

```python
from aib.sys import BaseModel
from aib.cv.img import Image2D

class CustomDetector(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _preproc(self, image: Image2D):
        # Custom preprocessing
        return processed_data
        
    def _postproc(self, predictions):
        # Custom postprocessing
        return results
        
    def infer(self, image: Image2D):
        input_data = self._preproc(image)
        predictions = self.model(input_data)
        return self._postproc(predictions)
```

## 🔧 Configuration

AIBrain supports flexible configuration management:

```python
from aib.cfg import Configuration

# Load configuration
config = Configuration("config.properties")

# Access nested configurations
detection_config = config.models.detection
```

## 🏗️ Model Support

### Object Detection
- **YOLO** (v8, v11): OpenVINO optimized implementation
- **RT-DETR**: Real-time detection transformer

### Pose Estimation  
- **RTMPose**: High-performance pose estimation
- **MoveNet**: Single person pose detection

### Object Tracking
- **OCSORT**: Observation-centric sort tracking
- **Re-identification**: Feature extraction and matching

### Legacy Models
Backward compatibility support for various model architectures in the `legacy/` module.

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/salimnamvar/AIBrain.git
cd AIBrain
```

### Running Examples

```bash
# Pose denoising example
cd sln/pose_denoising/
python main.py
```

## 📊 Performance

AIBrain is optimized for performance:

- **OpenVINO Integration**: Hardware-accelerated inference
- **Async Processing**: Non-blocking pipeline execution  
- **Memory Efficient**: Optimized data structures and processing
- **Profiling Tools**: Built-in performance monitoring

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Dependencies

This project uses several third-party libraries. The licenses for these dependencies can be found in the [`licenses/`](licenses/) directory. The license information is provided for transparency and to comply with open-source licensing requirements.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/salimnamvar/AIBrain/issues)
- **Documentation**: [Coming Soon]
- **Email**: salim.namvar@gmail.com

## 🙏 Acknowledgments

- OpenVINO team for the excellent inference runtime
- Ultralytics for YOLO implementations
- All contributors and the open-source community

---

**Made with ❤️ by [Salim Namvar](https://github.com/salimnamvar)**
