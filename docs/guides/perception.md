# Perception & Vision Integration Guide

Learn how to integrate computer vision, object detection, and vision-language models (VLMs) into DIMOS agents.

## Table of Contents

1. [Overview](#overview)
2. [Video Stream Integration](#video-stream-integration)
3. [Vision-Language Models (VLMs)](#vision-language-models-vlms)
4. [Object Detection](#object-detection)
5. [Custom Perception Modules](#custom-perception-modules)
6. [Advanced Patterns](#advanced-patterns)
7. [Best Practices](#best-practices)

## Overview

DIMOS provides multiple ways to integrate vision capabilities:

1. **Video Streams**: ROS2 camera topics → RxPY observables
2. **VLM Agents**: GPT-4o, Qwen-VL for vision-based reasoning
3. **Detection Modules**: Detic, YOLO for object detection
4. **Custom Perception**: Build your own vision pipelines

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Camera (ROS2 Topic)                      │
└──────────────────────┬─────────────────────────────────────┘
                       │ Image Messages
                       ▼
┌────────────────────────────────────────────────────────────┐
│               ROSVideoProvider                              │
│           (ROS2 → RxPY Observable)                          │
└──────────────────────┬─────────────────────────────────────┘
                       │ Observable<np.ndarray>
                       ▼
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐      ┌──────────────────────┐
│   VLM Agent     │      │  Detection Module    │
│   (GPT-4o)      │      │  (Detic, YOLO)       │
└─────────────────┘      └──────────────────────┘
         │                           │
         │                           │
         └───────────┬───────────────┘
                     ▼
         ┌───────────────────────┐
         │    Robot Skills       │
         │  (Navigate, Grasp)    │
         └───────────────────────┘
```

## Video Stream Integration

### Getting Video Streams from Robot

All DIMOS robots provide video streams as observables:

```python
from dimos.robot.unitree.unitree_go2 import UnitreeGo2

# Initialize robot
robot = UnitreeGo2(ip="192.168.123.161")

# Get video stream (Observable<np.ndarray>)
video_stream = robot.get_ros_video_stream()

# Subscribe to frames
video_stream.subscribe(
    on_next=lambda frame: print(f"Frame shape: {frame.shape}")
)
```

### Configuring Video Stream

```python
# Get stream with specific FPS
video_stream = robot.get_ros_video_stream(fps=10)  # Sample at 10 FPS

# Multiple camera streams (if available)
front_camera = robot.get_ros_video_stream(camera="front")
rear_camera = robot.get_ros_video_stream(camera="rear")
```

### Processing Video Frames

```python
import cv2
from reactivex import operators as ops

# Process each frame
processed_stream = video_stream.pipe(
    # Resize
    ops.map(lambda frame: cv2.resize(frame, (640, 480))),
    
    # Convert color space
    ops.map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
    
    # Apply filters
    ops.map(lambda frame: cv2.GaussianBlur(frame, (5, 5), 0))
)

# Display frames
processed_stream.subscribe(
    on_next=lambda frame: cv2.imshow("Processed", frame)
)
```

## Vision-Language Models (VLMs)

VLMs combine vision and language understanding, enabling agents to reason about visual input.

### Using GPT-4o (Vision-Capable)

```python
from dimos.agents.agent import OpenAIAgent
from dimos.robot.unitree.unitree_go2 import UnitreeGo2

# Initialize robot
robot = UnitreeGo2(ip="192.168.123.161")

# Create VLM agent
agent = OpenAIAgent(
    dev_name="VisionAgent",
    input_query_stream=query_stream,
    input_video_stream=robot.get_ros_video_stream(),  # Add video!
    skills=robot.get_skills(),
    system_query="""
    You are a robot with vision capabilities.
    Analyze the camera feed and respond to visual queries.
    Use your skills to navigate and interact with objects you see.
    """,
    model_name="gpt-4o"  # Vision-capable model
)

# Example queries
# "What do you see?"
# "Navigate towards the red object"
# "Is there a person in front of you?"
# "Describe the room you're in"
```

### Vision Agent Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_video_stream` | Video feed observable | `robot.get_ros_video_stream()` |
| `image_detail` | Image quality | `"high"`, `"low"`, `"auto"` |
| `model_name` | VLM model | `"gpt-4o"`, `"gpt-4o-mini"` |

### Image Processing Options

```python
agent = OpenAIAgent(
    dev_name="VisionAgent",
    input_video_stream=video_stream,
    image_detail="high",  # Higher quality, more tokens
    max_input_tokens_per_request=5000,  # Increase for vision
    model_name="gpt-4o"
)
```

### Example: Vision-Based Navigation

```python
from dimos.agents.agent import OpenAIAgent
from reactivex.subject import Subject

# Query subject
query_subject = Subject()

# Vision agent
vision_agent = OpenAIAgent(
    dev_name="NavVisionAgent",
    input_query_stream=query_subject.pipe(),
    input_video_stream=robot.get_ros_video_stream(fps=5),  # 5 FPS for processing
    skills=robot.get_skills(),
    system_query="""
    You are a navigation robot with vision.
    - Analyze the camera feed for obstacles
    - Navigate safely towards goals
    - Describe what you see
    - Avoid collisions
    """,
    model_name="gpt-4o"
)

# Commands
query_subject.on_next("Navigate forward while avoiding obstacles")
# Agent will:
# 1. See current frame
# 2. Identify obstacles
# 3. Choose safe path
# 4. Execute Move skill
# 5. Repeat with next frame
```

### Example: Object-Seeking Behavior

```python
agent = OpenAIAgent(
    dev_name="ObjectSeeker",
    input_query_stream=query_subject.pipe(),
    input_video_stream=robot.get_ros_video_stream(),
    skills=robot.get_skills(),
    system_query="""
    You are searching for objects.
    When you see the target object:
    1. Confirm you found it
    2. Navigate towards it
    3. Stop when close
    """,
    model_name="gpt-4o"
)

# Seek specific object
query_subject.on_next("Find and approach the red ball")
```

## Object Detection

DIMOS includes object detection modules for programmatic visual understanding.

### Detic (Open-Vocabulary Detection)

Detic can detect any object described in text, without pre-training on specific classes.

#### Setup

```python
from dimos.perception.detection2d.detic_2d_det import Detic2DDetector

# Initialize Detic
detector = Detic2DDetector(
    config_file="path/to/config.yaml",
    model_weights="path/to/weights.pth",
    vocabulary="lvis",  # or "custom"
    confidence_threshold=0.5
)
```

#### Using Detic

```python
import cv2

# Get frame
frame = robot.get_latest_frame()

# Detect objects
detections = detector.detect(
    frame,
    classes=["person", "ball", "chair", "dog"]  # Any objects!
)

# Process detections
for det in detections:
    bbox = det['bbox']  # [x1, y1, x2, y2]
    class_name = det['class']
    confidence = det['score']
    
    print(f"Detected {class_name} at {bbox} (conf: {confidence:.2f})")
    
    # Draw on frame
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, class_name, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

#### Detic with Observable Streams

```python
from reactivex import operators as ops

# Detection stream
detection_stream = robot.get_ros_video_stream().pipe(
    ops.sample(1.0),  # Detect at 1 Hz
    ops.map(lambda frame: detector.detect(frame, classes=["person", "ball"]))
)

# Monitor detections
detection_stream.subscribe(
    on_next=lambda detections: 
        print(f"Found {len(detections)} objects")
)
```

### YOLO Detection

```python
from dimos.perception.detection2d.yolo_2d_det import YOLO2DDetector

# Initialize YOLO
detector = YOLO2DDetector(
    model_path="yolov8n.pt",  # Nano model for speed
    confidence_threshold=0.5
)

# Detect
detections = detector.detect(frame)

# YOLO has fixed classes (COCO dataset)
# "person", "bicycle", "car", "dog", "cat", etc.
```

### Creating Detection Skills

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field

class DetectObject(AbstractRobotSkill):
    """Detect a specific object in the camera view."""
    
    object_name: str = Field(..., description="Object to detect (e.g., 'person', 'ball')")
    
    def __init__(self, robot=None, detector=None, **data):
        super().__init__(robot=robot, **data)
        self.detector = detector
    
    def validate(self) -> tuple[bool, str]:
        if self._robot is None:
            return False, "Robot not initialized"
        if self.detector is None:
            return False, "Detector not available"
        return True, ""
    
    def __call__(self):
        super().__call__()
        
        # Get frame
        frame = self._robot.get_latest_frame()
        if frame is None:
            return "No camera feed available"
        
        # Detect
        detections = self.detector.detect(frame, classes=[self.object_name])
        
        if not detections:
            return f"No {self.object_name} detected"
        
        # Return results
        count = len(detections)
        locations = [det['bbox'] for det in detections]
        
        return f"Detected {count} {self.object_name}(s) at {locations}"
```

### Tracking Objects

```python
from dimos.perception.detection2d.detic_2d_det import SimpleTracker

# Initialize tracker
tracker = SimpleTracker(iou_threshold=0.3, max_age=5)

# Track over multiple frames
def process_frame(frame):
    # Detect
    detections = detector.detect(frame, classes=["person"])
    
    # Convert to tracking format [x1,y1,x2,y2,score,class_id]
    tracking_input = [
        [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
         d['score'], 0]  # class_id=0 for person
        for d in detections
    ]
    
    # Update tracker
    tracks = tracker.update(tracking_input)
    
    # Each track has: [track_id, bbox, score, class_id]
    for track in tracks:
        track_id = track[0]
        print(f"Tracking person {track_id}")

# Apply to stream
video_stream.pipe(
    ops.sample(0.5)  # 2 Hz
).subscribe(on_next=process_frame)
```

## Custom Perception Modules

Create your own perception modules for specialized tasks.

### Example: Depth Estimation

```python
import torch
from torchvision import transforms

class DepthEstimator:
    """Estimate depth from RGB images."""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        # Load your depth estimation model
        model = torch.load(model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def estimate(self, frame):
        """Estimate depth map from RGB frame."""
        # Preprocess
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            depth_map = self.model(input_tensor)
        
        # Postprocess
        depth_map = depth_map.squeeze().cpu().numpy()
        
        return depth_map

# Usage
depth_estimator = DepthEstimator("path/to/model.pth")

# Apply to stream
depth_stream = video_stream.pipe(
    ops.sample(2.0),  # 0.5 Hz
    ops.map(lambda frame: depth_estimator.estimate(frame))
)

depth_stream.subscribe(
    on_next=lambda depth: print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")
)
```

### Example: Segmentation

```python
class SemanticSegmentation:
    """Segment images into semantic regions."""
    
    def __init__(self, model_path, classes):
        self.model = self._load_model(model_path)
        self.classes = classes
    
    def segment(self, frame):
        """Segment frame into regions."""
        # Run segmentation model
        output = self.model(frame)
        
        # Parse results
        segmentation_map = output.argmax(dim=1).squeeze().cpu().numpy()
        
        return segmentation_map
    
    def get_region_info(self, segmentation_map):
        """Get info about segmented regions."""
        regions = {}
        for class_id, class_name in enumerate(self.classes):
            mask = (segmentation_map == class_id)
            pixel_count = mask.sum()
            
            if pixel_count > 0:
                regions[class_name] = {
                    'pixels': int(pixel_count),
                    'percentage': float(pixel_count / segmentation_map.size * 100)
                }
        
        return regions

# Usage
segmenter = SemanticSegmentation(
    "path/to/model.pth",
    classes=["sky", "road", "grass", "building", "person"]
)

frame = robot.get_latest_frame()
seg_map = segmenter.segment(frame)
regions = segmenter.get_region_info(seg_map)

print(regions)
# {'sky': {'pixels': 12000, 'percentage': 15.5}, ...}
```

## Advanced Patterns

### Pattern 1: Multi-Modal Fusion

Combine multiple perception modules:

```python
class MultiModalPerception:
    """Combine RGB, depth, and detection."""
    
    def __init__(self, detector, depth_estimator):
        self.detector = detector
        self.depth_estimator = depth_estimator
    
    def analyze(self, frame):
        """Comprehensive scene analysis."""
        # Object detection
        detections = self.detector.detect(frame, classes=["person", "obstacle"])
        
        # Depth estimation
        depth_map = self.depth_estimator.estimate(frame)
        
        # Combine information
        results = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get depth of detected object
            roi_depth = depth_map[y1:y2, x1:x2]
            avg_depth = roi_depth.mean()
            
            results.append({
                'class': det['class'],
                'bbox': bbox,
                'confidence': det['score'],
                'depth': float(avg_depth),
                'distance': float(avg_depth)  # In meters (after calibration)
            })
        
        return results

# Usage
perception = MultiModalPerception(detector, depth_estimator)

analysis = perception.analyze(frame)
for obj in analysis:
    print(f"{obj['class']} at {obj['distance']:.2f}m")
```

### Pattern 2: Attention-Based Processing

Focus processing on regions of interest:

```python
from reactivex import operators as ops

class AttentionProcessor:
    """Process only interesting regions."""
    
    def __init__(self, detector):
        self.detector = detector
    
    def is_interesting(self, frame):
        """Check if frame contains objects of interest."""
        detections = self.detector.detect(frame, classes=["person", "ball"])
        return len(detections) > 0
    
    def process_roi(self, frame):
        """Detailed processing of interesting frames."""
        detections = self.detector.detect(frame, classes=["person", "ball"])
        
        # Detailed analysis only for interesting frames
        detailed_results = []
        for det in detections:
            # Extract ROI
            x1, y1, x2, y2 = det['bbox']
            roi = frame[y1:y2, x1:x2]
            
            # Further analysis on ROI
            # (e.g., pose estimation, attribute recognition)
            
            detailed_results.append(det)
        
        return detailed_results

# Usage
attention = AttentionProcessor(detector)

# Process only interesting frames
interesting_stream = video_stream.pipe(
    ops.filter(lambda frame: attention.is_interesting(frame)),
    ops.map(lambda frame: attention.process_roi(frame))
)

interesting_stream.subscribe(
    on_next=lambda results: print(f"Found {len(results)} interesting objects")
)
```

### Pattern 3: Temporal Coherence

Track consistency across frames:

```python
class TemporalCoherence:
    """Maintain coherent perception over time."""
    
    def __init__(self, detector, window_size=5):
        self.detector = detector
        self.window_size = window_size
        self.detection_history = []
    
    def process(self, frame):
        """Process with temporal context."""
        # Current detections
        detections = self.detector.detect(frame, classes=["person"])
        
        # Add to history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.window_size:
            self.detection_history.pop(0)
        
        # Verify detections appear in multiple frames
        consistent_detections = self._find_consistent()
        
        return consistent_detections
    
    def _find_consistent(self):
        """Find objects detected in multiple frames."""
        if len(self.detection_history) < 3:
            return []
        
        # Simple consistency check (can be more sophisticated)
        recent = self.detection_history[-3:]
        
        # Objects present in at least 2 of last 3 frames
        consistent = []
        # Implementation details...
        
        return consistent

# Usage
temporal = TemporalCoherence(detector, window_size=5)

coherent_stream = video_stream.pipe(
    ops.map(lambda frame: temporal.process(frame))
)
```

## Best Practices

### 1. Frame Rate Management

✅ **Do**:
```python
# Sample appropriately for task
video_stream.pipe(
    ops.sample(0.5)  # 2 Hz for object detection (sufficient)
)

# Different rates for different tasks
fast_stream = video_stream.pipe(ops.sample(0.1))   # 10 Hz for navigation
slow_stream = video_stream.pipe(ops.sample(2.0))   # 0.5 Hz for scene understanding
```

❌ **Don't**:
```python
# Process every frame (30+ FPS)
video_stream.subscribe(on_next=expensive_detection)  # Too much!
```

### 2. GPU Memory Management

✅ **Do**:
```python
import torch

# Clear cache periodically
def process_batch(frames):
    results = model(frames)
    torch.cuda.empty_cache()  # Free GPU memory
    return results
```

### 3. Error Handling

✅ **Do**:
```python
def safe_detect(frame):
    try:
        return detector.detect(frame)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return []  # Return empty instead of crashing

video_stream.pipe(
    ops.map(lambda frame: safe_detect(frame))
).subscribe(...)
```

### 4. Visualization

```python
def visualize_detections(frame, detections):
    """Draw detections on frame."""
    vis_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['score']:.2f}"
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_frame

# Apply to stream
video_stream.pipe(
    ops.map(lambda frame: detector.detect(frame)),
    ops.map(lambda detections: visualize_detections(frame, detections))
).subscribe(
    on_next=lambda frame: cv2.imshow("Detections", frame)
)
```

## Troubleshooting

### No Video Feed

**Problem**: Stream not producing frames

**Solutions**:
```python
# 1. Check ROS topic
# ros2 topic list | grep image

# 2. Verify stream
robot.get_ros_video_stream().subscribe(
    on_next=lambda f: print(f"Frame: {f.shape}"),
    on_error=lambda e: print(f"Error: {e}")
)

# 3. Check camera configuration
print(robot.ros_control.camera_topics)
```

### Detection Not Working

**Problem**: No objects detected

**Solutions**:
```python
# 1. Check confidence threshold
detector.confidence_threshold = 0.3  # Lower threshold

# 2. Verify classes
print(detector.available_classes)

# 3. Test on single frame
frame = cv2.imread("test_image.jpg")
detections = detector.detect(frame)
print(f"Detections: {detections}")
```

### GPU Out of Memory

**Problem**: CUDA out of memory

**Solutions**:
```python
# 1. Reduce batch size
# 2. Lower image resolution
video_stream.pipe(
    ops.map(lambda frame: cv2.resize(frame, (640, 480)))
)

# 3. Use smaller model
detector = Detic2DDetector(model="detic_small")

# 4. Clear cache
torch.cuda.empty_cache()
```

## Next Steps

- **Skills**: Create vision-based skills → [Skills Guide](skills.md)
- **Memory**: Store visual observations → [Memory Guide](memory.md)
- **API Reference**: Perception API details → [API Reference](../api/agents.md)

---

**Previous**: [Memory & RAG Guide](memory.md) | **Next**: [Robot Platform Abstraction Guide](robot-platforms.md)
