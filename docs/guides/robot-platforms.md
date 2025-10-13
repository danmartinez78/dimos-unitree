# Robot Platform Abstraction Guide

Learn how to adapt DIMOS to new robot platforms beyond Unitree Go2.

## Table of Contents

1. [Overview](#overview)
2. [Robot Interface Requirements](#robot-interface-requirements)
3. [Implementation Guide](#implementation-guide)
4. [ROS2 Integration](#ros2-integration)
5. [Non-ROS Integration](#non-ros-integration)
6. [Skill Library Creation](#skill-library-creation)
7. [Testing](#testing)
8. [Examples](#examples)

## Overview

DIMOS is designed to be **platform-agnostic**. The `Robot` base class provides a common interface that can be implemented for any robot platform.

### Supported Integration Methods

1. **ROS2 Integration**: For robots with ROS2 support (recommended)
2. **Direct SDK Integration**: For robots with proprietary SDKs
3. **REST API Integration**: For robots with HTTP APIs
4. **Simulation Integration**: For simulators (Genesis, Isaac Sim)

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DIMOS Agent Layer                      │
│              (Platform-Independent)                       │
└────────────────────┬─────────────────────────────────────┘
                     │ Skills
                     ▼
┌──────────────────────────────────────────────────────────┐
│                   Robot Base Class                        │
│              (Abstract Interface)                         │
└────────────────────┬─────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Unitree  │  │  Custom  │  │Simulator │
│   Go2    │  │  Robot   │  │  Robot   │
└──────────┘  └──────────┘  └──────────┘
```

## Robot Interface Requirements

To integrate a new robot, implement these core methods:

### Required Methods

```python
from dimos.robot.robot import Robot
import reactivex as rx

class MyRobot(Robot):
    """Your custom robot implementation."""
    
    # Motion Control (Required)
    def move(self, distance: float, speed: float = 0.5):
        """Move robot forward/backward."""
        pass
    
    # State Information (Required)
    def get_pose(self):
        """Get current robot pose."""
        pass
    
    # Video Streams (Required for vision agents)
    def get_ros_video_stream(self, fps: int = 30) -> rx.Observable:
        """Get video stream as observable."""
        pass
    
    # Skills (Required)
    def get_skills(self):
        """Get available skills."""
        pass
```

### Optional Methods

```python
class MyRobot(Robot):
    # Additional motion
    def rotate(self, angle: float):
        """Rotate robot."""
        pass
    
    def move_to_pose(self, x: float, y: float, theta: float):
        """Move to specific pose."""
        pass
    
    # State queries
    def get_battery_level(self) -> float:
        """Get battery percentage."""
        pass
    
    def get_temperature(self) -> float:
        """Get robot temperature."""
        pass
    
    # Safety
    def stop(self):
        """Emergency stop."""
        pass
    
    def is_moving(self) -> bool:
        """Check if robot is in motion."""
        pass
```

## Implementation Guide

### Step 1: Create Robot Class

```python
from dimos.robot.robot import Robot
from dimos.hardware.interface import HardwareInterface
from dimos.skills.skills import SkillLibrary
import reactivex as rx
from reactivex.subject import Subject
import logging

logger = logging.getLogger(__name__)

class MyCustomRobot(Robot):
    """Custom robot implementation.
    
    This example shows a basic robot with:
    - Direct SDK control
    - Video streaming
    - State monitoring
    """
    
    def __init__(self,
                 ip: str,
                 port: int = 8080,
                 skills: SkillLibrary = None,
                 **kwargs):
        """Initialize custom robot.
        
        Args:
            ip: Robot IP address
            port: Communication port
            skills: Robot skill library
            **kwargs: Additional arguments for Robot base class
        """
        # Initialize your robot SDK/API
        from my_robot_sdk import RobotSDK
        
        self.sdk = RobotSDK(ip=ip, port=port)
        self.sdk.connect()
        
        # Initialize video stream subject
        self._video_subject = Subject()
        
        # Initialize base Robot class
        super().__init__(
            skill_library=skills,
            **kwargs
        )
        
        # Start video streaming
        self._start_video_stream()
        
        logger.info(f"MyCustomRobot initialized at {ip}:{port}")
    
    def _start_video_stream(self):
        """Start video streaming from robot camera."""
        import threading
        
        def stream_loop():
            while True:
                try:
                    frame = self.sdk.get_camera_frame()
                    self._video_subject.on_next(frame)
                except Exception as e:
                    logger.error(f"Video stream error: {e}")
                    self._video_subject.on_error(e)
                    break
        
        thread = threading.Thread(target=stream_loop, daemon=True)
        thread.start()
```

### Step 2: Implement Motion Control

```python
class MyCustomRobot(Robot):
    # ... (continued from above)
    
    def move(self, distance: float, speed: float = 0.5):
        """Move robot forward or backward.
        
        Args:
            distance: Distance in meters (positive=forward, negative=backward)
            speed: Movement speed in m/s
        """
        try:
            # Calculate duration
            duration = abs(distance) / speed
            
            # Send velocity command via SDK
            if distance > 0:
                self.sdk.move_forward(speed=speed, duration=duration)
            else:
                self.sdk.move_backward(speed=speed, duration=duration)
            
            logger.info(f"Moving {distance}m at {speed}m/s")
            
        except Exception as e:
            logger.error(f"Move failed: {e}")
            raise
    
    def rotate(self, angle: float, speed: float = 45.0):
        """Rotate robot.
        
        Args:
            angle: Rotation angle in degrees (positive=left, negative=right)
            speed: Rotation speed in degrees/second
        """
        try:
            duration = abs(angle) / speed
            
            if angle > 0:
                self.sdk.turn_left(speed=speed, duration=duration)
            else:
                self.sdk.turn_right(speed=speed, duration=duration)
            
            logger.info(f"Rotating {angle}° at {speed}°/s")
            
        except Exception as e:
            logger.error(f"Rotation failed: {e}")
            raise
    
    def stop(self):
        """Emergency stop."""
        try:
            self.sdk.stop_all_motion()
            logger.info("Emergency stop executed")
        except Exception as e:
            logger.error(f"Stop failed: {e}")
            raise
```

### Step 3: Implement State Queries

```python
class MyCustomRobot(Robot):
    # ... (continued)
    
    def get_pose(self):
        """Get current robot pose.
        
        Returns:
            Pose object with x, y, theta attributes
        """
        try:
            pose_data = self.sdk.get_odometry()
            
            # Create pose object
            from collections import namedtuple
            Pose = namedtuple('Pose', ['x', 'y', 'theta'])
            
            return Pose(
                x=pose_data['x'],
                y=pose_data['y'],
                theta=pose_data['yaw']
            )
        except Exception as e:
            logger.error(f"Failed to get pose: {e}")
            return None
    
    def get_battery_level(self) -> float:
        """Get battery percentage.
        
        Returns:
            Battery level (0-100)
        """
        try:
            battery_data = self.sdk.get_battery_status()
            return battery_data['percentage']
        except Exception as e:
            logger.error(f"Failed to get battery: {e}")
            return 0.0
    
    def is_moving(self) -> bool:
        """Check if robot is currently moving.
        
        Returns:
            True if moving, False otherwise
        """
        try:
            status = self.sdk.get_motion_status()
            return status['is_moving']
        except Exception as e:
            logger.error(f"Failed to check motion status: {e}")
            return False
```

### Step 4: Implement Video Streaming

```python
class MyCustomRobot(Robot):
    # ... (continued)
    
    def get_ros_video_stream(self, fps: int = 30) -> rx.Observable:
        """Get video stream as RxPY observable.
        
        Args:
            fps: Target frames per second
        
        Returns:
            Observable that emits numpy arrays (frames)
        """
        from reactivex import operators as ops
        
        # Return the video subject as observable
        # Sample at specified FPS
        return self._video_subject.pipe(
            ops.sample(1.0 / fps)  # Sample interval
        )
    
    def get_latest_frame(self):
        """Get the most recent camera frame.
        
        Returns:
            Numpy array representing the frame, or None if unavailable
        """
        try:
            return self.sdk.get_camera_frame()
        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            return None
```

### Step 5: Cleanup

```python
class MyCustomRobot(Robot):
    # ... (continued)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.dispose()
    
    def dispose(self):
        """Dispose resources."""
        try:
            # Stop video stream
            self._video_subject.on_completed()
            
            # Disconnect SDK
            if hasattr(self, 'sdk'):
                self.sdk.disconnect()
            
            # Call parent dispose
            super().dispose()
            
            logger.info("MyCustomRobot disposed")
        except Exception as e:
            logger.error(f"Disposal error: {e}")
```

## ROS2 Integration

For robots with ROS2 support, inherit from `ROSControl`:

```python
from dimos.robot.ros_control import ROSControl, RobotMode
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rclpy

class MyROS2Robot(ROSControl):
    """Custom robot with ROS2 integration."""
    
    def __init__(self,
                 node_name: str = "my_robot",
                 camera_topics: dict = None,
                 **kwargs):
        """Initialize ROS2 robot.
        
        Args:
            node_name: ROS2 node name
            camera_topics: Dictionary of camera configurations
            **kwargs: Additional ROSControl arguments
        """
        # Define camera topics
        if camera_topics is None:
            camera_topics = {
                "front": {
                    "topic": "/camera/image_raw",
                    "type": Image
                }
            }
        
        # Initialize ROSControl
        super().__init__(
            node_name=node_name,
            camera_topics=camera_topics,
            state_topic="/robot_state",
            move_vel_topic="/cmd_vel",
            odom_topic="/odom",
            **kwargs
        )
        
        # Create velocity publisher
        self._vel_pub = self._node.create_publisher(
            Twist,
            self._move_vel_topic,
            10
        )
    
    def move(self, distance: float, speed: float = 0.5):
        """Move robot using ROS2."""
        import time
        
        # Create Twist message
        twist = Twist()
        twist.linear.x = speed if distance > 0 else -speed
        
        # Publish for duration
        duration = abs(distance) / speed
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self._vel_pub.publish(twist)
            time.sleep(0.1)
        
        # Stop
        twist.linear.x = 0.0
        self._vel_pub.publish(twist)
    
    def _update_mode(self, msg):
        """Update robot mode from state message.
        
        Implement based on your robot's state message format.
        """
        # Example implementation
        if msg.velocity > 0.1:
            self._mode = RobotMode.MOVING
        else:
            self._mode = RobotMode.IDLE
```

## Non-ROS Integration

For robots without ROS2:

```python
class NonROSRobot(Robot):
    """Robot without ROS2 support."""
    
    def __init__(self, connection_config, **kwargs):
        """Initialize non-ROS robot.
        
        Args:
            connection_config: SDK/API connection configuration
        """
        # Initialize your SDK
        self.api = MyRobotAPI(connection_config)
        self.api.connect()
        
        # Initialize video streaming
        self._setup_video_stream()
        
        # Initialize base class
        super().__init__(**kwargs)
    
    def _setup_video_stream(self):
        """Setup video streaming."""
        from reactivex.subject import Subject
        import threading
        
        self._video_subject = Subject()
        
        def video_loop():
            while True:
                frame = self.api.get_frame()
                self._video_subject.on_next(frame)
        
        thread = threading.Thread(target=video_loop, daemon=True)
        thread.start()
    
    def get_ros_video_stream(self, fps: int = 30) -> rx.Observable:
        """Get video stream."""
        from reactivex import operators as ops
        return self._video_subject.pipe(ops.sample(1.0 / fps))
    
    # Implement other required methods...
```

## Skill Library Creation

Create robot-specific skills:

```python
from dimos.skills.skills import SkillLibrary, AbstractRobotSkill
from pydantic import Field

# Define skills
class MyRobotMove(AbstractRobotSkill):
    """Move the robot forward or backward."""
    distance: float = Field(..., description="Distance in meters")
    speed: float = Field(default=0.5, description="Speed in m/s")
    
    def __call__(self):
        super().__call__()
        return self._robot.move(self.distance, self.speed)

class MyRobotRotate(AbstractRobotSkill):
    """Rotate the robot."""
    angle: float = Field(..., description="Angle in degrees")
    
    def __call__(self):
        super().__call__()
        return self._robot.rotate(self.angle)

class MyRobotGetPose(AbstractRobotSkill):
    """Get current robot pose."""
    
    def __call__(self):
        super().__call__()
        pose = self._robot.get_pose()
        return f"Position: x={pose.x:.2f}, y={pose.y:.2f}, θ={pose.theta:.2f}"

# Create skill library
class MyRobotSkills(SkillLibrary):
    """Skill library for MyCustomRobot."""
    
    def __init__(self, robot=None):
        super().__init__()
        self._robot = robot
        
        if robot:
            self.initialize_skills(robot)
    
    def initialize_skills(self, robot):
        """Initialize all skills."""
        self.add(MyRobotMove(robot=robot))
        self.add(MyRobotRotate(robot=robot))
        self.add(MyRobotGetPose(robot=robot))
```

## Testing

### Unit Tests

```python
import unittest
from unittest.mock import Mock, patch

class TestMyCustomRobot(unittest.TestCase):
    """Test MyCustomRobot implementation."""
    
    def setUp(self):
        """Setup test fixtures."""
        # Mock SDK
        with patch('my_robot_sdk.RobotSDK') as mock_sdk:
            self.robot = MyCustomRobot(ip="192.168.1.100")
            self.mock_sdk = self.robot.sdk
    
    def test_move_forward(self):
        """Test forward movement."""
        self.robot.move(distance=2.0, speed=0.5)
        
        # Verify SDK was called correctly
        self.mock_sdk.move_forward.assert_called_once()
    
    def test_rotate_left(self):
        """Test left rotation."""
        self.robot.rotate(angle=90.0)
        
        self.mock_sdk.turn_left.assert_called_once()
    
    def test_get_pose(self):
        """Test pose retrieval."""
        # Mock odometry data
        self.mock_sdk.get_odometry.return_value = {
            'x': 1.0,
            'y': 2.0,
            'yaw': 0.5
        }
        
        pose = self.robot.get_pose()
        
        self.assertEqual(pose.x, 1.0)
        self.assertEqual(pose.y, 2.0)
        self.assertEqual(pose.theta, 0.5)
    
    def test_video_stream(self):
        """Test video streaming."""
        import numpy as np
        
        # Mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_sdk.get_camera_frame.return_value = mock_frame
        
        # Get stream
        stream = self.robot.get_ros_video_stream()
        
        # Subscribe and verify
        received_frames = []
        stream.subscribe(on_next=lambda f: received_frames.append(f))
        
        # Trigger frame
        self.robot._video_subject.on_next(mock_frame)
        
        self.assertEqual(len(received_frames), 1)
        self.assertTrue(np.array_equal(received_frames[0], mock_frame))

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
def test_robot_with_agent():
    """Test robot integration with agent."""
    from dimos.agents.agent import OpenAIAgent
    from reactivex.subject import Subject
    
    # Initialize robot
    robot = MyCustomRobot(ip="192.168.1.100")
    
    # Create query stream
    query_subject = Subject()
    
    # Create agent
    agent = OpenAIAgent(
        dev_name="TestAgent",
        input_query_stream=query_subject.pipe(),
        skills=robot.get_skills(),
        model_name="gpt-4o-mini"
    )
    
    # Test command
    query_subject.on_next("Move forward 1 meter")
    
    # Wait and verify
    import time
    time.sleep(2)
    
    # Check robot moved
    pose = robot.get_pose()
    assert pose.x > 0, "Robot should have moved forward"
```

## Examples

### Example 1: Simple Wheeled Robot

```python
class SimpleWheeledRobot(Robot):
    """Simple differential drive robot."""
    
    def __init__(self, serial_port: str, **kwargs):
        import serial
        
        self.serial = serial.Serial(serial_port, 115200)
        super().__init__(**kwargs)
    
    def move(self, distance: float, speed: float = 0.5):
        """Send move command via serial."""
        command = f"MOVE {distance} {speed}\n"
        self.serial.write(command.encode())
    
    def rotate(self, angle: float):
        """Send rotation command."""
        command = f"ROTATE {angle}\n"
        self.serial.write(command.encode())
    
    def get_pose(self):
        """Query pose via serial."""
        self.serial.write(b"GET_POSE\n")
        response = self.serial.readline().decode().strip()
        
        # Parse: "x,y,theta"
        x, y, theta = map(float, response.split(','))
        
        from collections import namedtuple
        Pose = namedtuple('Pose', ['x', 'y', 'theta'])
        return Pose(x, y, theta)
```

### Example 2: REST API Robot

```python
import requests

class RESTAPIRobot(Robot):
    """Robot controlled via REST API."""
    
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url
        self.session = requests.Session()
        super().__init__(**kwargs)
    
    def move(self, distance: float, speed: float = 0.5):
        """Send move command via POST."""
        response = self.session.post(
            f"{self.base_url}/move",
            json={"distance": distance, "speed": speed}
        )
        response.raise_for_status()
        return response.json()['message']
    
    def rotate(self, angle: float):
        """Send rotation command."""
        response = self.session.post(
            f"{self.base_url}/rotate",
            json={"angle": angle}
        )
        response.raise_for_status()
    
    def get_pose(self):
        """Get pose via GET."""
        response = self.session.get(f"{self.base_url}/pose")
        response.raise_for_status()
        
        data = response.json()
        from collections import namedtuple
        Pose = namedtuple('Pose', ['x', 'y', 'theta'])
        return Pose(data['x'], data['y'], data['theta'])
    
    def get_ros_video_stream(self, fps: int = 30) -> rx.Observable:
        """Stream video via HTTP."""
        from reactivex.subject import Subject
        import threading
        
        video_subject = Subject()
        
        def stream_loop():
            response = self.session.get(
                f"{self.base_url}/video_stream",
                stream=True
            )
            
            for chunk in response.iter_content(chunk_size=None):
                # Parse MJPEG stream
                frame = self._parse_mjpeg_frame(chunk)
                if frame is not None:
                    video_subject.on_next(frame)
        
        thread = threading.Thread(target=stream_loop, daemon=True)
        thread.start()
        
        from reactivex import operators as ops
        return video_subject.pipe(ops.sample(1.0 / fps))
```

### Example 3: Simulation Robot

```python
class SimulationRobot(Robot):
    """Robot in simulation environment."""
    
    def __init__(self, sim_client, **kwargs):
        """Initialize simulation robot.
        
        Args:
            sim_client: Simulation environment client
        """
        self.sim = sim_client
        super().__init__(**kwargs)
    
    def move(self, distance: float, speed: float = 0.5):
        """Move in simulation."""
        self.sim.set_velocity(linear=speed)
        
        # Simulate for duration
        duration = abs(distance) / speed
        self.sim.step(duration)
        
        self.sim.set_velocity(linear=0.0)
    
    def get_pose(self):
        """Get pose from simulation."""
        sim_pose = self.sim.get_robot_pose()
        
        from collections import namedtuple
        Pose = namedtuple('Pose', ['x', 'y', 'theta'])
        return Pose(sim_pose.x, sim_pose.y, sim_pose.yaw)
    
    def get_ros_video_stream(self, fps: int = 30) -> rx.Observable:
        """Get simulated camera."""
        from reactivex.subject import Subject
        import threading
        
        video_subject = Subject()
        
        def render_loop():
            while True:
                frame = self.sim.render_camera()
                video_subject.on_next(frame)
                time.sleep(1.0 / fps)
        
        thread = threading.Thread(target=render_loop, daemon=True)
        thread.start()
        
        return video_subject.pipe()
```

## Next Steps

- **Skills**: Create platform-specific skills → [Skills Guide](skills.md)
- **Integration**: Connect with agents → [Integration Guide](integration.md)
- **Testing**: Validate your implementation → Write comprehensive tests

## Checklist

Before deploying your custom robot:

- [ ] All required methods implemented
- [ ] Video streaming working
- [ ] Skills library created
- [ ] Unit tests passing
- [ ] Integration tests with agents passing
- [ ] Error handling comprehensive
- [ ] Cleanup/dispose working
- [ ] Documentation complete

---

**Previous**: [Perception & Vision Guide](perception.md) | **Next**: [API Reference](../api/agents.md)
