# Skills Extension Guide

Learn how to create custom skills to extend DIMOS robot capabilities beyond the built-in functionality.

## Table of Contents

1. [Overview](#overview)
2. [Skill Types](#skill-types)
3. [Creating Class-Based Skills](#creating-class-based-skills)
4. [Skill Libraries](#skill-libraries)
5. [WebRTC API Skills](#webrtc-api-skills)
6. [Composite Skills](#composite-skills)
7. [Best Practices](#best-practices)

## Overview

Skills in DIMOS are **Pydantic-based callable objects** that represent discrete robot capabilities. They serve as the interface between agent decisions (LLM tool calls) and robot actions.

### Key Concepts

- **AbstractSkill**: Base class for all skills
- **AbstractRobotSkill**: Base class for robot-specific skills
- **SkillLibrary**: Collection and manager for skills
- **Pydantic Models**: Automatic parameter validation and schema generation

### Skill Lifecycle

```
1. Agent receives query
2. LLM decides to invoke skill
3. Skill instantiated with parameters (Pydantic validation)
4. skill.validate() called for pre-execution checks
5. skill.__call__() executes the action
6. Result returned to agent
```

## Skill Types

DIMOS supports three types of skills:

### 1. Class-Based Skills

Custom Python classes that implement robot behavior using ROS2, SDKs, or APIs.

**Use cases**:
- Custom navigation patterns
- Sensor processing
- Complex multi-step behaviors
- Platform-specific integrations

### 2. WebRTC API Skills

Pre-defined robot behaviors accessed via API IDs (Unitree Go2 specific).

**Use cases**:
- Robot-specific motions (flips, jumps, dances)
- Built-in behaviors from manufacturer
- Quick prototyping

### 3. Composite Skills

Skills that chain multiple sub-skills together.

**Use cases**:
- Reusable behavior sequences
- Hierarchical task decomposition
- Parameterized routines

## Creating Class-Based Skills

### Basic Skill Structure

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field
from typing import Optional
from dimos.robot.robot import Robot

class MyCustomSkill(AbstractRobotSkill):
    """
    Brief description of what the skill does.
    
    The LLM reads this docstring to understand when to use the skill,
    so be clear and specific!
    """
    
    # Pydantic fields define parameters
    param1: float = Field(..., description="Description for LLM")
    param2: str = Field(default="default_value", description="Optional parameter")
    
    def __init__(self, robot: Optional[Robot] = None, **data):
        """Initialize the skill."""
        super().__init__(robot=robot, **data)
        # Additional initialization if needed
    
    def validate(self) -> tuple[bool, str]:
        """
        Pre-execution validation.
        
        Returns:
            (is_valid, error_message)
        """
        # Check parameters
        if self.param1 <= 0:
            return False, "param1 must be positive"
        
        # Check robot state
        if self._robot is None:
            return False, "Robot not initialized"
        
        return True, ""  # Valid
    
    def __call__(self):
        """
        Execute the skill.
        
        Returns:
            str: Human-readable result message
        """
        # Call parent (validates robot instance)
        super().__call__()
        
        # Implement your skill logic here
        result = self._robot.some_action(self.param1, self.param2)
        
        return f"Skill completed: {result}"
```

### Example 1: Simple Movement Skill

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field, validator
from typing import Optional

class MoveDistance(AbstractRobotSkill):
    """Move the robot forward or backward a specified distance."""
    
    distance: float = Field(
        ...,
        description="Distance to move in meters. Positive for forward, negative for backward."
    )
    speed: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Movement speed in m/s (0.1 to 2.0)"
    )
    
    def validate(self) -> tuple[bool, str]:
        """Validate movement parameters."""
        # Check robot availability
        if self._robot is None:
            return False, "Robot instance not provided"
        
        # Check for ROS control
        if not hasattr(self._robot, 'ros_control') or self._robot.ros_control is None:
            return False, "ROS control not available"
        
        # Check distance limits
        max_distance = 10.0  # Safety limit
        if abs(self.distance) > max_distance:
            return False, f"Distance {self.distance}m exceeds max {max_distance}m"
        
        return True, ""
    
    def __call__(self):
        """Execute the movement."""
        super().__call__()
        
        direction = "forward" if self.distance > 0 else "backward"
        abs_distance = abs(self.distance)
        
        try:
            # Calculate duration
            duration = abs_distance / self.speed
            
            # Send movement command
            if self.distance > 0:
                self._robot.move(distance=abs_distance, speed=self.speed)
            else:
                self._robot.reverse(distance=abs_distance, speed=self.speed)
            
            return f"Moving {direction} {abs_distance}m at {self.speed}m/s"
        
        except Exception as e:
            return f"Movement failed: {str(e)}"
```

### Example 2: Vision-Based Skill

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field
import cv2
import numpy as np

class DetectAndApproach(AbstractRobotSkill):
    """Detect an object and move towards it."""
    
    object_name: str = Field(..., description="Name of object to detect (e.g., 'person', 'ball')")
    approach_distance: float = Field(default=1.0, description="Distance to stop from object in meters")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        # Initialize detector if available
        self.detector = None
        if robot and hasattr(robot, 'detection_module'):
            self.detector = robot.detection_module
    
    def validate(self) -> tuple[bool, str]:
        """Check if detection is available."""
        if self._robot is None:
            return False, "Robot not initialized"
        
        if self.detector is None:
            return False, "Object detection not available"
        
        return True, ""
    
    def __call__(self):
        """Detect object and approach."""
        super().__call__()
        
        # Get current video frame
        frame = self._robot.get_latest_frame()
        if frame is None:
            return "No video frame available"
        
        # Detect objects
        detections = self.detector.detect(frame, classes=[self.object_name])
        
        if not detections:
            return f"No {self.object_name} detected"
        
        # Get largest detection
        largest = max(detections, key=lambda d: d.area)
        
        # Calculate approach distance
        # (Simple heuristic based on bounding box size)
        bbox_height = largest.bbox[3] - largest.bbox[1]
        frame_height = frame.shape[0]
        
        # Estimate distance (this is simplified)
        estimated_distance = (frame_height / bbox_height) * 0.5
        move_distance = max(0, estimated_distance - self.approach_distance)
        
        if move_distance > 0.1:
            # Move towards object
            self._robot.move(distance=move_distance, speed=0.3)
            return f"Approaching {self.object_name}, moving {move_distance:.2f}m"
        else:
            return f"Already close to {self.object_name}"
```

### Example 3: Asynchronous Skill

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field
import threading
import time

class PatrolArea(AbstractRobotSkill):
    """Patrol an area continuously for a specified duration."""
    
    duration: float = Field(..., description="Duration to patrol in seconds")
    patrol_distance: float = Field(default=2.0, description="Distance per patrol leg in meters")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        self._stop_flag = threading.Event()
        self._patrol_thread = None
    
    def validate(self) -> tuple[bool, str]:
        if self._robot is None:
            return False, "Robot not initialized"
        
        if self.duration <= 0:
            return False, "Duration must be positive"
        
        return True, ""
    
    def _patrol_loop(self):
        """Background patrol loop."""
        start_time = time.time()
        
        while not self._stop_flag.is_set() and (time.time() - start_time) < self.duration:
            # Move forward
            self._robot.move(distance=self.patrol_distance, speed=0.5)
            time.sleep(3)
            
            if self._stop_flag.is_set():
                break
            
            # Turn 90 degrees
            self._robot.spin_left(degrees=90)
            time.sleep(2)
        
        print("Patrol completed")
    
    def __call__(self):
        """Start patrol."""
        super().__call__()
        
        # Start patrol in background thread
        self._patrol_thread = threading.Thread(target=self._patrol_loop)
        self._patrol_thread.daemon = True
        self._patrol_thread.start()
        
        return f"Started patrol for {self.duration}s"
    
    def stop(self):
        """Stop patrol early."""
        self._stop_flag.set()
        if self._patrol_thread:
            self._patrol_thread.join(timeout=5)
```

## Skill Libraries

**SkillLibrary** is a container that manages collections of skills and provides them to agents.

### Creating a Custom Skill Library

```python
from dimos.skills.skills import SkillLibrary, AbstractRobotSkill
from pydantic import Field

# Define your skills
class NavigateToPoint(AbstractRobotSkill):
    """Navigate to a specific (x, y) coordinate."""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    
    def __call__(self):
        super().__call__()
        # Use robot's navigation stack
        return self._robot.navigate_to(x=self.x, y=self.y)

class GetCurrentPosition(AbstractRobotSkill):
    """Get the robot's current position."""
    
    def __call__(self):
        super().__call__()
        pose = self._robot.get_pose()
        return f"Position: x={pose.x:.2f}, y={pose.y:.2f}, theta={pose.theta:.2f}"

class SearchForObject(AbstractRobotSkill):
    """Search for an object by rotating and scanning."""
    object_name: str = Field(..., description="Object to search for")
    
    def __call__(self):
        super().__call__()
        # Implement search behavior
        return f"Searching for {self.object_name}"

# Create custom library
class NavigationSkills(SkillLibrary):
    """Library of navigation-related skills."""
    
    def __init__(self, robot=None):
        super().__init__()
        self._robot = robot
        
        # Initialize skills (optional - can be done later)
        if robot:
            self.initialize_skills(robot)
    
    def initialize_skills(self, robot):
        """Initialize all skills with robot instance."""
        self._robot = robot
        
        # Add skills to library
        self.add(NavigateToPoint(robot=robot))
        self.add(GetCurrentPosition(robot=robot))
        self.add(SearchForObject(robot=robot))
        
        print(f"Initialized {len(self.get_all_skills())} navigation skills")
```

### Using the Skill Library

```python
from dimos.robot.unitree.unitree_go2 import UnitreeGo2
from dimos.agents.agent import OpenAIAgent

# 1. Create robot
robot = UnitreeGo2(ip="192.168.123.161")

# 2. Create and configure skill library
nav_skills = NavigationSkills(robot=robot)

# 3. Create agent with skill library
agent = OpenAIAgent(
    dev_name="NavigationAgent",
    input_query_stream=query_stream,
    skills=nav_skills,  # Pass the SkillLibrary instance
    system_query="You are a navigation agent. Help the user move around.",
    model_name="gpt-4o"
)

# Now agent can use all skills in the library
```

### Extending Built-in Skill Libraries

```python
from dimos.robot.unitree.unitree_skills import MyUnitreeSkills

class ExtendedUnitreeSkills(MyUnitreeSkills):
    """Extended Unitree skills with custom additions."""
    
    def __init__(self, robot=None):
        # Initialize parent (gets all default Unitree skills)
        super().__init__(robot=robot)
        
        # Add custom skills
        if robot:
            self.add(MyCustomSkill(robot=robot))
            self.add(AnotherCustomSkill(robot=robot))

# Use extended library
robot = UnitreeGo2(
    ip="192.168.123.161",
    skills=ExtendedUnitreeSkills()  # Pass custom library
)
```

## WebRTC API Skills

Unitree Go2 robots support WebRTC API commands for built-in behaviors.

### Available WebRTC Commands

```python
# Unitree Go2 WebRTC API IDs (examples)
WEBRTC_COMMANDS = {
    # Movement
    1001: "FrontFlip",
    1002: "FrontJump",
    1003: "FrontPounce",
    
    # Greetings
    1004: "Hello",
    1005: "Wave",
    
    # Dance moves
    1006: "Dance1",
    1007: "Dance2",
    
    # Other
    1008: "Sit",
    1009: "StandUp",
}
```

### Creating WebRTC Skills

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field

class FrontFlip(AbstractRobotSkill):
    """Perform a front flip using the WebRTC API."""
    
    def validate(self) -> tuple[bool, str]:
        if self._robot is None:
            return False, "Robot not initialized"
        
        if not hasattr(self._robot, 'webrtc_req'):
            return False, "Robot does not support WebRTC commands"
        
        return True, ""
    
    def __call__(self):
        super().__call__()
        
        # WebRTC API ID for front flip
        api_id = 1001
        
        # Send command
        try:
            self._robot.webrtc_req(api_id=api_id)
            return "Executing front flip"
        except Exception as e:
            return f"Front flip failed: {str(e)}"

class PerformDance(AbstractRobotSkill):
    """Perform a dance routine."""
    
    dance_number: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Dance routine number (1 or 2)"
    )
    
    def __call__(self):
        super().__call__()
        
        # Map dance number to API ID
        api_id = 1005 + self.dance_number
        
        self._robot.webrtc_req(api_id=api_id)
        return f"Performing dance routine {self.dance_number}"
```

## Composite Skills

Composite skills combine multiple sub-skills into reusable sequences.

### Example 1: Simple Sequence

```python
from dimos.skills.skills import AbstractRobotSkill

class GreetingRoutine(AbstractRobotSkill):
    """Perform a greeting routine: wave, say hello, and bow."""
    
    def __call__(self):
        super().__call__()
        
        results = []
        
        # Step 1: Wave
        wave_skill = Wave(robot=self._robot)
        results.append(wave_skill())
        
        # Step 2: Play sound (if available)
        if hasattr(self._robot, 'play_sound'):
            self._robot.play_sound("hello.wav")
            results.append("Played greeting sound")
        
        # Step 3: Bow
        bow_skill = Bow(robot=self._robot)
        results.append(bow_skill())
        
        return "Greeting complete: " + " -> ".join(results)
```

### Example 2: Parameterized Sequence

```python
class PatrolSquare(AbstractRobotSkill):
    """Patrol in a square pattern."""
    
    side_length: float = Field(default=2.0, description="Length of each side in meters")
    speed: float = Field(default=0.5, description="Movement speed")
    
    def __call__(self):
        super().__call__()
        
        for i in range(4):
            # Move forward
            move_skill = Move(
                robot=self._robot,
                distance=self.side_length,
                speed=self.speed
            )
            move_skill()
            
            # Turn 90 degrees
            turn_skill = SpinLeft(
                robot=self._robot,
                degrees=90
            )
            turn_skill()
        
        return f"Completed square patrol ({self.side_length}m sides)"
```

### Example 3: Conditional Composite

```python
class SearchAndApproach(AbstractRobotSkill):
    """Search for an object and approach it if found."""
    
    object_name: str = Field(..., description="Object to search for")
    search_duration: float = Field(default=30.0, description="Max search time in seconds")
    
    def __call__(self):
        super().__call__()
        
        import time
        start_time = time.time()
        
        # Search phase
        while (time.time() - start_time) < self.search_duration:
            # Check for object
            detections = self._robot.detect_objects([self.object_name])
            
            if detections:
                # Found it! Approach
                approach_skill = DetectAndApproach(
                    robot=self._robot,
                    object_name=self.object_name
                )
                result = approach_skill()
                return f"Found and approached {self.object_name}: {result}"
            
            # Not found, turn and search
            turn_skill = SpinLeft(robot=self._robot, degrees=45)
            turn_skill()
            time.sleep(2)
        
        return f"Search timeout: {self.object_name} not found"
```

## Best Practices

### 1. Documentation

✅ **Do**:
```python
class MySkill(AbstractRobotSkill):
    """
    Clear, concise description of what the skill does.
    
    The LLM uses this to decide when to invoke the skill.
    Be specific about:
    - What the skill accomplishes
    - When it should be used
    - Any preconditions or limitations
    """
    
    param: float = Field(
        ...,
        description="Detailed parameter description for the LLM"
    )
```

❌ **Don't**:
```python
class MySkill(AbstractRobotSkill):
    """Does something."""  # Too vague!
    param: float  # No description for LLM
```

### 2. Validation

✅ **Do**:
```python
def validate(self) -> tuple[bool, str]:
    """Comprehensive validation."""
    # Check robot
    if self._robot is None:
        return False, "Robot not initialized"
    
    # Check parameters
    if self.distance < 0:
        return False, "Distance must be non-negative"
    
    # Check robot state
    if hasattr(self._robot, 'get_battery_level'):
        if self._robot.get_battery_level() < 20:
            return False, "Battery too low"
    
    # Check hardware availability
    if not hasattr(self._robot, 'required_method'):
        return False, "Robot missing required capability"
    
    return True, ""
```

❌ **Don't**:
```python
def validate(self) -> tuple[bool, str]:
    return True, ""  # No validation!
```

### 3. Error Handling

✅ **Do**:
```python
def __call__(self):
    super().__call__()
    
    try:
        result = self._robot.some_action()
        return f"Success: {result}"
    
    except RobotException as e:
        # Handle specific error
        self._robot.stop()
        return f"Action failed: {str(e)}"
    
    except Exception as e:
        # Handle unexpected errors
        self._robot.emergency_stop()
        return f"Unexpected error: {str(e)}"
```

❌ **Don't**:
```python
def __call__(self):
    super().__call__()
    result = self._robot.some_action()  # No error handling!
    return result
```

### 4. Parameter Design

✅ **Do**:
```python
class MySkill(AbstractRobotSkill):
    # Use meaningful names
    target_distance: float = Field(..., description="Distance to target in meters")
    
    # Provide reasonable defaults
    speed: float = Field(default=0.5, description="Movement speed in m/s")
    
    # Add constraints
    retry_count: int = Field(default=3, ge=1, le=10, description="Number of retries")
    
    # Use enums for discrete choices
    direction: Literal["forward", "backward"] = Field(
        default="forward",
        description="Movement direction"
    )
```

❌ **Don't**:
```python
class MySkill(AbstractRobotSkill):
    d: float  # Unclear name
    s: float = 0.5  # No description
    r: int = 999  # Unreasonable default
```

### 5. Skill Naming

✅ **Do**:
- Use verb-noun patterns: `MoveForward`, `DetectObjects`, `NavigateToGoal`
- Be specific: `SpinLeft90` instead of just `Turn`
- Indicate modality: `VisionBasedNavigation` vs `GPSBasedNavigation`

❌ **Don't**:
- Use vague names: `Action1`, `DoThing`
- Use acronyms: `MVFD` instead of `MoveForward`
- Be overly generic: `Process`, `Execute`

### 6. Testing Skills

```python
def test_skill_independently():
    """Test skill without agent."""
    from unittest.mock import Mock
    
    # Create mock robot
    mock_robot = Mock()
    mock_robot.move.return_value = True
    
    # Create skill
    skill = MoveDistance(
        robot=mock_robot,
        distance=2.0,
        speed=0.5
    )
    
    # Validate
    is_valid, msg = skill.validate()
    assert is_valid, f"Validation failed: {msg}"
    
    # Execute
    result = skill()
    print(f"Result: {result}")
    
    # Verify robot method was called
    mock_robot.move.assert_called_once()
```

## Common Patterns

### Pattern 1: Stateful Skills

```python
class MonitorTemperature(AbstractRobotSkill):
    """Monitor temperature and alert if threshold exceeded."""
    
    threshold: float = Field(..., description="Temperature threshold in Celsius")
    duration: float = Field(..., description="Monitoring duration in seconds")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        self._measurements = []
        self._alerts = []
    
    def __call__(self):
        super().__call__()
        
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < self.duration:
            temp = self._robot.get_temperature()
            self._measurements.append(temp)
            
            if temp > self.threshold:
                self._alerts.append((time.time(), temp))
            
            time.sleep(1)
        
        avg_temp = sum(self._measurements) / len(self._measurements)
        return f"Monitoring complete. Avg: {avg_temp:.1f}°C, Alerts: {len(self._alerts)}"
```

### Pattern 2: Skill with Cleanup

```python
class RecordVideo(AbstractRobotSkill):
    """Record video for a specified duration."""
    
    duration: float = Field(..., description="Recording duration in seconds")
    filename: str = Field(default="recording.mp4", description="Output filename")
    
    def __init__(self, robot=None, **data):
        super().__init__(robot=robot, **data)
        self._recorder = None
    
    def __call__(self):
        super().__call__()
        
        try:
            # Start recording
            self._recorder = self._robot.start_video_recording(self.filename)
            
            # Record for duration
            time.sleep(self.duration)
            
            return f"Recorded {self.duration}s to {self.filename}"
        
        finally:
            # Always cleanup
            if self._recorder:
                self._recorder.stop()
                self._recorder = None
```

### Pattern 3: Skill with Callbacks

```python
class NavigateWithProgress(AbstractRobotSkill):
    """Navigate with progress updates."""
    
    x: float = Field(..., description="Target X coordinate")
    y: float = Field(..., description="Target Y coordinate")
    
    def __init__(self, robot=None, progress_callback=None, **data):
        super().__init__(robot=robot, **data)
        self.progress_callback = progress_callback or (lambda p: None)
    
    def __call__(self):
        super().__call__()
        
        # Start navigation
        self._robot.navigate_to(self.x, self.y)
        
        # Monitor progress
        while not self._robot.navigation_complete():
            progress = self._robot.get_navigation_progress()
            self.progress_callback(progress)
            time.sleep(0.5)
        
        return f"Reached destination ({self.x}, {self.y})"
```

## Next Steps

- **Observable Streams**: Learn about data flow → [Observables Guide](observables.md)
- **Perception Integration**: Add vision to skills → [Perception Guide](perception.md)
- **API Reference**: Detailed skill API → [Skills API](../api/skills.md)

## Troubleshooting

### Skill Not Appearing in Agent

**Problem**: Agent doesn't recognize skill

**Solutions**:
1. Check skill is added to library: `library.add(MySkill(robot=robot))`
2. Verify Pydantic schema: `print(MySkill.model_json_schema())`
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

### Validation Always Failing

**Problem**: `validate()` returns `False`

**Solutions**:
1. Check robot initialization: `print(skill._robot)`
2. Verify robot attributes: `print(dir(skill._robot))`
3. Add debug prints in `validate()`

### Skill Execution Hangs

**Problem**: Skill never returns

**Solutions**:
1. Add timeouts to blocking operations
2. Use threading for long-running tasks
3. Monitor robot state in loops
4. Add early exit conditions

---

**Previous**: [Integration Guide](integration.md) | **Next**: [Observable Streams Guide](observables.md)
