# Skills API Reference

Complete API documentation for DIMOS skills system.

## Table of Contents

1. [Base Classes](#base-classes)
2. [SkillLibrary](#skilllibrary)
3. [Built-in Skills](#built-in-skills)
4. [Creating Custom Skills](#creating-custom-skills)

## Base Classes

### AbstractSkill

Base class for all skills in DIMOS.

```python
class AbstractSkill(BaseModel):
    def __init__(self, *args, **kwargs)
    
    def clone(self) -> "AbstractSkill"
    
    def register_as_running(self, name: str, 
                           skill_library: SkillLibrary,
                           subscription=None)
```

**Inheritance:** Pydantic `BaseModel` for automatic validation and schema generation

**Methods:**

#### \_\_init\_\_(\*args, \*\*kwargs)
Initialize the skill with parameters.

**Parameters:** Defined by child class fields

#### clone() → AbstractSkill
Create a copy of the skill instance.

**Returns:** New skill instance

#### register_as_running(name, skill_library, subscription)
Register skill as currently executing.

**Parameters:**
- `name` (str): Skill identifier
- `skill_library` (SkillLibrary): Library managing the skill
- `subscription` (Disposable): Optional observable subscription

---

### AbstractRobotSkill

Base class for robot-specific skills.

```python
class AbstractRobotSkill(AbstractSkill):
    _robot: Robot = None
    
    def __init__(self, robot: Optional[Robot] = None, **data)
    
    def set_robot(self, robot: Robot) -> None
    
    def validate(self) -> tuple[bool, str]
    
    def __call__(self)
```

**Attributes:**
- `_robot` (Robot): Robot instance (protected)

**Methods:**

#### \_\_init\_\_(robot, \*\*data)
Initialize skill with robot reference.

**Parameters:**
- `robot` (Robot): Robot instance to control
- `**data`: Pydantic field values

#### set_robot(robot)
Set or update the robot reference.

**Parameters:**
- `robot` (Robot): New robot instance

#### validate() → tuple[bool, str]
Pre-execution validation.

**Returns:** 
- `(True, "")` if valid
- `(False, "error message")` if invalid

**Example:**
```python
def validate(self) -> tuple[bool, str]:
    if self._robot is None:
        return False, "Robot not initialized"
    
    if self.distance > 10.0:
        return False, "Distance exceeds maximum"
    
    return True, ""
```

#### \_\_call\_\_()
Execute the skill.

**Returns:** str - Result message

**Raises:** RuntimeError if robot not initialized

**Example:**
```python
def __call__(self):
    super().__call__()  # Validates robot
    
    result = self._robot.move(self.distance, self.speed)
    return f"Moved {self.distance}m at {self.speed}m/s"
```

## SkillLibrary

Container and manager for skill collections.

```python
class SkillLibrary:
    def __init__(self)
    
    def add(self, skill: AbstractSkill)
    
    def get_all_skills(self) -> List[AbstractSkill]
    
    def get_skill(self, name: str) -> Optional[AbstractSkill]
    
    def register_running_skill(self, name: str, 
                              skill: AbstractSkill,
                              subscription=None)
    
    def unregister_running_skill(self, name: str)
    
    def get_running_skills(self) -> Dict[str, AbstractSkill]
```

### Methods

#### add(skill)
Add a skill to the library.

**Parameters:**
- `skill` (AbstractSkill): Skill instance to add

**Example:**
```python
library = SkillLibrary()
library.add(MoveSkill(robot=robot))
library.add(RotateSkill(robot=robot))
```

#### get_all_skills() → List[AbstractSkill]
Get all registered skills.

**Returns:** List of skill instances

#### get_skill(name) → Optional[AbstractSkill]
Get skill by name.

**Parameters:**
- `name` (str): Skill class name

**Returns:** Skill instance or None

#### register_running_skill(name, skill, subscription)
Register a skill as currently executing.

**Parameters:**
- `name` (str): Skill name
- `skill` (AbstractSkill): Skill instance
- `subscription` (Disposable): Observable subscription

#### unregister_running_skill(name)
Unregister a running skill.

**Parameters:**
- `name` (str): Skill name to unregister

#### get_running_skills() → Dict[str, AbstractSkill]
Get all currently running skills.

**Returns:** Dictionary mapping names to skill instances

### Usage Example

```python
from dimos.skills.skills import SkillLibrary, AbstractRobotSkill

class CustomLibrary(SkillLibrary):
    def __init__(self, robot=None):
        super().__init__()
        self._robot = robot
        
        if robot:
            self.initialize_skills(robot)
    
    def initialize_skills(self, robot):
        self.add(MoveSkill(robot=robot))
        self.add(RotateSkill(robot=robot))
        self.add(GetPoseSkill(robot=robot))

# Usage
library = CustomLibrary(robot=my_robot)
skills = library.get_all_skills()
print(f"Loaded {len(skills)} skills")
```

## Built-in Skills

### Navigation Skills

#### Move
```python
class Move(AbstractRobotSkill):
    """Move the robot forward or backward."""
    
    distance: float = Field(..., description="Distance in meters")
    speed: float = Field(default=0.5, description="Speed in m/s")
```

**Fields:**
- `distance` (float): Distance to move (positive=forward, negative=backward)
- `speed` (float): Movement speed in m/s (default: 0.5)

**Example:**
```python
move = Move(robot=robot, distance=2.0, speed=0.5)
result = move()
```

#### Reverse
```python
class Reverse(AbstractRobotSkill):
    """Move the robot backward."""
    
    distance: float = Field(..., description="Distance in meters")
    speed: float = Field(default=0.5, description="Speed in m/s")
```

#### SpinLeft
```python
class SpinLeft(AbstractRobotSkill):
    """Rotate the robot left."""
    
    degrees: float = Field(..., description="Rotation angle in degrees")
```

#### SpinRight
```python
class SpinRight(AbstractRobotSkill):
    """Rotate the robot right."""
    
    degrees: float = Field(..., description="Rotation angle in degrees")
```

#### NavigateToGoal
```python
class NavigateToGoal(AbstractRobotSkill):
    """Navigate to a specific (x, y) coordinate."""
    
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
```

### Observation Skills

#### ObserveStream
```python
class ObserveStream(AbstractRobotSkill):
    """Periodically send images from robot camera to agent."""
    
    timestep: float = Field(default=5.0, 
                           description="Interval between observations")
    query_text: str = Field(default="What do you observe?",
                           description="Query to send with each frame")
    duration: float = Field(default=60.0,
                           description="Total observation duration")
```

**Fields:**
- `timestep` (float): Seconds between observations (default: 5.0)
- `query_text` (str): Prompt for each frame
- `duration` (float): Total monitoring time (default: 60.0)

**Special Methods:**
- `stop()`: Stop observation early

**Example:**
```python
observe = ObserveStream(
    robot=robot,
    agent=vision_agent,
    timestep=10.0,
    query_text="Describe what you see",
    duration=120.0
)
observe()  # Start observing

# Later...
observe.stop()  # Stop observation
```

### Utility Skills

#### GetPose
```python
class GetPose(AbstractRobotSkill):
    """Get the robot's current position."""
```

**Returns:** String describing current pose (x, y, theta)

**Example:**
```python
pose_skill = GetPose(robot=robot)
result = pose_skill()
# "Current position: x=2.5, y=1.3, theta=0.78"
```

#### KillSkill
```python
class KillSkill(AbstractRobotSkill):
    """Stop a running skill by name."""
    
    skill_name: str = Field(..., 
                           description="Name of skill to stop")
```

## Creating Custom Skills

### Basic Skill Template

```python
from dimos.skills.skills import AbstractRobotSkill
from pydantic import Field
from typing import Optional
from dimos.robot.robot import Robot

class MyCustomSkill(AbstractRobotSkill):
    """
    [Description for LLM - be clear and specific]
    
    This skill [what it does].
    Use it when [conditions for use].
    """
    
    # Define parameters with Pydantic Fields
    param1: float = Field(
        ...,  # Required field
        description="[Description for LLM]",
        ge=0,  # Greater than or equal to 0 (optional)
        le=10  # Less than or equal to 10 (optional)
    )
    
    param2: str = Field(
        default="default_value",  # Optional with default
        description="[Description for LLM]"
    )
    
    def __init__(self, robot: Optional[Robot] = None, **data):
        """Initialize the skill."""
        super().__init__(robot=robot, **data)
        # Additional initialization if needed
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate parameters and robot state before execution.
        
        Returns:
            (is_valid, error_message)
        """
        # Check robot
        if self._robot is None:
            return False, "Robot not initialized"
        
        # Check parameters
        if self.param1 < 0:
            return False, "param1 must be non-negative"
        
        # Check robot state/capabilities
        if not hasattr(self._robot, 'required_method'):
            return False, "Robot missing required capability"
        
        return True, ""
    
    def __call__(self):
        """
        Execute the skill.
        
        Returns:
            str: Human-readable result message
        """
        # Call parent (validates robot)
        super().__call__()
        
        try:
            # Implement your skill logic
            result = self._robot.custom_action(
                self.param1,
                self.param2
            )
            
            return f"Skill executed successfully: {result}"
        
        except Exception as e:
            error_msg = f"Skill failed: {str(e)}"
            print(error_msg)
            return error_msg
```

### Field Types and Validation

#### Numeric Fields
```python
# Integer
count: int = Field(..., description="Number of items", ge=1, le=10)

# Float
distance: float = Field(..., description="Distance in meters", gt=0.0)

# With constraints
speed: float = Field(default=0.5, ge=0.1, le=2.0)
```

#### String Fields
```python
# Basic string
name: str = Field(..., description="Object name")

# With regex pattern
id: str = Field(..., pattern=r"^[A-Z]{3}\d{3}$")

# With max length
description: str = Field(..., max_length=100)
```

#### Enum Fields
```python
from typing import Literal

direction: Literal["forward", "backward", "left", "right"] = Field(
    ...,
    description="Movement direction"
)
```

#### List Fields
```python
from typing import List

waypoints: List[tuple] = Field(
    default=[],
    description="List of (x, y) coordinates"
)
```

### Validation Patterns

#### Parameter Validation
```python
def validate(self) -> tuple[bool, str]:
    # Range check
    if not 0 <= self.value <= 100:
        return False, "Value must be between 0 and 100"
    
    # Type check
    if not isinstance(self.items, list):
        return False, "Items must be a list"
    
    # Dependency check
    if self.use_advanced and not self.advanced_param:
        return False, "advanced_param required when use_advanced=True"
    
    return True, ""
```

#### Robot State Validation
```python
def validate(self) -> tuple[bool, str]:
    # Check robot exists
    if self._robot is None:
        return False, "Robot not initialized"
    
    # Check capabilities
    if not hasattr(self._robot, 'move'):
        return False, "Robot cannot move"
    
    # Check state
    if self._robot.is_moving():
        return False, "Robot is already moving"
    
    # Check battery
    if hasattr(self._robot, 'get_battery_level'):
        if self._robot.get_battery_level() < 20:
            return False, "Battery too low"
    
    return True, ""
```

### Execution Patterns

#### Synchronous Execution
```python
def __call__(self):
    super().__call__()
    
    result = self._robot.some_action()
    return f"Result: {result}"
```

#### Asynchronous Execution
```python
import threading

def __call__(self):
    super().__call__()
    
    def execute_async():
        self._robot.long_running_action()
    
    thread = threading.Thread(target=execute_async, daemon=True)
    thread.start()
    
    return "Action started in background"
```

#### With Timeout
```python
import signal

def __call__(self):
    super().__call__()
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Skill execution timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        result = self._robot.action()
        signal.alarm(0)  # Cancel timeout
        return f"Success: {result}"
    except TimeoutError:
        return "Skill timed out"
```

#### With Retry Logic
```python
def __call__(self):
    super().__call__()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = self._robot.unreliable_action()
            return f"Success on attempt {attempt + 1}: {result}"
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts: {e}"
            time.sleep(1)  # Wait before retry
```

## Tool Schema Generation

Skills are automatically converted to OpenAI tool schemas:

```python
from openai import pydantic_function_tool

class MySkill(AbstractRobotSkill):
    """Move to a specific location."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

# Auto-generated schema:
tool_schema = pydantic_function_tool(MySkill)
print(tool_schema)
```

Output:
```json
{
  "type": "function",
  "function": {
    "name": "MySkill",
    "description": "Move to a specific location.",
    "parameters": {
      "type": "object",
      "properties": {
        "x": {"type": "number", "description": "X coordinate"},
        "y": {"type": "number", "description": "Y coordinate"}
      },
      "required": ["x", "y"]
    }
  }
}
```

## Best Practices

### Documentation
- Write clear docstrings (LLM reads them!)
- Add detailed Field descriptions
- Include usage examples

### Validation
- Always implement `validate()`
- Check robot state
- Validate parameter ranges

### Error Handling
- Use try-except blocks
- Return meaningful error messages
- Don't raise exceptions from `__call__()`

### Testing
```python
def test_skill():
    from unittest.mock import Mock
    
    # Mock robot
    mock_robot = Mock()
    mock_robot.move.return_value = True
    
    # Create skill
    skill = MySkill(robot=mock_robot, distance=2.0)
    
    # Validate
    is_valid, msg = skill.validate()
    assert is_valid, msg
    
    # Execute
    result = skill()
    assert "Success" in result
    
    # Verify robot called
    mock_robot.move.assert_called_once_with(2.0)
```

---

**Related Documentation:**
- [Skills Extension Guide](../guides/skills.md)
- [Integration Guide](../guides/integration.md)
- [Agents API](agents.md)
