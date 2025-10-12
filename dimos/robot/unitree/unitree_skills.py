# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
import time
from pydantic import Field
import threading

if TYPE_CHECKING:
    from dimos.robot.robot import Robot, MockRobot
else:
    Robot = "Robot"
    MockRobot = "MockRobot"

from dimos.skills.skills import AbstractRobotSkill, AbstractSkill, SkillLibrary
from dimos.types.constants import Colors
from inspect import signature, Parameter
from typing import Callable, Any, get_type_hints

# Module-level constant for Unitree ROS control definitions
# TEMPORARY: Reduced skill set for Ollama testing (Issue #12)
# Full list commented out due to qwen2.5-coder:7b having issues with 47 tools  
# Uncomment when switching to llama.cpp or vLLM
# Only using class-based nav2 skills: Move, Reverse, SpinLeft, SpinRight, Wait
UNITREE_ROS_CONTROLS: List[Tuple[str, int, str]] = []

# FULL LIST (commented out for Ollama testing):
# [
#     ("Damp", 1001, "Lowers the robot to the ground fully."),
#     ("BalanceStand", 1002, "Maintains balanced standing position."),
#     ("StandUp", 1004, "Transition from sitting to standing."),
#     ("StandDown", 1005, "Move from standing to sitting."),
#     ("Sit", 1009, "Sit down from standing."),
#     ("RiseSit", 1010, "Rise from sitting to standing."),
#     ... and 41 more skills ...
# ]

# region MyUnitreeSkills


class MyUnitreeSkills(SkillLibrary):
    """My Unitree Skills."""

    _robot: Optional[Robot] = None

    @classmethod
    def register_skills(
        cls, skill_classes: Union["AbstractSkill", list["AbstractSkill"]]
    ):
        """Add multiple skill classes as class attributes.

        Args:
            skill_classes: List of skill classes to add
        """
        if isinstance(skill_classes, list):
            for skill_class in skill_classes:
                setattr(cls, skill_class.__name__, skill_class)
        else:
            setattr(cls, skill_classes.__name__, skill_classes)

    def __init__(self, robot: Optional[Robot] = None):
        super().__init__()
        self._robot: Robot = None

        # Add dynamic skills to this class
        self.register_skills(self.create_skills_live())

        if robot is not None:
            self._robot = robot
            self.initialize_skills()

    def initialize_skills(self):
        # Create the skills and add them to the list of skills
        self.register_skills(self.create_skills_live())

        # Provide the robot instance to each skill
        for skill_class in self:
            print(
                f"{Colors.GREEN_PRINT_COLOR}Creating instance for skill: {skill_class}{Colors.RESET_COLOR}"
            )
            self.create_instance(skill_class.__name__, robot=self._robot)

        # Refresh the class skills
        self.refresh_class_skills()

    def create_skills_live(self) -> List[AbstractRobotSkill]:
        # ================================================
        # Procedurally created skills
        # ================================================
        class BaseUnitreeSkill(AbstractRobotSkill):
            """Base skill for dynamic skill creation."""

            def __call__(self):
                string = f"{Colors.GREEN_PRINT_COLOR}This is a base skill, created for the specific skill: {self._app_id}{Colors.RESET_COLOR}"
                print(string)
                super().__call__()
                if self._app_id is None:
                    raise RuntimeError(
                        f"{Colors.RED_PRINT_COLOR}"
                        f"No App ID provided to {self.__class__.__name__} Skill"
                        f"{Colors.RESET_COLOR}"
                    )
                else:
                    self._robot.webrtc_req(api_id=self._app_id)
                    string = f"{Colors.GREEN_PRINT_COLOR}{self.__class__.__name__} was successful: id={self._app_id}{Colors.RESET_COLOR}"
                    print(string)
                    return string

        skills_classes = []
        for name, app_id, description in UNITREE_ROS_CONTROLS:
            skill_class = type(
                name,  # Name of the class
                (BaseUnitreeSkill,),  # Base classes
                {"__doc__": description, "_app_id": app_id},
            )
            skills_classes.append(skill_class)

        return skills_classes

    # region Class-based Skills

    class Move(AbstractRobotSkill):
        """Move the robot using direct velocity commands."""

        x: float = Field(..., description="Forward velocity (m/s).")
        y: float = Field(default=0.0, description="Left/right velocity (m/s)")
        yaw: float = Field(default=0.0, description="Rotational velocity (rad/s)")
        duration: float = Field(
            default=0.0,
            description="How long to move (seconds). If 0, command is continuous",
        )

        def __call__(self):
            super().__call__()
            return self._robot.move_vel(
                x=self.x, y=self.y, yaw=self.yaw, duration=self.duration
            )

    class Reverse(AbstractRobotSkill):
        """Reverse the robot using direct velocity commands."""

        x: float = Field(
            ..., description="Backward velocity (m/s). Positive values move backward."
        )
        y: float = Field(default=0.0, description="Left/right velocity (m/s)")
        yaw: float = Field(default=0.0, description="Rotational velocity (rad/s)")
        duration: float = Field(
            default=0.0,
            description="How long to move (seconds). If 0, command is continuous",
        )

        def __call__(self):
            super().__call__()
            # Use move_vel with negative x for backward movement
            return self._robot.move_vel(
                x=-self.x, y=self.y, yaw=self.yaw, duration=self.duration
            )

    class SpinLeft(AbstractRobotSkill):
        """Spin the robot left using degree commands."""

        degrees: float = Field(..., description="Distance to spin left in degrees")

        def __call__(self):
            super().__call__()
            return self._robot.spin(
                degrees=self.degrees
            )  # Spinning left is positive degrees

    class SpinRight(AbstractRobotSkill):
        """Spin the robot right using degree commands."""

        degrees: float = Field(..., description="Distance to spin right in degrees")

        def __call__(self):
            super().__call__()
            return self._robot.spin(
                degrees=-self.degrees
            )  # Spinning right is negative degrees

    class Wait(AbstractSkill):
        """Wait for a specified amount of time."""

        seconds: float = Field(..., description="Seconds to wait")

        def __call__(self):
            time.sleep(self.seconds)
            return f"Wait completed with length={self.seconds}s"
