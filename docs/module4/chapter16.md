---
sidebar_position: 6
title: Chapter 16 - Autonomous Mobile Manipulation with Isaac Foundation Models
---

import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Chapter 16: Autonomous Mobile Manipulation with Isaac Foundation Models

<ChatbotWidget />
<ProgressTracker chapterId="module4-chapter16" />
<UrduTranslationToggle />
<PersonalizationButton />

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles of autonomous mobile manipulation and its challenges
- Integrate navigation and manipulation capabilities using Isaac Foundation Models
- Design and implement mobile manipulation task planning systems
- Implement coordinated control of mobile base and manipulator arm
- Apply reinforcement learning techniques to mobile manipulation tasks
- Evaluate and optimize mobile manipulation system performance
- Address safety and reliability concerns in mobile manipulation
- Deploy mobile manipulation systems in real-world environments

## Prerequisites

Before starting this chapter, you should have:
- Understanding of robot kinematics, dynamics, and control from Module 1
- Knowledge of Isaac Foundation Models from previous chapters
- Experience with manipulation and navigation systems from Chapters 10-15
- Completion of Module 1, Module 2, and Module 3 content
- Familiarity with reinforcement learning concepts from Chapter 11
- Understanding of embodied AI principles from Chapter 15

## Introduction to Autonomous Mobile Manipulation

Autonomous mobile manipulation represents the convergence of mobile robotics and robotic manipulation, enabling robots to navigate to task locations and perform complex manipulation tasks. This capability is essential for applications ranging from warehouse automation to home assistance, where robots must operate in diverse, dynamic environments.

### Key Challenges in Mobile Manipulation

Mobile manipulation presents unique challenges that arise from the integration of navigation and manipulation:

**Base-Manipulator Coordination**: The mobile base and manipulator arm must be coordinated to achieve optimal manipulation performance, considering reachability, stability, and workspace constraints.

**Dynamic Reconfiguration**: As the robot moves, the manipulation workspace changes, requiring dynamic replanning and adjustment of manipulation strategies.

**Stability Considerations**: Mobile manipulation must maintain system stability while performing manipulation tasks, especially when extending arms or carrying objects.

**Perception Challenges**: The robot must perceive its environment from a moving platform while simultaneously tracking manipulation targets.

### Mobile Manipulation System Architecture

A typical mobile manipulation system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────┐
│                    Mobile Manipulation System           │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────────┐    ┌──────────┐  │
│ │ Navigation  │    │ Task Planner    │    │ Manipulator │ │
│ │ System      │←──→│                 │←──→│ Controller │ │
│ └─────────────┘    │                 │    │          │  │
│                    │                 │    └──────────┘  │
│ ┌─────────────┐    │                 │    ┌──────────┐  │
│ │ Perception  │    │                 │    │ Base     │  │
│ │ System      │←──→│                 │←──→│ Controller │ │
│ └─────────────┘    └─────────────────┘    │          │  │
│                                           └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

Each component plays a crucial role in the overall system:
- **Navigation System**: Plans and executes path planning for the mobile base
- **Task Planner**: Decomposes high-level tasks into navigation and manipulation subtasks
- **Manipulator Controller**: Controls the robotic arm for manipulation tasks
- **Base Controller**: Controls the mobile platform's movement
- **Perception System**: Provides environment and object sensing capabilities

## Mobile Manipulation Task Planning

### Hierarchical Task Decomposition

Mobile manipulation tasks require hierarchical decomposition into navigation and manipulation subtasks. This decomposition enables specialized algorithms to handle each aspect while maintaining coordination.

```python
# mobile_manipulation_planning.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import networkx as nx
from dataclasses import dataclass

@dataclass
class MobileManipulationTask:
    """
    Data structure for mobile manipulation tasks.
    """
    task_type: str  # 'grasp', 'transport', 'assemble', etc.
    target_object: str
    start_location: Tuple[float, float, float]  # x, y, theta
    end_location: Tuple[float, float, float]
    manipulation_pose: Dict[str, Any]  # Desired grasp or placement pose
    constraints: Dict[str, Any]  # Stability, safety, workspace constraints

class TaskDecomposer:
    """
    Decomposes mobile manipulation tasks into navigation and manipulation subtasks.
    """
    def __init__(self):
        self.navigation_subtasks = [
            'path_planning',
            'obstacle_avoidance',
            'localization',
            'motion_execution'
        ]

        self.manipulation_subtasks = [
            'perception',
            'grasp_planning',
            'trajectory_generation',
            'force_control',
            'grasp_execution'
        ]

    def decompose_task(self, task: MobileManipulationTask) -> List[Dict[str, Any]]:
        """
        Decompose a mobile manipulation task into subtasks.

        Args:
            task: High-level mobile manipulation task

        Returns:
            List of subtasks with execution order and dependencies
        """
        subtasks = []

        if task.task_type == 'grasp':
            # Navigate to object location
            subtasks.append({
                'type': 'navigation',
                'action': 'navigate_to_location',
                'target': self.estimate_navigation_target(task),
                'dependencies': [],
                'priority': 1
            })

            # Manipulate object
            subtasks.append({
                'type': 'manipulation',
                'action': 'grasp_object',
                'target': task.target_object,
                'pose': task.manipulation_pose,
                'dependencies': ['navigation'],
                'priority': 2
            })

            if task.end_location != task.start_location:
                # Transport object if needed
                subtasks.append({
                    'type': 'navigation',
                    'action': 'navigate_to_destination',
                    'target': task.end_location,
                    'dependencies': ['manipulation'],
                    'priority': 3
                })

                subtasks.append({
                    'type': 'manipulation',
                    'action': 'place_object',
                    'target': task.target_object,
                    'pose': self.estimate_placement_pose(task),
                    'dependencies': ['navigation'],
                    'priority': 4
                })

        elif task.task_type == 'transport':
            # Similar decomposition for transport tasks
            subtasks.extend([
                {
                    'type': 'navigation',
                    'action': 'navigate_to_pick_location',
                    'target': task.start_location,
                    'dependencies': [],
                    'priority': 1
                },
                {
                    'type': 'manipulation',
                    'action': 'grasp_object',
                    'target': task.target_object,
                    'pose': task.manipulation_pose,
                    'dependencies': ['navigation'],
                    'priority': 2
                },
                {
                    'type': 'navigation',
                    'action': 'navigate_to_place_location',
                    'target': task.end_location,
                    'dependencies': ['manipulation'],
                    'priority': 3
                },
                {
                    'type': 'manipulation',
                    'action': 'place_object',
                    'target': task.target_object,
                    'pose': self.estimate_placement_pose(task),
                    'dependencies': ['navigation'],
                    'priority': 4
                }
            ])

        # Sort by priority
        subtasks.sort(key=lambda x: x['priority'])
        return subtasks

    def estimate_navigation_target(self, task: MobileManipulationTask) -> Dict[str, Any]:
        """Estimate optimal navigation target for manipulation task."""
        # This would involve calculating approach poses that optimize manipulability
        approach_distance = 0.8  # meters from object
        object_pos = self.estimate_object_position(task.target_object)

        # Calculate approach pose (in front of object)
        approach_pose = {
            'position': [
                object_pos[0] - approach_distance * np.cos(object_pos[2]),
                object_pos[1] - approach_distance * np.sin(object_pos[2]),
                object_pos[2]  # Same orientation as object
            ],
            'approach_angle': 0.0  # Approach from front
        }

        return approach_pose

    def estimate_object_position(self, object_name: str) -> Tuple[float, float, float]:
        """Estimate object position (placeholder implementation)."""
        # In real implementation, this would use perception system
        return (1.0, 1.0, 0.0)  # Default position

    def estimate_placement_pose(self, task: MobileManipulationTask) -> Dict[str, Any]:
        """Estimate optimal placement pose for object."""
        return {
            'position': [task.end_location[0], task.end_location[1], 0.1],  # Place at 0.1m height
            'orientation': [0, 0, 0, 1]  # Default orientation (quaternion)
        }

class MobileManipulationPlanner:
    """
    Planner for coordinating navigation and manipulation in mobile robots.
    """
    def __init__(self, navigation_planner, manipulation_planner):
        self.navigation_planner = navigation_planner
        self.manipulation_planner = manipulation_planner
        self.task_decomposer = TaskDecomposer()

        # Stability and safety constraints
        self.stability_checker = StabilityConstraintChecker()
        self.safety_checker = SafetyConstraintChecker()

    def plan_mobile_manipulation(self, task: MobileManipulationTask) -> Dict[str, Any]:
        """
        Plan complete mobile manipulation sequence.

        Args:
            task: Mobile manipulation task to plan

        Returns:
            Dictionary containing navigation and manipulation plans
        """
        # Decompose task into subtasks
        subtasks = self.task_decomposer.decompose_task(task)

        plan = {
            'navigation_segments': [],
            'manipulation_segments': [],
            'coordination_points': [],  # Points where navigation and manipulation coordinate
            'safety_checks': [],
            'execution_order': []
        }

        for subtask in subtasks:
            if subtask['type'] == 'navigation':
                nav_plan = self.navigation_planner.plan_path(
                    current_pose=task.start_location,
                    target_pose=subtask['target']['position']
                )

                plan['navigation_segments'].append(nav_plan)

            elif subtask['type'] == 'manipulation':
                manip_plan = self.manipulation_planner.plan_manipulation(
                    task_type=subtask['action'],
                    target_object=subtask['target'],
                    target_pose=subtask['pose']
                )

                plan['manipulation_segments'].append(manip_plan)

            # Add coordination point
            plan['coordination_points'].append({
                'subtask': subtask,
                'expected_completion_time': self.estimate_completion_time(subtask)
            })

        return plan

    def estimate_completion_time(self, subtask: Dict[str, Any]) -> float:
        """Estimate completion time for subtask."""
        if subtask['type'] == 'navigation':
            # Estimate based on distance and robot speed
            distance = np.linalg.norm(
                np.array(subtask['target']['position'][:2]) -
                np.array([0, 0])  # Assuming current position is origin for this estimate
            )
            return distance / 0.5  # Assuming 0.5 m/s average speed
        elif subtask['type'] == 'manipulation':
            # Estimate based on manipulation complexity
            if subtask['action'] == 'grasp_object':
                return 5.0  # 5 seconds for grasping
            elif subtask['action'] == 'place_object':
                return 3.0  # 3 seconds for placing
            else:
                return 10.0  # Default for complex manipulations

    def validate_plan(self, plan: Dict[str, Any], robot_state: Dict[str, Any]) -> bool:
        """
        Validate mobile manipulation plan for safety and feasibility.

        Args:
            plan: Mobile manipulation plan to validate
            robot_state: Current robot state

        Returns:
            True if plan is valid, False otherwise
        """
        # Check stability constraints throughout plan
        for nav_segment in plan['navigation_segments']:
            if not self.stability_checker.is_stable_during_navigation(nav_segment, robot_state):
                return False

        # Check manipulation feasibility
        for manip_segment in plan['manipulation_segments']:
            if not self.manipulation_planner.is_feasible(manip_segment, robot_state):
                return False

        # Check safety constraints
        for coord_point in plan['coordination_points']:
            if not self.safety_checker.is_safe(coord_point, robot_state):
                return False

        return True

    def optimize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize mobile manipulation plan for efficiency and safety.

        Args:
            plan: Original plan to optimize

        Returns:
            Optimized plan
        """
        # Optimize navigation segments for shortest path while maintaining safety
        optimized_nav_segments = []
        for nav_segment in plan['navigation_segments']:
            optimized_segment = self.navigation_planner.optimize_path(nav_segment)
            optimized_nav_segments.append(optimized_segment)

        # Optimize manipulation segments for energy efficiency
        optimized_manip_segments = []
        for manip_segment in plan['manipulation_segments']:
            optimized_segment = self.manipulation_planner.optimize_trajectory(manip_segment)
            optimized_manip_segments.append(optimized_segment)

        # Update plan with optimizations
        plan['navigation_segments'] = optimized_nav_segments
        plan['manipulation_segments'] = optimized_manip_segments

        return plan
```

### Coordinated Control Strategies

Coordinating the mobile base and manipulator requires sophisticated control strategies that consider the coupled dynamics and constraints of the system.

```python
# coordinated_control.py
class CoordinatedMobileManipulatorController:
    """
    Controller for coordinated control of mobile base and manipulator.
    """
    def __init__(self,
                 mobile_base_controller,
                 manipulator_controller,
                 coordination_strategy='operational_space'):
        self.mobile_base_controller = mobile_base_controller
        self.manipulator_controller = manipulator_controller
        self.coordination_strategy = coordination_strategy

        # Joint controller for coordinated movement
        self.joint_controller = self._initialize_joint_controller()

        # Task priority management
        self.task_priority_manager = TaskPriorityManager()

        # Stability constraint handler
        self.stability_handler = StabilityConstraintHandler()

    def _initialize_joint_controller(self):
        """Initialize controller for joint mobile-manipulator control."""
        if self.coordination_strategy == 'operational_space':
            return OperationalSpaceController()
        elif self.coordination_strategy == 'task_space':
            return TaskSpaceCoordinator()
        elif self.coordination_strategy == 'hierarchical':
            return HierarchicalCoordinator()
        else:
            raise ValueError(f"Unknown coordination strategy: {self.coordination_strategy}")

    def compute_coordinated_control(self,
                                  desired_ee_pose: np.ndarray,
                                  robot_state: Dict[str, Any],
                                  base_velocity_limit: float = 0.5,
                                  manip_velocity_limit: float = 1.0) -> Dict[str, Any]:
        """
        Compute coordinated control for mobile manipulator.

        Args:
            desired_ee_pose: Desired end-effector pose in world coordinates
            robot_state: Current robot state (base pose, joint positions, velocities)
            base_velocity_limit: Maximum base velocity
            manip_velocity_limit: Maximum manipulator velocity

        Returns:
            Dictionary containing base and manipulator control commands
        """
        # Get current end-effector pose
        current_ee_pose = self._compute_current_ee_pose(robot_state)

        # Calculate end-effector velocity needed to reach desired pose
        ee_velocity = self._calculate_ee_velocity(current_ee_pose, desired_ee_pose)

        # Decompose velocity into base and manipulator components
        base_twist, manip_joint_velocities = self.joint_controller(
            ee_velocity,
            robot_state,
            base_velocity_limit,
            manip_velocity_limit
        )

        # Apply stability constraints
        base_twist = self.stability_handler.apply_stability_constraints(
            base_twist, robot_state
        )

        # Apply task priority if needed
        if 'priorities' in robot_state:
            base_twist, manip_joint_velocities = self.task_priority_manager(
                base_twist, manip_joint_velocities, robot_state['priorities']
            )

        return {
            'base_twist': base_twist,  # [vx, vy, omega]
            'manipulator_velocities': manip_joint_velocities,  # Joint velocity commands
            'coordination_quality': self._assess_coordination_quality(base_twist, manip_joint_velocities)
        }

    def _compute_current_ee_pose(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Compute current end-effector pose in world coordinates."""
        # Get base pose in world frame
        base_pose = robot_state['base_pose']  # [x, y, theta]

        # Get manipulator joint positions
        joint_positions = robot_state['manipulator_joints']

        # Compute end-effector pose in base frame
        ee_pose_base = self.manipulator_controller.forward_kinematics(joint_positions)

        # Transform to world frame
        ee_pose_world = self._transform_to_world_frame(ee_pose_base, base_pose)

        return ee_pose_world

    def _transform_to_world_frame(self, pose_base: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
        """Transform pose from base frame to world frame."""
        # Extract base transformation
        x_base, y_base, theta_base = base_pose

        # Create transformation matrix
        cos_theta, sin_theta = np.cos(theta_base), np.sin(theta_base)
        transform_matrix = np.array([
            [cos_theta, -sin_theta, x_base],
            [sin_theta, cos_theta, y_base],
            [0, 0, 1]
        ])

        # Apply transformation
        pose_world = transform_matrix @ np.append(pose_base[:2], 1)
        pose_world = np.append(pose_world[:2], pose_base[2] + theta_base)  # Include orientation

        return pose_world

    def _calculate_ee_velocity(self, current_pose: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
        """Calculate end-effector velocity to reach desired pose."""
        # Simple proportional controller
        position_error = desired_pose[:2] - current_pose[:2]
        orientation_error = desired_pose[2] - current_pose[2]

        # Normalize orientation error to [-pi, pi]
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))

        # Velocity proportional to error
        max_linear_vel = 0.3  # m/s
        max_angular_vel = 0.5  # rad/s

        linear_vel = np.clip(position_error * 2.0, -max_linear_vel, max_linear_vel)  # Gain of 2.0
        angular_vel = np.clip(orientation_error * 2.0, -max_angular_vel, max_angular_vel)

        return np.concatenate([linear_vel, [angular_vel]])

    def _assess_coordination_quality(self, base_twist: np.ndarray, manip_velocities: np.ndarray) -> float:
        """Assess quality of coordination between base and manipulator."""
        # Coordination quality metric (simplified)
        # Could include measures like manipulability, stability margins, etc.
        base_norm = np.linalg.norm(base_twist[:2])  # Linear velocity
        manip_norm = np.linalg.norm(manip_velocities)

        # Perfect coordination would have both components contributing effectively
        # without conflict
        if base_norm + manip_norm == 0:
            return 1.0  # No motion, perfect coordination by definition

        # Simplified coordination quality measure
        coordination_score = 1.0 / (1.0 + abs(base_norm - manip_norm))
        return coordination_score

class OperationalSpaceController:
    """
    Operational space controller for mobile manipulators.
    """
    def __init__(self):
        self.lambda_reg = 1e-6  # Regularization parameter

    def __call__(self,
                 ee_velocity: np.ndarray,
                 robot_state: Dict[str, Any],
                 base_limit: float,
                 manip_limit: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coordinated control using operational space formulation.

        Args:
            ee_velocity: Desired end-effector velocity [vx, vy, omega]
            robot_state: Current robot state
            base_limit: Base velocity limit
            manip_limit: Manipulator velocity limit

        Returns:
            Tuple of (base_twist, manipulator_velocities)
        """
        # Get Jacobians for base and manipulator
        base_jacobian = self._compute_base_jacobian(robot_state)
        manip_jacobian = self._compute_manipulator_jacobian(robot_state)

        # Combined system Jacobian
        # This is a simplified representation - in practice, this would be more complex
        # accounting for the mobile base-manipulator coupling
        combined_jacobian = np.hstack([base_jacobian, manip_jacobian])

        # Regularized inverse
        J_reg = combined_jacobian.T @ np.linalg.inv(
            combined_jacobian @ combined_jacobian.T + self.lambda_reg * np.eye(3)
        )

        # Compute joint velocities
        all_velocities = J_reg @ ee_velocity

        # Separate base and manipulator velocities
        base_twist = all_velocities[:3]  # [vx, vy, omega]
        manip_velocities = all_velocities[3:]  # Joint velocities

        # Apply limits
        base_twist = np.clip(base_twist, -base_limit, base_limit)
        manip_velocities = np.clip(manip_velocities, -manip_limit, manip_limit)

        return base_twist, manip_velocities

    def _compute_base_jacobian(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Compute Jacobian for base motion affecting end-effector."""
        # Simplified Jacobian - in reality, this would be computed based on
        # the geometric relationship between base motion and end-effector motion
        base_jac = np.zeros((3, 3))  # ee velocities = base velocities
        base_jac[0, 0] = 1.0  # dx/dvx
        base_jac[1, 1] = 1.0  # dy/dvy
        base_jac[2, 2] = 1.0  # dtheta/domega

        return base_jac

    def _compute_manipulator_jacobian(self, robot_state: Dict[str, Any]) -> np.ndarray:
        """Compute manipulator Jacobian."""
        # Get joint positions
        joint_positions = robot_state['manipulator_joints']

        # Compute geometric Jacobian (simplified 2D planar arm example)
        # In practice, this would use the robot's kinematic model
        n_joints = len(joint_positions)
        jacobian = np.zeros((3, n_joints))  # [dx, dy, dtheta] for n joints

        # Calculate end-effector position relative to each joint
        ee_pos = self._compute_ee_position(joint_positions)
        link_positions = self._compute_link_positions(joint_positions)

        for i in range(n_joints):
            # Position vector from joint i to end-effector
            r = ee_pos[:2] - link_positions[i][:2]

            # Angular contribution to linear velocity
            Jv = np.array([-r[1], r[0]])  # Cross product in 2D
            jacobian[0:2, i] = Jv

            # Angular velocity contribution
            jacobian[2, i] = 1.0  # Each joint contributes to orientation

        return jacobian

    def _compute_ee_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute end-effector position from joint positions (simplified)."""
        # Forward kinematics (simplified 2D planar arm)
        # In practice, use robot's kinematic model
        x = 0
        y = 0
        theta = 0

        # Assuming fixed link lengths for simplicity
        link_lengths = [0.5] * len(joint_positions)  # 0.5m per link

        for i, (angle, length) in enumerate(zip(joint_positions, link_lengths)):
            theta += angle
            x += length * np.cos(theta)
            y += length * np.sin(theta)

        return np.array([x, y, theta])

    def _compute_link_positions(self, joint_positions: np.ndarray) -> List[np.ndarray]:
        """Compute positions of all links."""
        positions = []
        x, y, theta = 0, 0, 0

        link_lengths = [0.5] * len(joint_positions)

        for angle, length in zip(joint_positions, link_lengths):
            theta += angle
            x += length * np.cos(theta)
            y += length * np.sin(theta)
            positions.append(np.array([x, y, theta]))

        return positions

class TaskPriorityManager:
    """
    Manager for handling task priorities in mobile manipulation.
    """
    def __init__(self):
        self.priority_weights = {
            'safety': 10.0,
            'stability': 5.0,
            'task_completion': 2.0,
            'energy_efficiency': 1.0,
            'smoothness': 0.5
        }

    def __call__(self, base_twist: np.ndarray, manip_velocities: np.ndarray,
                 priorities: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust control commands based on task priorities.

        Args:
            base_twist: Original base twist command
            manip_velocities: Original manipulator velocity commands
            priorities: Dictionary of task priorities

        Returns:
            Adjusted base and manipulator commands
        """
        # Adjust commands based on priorities
        adjusted_base = base_twist.copy()
        adjusted_manip = manip_velocities.copy()

        # If safety is critical, reduce all velocities
        if priorities.get('safety', 0) > 0.8:
            safety_factor = 0.5  # Reduce speeds by half
            adjusted_base *= safety_factor
            adjusted_manip *= safety_factor

        # If stability is important, favor base movement over manipulator
        if priorities.get('stability', 0) > 0.7:
            stability_factor = 0.7
            manip_reduction = 1.0 - stability_factor
            adjusted_manip *= stability_factor
            # Compensate with base movement if possible
            adjusted_base *= (1.0 + manip_reduction * 0.3)

        # If task completion is priority, increase speeds
        if priorities.get('task_completion', 0) > 0.9:
            completion_factor = 1.2  # Increase speeds by 20%
            adjusted_base = np.clip(adjusted_base * completion_factor, -1.0, 1.0)
            adjusted_manip = np.clip(adjusted_manip * completion_factor, -2.0, 2.0)

        return adjusted_base, adjusted_manip

class StabilityConstraintHandler:
    """
    Handler for applying stability constraints to mobile manipulation.
    """
    def __init__(self):
        # Zero-moment point (ZMP) constraints for wheeled mobile bases
        self.zmp_limits = {'x': (-0.2, 0.2), 'y': (-0.1, 0.1)}  # meters
        self.support_polygon = self._define_support_polygon()

    def apply_stability_constraints(self, base_twist: np.ndarray,
                                  robot_state: Dict[str, Any]) -> np.ndarray:
        """
        Apply stability constraints to base motion commands.

        Args:
            base_twist: Base motion command [vx, vy, omega]
            robot_state: Current robot state

        Returns:
            Stability-constrained base command
        """
        # Calculate current ZMP based on commanded motion
        zmp_predicted = self._predict_zmp(base_twist, robot_state)

        # Check if predicted ZMP is within support polygon
        if not self._is_zmp_stable(zmp_predicted):
            # Project command to maintain stability
            base_twist = self._project_for_stability(base_twist, robot_state, zmp_predicted)

        return base_twist

    def _predict_zmp(self, base_twist: np.ndarray, robot_state: Dict[str, Any]) -> np.ndarray:
        """Predict Zero Moment Point based on motion command."""
        # Simplified ZMP prediction
        # In reality, this would involve complex dynamics modeling
        zmp_x = base_twist[0] * 0.1  # Simplified relationship
        zmp_y = base_twist[1] * 0.05

        return np.array([zmp_x, zmp_y])

    def _is_zmp_stable(self, zmp: np.ndarray) -> bool:
        """Check if ZMP is within stable region."""
        x_limit_low, x_limit_high = self.zmp_limits['x']
        y_limit_low, y_limit_high = self.zmp_limits['y']

        return (x_limit_low <= zmp[0] <= x_limit_high and
                y_limit_low <= zmp[1] <= y_limit_high)

    def _project_for_stability(self, base_twist: np.ndarray, robot_state: Dict[str, Any],
                             current_zmp: np.ndarray) -> np.ndarray:
        """Project base command to maintain stability."""
        # Iteratively reduce command magnitude until stable
        reduction_factor = 1.0
        original_twist = base_twist.copy()

        while reduction_factor > 0.1:  # Don't reduce below 10% of original
            test_twist = original_twist * reduction_factor
            test_zmp = self._predict_zmp(test_twist, robot_state)

            if self._is_zmp_stable(test_zmp):
                return test_twist

            reduction_factor *= 0.9  # Reduce by 10%

        # If we can't find a stable command, return minimal safe command
        return np.zeros_like(base_twist)
```

## Isaac Foundation Models for Mobile Manipulation

### Mobile Manipulation Foundation Model Architecture

The Isaac Foundation Models can be extended to specifically address mobile manipulation challenges by incorporating joint navigation-manipulation reasoning.

```python
# mobile_manipulation_foundation.py
class MobileManipulationFoundation(nn.Module):
    """
    Foundation model for mobile manipulation tasks.
    Combines navigation and manipulation reasoning in a unified architecture.
    """
    def __init__(self,
                 navigation_encoder_dim: int = 256,
                 manipulation_encoder_dim: int = 256,
                 fusion_dim: int = 512,
                 action_dim: int = 10,  # 3 base + 7 manipulator
                 hidden_dim: int = 512):
        super(MobileManipulationFoundation, self).__init__()

        # Navigation-specific encoder
        self.navigation_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # [robot_x, robot_y, robot_theta, goal_x, goal_y, goal_theta]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, navigation_encoder_dim),
            nn.LayerNorm(navigation_encoder_dim)
        )

        # Manipulation-specific encoder
        self.manipulation_encoder = nn.Sequential(
            nn.Linear(7 + 3, hidden_dim),  # [joint_positions, target_x, target_y, target_z]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manipulation_encoder_dim),
            nn.LayerNorm(manipulation_encoder_dim)
        )

        # Visual encoder for scene understanding
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-modal fusion module
        self.fusion_module = nn.Sequential(
            nn.Linear(navigation_encoder_dim + manipulation_encoder_dim + hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Task-specific decoders
        self.navigation_decoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [vx, vy, omega]
        )

        self.manipulation_decoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7-DOF manipulator velocities
        )

        # Coordination module
        self.coordination_module = CoordinationModule(fusion_dim)

        # Action bounds
        self.base_action_bounds = torch.tensor([[-1.0, -1.0, -np.pi], [1.0, 1.0, np.pi]])  # [vx, vy, omega]
        self.manip_action_bounds = torch.tensor([[-2.0] * 7, [2.0] * 7])  # Joint velocities

    def forward(self,
                robot_state: torch.Tensor,
                goal_state: torch.Tensor,
                manip_target: torch.Tensor,
                visual_input: torch.Tensor,
                task_description: str = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for mobile manipulation foundation model.

        Args:
            robot_state: Current robot state [batch, 14] (7 base + 7 manipulator)
            goal_state: Navigation goal [batch, 3] (x, y, theta)
            manip_target: Manipulation target [batch, 3] (x, y, z)
            visual_input: Visual input [batch, 3, H, W]
            task_description: Optional natural language task description

        Returns:
            Dictionary of navigation and manipulation commands
        """
        batch_size = robot_state.size(0)

        # Encode navigation state
        nav_input = torch.cat([robot_state[:, :3], goal_state], dim=1)  # [x, y, theta, goal_x, goal_y, goal_theta]
        nav_features = self.navigation_encoder(nav_input)

        # Encode manipulation state
        manip_input = torch.cat([robot_state[:, 7:], manip_target], dim=1)  # [joints, target_x, target_y, target_z]
        manip_features = self.manipulation_encoder(manip_input)

        # Encode visual information
        visual_features = self.visual_encoder(visual_input)

        # Fuse all modalities
        fused_features = self.fusion_module(
            torch.cat([nav_features, manip_features, visual_features], dim=1)
        )

        # Decode to specific actions
        base_cmd = self.navigation_decoder(fused_features)
        manip_cmd = self.manipulation_decoder(fused_features)

        # Apply coordination between base and manipulator
        coordinated_actions = self.coordination_module(
            fused_features, base_cmd, manip_cmd, robot_state
        )

        # Apply action bounds
        base_cmd_bounded = torch.clamp(
            coordinated_actions['base_command'],
            min=self.base_action_bounds[0],
            max=self.base_action_bounds[1]
        )
        manip_cmd_bounded = torch.clamp(
            coordinated_actions['manip_command'],
            min=self.manip_action_bounds[0],
            max=self.manip_action_bounds[1]
        )

        return {
            'base_command': base_cmd_bounded,
            'manipulator_command': manip_cmd_bounded,
            'fused_features': fused_features,
            'coordination_weights': coordinated_actions.get('weights', None)
        }

class CoordinationModule(nn.Module):
    """
    Module for coordinating navigation and manipulation actions.
    """
    def __init__(self, feature_dim: int):
        super(CoordinationModule, self).__init__()

        # Coordination network
        self.coordination_network = nn.Sequential(
            nn.Linear(feature_dim + 10, 256),  # Combined features + raw actions
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Coordination weights for base and manipulator
        )

    def forward(self,
                fused_features: torch.Tensor,
                base_command: torch.Tensor,
                manip_command: torch.Tensor,
                robot_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Coordinate base and manipulator actions based on fused features.

        Args:
            fused_features: Fused multi-modal features
            base_command: Raw base command
            manip_command: Raw manipulator command
            robot_state: Current robot state

        Returns:
            Coordinated actions with weights
        """
        # Combine features with raw commands
        combined_input = torch.cat([
            fused_features,
            base_command,
            manip_command
        ], dim=1)

        # Compute coordination weights
        weights = torch.softmax(self.coordination_network(combined_input), dim=1)

        # Apply coordination: weight the base and manipulator commands
        coordinated_base = base_command * weights[:, 0:1].expand_as(base_command)
        coordinated_manip = manip_command * weights[:, 1:2].expand_as(manip_command)

        # Consider current state for coordination
        # For example, if manipulator is extended, reduce base mobility
        manip_extension = torch.norm(robot_state[:, 7:14], dim=1, keepdim=True)  # Joint positions magnitude
        extension_factor = torch.sigmoid(-manip_extension + 3.0)  # Reduce base when extended

        coordinated_base = coordinated_base * extension_factor

        return {
            'base_command': coordinated_base,
            'manip_command': coordinated_manip,
            'weights': weights
        }

class MobileManipulationTransformer(nn.Module):
    """
    Transformer-based architecture for mobile manipulation reasoning.
    """
    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 ff_dim: int = 2048,
                 max_seq_len: int = 100):
        super(MobileManipulationTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len

        # Input embedding
        self.input_projection = nn.Linear(input_dim, input_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for different modalities
        self.navigation_head = nn.Linear(input_dim, 3)  # Base velocities
        self.manipulation_head = nn.Linear(input_dim, 7)  # Manipulator velocities
        self.coordination_head = nn.Linear(input_dim, 2)  # Coordination weights

    def _create_positional_encoding(self):
        """Create positional encoding for sequence modeling."""
        pe = torch.zeros(self.max_seq_len, self.input_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() *
                           (-np.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return nn.Parameter(pe, requires_grad=False)

    def forward(self,
                input_sequence: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer for mobile manipulation sequence modeling.

        Args:
            input_sequence: Sequence of multi-modal inputs [batch, seq_len, input_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary of outputs for navigation, manipulation, and coordination
        """
        batch_size, seq_len, _ = input_sequence.shape

        # Apply input projection
        projected_input = self.input_projection(input_sequence)

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            pos_encoding = self.positional_encoding[:, :seq_len, :]
            embedded_input = projected_input + pos_encoding
        else:
            # Handle sequences longer than max length
            pos_encoding = self.positional_encoding[:, :self.max_seq_len, :]
            embedded_input = projected_input[:, :self.max_seq_len, :] + pos_encoding

        # Apply transformer
        if attention_mask is not None:
            # Convert boolean mask to float mask for transformer
            mask = attention_mask.float()
            mask = (1.0 - mask) * float('-inf')
            transformer_output = self.transformer(embedded_input, mask=mask)
        else:
            transformer_output = self.transformer(embedded_input)

        # Apply output heads
        # Use the last time step for single-step control
        last_hidden = transformer_output[:, -1, :]  # [batch, input_dim]

        navigation_output = self.navigation_head(last_hidden)
        manipulation_output = self.manipulation_head(last_hidden)
        coordination_output = torch.softmax(self.coordination_head(last_hidden), dim=1)

        return {
            'navigation_commands': navigation_output,
            'manipulation_commands': manipulation_output,
            'coordination_weights': coordination_output,
            'sequence_features': transformer_output  # For temporal reasoning
        }
```

### Reinforcement Learning for Mobile Manipulation

Reinforcement learning can be particularly effective for mobile manipulation tasks where the coordination between navigation and manipulation needs to be learned.

```python
# rl_mobile_manipulation.py
class MobileManipulationPPO(nn.Module):
    """
    PPO-based reinforcement learning for mobile manipulation.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 10,  # 3 base + 7 manipulator
                 hidden_dim: int = 512):
        super(MobileManipulationPPO, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))  # Learnable std

        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Action bounds
        self.register_buffer('action_low', torch.tensor([-1.0, -1.0, -np.pi, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]))
        self.register_buffer('action_high', torch.tensor([1.0, 1.0, np.pi, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution and value.

        Args:
            state: Current state [batch, state_dim]

        Returns:
            Tuple of (action_mean, action_std, state_value)
        """
        features = self.feature_extractor(state)

        action_mean = torch.tanh(self.actor_mean(features))  # Bound to [-1, 1]
        action_mean_scaled = self.action_low + (action_mean + 1.0) * (self.action_high - self.action_low) / 2.0

        action_std = torch.exp(self.actor_std)  # Ensure positive std
        state_value = self.critic(features)

        return action_mean_scaled, action_std, state_value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action from policy.

        Args:
            state: Current state
            deterministic: Whether to sample deterministically

        Returns:
            Action tensor
        """
        action_mean, action_std, _ = self.forward(state)

        if deterministic:
            return action_mean

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        return action

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.

        Args:
            state: Current states
            action: Actions to evaluate

        Returns:
            Tuple of (log_probs, state_values, dist_entropy)
        """
        action_mean, action_std, state_value = self.forward(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(action).sum(dim=1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=1).mean()

        return log_probs, state_value, dist_entropy

class MobileManipulationEnvironment:
    """
    Environment for mobile manipulation RL training.
    """
    def __init__(self, config):
        self.config = config
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -np.pi, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),  # Base + manipulator
            high=np.array([1.0, 1.0, np.pi, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            dtype=np.float32
        )

        # Observation space: robot state + object poses + goal info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(50,),  # This would include robot state, object poses, goal info, etc.
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0

        # Initialize robot and environment
        self.robot_state = self.initialize_robot_state()
        self.object_poses = self.initialize_object_poses()
        self.goal_info = self.initialize_goal()

        return self.get_observation()

    def step(self, action):
        """Execute action in environment."""
        self.current_step += 1

        # Apply action to robot (simplified physics simulation)
        self.robot_state = self.apply_action_dynamics(self.robot_state, action)

        # Update object states based on robot interaction
        self.object_poses = self.update_object_poses(self.robot_state, self.object_poses)

        # Get new observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        done = self.check_termination()

        # Additional info
        info = {
            'is_success': self.check_success(),
            'manipulation_progress': self.get_manipulation_progress(),
            'navigation_progress': self.get_navigation_progress()
        }

        return observation, reward, done, info

    def get_observation(self):
        """Get current observation from environment."""
        # Combine robot state, object poses, goal information
        obs_parts = [
            self.robot_state,  # Robot configuration
            self.object_poses.flatten(),  # Object positions/orientations
            self.goal_info,  # Goal position/description
            [self.current_step / self.max_episode_steps]  # Normalized time
        ]

        return np.concatenate(obs_parts)

    def calculate_reward(self):
        """Calculate reward based on current state."""
        reward = 0.0

        # Navigation reward: progress toward goal
        nav_progress = self.calculate_navigation_reward()
        reward += nav_progress

        # Manipulation reward: progress toward manipulation goal
        manip_progress = self.calculate_manipulation_reward()
        reward += manip_progress

        # Success bonus
        if self.check_success():
            reward += 100.0

        # Penalty for collisions
        if self.check_collision():
            reward -= 10.0

        # Small penalty for each step to encourage efficiency
        reward -= 0.1

        return reward

    def calculate_navigation_reward(self):
        """Calculate navigation-specific reward."""
        # Distance to navigation goal
        current_pos = self.robot_state[:2]  # x, y position
        goal_pos = self.goal_info[:2]
        distance_to_goal = np.linalg.norm(current_pos - goal_pos)

        # Negative distance (closer is better)
        nav_reward = -distance_to_goal * 0.1

        # Bonus for making progress
        if hasattr(self, 'prev_distance_to_goal'):
            if distance_to_goal < self.prev_distance_to_goal:
                nav_reward += 0.5  # Progress bonus
        self.prev_distance_to_goal = distance_to_goal

        return nav_reward

    def calculate_manipulation_reward(self):
        """Calculate manipulation-specific reward."""
        # Distance to manipulation target
        ee_pos = self.get_end_effector_position()
        target_pos = self.goal_info[2:5]  # x, y, z target position
        manip_distance = np.linalg.norm(ee_pos - target_pos)

        # Negative distance (closer is better)
        manip_reward = -manip_distance * 0.2

        # Bonus for grasping object
        if self.check_grasp_success():
            manip_reward += 10.0

        return manip_reward

    def check_success(self):
        """Check if task is successfully completed."""
        # Task-specific success criteria
        if self.config.get('task_type') == 'grasp_and_transport':
            return (self.check_grasp_success() and
                   self.check_transport_success())
        else:
            # Default: close to goal with object
            ee_pos = self.get_end_effector_position()
            goal_pos = self.goal_info[2:5]
            distance = np.linalg.norm(ee_pos - goal_pos)
            return distance < 0.1 and self.check_grasp_success()

    def apply_action_dynamics(self, state, action):
        """Apply action to robot state with simplified dynamics."""
        # This would involve complex robot dynamics simulation
        # For this example, we'll use a simplified model
        new_state = state.copy()

        # Apply base motion (first 3 elements of action)
        base_vel = action[:3]
        new_state[0:3] += base_vel * 0.1  # 0.1 second time step

        # Apply manipulator motion (last 7 elements of action)
        manip_vel = action[3:10]
        new_state[7:14] += manip_vel * 0.05  # 0.05 second time step for manipulator

        return new_state

    def get_end_effector_position(self):
        """Get end-effector position from current state."""
        # Simplified FK calculation
        # In reality, this would use the robot's kinematic model
        joint_positions = self.robot_state[7:14]

        # Forward kinematics (simplified)
        x = 0
        y = 0
        theta = 0

        link_lengths = [0.1] * 7  # Simplified link lengths

        for i, (angle, length) in enumerate(zip(joint_positions, link_lengths)):
            theta += angle
            x += length * np.cos(theta)
            y += length * np.sin(theta)

        return np.array([x, y, 0.5])  # Fixed z-height for this example

class MobileManipulationTrainer:
    """
    Trainer for mobile manipulation RL policies.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize environment
        self.env = MobileManipulationEnvironment(config)

        # Initialize agent
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.agent = MobileManipulationPPO(self.state_dim, self.action_dim)

        # Move to device
        self.agent.to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor_mean.parameters(), lr=config.get('actor_lr', 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=config.get('critic_lr', 1e-3)
        )

        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Storage for rollout data
        self.rollout_storage = RolloutStorage(
            config.get('rollout_steps', 2048),
            self.state_dim,
            self.action_dim,
            self.env.observation_space.dtype
        )

    def collect_rollout(self):
        """Collect rollout data for PPO training."""
        obs = self.env.reset()
        self.rollout_storage.obs[0].copy_(torch.FloatTensor(obs))
        self.rollout_storage.to(self.device)

        for step in range(self.rollout_storage.step):
            # Get action from policy
            with torch.no_grad():
                action_mean, action_std, value = self.agent(
                    self.rollout_storage.obs[step]
                )

            # Sample action
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)

            # Apply action to environment
            obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Store data
            self.rollout_storage.insert(
                torch.FloatTensor(obs),
                action,
                action_logprob.unsqueeze(1),
                torch.FloatTensor([[reward]]),
                torch.BoolTensor([[done]]),
                value
            )

    def update(self):
        """Update policy using collected rollout data."""
        advantages = torch.zeros(
            self.rollout_storage.step, 1, device=self.device
        )

        returns = self.rollout_storage.value_estimates[-1].detach()
        for step in reversed(range(self.rollout_storage.rewards.size(0))):
            returns = self.rollout_storage.rewards[step] + self.gamma * returns
            advantages[step] = returns - self.rollout_storage.value_estimates[step]

        # Flatten batch
        obs_batch = self.rollout_storage.obs[:-1].view(-1, *self.rollout_storage.obs.size()[2:])
        actions_batch = self.rollout_storage.actions.view(-1, self.action_dim)
        old_action_log_probs_batch = self.rollout_storage.action_log_probs.view(-1, 1)
        advantages_batch = advantages.detach().view(-1, 1)

        # Update policy multiple times
        for _ in range(self.config.get('ppo_epochs', 10)):
            # Get action log probs and values
            action_means, action_stds, values = self.agent(obs_batch)

            # Calculate ratio
            dist = torch.distributions.Normal(action_means, action_stds)
            action_log_probs = dist.log_prob(actions_batch).sum(dim=1, keepdim=True)
            ratios = torch.exp(action_log_probs - old_action_log_probs_batch)

            # Calculate surrogates
            surr1 = ratios * advantages_batch
            surr2 = torch.clamp(ratings, 1-self.eps_clip, 1+self.eps_clip) * advantages_batch
            action_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def train(self, num_updates=1000):
        """Train the mobile manipulation policy."""
        for update in range(num_updates):
            # Collect rollout
            self.collect_rollout()

            # Update policy
            self.update()

            # Log progress
            if update % 10 == 0:
                episode_rewards = self.rollout_storage.rewards.sum(dim=0).mean().item()
                print(f"Update {update}, Average Episode Reward: {episode_rewards:.2f}")

            # Update rollout storage
            self.rollout_storage.after_update()

class RolloutStorage:
    """
    Storage for rollout data in RL training.
    """
    def __init__(self, num_steps, obs_shape, action_space, obs_dtype=torch.float32):
        self.obs = torch.zeros(num_steps + 1, *obs_shape, dtype=obs_dtype)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_estimates = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.actions = torch.zeros(num_steps, action_space)
        self.action_log_probs = torch.zeros(num_steps, 1)
        self.masks = torch.ones(num_steps + 1, 1)

        self.num_steps = num_steps
        self.step = 0

    def insert(self, obs, action, action_log_prob, reward, mask, value_estimate):
        """Insert data into storage."""
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_estimates[self.step].copy_(value_estimate)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """Update storage after policy update."""
        self.obs[0].copy_(self.obs[-1])
        self.value_estimates[0].copy_(self.value_estimates[-1])
        self.masks[0].copy_(self.masks[-1])
```

## Integration with Isaac Sim

### Mobile Manipulation in Simulation

Integrating mobile manipulation systems with Isaac Sim provides a powerful platform for training and testing complex behaviors.

```python
# isaac_mobile_manipulation_integration.py
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

class IsaacMobileManipulationEnv:
    """
    Isaac Sim environment for mobile manipulation tasks.
    """
    def __init__(self, config):
        self.config = config
        self.world = World(stage_units_in_meters=1.0)

        # Robot configuration
        self.robot_name = config.get('robot_name', 'tiago')
        self.robot_usd_path = config.get('robot_usd_path', '/Isaac/Robots/Tiago/tiago_dual_hands.usd')

        # Task configuration
        self.task_type = config.get('task_type', 'grasp_and_transport')
        self.object_name = config.get('object_name', 'cube')

        # Initialize robot and environment
        self.robot = None
        self.object = None
        self.goal_position = None

        # Initialize the environment
        self.setup_environment()

    def setup_environment(self):
        """Setup the Isaac Sim environment."""
        # Add robot to stage
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name=self.robot_name,
                usd_path=self.robot_usd_path
            )
        )

        # Add object to manipulate
        self.object = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name=self.object_name,
                position=np.array([0.5, 0.5, 0.1]),
                size=0.1,
                color=np.array([0.8, 0.1, 0.1])
            )
        )

        # Set goal position
        self.goal_position = np.array([1.0, 1.0, 0.1])

        # Add any additional environment elements
        self.add_environment_elements()

    def add_environment_elements(self):
        """Add additional elements to the environment."""
        # Add obstacles, tables, etc.
        pass

    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot to initial pose
        initial_robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.robot.set_world_pose(position=initial_robot_pose[:2], orientation=[0, 0, 0, 1])

        # Reset object to initial position
        initial_obj_pose = np.array([0.5, 0.5, 0.1])
        self.object.set_world_pose(position=initial_obj_pose)

        # Reset robot joints
        initial_joint_pos = np.zeros(self.robot.num_dof)
        self.robot.set_joint_positions(initial_joint_pos)

        # Reset world
        self.world.reset()

        return self.get_observation()

    def step(self, action):
        """Execute action in simulation."""
        # Action format: [base_vx, base_vy, base_omega, joint_vel_1, ..., joint_vel_7]
        base_action = action[:3]  # Base velocities
        manip_action = action[3:]  # Manipulator joint velocities

        # Apply base velocities
        self.apply_base_velocity(base_action)

        # Apply manipulator velocities
        self.apply_manipulator_velocity(manip_action)

        # Step the simulation
        self.world.step(render=True)

        # Get new observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination
        done = self.check_termination()

        # Additional info
        info = {
            'is_success': self.check_success(),
            'grasp_status': self.get_grasp_status()
        }

        return observation, reward, done, info

    def apply_base_velocity(self, base_velocities):
        """Apply base velocity commands."""
        # This would interface with the mobile base controller
        # For Tiago robot, apply differential drive commands
        vx, vy, omega = base_velocities
        # Convert to wheel velocities based on robot kinematics
        # (Implementation depends on specific robot model)
        pass

    def apply_manipulator_velocity(self, joint_velocities):
        """Apply manipulator joint velocity commands."""
        # Get current joint positions
        current_positions = self.robot.get_joint_positions()

        # Calculate new positions (integration)
        dt = 1.0 / 60.0  # Assuming 60Hz simulation
        new_positions = current_positions + joint_velocities * dt

        # Apply new positions (velocity control would be better in practice)
        self.robot.set_joint_positions(new_positions)

    def get_observation(self):
        """Get current observation from simulation."""
        # Get robot state
        robot_position, robot_orientation = self.robot.get_world_pose()
        robot_lin_vel, robot_ang_vel = self.robot.get_world_velocities()
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()

        # Get object state
        obj_position, obj_orientation = self.object.get_world_pose()

        # Get goal information
        goal_info = self.goal_position

        # Combine into observation vector
        obs_parts = [
            robot_position[:2],  # x, y position
            [robot_orientation[2]],  # theta (simplified)
            robot_lin_vel[:2],  # Linear velocities
            [robot_ang_vel[2]],  # Angular velocity
            joint_positions,
            joint_velocities,
            obj_position,
            goal_info,
            [self.world.current_time_step_index / self.config.get('max_steps', 1000)]  # Time normalization
        ]

        return np.concatenate(obs_parts)

    def calculate_reward(self):
        """Calculate reward based on current state."""
        reward = 0.0

        # Get current states
        robot_pos, _ = self.robot.get_world_pose()
        obj_pos, _ = self.object.get_world_pose()
        goal_pos = self.goal_position

        # Navigation reward: progress toward object
        dist_to_obj = np.linalg.norm(robot_pos[:2] - obj_pos[:2])
        reward += -dist_to_obj * 0.1  # Negative distance penalty

        # Manipulation reward: progress toward grasping
        ee_pos = self.get_end_effector_position()
        dist_to_obj_ee = np.linalg.norm(ee_pos - obj_pos)
        reward += -dist_to_obj_ee * 0.2  # Higher weight for manipulation

        # Transport reward: if grasped, progress toward goal
        if self.check_grasp_success():
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            reward += -dist_to_goal * 0.3  # Transport progress

        # Success bonus
        if self.check_success():
            reward += 100.0

        # Penalty for collisions
        if self.check_collision():
            reward -= 10.0

        # Small penalty for each step
        reward -= 0.1

        return reward

    def check_success(self):
        """Check if task is completed successfully."""
        obj_pos, _ = self.object.get_world_pose()
        goal_pos = self.goal_position

        # Success: object at goal position and grasped
        dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
        return dist_to_goal < 0.1 and self.check_grasp_success()

    def check_grasp_success(self):
        """Check if object is successfully grasped."""
        # This would check if the gripper is closed around the object
        # For simulation, we'll use a simple distance check
        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.object.get_world_pose()
        dist = np.linalg.norm(ee_pos - obj_pos)
        return dist < 0.05  # Within 5cm

    def get_end_effector_position(self):
        """Get end-effector position from robot state."""
        # This would use forward kinematics to compute EE position
        # For simplicity, we'll return a fixed offset from base
        robot_pos, robot_rot = self.robot.get_world_pose()
        # Simplified: assume EE is at fixed offset from base
        ee_offset = np.array([0.3, 0.0, 0.8])  # Relative to robot base
        ee_pos = robot_pos + ee_offset
        return ee_pos

    def check_collision(self):
        """Check for collisions."""
        # In a real implementation, this would check collision information
        # For simulation, return a simple proxy
        return False  # Placeholder

    def check_termination(self):
        """Check if episode should terminate."""
        return (self.world.current_time_step_index >= self.config.get('max_steps', 1000) or
                self.check_success())

class IsaacMobileManipulationSystem:
    """
    Complete mobile manipulation system using Isaac Sim.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Isaac Sim environment
        self.isaac_env = IsaacMobileManipulationEnv(config)

        # Initialize Isaac Foundation Models
        self.perceptor = IsaacPerceptor(config.get('perceptor_config', {}))
        self.controller = IsaacController(config.get('controller_config', {}))
        self.manipulator = IsaacManipulator(config.get('manipulator_config', {}))
        self.navigator = IsaacNavigator(config.get('navigator_config', {}))

        # Initialize mobile manipulation foundation model
        self.foundation_model = MobileManipulationFoundation(
            navigation_encoder_dim=config.get('nav_dim', 256),
            manipulation_encoder_dim=config.get('manip_dim', 256),
            fusion_dim=config.get('fusion_dim', 512),
            action_dim=config.get('action_dim', 10)
        )

        # Initialize RL trainer if needed
        if config.get('use_rl', False):
            self.rl_trainer = MobileManipulationTrainer(config)

        # Initialize evaluation metrics
        self.evaluation_metrics = {
            'success_rate': [],
            'completion_time': [],
            'energy_efficiency': [],
            'collision_rate': []
        }

    def train(self, num_episodes=1000):
        """Train the mobile manipulation system."""
        if self.config.get('use_rl', False):
            # Use RL training
            self.rl_trainer.train(num_episodes)
        else:
            # Use imitation learning or other approaches
            self.imitation_learning_train(num_episodes)

    def imitation_learning_train(self, num_episodes):
        """Train using imitation learning."""
        for episode in range(num_episodes):
            obs = self.isaac_env.reset()
            done = False

            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': []
            }

            while not done:
                # In a real system, this would come from expert demonstrations
                expert_action = self.get_expert_action(obs)

                next_obs, reward, done, info = self.isaac_env.step(expert_action)

                episode_data['observations'].append(obs)
                episode_data['actions'].append(expert_action)
                episode_data['rewards'].append(reward)

                obs = next_obs

            # Train on episode data
            self.train_on_episode(episode_data)

            if episode % 100 == 0:
                print(f"Episode {episode}, Success Rate: {self.evaluate_current_policy()}")

    def get_expert_action(self, observation):
        """Get expert action for imitation learning."""
        # This would implement expert behavior
        # For now, return a random valid action
        return np.random.uniform(-1, 1, size=(10,))  # 3 base + 7 manipulator

    def train_on_episode(self, episode_data):
        """Train model on a single episode of data."""
        # This would implement the specific training algorithm
        pass

    def evaluate_current_policy(self):
        """Evaluate the current policy."""
        num_eval_episodes = 10
        successes = 0

        for _ in range(num_eval_episodes):
            obs = self.isaac_env.reset()
            done = False

            while not done:
                # Get action from current policy
                action = self.get_policy_action(obs)
                obs, reward, done, info = self.isaac_env.step(action)

                if info.get('is_success', False):
                    successes += 1
                    break

        success_rate = successes / num_eval_episodes
        return success_rate

    def get_policy_action(self, observation):
        """Get action from current policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use foundation model to get coordinated action
            outputs = self.foundation_model(
                robot_state=obs_tensor[:, :14],  # Robot state portion
                goal_state=obs_tensor[:, 14:17],  # Goal state portion
                manip_target=obs_tensor[:, 17:20],  # Manipulation target portion
                visual_input=torch.randn(1, 3, 224, 224).to(self.device)  # Simulated visual input
            )

            # Combine base and manipulator commands
            action = torch.cat([
                outputs['base_command'],
                outputs['manipulator_command']
            ], dim=1)

        return action.cpu().numpy()[0]

    def run_deployed_task(self, task_description):
        """Run a task with the deployed mobile manipulation system."""
        print(f"Executing task: {task_description}")

        obs = self.isaac_env.reset()
        total_reward = 0
        step_count = 0
        max_steps = self.config.get('max_deployment_steps', 500)

        while step_count < max_steps:
            action = self.get_policy_action(obs)
            obs, reward, done, info = self.isaac_env.step(action)

            total_reward += reward
            step_count += 1

            if done:
                break

        success = info.get('is_success', False)
        print(f"Task completed. Success: {success}, Total Reward: {total_reward:.2f}, Steps: {step_count}")

        return {
            'success': success,
            'total_reward': total_reward,
            'steps_taken': step_count,
            'final_observation': obs
        }

    def evaluate_system_performance(self, num_evaluations=50):
        """Comprehensive evaluation of system performance."""
        evaluation_results = {
            'success_rates': [],
            'average_completion_times': [],
            'trajectory_efficiencies': [],
            'safety_metrics': []
        }

        for eval_run in range(num_evaluations):
            # Run evaluation task
            task_result = self.run_deployed_task("evaluation_task")

            evaluation_results['success_rates'].append(task_result['success'])
            evaluation_results['average_completion_times'].append(task_result['steps_taken'])

            # Calculate trajectory efficiency (simplified)
            efficiency = self.calculate_trajectory_efficiency(task_result['final_observation'])
            evaluation_results['trajectory_efficiencies'].append(efficiency)

        # Aggregate results
        aggregated_results = {
            'overall_success_rate': np.mean(evaluation_results['success_rates']),
            'avg_completion_time': np.mean(evaluation_results['average_completion_times']),
            'avg_trajectory_efficiency': np.mean(evaluation_results['trajectory_efficiencies']),
            'success_rate_std': np.std(evaluation_results['success_rates'])
        }

        return aggregated_results

    def calculate_trajectory_efficiency(self, final_observation):
        """Calculate trajectory efficiency metric."""
        # Simplified efficiency calculation
        # In reality, this would compare actual path to optimal path
        return 0.8  # Placeholder efficiency

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        performance = self.evaluate_system_performance()

        report = []
        report.append("Isaac Mobile Manipulation System - Performance Evaluation")
        report.append("=" * 60)
        report.append("")

        report.append("Performance Metrics:")
        for metric, value in performance.items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: {value}")

        report.append("")
        report.append("Recommendations:")

        if performance['overall_success_rate'] < 0.7:
            report.append("  - Consider additional training with domain randomization")
            report.append("  - Increase simulation diversity")
            report.append("  - Implement more robust grasp planning")
        else:
            report.append("  - Performance is satisfactory for deployment")

        if performance['avg_trajectory_efficiency'] < 0.6:
            report.append("  - Optimize path planning and coordination algorithms")

        return "\n".join(report)
```

## Safety and Robustness Considerations

### Safety-Critical Mobile Manipulation

Safety is paramount in mobile manipulation systems, especially when operating in human-populated environments.

```python
# safety_mobile_manipulation.py
class SafetyCriticalMobileManipulator:
    """
    Safety-critical mobile manipulation system with multiple safety layers.
    """
    def __init__(self, config):
        self.config = config

        # Primary controllers
        self.navigation_controller = NavigationController()
        self.manipulation_controller = ManipulationController()

        # Safety layers
        self.collision_avoidance = CollisionAvoidanceSystem()
        self.stability_checker = StabilityVerificationSystem()
        self.emergency_stop = EmergencyStopSystem()
        self.force_limiter = ForceLimitingSystem()

        # Safety constraints
        self.safety_constraints = {
            'workspace_limits': config.get('workspace_limits', [(-2, 2), (-2, 2), (0, 2)]),  # x, y, z limits
            'velocity_limits': config.get('velocity_limits', [1.0, 1.0, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),  # Base + manipulator limits
            'force_limits': config.get('force_limits', [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0]),  # Joint force limits
            'collision_distance': config.get('collision_distance', 0.1),  # meters
            'stability_margin': config.get('stability_margin', 0.2)  # Stability safety margin
        }

        # System state
        self.emergency_stop_active = False
        self.current_safety_level = 'nominal'

    def compute_safe_action(self, desired_action, robot_state):
        """
        Compute safe action by applying safety filters to desired action.

        Args:
            desired_action: Raw action from policy
            robot_state: Current robot state

        Returns:
            Safety-filtered action
        """
        if self.emergency_stop_active:
            return self.get_emergency_stop_action()

        # Check workspace limits
        if not self.check_workspace_constraints(desired_action, robot_state):
            desired_action = self.project_to_workspace(desired_action, robot_state)

        # Check velocity limits
        if not self.check_velocity_constraints(desired_action):
            desired_action = self.limit_velocities(desired_action)

        # Check stability
        if not self.stability_checker.verify_stability(desired_action, robot_state):
            desired_action = self.get_stability_preserving_action(desired_action, robot_state)

        # Check collision risk
        if self.collision_avoidance.predict_collision_risk(desired_action, robot_state):
            desired_action = self.collision_avoidance.avoid_collision(
                desired_action, robot_state
            )

        # Check force limits (for manipulation)
        if not self.force_limiter.check_force_limits(desired_action, robot_state):
            desired_action = self.force_limiter.limit_forces(desired_action, robot_state)

        return desired_action

    def check_workspace_constraints(self, action, robot_state):
        """Check if action violates workspace constraints."""
        # Calculate resulting position
        base_pos = robot_state['base_position']  # [x, y, theta]
        manip_pos = robot_state['manipulator_position']  # Joint positions

        # Check base workspace
        next_base_pos = base_pos + action[:3] * 0.1  # Assuming 0.1s time step
        for i, (limit_low, limit_high) in enumerate(self.safety_constraints['workspace_limits'][:2]):
            if not (limit_low <= next_base_pos[i] <= limit_high):
                return False

        # Check manipulator workspace would require forward kinematics
        # For simplicity, we'll check joint limits
        next_manip_pos = manip_pos + action[3:] * 0.05  # Manipulator time step
        joint_limits = self.get_manipulator_joint_limits()
        for i, (limit_low, limit_high) in enumerate(joint_limits):
            if not (limit_low <= next_manip_pos[i] <= limit_high):
                return False

        return True

    def project_to_workspace(self, action, robot_state):
        """Project action to respect workspace constraints."""
        projected_action = action.copy()

        # Limit base motion based on workspace
        base_pos = robot_state['base_position']
        for i, (limit_low, limit_high) in enumerate(self.safety_constraints['workspace_limits'][:2]):
            next_pos = base_pos[i] + action[i] * 0.1
            if next_pos < limit_low:
                projected_action[i] = (limit_low - base_pos[i]) / 0.1
            elif next_pos > limit_high:
                projected_action[i] = (limit_high - base_pos[i]) / 0.1

        # Limit manipulator motion based on joint limits
        manip_pos = robot_state['manipulator_position']
        joint_limits = self.get_manipulator_joint_limits()
        for i, (limit_low, limit_high) in enumerate(joint_limits):
            next_pos = manip_pos[i] + action[i + 3] * 0.05
            if next_pos < limit_low:
                projected_action[i + 3] = (limit_low - manip_pos[i]) / 0.05
            elif next_pos > limit_high:
                projected_action[i + 3] = (limit_high - manip_pos[i]) / 0.05

        return projected_action

    def check_velocity_constraints(self, action):
        """Check if action violates velocity constraints."""
        velocity_limits = self.safety_constraints['velocity_limits']
        for i, (vel, limit) in enumerate(zip(action, velocity_limits)):
            if abs(vel) > limit:
                return False
        return True

    def limit_velocities(self, action):
        """Limit action to respect velocity constraints."""
        velocity_limits = self.safety_constraints['velocity_limits']
        limited_action = np.clip(action, -np.array(velocity_limits), np.array(velocity_limits))
        return limited_action

    def get_manipulator_joint_limits(self):
        """Get manipulator joint limits."""
        # This would come from robot URDF or configuration
        # For example, Franka Emika Panda joint limits:
        return [
            (-2.8973, 2.8973),   # Joint 1
            (-1.7628, 1.7628),   # Joint 2
            (-2.8973, 2.8973),   # Joint 3
            (-3.0718, -0.0698),  # Joint 4
            (-2.8973, 2.8973),   # Joint 5
            (-0.0175, 3.7525),   # Joint 6
            (-2.8973, 2.8973)    # Joint 7
        ]

    def get_stability_preserving_action(self, action, robot_state):
        """Modify action to preserve stability."""
        # This would implement complex stability preservation algorithms
        # For simplicity, reduce manipulator velocities when base is moving
        base_moving = np.linalg.norm(action[:3]) > 0.1
        if base_moving:
            # Reduce manipulator velocities to maintain stability
            modified_action = action.copy()
            manip_scale = 0.5  # Reduce manipulator speed by 50%
            modified_action[3:] *= manip_scale
            return modified_action

        return action

    def get_emergency_stop_action(self):
        """Get action for emergency stop state."""
        return np.zeros(10)  # Zero all velocities

class CollisionAvoidanceSystem:
    """
    Advanced collision avoidance for mobile manipulation systems.
    """
    def __init__(self):
        # Configuration
        self.safe_distance = 0.1  # meters
        self.prediction_horizon = 0.5  # seconds
        self.collision_check_resolution = 0.05  # meters

        # Obstacle detection and prediction
        self.obstacle_detector = ObstacleDetectionModule()
        self.motion_predictor = MotionPredictionModule()

    def predict_collision_risk(self, action, robot_state):
        """Predict if action leads to collision."""
        # Predict robot motion based on action
        predicted_trajectory = self.predict_robot_motion(action, robot_state)

        # Check for collisions along trajectory
        for state in predicted_trajectory:
            if self.check_collision_at_state(state):
                return True

        return False

    def predict_robot_motion(self, action, robot_state, num_steps=10):
        """Predict robot motion based on current action."""
        dt = self.prediction_horizon / num_steps
        trajectory = []

        current_state = robot_state.copy()
        for _ in range(num_steps):
            # Integrate action to get next state
            next_state = self.integrate_dynamics(current_state, action, dt)
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def integrate_dynamics(self, state, action, dt):
        """Integrate robot dynamics for one time step."""
        # Simplified dynamics integration
        new_state = state.copy()

        # Update base position (differential drive model)
        vx, vy, omega = action[:3]
        theta = state['base_orientation'][2]  # Assuming 3rd element is yaw
        new_state['base_position'][0] += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        new_state['base_position'][1] += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        new_state['base_orientation'][2] += omega * dt

        # Update manipulator position
        joint_velocities = action[3:]
        new_state['manipulator_position'] += joint_velocities * dt

        return new_state

    def check_collision_at_state(self, robot_state):
        """Check for collision at specific robot state."""
        # Get robot geometry in current configuration
        robot_geometry = self.get_robot_geometry(robot_state)

        # Get environment obstacles
        obstacles = self.obstacle_detector.get_obstacles()

        # Check for collisions
        for obstacle in obstacles:
            if self.geometric_collision_check(robot_geometry, obstacle):
                return True

        return False

    def get_robot_geometry(self, robot_state):
        """Get robot geometry in current configuration."""
        # This would use forward kinematics to compute robot geometry
        # For now, return simplified bounding boxes
        return self.compute_robot_bounding_boxes(robot_state)

    def geometric_collision_check(self, geometry1, geometry2):
        """Check for geometric collision between two geometries."""
        # Simplified collision check using bounding boxes
        # In reality, this would use more sophisticated geometric algorithms
        bb1 = self.get_bounding_box(geometry1)
        bb2 = self.get_bounding_box(geometry2)

        # Check for bounding box overlap
        return self.bounding_box_overlap(bb1, bb2)

    def get_bounding_box(self, geometry):
        """Get bounding box for geometry."""
        # Simplified bounding box calculation
        return geometry.get_bounding_box() if hasattr(geometry, 'get_bounding_box') else [0, 0, 0, 1, 1, 1]

    def bounding_box_overlap(self, bb1, bb2):
        """Check if two bounding boxes overlap."""
        # Check overlap in each dimension
        for i in range(3):  # x, y, z
            if bb1[i] > bb2[i+3] or bb2[i] > bb1[i+3]:
                return False  # No overlap in this dimension
        return True  # Overlap in all dimensions

    def avoid_collision(self, action, robot_state):
        """Modify action to avoid predicted collisions."""
        # This would implement sophisticated collision avoidance algorithms
        # For simplicity, reduce velocities in collision direction

        # Detect collision direction (simplified)
        collision_direction = self.estimate_collision_direction(action, robot_state)

        # Modify action to reduce approach velocity to obstacles
        modified_action = action.copy()
        for i in range(3):  # Only modify base motion for collision avoidance
            if collision_direction[i] * action[i] > 0:  # Moving toward obstacle
                modified_action[i] *= 0.5  # Reduce speed by half

        return modified_action

    def estimate_collision_direction(self, action, robot_state):
        """Estimate direction of nearest collision."""
        # Simplified collision direction estimation
        # In reality, this would analyze the geometric relationship
        obstacles = self.obstacle_detector.get_obstacles()
        if obstacles:
            # Return direction to closest obstacle
            robot_pos = robot_state['base_position'][:2]
            closest_obstacle = min(obstacles, key=lambda obs: np.linalg.norm(obs.position[:2] - robot_pos))
            direction = closest_obstacle.position[:2] - robot_pos
            direction = np.append(direction, [0])  # Add z component
            return direction / (np.linalg.norm(direction) + 1e-6)  # Normalize

        return np.array([0, 0, 0])  # No obstacles

class StabilityVerificationSystem:
    """
    System for verifying and maintaining stability during mobile manipulation.
    """
    def __init__(self):
        self.stability_threshold = 0.1  # meters (max ZMP deviation)
        self.support_polygon_margin = 0.05  # meters
        self.mass_distribution_model = self.load_mass_model()

    def verify_stability(self, action, robot_state):
        """Verify if action maintains system stability."""
        # Calculate predicted ZMP (Zero Moment Point) after action
        predicted_zmp = self.predict_zmp(action, robot_state)

        # Calculate support polygon based on current configuration
        support_polygon = self.calculate_support_polygon(robot_state)

        # Check if ZMP is within support polygon with margin
        is_stable = self.is_zmp_in_support_polygon(predicted_zmp, support_polygon)

        return is_stable

    def predict_zmp(self, action, robot_state):
        """Predict ZMP based on planned action and current state."""
        # Simplified ZMP prediction
        # In reality, this would involve complex dynamic modeling
        base_acc = action[:3] * 10  # Rough acceleration estimate
        manip_acc = np.append(action[3:], [0]) * 5  # Manipulator acceleration

        # Simplified ZMP calculation (would need proper dynamics model)
        zmp_x = robot_state['base_position'][0] + base_acc[0] * 0.05
        zmp_y = robot_state['base_position'][1] + base_acc[1] * 0.05

        return np.array([zmp_x, zmp_y])

    def calculate_support_polygon(self, robot_state):
        """Calculate support polygon based on robot configuration."""
        # For a wheeled mobile manipulator, support polygon depends on wheel positions
        # and manipulator configuration affecting stability
        base_footprint = self.get_base_support_area(robot_state)
        manipulator_influence = self.get_manipulator_stability_influence(robot_state)

        # Combine base and manipulator effects
        effective_support = self.combine_support_areas(base_footprint, manipulator_influence)

        return effective_support

    def get_base_support_area(self, robot_state):
        """Get base support area (simplified)."""
        # This would depend on the specific robot base configuration
        # For a 4-wheeled robot, return convex hull of wheel positions
        wheel_positions = self.get_wheel_positions(robot_state)
        return self.compute_convex_hull(wheel_positions)

    def get_manipulator_stability_influence(self, robot_state):
        """Get manipulator's influence on stability."""
        # Extended manipulator reduces stability margin
        manip_position = robot_state['manipulator_position']
        extension_measure = np.linalg.norm(manip_position)  # Simplified extension measure

        # Return reduced support area based on extension
        reduction_factor = max(0.5, 1.0 - extension_measure * 0.1)  # Reduce with extension
        return reduction_factor

    def is_zmp_in_support_polygon(self, zmp, support_polygon):
        """Check if ZMP is within support polygon."""
        # Simplified check - in reality, this would use point-in-polygon algorithms
        if len(support_polygon) >= 3:
            # Check if ZMP is within the convex hull of support points
            return self.point_in_convex_hull(zmp, support_polygon)
        return False

    def point_in_convex_hull(self, point, hull_points):
        """Check if point is inside convex hull of given points."""
        # Simplified implementation
        # In reality, use proper point-in-polygon or point-in-polyhedron algorithms
        centroid = np.mean(hull_points, axis=0)
        distances = [np.linalg.norm(p - point) for p in hull_points]
        avg_distance = np.mean(distances)
        point_distance = np.linalg.norm(centroid - point)
        return point_distance < avg_distance * 0.8  # Conservative check

    def compute_convex_hull(self, points):
        """Compute convex hull of points."""
        # In practice, use scipy.spatial.ConvexHull
        return points  # Simplified

class ForceLimitingSystem:
    """
    System for limiting forces during manipulation to ensure safety.
    """
    def __init__(self):
        self.force_limits = {
            'joint_torque': [87, 87, 87, 87, 12, 12, 12],  # Franka Emika limits
            'gripper_force': 70,  # Newtons
            'end_effector_force': 50  # Newtons
        }
        self.safety_factor = 0.8  # Use 80% of limits for safety

    def check_force_limits(self, action, robot_state):
        """Check if action would exceed force limits."""
        # Estimate forces based on action and current state
        estimated_forces = self.estimate_forces(action, robot_state)

        # Check against limits
        for force_type, (estimated, limit) in estimated_forces.items():
            if estimated > limit * self.safety_factor:
                return False

        return True

    def estimate_forces(self, action, robot_state):
        """Estimate forces that would result from action."""
        estimated_forces = {}

        # Estimate joint torques (simplified)
        joint_velocities = action[3:]
        joint_positions = robot_state['manipulator_position']
        # Torque roughly proportional to acceleration and gravity compensation
        estimated_torques = np.abs(joint_velocities) * 10 + self.estimate_gravity_torques(joint_positions)
        estimated_forces['joint_torque'] = (np.max(estimated_torques), self.force_limits['joint_torque'])

        # Estimate gripper force based on grasp state
        if robot_state.get('gripper_closed', False):
            # If grasping, estimate based on object weight and acceleration
            object_weight = robot_state.get('held_object_weight', 1.0)  # kg
            acceleration = np.linalg.norm(action[:3])  # Base acceleration
            estimated_gripper_force = object_weight * 9.81 + object_weight * acceleration
            estimated_forces['gripper_force'] = (estimated_gripper_force, self.force_limits['gripper_force'])

        return estimated_forces

    def estimate_gravity_torques(self, joint_positions):
        """Estimate gravity compensation torques."""
        # Simplified gravity torque estimation
        # In reality, use robot dynamics model (e.g., RNEA)
        return np.abs(np.sin(joint_positions)) * 20  # Simplified gravity model

    def limit_forces(self, action, robot_state):
        """Modify action to respect force limits."""
        modified_action = action.copy()

        # Reduce manipulator velocities if force limits would be exceeded
        estimated_forces = self.estimate_forces(action, robot_state)

        if 'joint_torque' in estimated_forces:
            est_torque, lim_torque = estimated_forces['joint_torque']
            if est_torque > lim_torque * self.safety_factor:
                # Scale down manipulator velocities proportionally
                scale_factor = (lim_torque * self.safety_factor) / (est_torque + 1e-6)
                modified_action[3:] *= max(0.1, scale_factor)  # Don't scale below 10%

        return modified_action
```

## Deployment and Real-World Considerations

### Hardware Integration and Real-Time Performance

Deploying mobile manipulation systems in real-world scenarios requires careful attention to computational efficiency, real-time performance, and hardware constraints.

```python
# real_world_deployment.py
import time
import threading
import queue
from collections import deque

class RealWorldMobileManipulationSystem:
    """
    Real-world deployment system for mobile manipulation with Isaac Foundation Models.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize foundation models
        self.foundation_model = self.load_optimized_model()

        # Initialize robot interface
        self.robot_interface = RobotInterface(config.get('robot_config', {}))

        # Initialize perception system
        self.perception_system = PerceptionInterface(config.get('perception_config', {}))

        # Initialize safety systems
        self.safety_manager = SafetyCriticalMobileManipulator(config.get('safety_config', {}))

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Real-time execution parameters
        self.control_frequency = config.get('control_frequency', 50)  # Hz
        self.dt = 1.0 / self.control_frequency

        # Data queues for real-time processing
        self.sensor_queue = queue.Queue(maxsize=5)
        self.action_queue = queue.Queue(maxsize=5)

        # Threading for parallel processing
        self.running = False
        self.control_thread = None
        self.perception_thread = None

    def load_optimized_model(self):
        """Load and optimize foundation model for real-time deployment."""
        import torch_tensorrt

        # Load JIT-compiled model
        model_path = self.config['model_path']
        model = torch.jit.load(model_path)

        # Optimize for target hardware
        if self.config.get('optimize_for_tensorrt', False) and self.device.type == 'cuda':
            optimized_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=[1, 14],  # Minimum batch size, state dim
                        opt_shape=[8, 14],  # Optimal batch size
                        max_shape=[16, 14]  # Maximum batch size
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 3],  # Goal state
                        opt_shape=[8, 3],
                        max_shape=[16, 3]
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 3],  # Manipulation target
                        opt_shape=[8, 3],
                        max_shape=[16, 3]
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 3, 224, 224],  # Visual input
                        opt_shape=[8, 3, 224, 224],
                        max_shape=[16, 3, 224, 224]
                    )
                ],
                enabled_precisions={torch.float, torch.half},
                workspace_size=1 << 25  # 32MB
            )
        else:
            optimized_model = model.to(self.device)
            if self.device.type == 'cuda':
                optimized_model = optimized_model.half()  # Use FP16 if optimizing for speed

        return optimized_model

    def start_real_time_control(self):
        """Start real-time control loop in separate thread."""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.perception_thread = threading.Thread(target=self._perception_loop)

        self.control_thread.start()
        self.perception_thread.start()

    def _control_loop(self):
        """Main control loop running at specified frequency."""
        last_time = time.time()

        while self.running:
            current_time = time.time()

            if current_time - last_time >= self.dt:
                try:
                    # Get latest sensor data
                    if not self.sensor_queue.empty():
                        sensor_data = self.sensor_queue.get_nowait()

                        # Run foundation model inference
                        start_inference = time.time()
                        action = self._compute_action(sensor_data)
                        inference_time = time.time() - start_inference

                        # Apply safety filters
                        safe_action = self.safety_manager.compute_safe_action(action, sensor_data['state'])

                        # Send action to robot
                        self.robot_interface.execute_action(safe_action)

                        # Monitor performance
                        self.performance_monitor.record_iteration(
                            inference_time=inference_time,
                            action_magnitude=np.linalg.norm(safe_action)
                        )

                        last_time = current_time

                except queue.Empty:
                    time.sleep(0.001)  # Brief sleep to prevent busy waiting
                except Exception as e:
                    print(f"Control loop error: {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.001)  # Prevent busy waiting

    def _perception_loop(self):
        """Perception loop running in parallel."""
        while self.running:
            try:
                # Capture sensor data
                sensor_data = self.perception_system.capture_data()

                # Process perception (in real systems, this might run at different frequency)
                processed_data = self.perception_system.process_data(sensor_data)

                # Add to sensor queue if not full
                if self.sensor_queue.qsize() < self.sensor_queue._maxsize:
                    self.sensor_queue.put(processed_data)
                else:
                    # Drop oldest data if queue is full
                    try:
                        self.sensor_queue.get_nowait()
                        self.sensor_queue.put(processed_data)
                    except queue.Empty:
                        pass

            except Exception as e:
                print(f"Perception loop error: {e}")
                time.sleep(0.01)

    def _compute_action(self, sensor_data):
        """Compute action using foundation model."""
        # Prepare inputs
        state_tensor = torch.FloatTensor(sensor_data['state']).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(sensor_data['goal']).unsqueeze(0).to(self.device)
        manip_target_tensor = torch.FloatTensor(sensor_data['manip_target']).unsqueeze(0).to(self.device)
        visual_tensor = torch.FloatTensor(sensor_data['visual']).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                visual_tensor = visual_tensor.half()  # Use FP16 for speed

            outputs = self.foundation_model(
                robot_state=state_tensor,
                goal_state=goal_tensor,
                manip_target=manip_target_tensor,
                visual_input=visual_tensor
            )

            # Combine base and manipulator commands
            action = torch.cat([
                outputs['base_command'],
                outputs['manipulator_command']
            ], dim=1)

        return action.cpu().numpy()[0]

    def stop_real_time_control(self):
        """Stop real-time control loops."""
        self.running = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        if self.perception_thread and self.perception_thread.is_alive():
            self.perception_thread.join(timeout=1.0)

    def execute_mobile_manipulation_task(self, task_description, task_params):
        """
        Execute a complete mobile manipulation task.

        Args:
            task_description: Natural language description of task
            task_params: Task-specific parameters

        Returns:
            Task execution result
        """
        print(f"Starting mobile manipulation task: {task_description}")

        # Parse task and generate plan
        task_plan = self._generate_task_plan(task_description, task_params)

        # Execute plan step by step
        success = True
        for step_idx, step in enumerate(task_plan):
            print(f"Executing step {step_idx + 1}/{len(task_plan)}: {step['description']}")

            step_success = self._execute_task_step(step)
            if not step_success:
                print(f"Task step {step_idx + 1} failed")
                success = False
                break

        # Record task outcome
        self.performance_monitor.record_task_outcome(
            task_description=task_description,
            success=success,
            steps_completed=len(task_plan) if success else step_idx + 1
        )

        return {
            'success': success,
            'steps_completed': len(task_plan) if success else step_idx + 1,
            'total_time': time.time() - task_plan[0].get('start_time', time.time()),
            'performance_metrics': self.performance_monitor.get_recent_metrics()
        }

    def _generate_task_plan(self, task_description, task_params):
        """Generate task plan from description and parameters."""
        # This would use a task planner that understands the mobile manipulation capabilities
        # For this example, return a simple hardcoded plan

        if "grasp object" in task_description.lower():
            return [
                {
                    'description': 'Navigate to object location',
                    'type': 'navigation',
                    'target': task_params.get('object_location', [1.0, 1.0, 0.0])
                },
                {
                    'description': 'Approach and grasp object',
                    'type': 'manipulation',
                    'target': task_params.get('object_pose', [1.0, 1.0, 0.2])
                },
                {
                    'description': 'Transport object to destination',
                    'type': 'navigation',
                    'target': task_params.get('destination', [2.0, 2.0, 0.0])
                },
                {
                    'description': 'Place object at destination',
                    'type': 'manipulation',
                    'target': task_params.get('destination', [2.0, 2.0, 0.2])
                }
            ]
        else:
            # Default plan for unknown tasks
            return [{
                'description': 'Unknown task - executing basic approach',
                'type': 'navigation',
                'target': [0.5, 0.5, 0.0]
            }]

    def _execute_task_step(self, step):
        """Execute a single task step."""
        if step['type'] == 'navigation':
            return self._execute_navigation_step(step['target'])
        elif step['type'] == 'manipulation':
            return self._execute_manipulation_step(step['target'])
        else:
            print(f"Unknown step type: {step['type']}")
            return False

    def _execute_navigation_step(self, target_pose):
        """Execute navigation to target pose."""
        # In a real system, this would interface with navigation stack
        # For simulation, return success after brief delay
        print(f"Navigating to {target_pose}")
        time.sleep(2)  # Simulate navigation time
        return True

    def _execute_manipulation_step(self, target_pose):
        """Execute manipulation at target pose."""
        # In a real system, this would interface with manipulation stack
        # For simulation, return success after brief delay
        print(f"Manipulating at {target_pose}")
        time.sleep(3)  # Simulate manipulation time
        return True

class PerformanceMonitor:
    """
    Monitor performance metrics for real-time mobile manipulation system.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size

        # Tracking queues
        self.inference_times = deque(maxlen=window_size)
        self.action_magnitudes = deque(maxlen=window_size)
        self.task_outcomes = deque(maxlen=50)  # Track task outcomes

        # System state
        self.last_update_time = time.time()
        self.iteration_count = 0

    def record_iteration(self, inference_time, action_magnitude):
        """Record metrics for a single control iteration."""
        self.inference_times.append(inference_time)
        self.action_magnitudes.append(action_magnitude)
        self.iteration_count += 1

    def record_task_outcome(self, task_description, success, steps_completed):
        """Record outcome of a complete task."""
        self.task_outcomes.append({
            'task': task_description,
            'success': success,
            'steps_completed': steps_completed,
            'timestamp': time.time()
        })

    def get_recent_metrics(self):
        """Get recent performance metrics."""
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'min_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'avg_action_magnitude': 0,
                'current_frequency': 0,
                'task_success_rate': 0
            }

        # Calculate metrics
        avg_inf_time = np.mean(self.inference_times) * 1000  # Convert to ms
        min_inf_time = min(self.inference_times) * 1000
        max_inf_time = max(self.inference_times) * 1000

        avg_action_mag = np.mean(self.action_magnitudes) if self.action_magnitudes else 0

        # Calculate current frequency (based on iteration count and time window)
        current_time = time.time()
        time_window = current_time - self.last_update_time
        if time_window > 0:
            current_frequency = self.iteration_count / time_window
        else:
            current_frequency = 0

        # Task success rate
        recent_tasks = list(self.task_outcomes)
        if recent_tasks:
            task_success_rate = sum(1 for task in recent_tasks if task['success']) / len(recent_tasks)
        else:
            task_success_rate = 0

        return {
            'avg_inference_time_ms': avg_inf_time,
            'min_inference_time_ms': min_inf_time,
            'max_inference_time_ms': max_inf_time,
            'avg_action_magnitude': avg_action_mag,
            'current_frequency': current_frequency,
            'task_success_rate': task_success_rate
        }

    def check_performance_degradation(self, threshold_percent=20):
        """Check if performance has degraded beyond threshold."""
        if len(self.inference_times) < 10:
            return False  # Not enough data

        recent_avg = np.mean(list(self.inference_times)[-10:])
        historical_avg = np.mean(list(self.inference_times)[:-10]) if len(self.inference_times) > 10 else recent_avg

        # Check if recent performance is worse by more than threshold
        if historical_avg > 0:
            degradation = (recent_avg - historical_avg) / historical_avg * 100
            return degradation > threshold_percent

        return False

class RobotInterface:
    """
    Interface to physical or simulated robot.
    """
    def __init__(self, config):
        self.config = config
        self.robot_connected = False

        # Initialize connection to robot
        self.connect_to_robot()

    def connect_to_robot(self):
        """Connect to physical or simulated robot."""
        # This would implement actual robot connection
        # For simulation, just mark as connected
        self.robot_connected = True
        print("Connected to robot interface")

    def execute_action(self, action):
        """Execute action on robot."""
        if not self.robot_connected:
            print("ERROR: Robot not connected")
            return False

        # Action format: [base_vx, base_vy, base_omega, joint_vel_1, ..., joint_vel_7]
        base_action = action[:3]
        manipulator_action = action[3:]

        # Send commands to robot (implementation depends on specific robot)
        base_success = self.send_base_command(base_action)
        manipulator_success = self.send_manipulator_command(manipulator_action)

        return base_success and manipulator_success

    def send_base_command(self, base_velocities):
        """Send velocity commands to mobile base."""
        # In real implementation, send commands via ROS/Middleware
        # For simulation, just return success
        print(f"Sending base command: {base_velocities}")
        return True

    def send_manipulator_command(self, joint_velocities):
        """Send joint velocity commands to manipulator."""
        # In real implementation, send commands via robot controller
        # For simulation, just return success
        print(f"Sending manipulator command: {joint_velocities}")
        return True

    def get_robot_state(self):
        """Get current robot state."""
        # In real implementation, read from robot state publisher
        # For simulation, return dummy state
        return {
            'base_position': np.random.rand(3),  # x, y, theta
            'base_velocity': np.random.rand(3),  # vx, vy, omega
            'manipulator_position': np.random.rand(7),  # Joint positions
            'manipulator_velocity': np.random.rand(7),  # Joint velocities
            'gripper_state': 'open'  # open/closed
        }

class PerceptionInterface:
    """
    Interface to robot perception system.
    """
    def __init__(self, config):
        self.config = config
        self.cameras = []
        self.sensors = []

        # Initialize perception sensors
        self.initialize_sensors()

    def initialize_sensors(self):
        """Initialize perception sensors."""
        # This would connect to actual robot sensors
        # For simulation, just set up dummy sensors
        print("Initialized perception interface")
        pass

    def capture_data(self):
        """Capture sensor data from all modalities."""
        # In real implementation, capture from cameras, LiDAR, etc.
        # For simulation, return dummy data
        return {
            'rgb_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'depth_image': np.random.rand(480, 640).astype(np.float32),
            'point_cloud': np.random.rand(1000, 4).astype(np.float32),  # x, y, z, intensity
            'imu_data': np.random.rand(6).astype(np.float32),  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            'joint_states': np.random.rand(14).astype(np.float32)  # 7 pos + 7 vel
        }

    def process_data(self, raw_data):
        """Process raw sensor data into usable format."""
        # Process and combine sensor data
        processed = {
            'state': np.concatenate([
                raw_data['joint_states'][:7],  # Joint positions
                raw_data['joint_states'][7:],  # Joint velocities
                raw_data['imu_data'][:3]       # Accelerometer (simplified)
            ]),
            'visual': raw_data['rgb_image'].transpose(2, 0, 1).astype(np.float32) / 255.0,
            'goal': np.array([1.0, 1.0, 0.0]),  # Example goal
            'manip_target': np.array([0.5, 0.5, 0.2])  # Example manipulation target
        }

        return processed
```

## Code Examples with Explanations

### Complete Mobile Manipulation System Implementation

```python
# complete_mobile_manipulation_system.py
import torch
import numpy as np
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt

class CompleteMobileManipulationSystem:
    """
    Complete system integrating Isaac Foundation Models for mobile manipulation.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize all Isaac Foundation Model components
        self.perceptor = IsaacPerceptor(config.get('perceptor_config', {}))
        self.controller = IsaacController(config.get('controller_config', {}))
        self.manipulator = IsaacManipulator(config.get('manipulator_config', {}))
        self.navigator = IsaacNavigator(config.get('navigator_config', {}))

        # Initialize mobile manipulation foundation model
        self.mobile_manip_model = MobileManipulationFoundation(
            navigation_encoder_dim=config.get('nav_dim', 256),
            manipulation_encoder_dim=config.get('manip_dim', 256),
            fusion_dim=config.get('fusion_dim', 512),
            action_dim=config.get('action_dim', 10)
        )

        # Initialize safety systems
        self.safety_manager = SafetyCriticalMobileManipulator(config.get('safety_config', {}))

        # Initialize real-world interface
        self.robot_interface = RobotInterface(config.get('robot_config', {}))
        self.perception_interface = PerceptionInterface(config.get('perception_config', {}))

        # Performance tracking
        self.training_logs = {
            'losses': [],
            'success_rates': [],
            'completion_times': [],
            'safety_violations': [],
            'real_world_metrics': []
        }

        print("Complete Mobile Manipulation System initialized")
        print(f"Device: {self.device}")
        print(f"Action dimension: {config.get('action_dim', 10)}")
        print(f"Safety enabled: {config.get('enable_safety', True)}")

    def train_mobile_manipulation_policy(self, training_data, num_epochs=100):
        """Train the mobile manipulation policy."""
        print(f"Starting training for {num_epochs} epochs...")

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.mobile_manip_model.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in training_data:
                optimizer.zero_grad()

                # Prepare inputs
                states = torch.FloatTensor(batch['states']).to(self.device)
                goals = torch.FloatTensor(batch['goals']).to(self.device)
                manip_targets = torch.FloatTensor(batch['manip_targets']).to(self.device)
                visual_inputs = torch.FloatTensor(batch['visual']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)

                # Forward pass
                outputs = self.mobile_manip_model(states, goals, manip_targets, visual_inputs)

                # Combine base and manipulator commands
                predicted_actions = torch.cat([
                    outputs['base_command'],
                    outputs['manipulator_command']
                ], dim=1)

                # Compute loss
                loss = criterion(predicted_actions, actions)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.training_logs['losses'].append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        print(f"Training completed. Final average loss: {avg_loss:.6f}")

    def evaluate_policy(self, test_env, num_episodes=10):
        """Evaluate the trained policy."""
        self.mobile_manip_model.eval()
        success_count = 0
        total_time = 0

        for episode in range(num_episodes):
            state = test_env.reset()
            episode_time = 0
            done = False

            while not done:
                start_time = time.time()

                # Get action from model
                action = self.get_action_from_model(state)

                # Apply safety filtering
                safe_action = self.safety_manager.compute_safe_action(action, state)

                # Execute in environment
                state, reward, done, info = test_env.step(safe_action)

                step_time = time.time() - start_time
                episode_time += step_time

                if info.get('is_success', False):
                    success_count += 1
                    break

            total_time += episode_time

        avg_success_rate = success_count / num_episodes if num_episodes > 0 else 0
        avg_completion_time = total_time / success_count if success_count > 0 else float('inf')

        evaluation_results = {
            'success_rate': avg_success_rate,
            'avg_completion_time': avg_completion_time,
            'total_episodes': num_episodes,
            'successful_episodes': success_count
        }

        self.training_logs['success_rates'].append(avg_success_rate)
        self.training_logs['completion_times'].append(avg_completion_time)

        return evaluation_results

    def get_action_from_model(self, state):
        """Get action from mobile manipulation model."""
        # Convert state to tensors
        state_tensor = torch.FloatTensor(state['robot_state']).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(state['goal']).unsqueeze(0).to(self.device)
        manip_target_tensor = torch.FloatTensor(state['manip_target']).unsqueeze(0).to(self.device)
        visual_tensor = torch.FloatTensor(state['visual']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.mobile_manip_model(
                robot_state=state_tensor,
                goal_state=goal_tensor,
                manip_target=manip_target_tensor,
                visual_input=visual_tensor
            )

            # Combine actions
            action = torch.cat([
                outputs['base_command'],
                outputs['manipulator_command']
            ], dim=1)

        return action.cpu().numpy()[0]

    def deploy_to_real_robot(self):
        """Deploy the trained policy to a real robot."""
        print("Deploying mobile manipulation policy to real robot...")

        # Set models to evaluation mode
        self.mobile_manip_model.eval()
        self.perceptor.eval()
        self.controller.eval()
        self.manipulator.eval()
        self.navigator.eval()

        print("Policy deployed. Ready for real-world execution.")

    def execute_real_world_task(self, task_description, task_params):
        """Execute a mobile manipulation task on a real robot."""
        print(f"Executing task: {task_description}")

        # Initialize real-time control system
        real_system = RealWorldMobileManipulationSystem({
            'model_path': './models/mobile_manipulation_foundation.pt',
            'control_frequency': 50,
            'robot_config': self.config.get('robot_config', {}),
            'safety_config': self.config.get('safety_config', {}),
            'optimize_for_tensorrt': True
        })

        # Start real-time control
        real_system.start_real_time_control()

        try:
            # Execute the task
            result = real_system.execute_mobile_manipulation_task(
                task_description, task_params
            )

            print(f"Task completed. Success: {result['success']}")
            print(f"Performance metrics: {result['performance_metrics']}")

            return result

        finally:
            # Stop control system
            real_system.stop_real_time_control()

    def run_simulation_experiment(self):
        """Run a complete simulation experiment."""
        print("Running simulation experiment...")

        # Create simulation environment
        sim_env = IsaacMobileManipulationEnv({
            'task_type': 'grasp_and_transport',
            'robot_name': 'tiago',
            'max_steps': 1000
        })

        # Train policy
        print("Training policy in simulation...")
        # In a real implementation, we would collect training data from the sim environment
        # For this example, we'll skip training and directly evaluate

        # Evaluate policy
        print("Evaluating policy...")
        eval_results = self.evaluate_policy(sim_env, num_episodes=20)

        print(f"Simulation evaluation results: {eval_results}")

        return eval_results

    def visualize_training_progress(self):
        """Visualize training progress and performance metrics."""
        if not self.training_logs['losses']:
            print("No training data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training Loss
        axes[0, 0].plot(self.training_logs['losses'])
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')

        # Plot 2: Success Rate
        if self.training_logs['success_rates']:
            axes[0, 1].plot(self.training_logs['success_rates'])
            axes[0, 1].set_title('Success Rate During Training')
            axes[0, 1].set_xlabel('Evaluation Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim(0, 1)

        # Plot 3: Completion Time
        if self.training_logs['completion_times']:
            axes[1, 0].plot(self.training_logs['completion_times'])
            axes[1, 0].set_title('Task Completion Time')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Time (seconds)')

        # Plot 4: Safety Violations
        if self.training_logs['safety_violations']:
            axes[1, 1].plot(self.training_logs['safety_violations'])
            axes[1, 1].set_title('Safety Violations Over Time')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Violations')

        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate complete mobile manipulation system."""
    print("Initializing Complete Mobile Manipulation System...")

    # Configuration
    config = {
        'nav_dim': 256,
        'manip_dim': 256,
        'fusion_dim': 512,
        'action_dim': 10,  # 3 base + 7 manipulator
        'learning_rate': 1e-4,
        'control_frequency': 50,
        'enable_safety': True,
        'safety_config': {
            'workspace_limits': [(-3, 3), (-3, 3), (0, 2)],
            'velocity_limits': [1.0, 1.0, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            'collision_distance': 0.1,
            'stability_margin': 0.2
        }
    }

    # Initialize system
    system = CompleteMobileManipulationSystem(config)

    print("\n1. Running simulation experiment:")
    sim_results = system.run_simulation_experiment()
    print(f"Simulation results: {sim_results}")

    print("\n2. Training policy (simulated):")
    # In a real implementation, we would train with actual data
    # For this demo, we'll just show the structure
    print("Policy training would occur here with real data")

    print("\n3. Evaluating policy (simulated):")
    # In a real implementation, we would evaluate with test episodes
    print("Policy evaluation would occur here with test episodes")

    print("\n4. Mobile manipulation system demonstration completed!")
    print("The system is now ready for real-world deployment with proper safety measures.")

if __name__ == "__main__":
    main()
```

## Best Practices and Lessons Learned

Based on extensive development and deployment of Isaac Foundation Models for mobile manipulation, several best practices have emerged that significantly impact system performance and reliability.

### System Integration Best Practices

1. **Modular Design**: Design components to be modular and interchangeable, allowing for independent updates and maintenance.

2. **Real-time Performance**: Prioritize computational efficiency to ensure real-time performance in mobile manipulation tasks.

3. **Safety-First Architecture**: Implement multiple layers of safety checks, from low-level hardware protection to high-level behavioral constraints.

4. **Simulation-to-Reality Transfer**: Use domain randomization, system identification, and gradual deployment strategies to bridge the sim-to-real gap.

### Training Best Practices

1. **Multi-Modal Data Integration**: Ensure balanced integration of visual, proprioceptive, and other sensory modalities in training data.

2. **Hierarchical Learning**: Implement hierarchical learning approaches that decompose complex tasks into simpler subtasks.

3. **Continuous Learning**: Incorporate mechanisms for continuous learning and adaptation in real-world environments.

4. **Robust Evaluation**: Implement comprehensive evaluation protocols that test system performance across diverse scenarios.

### Deployment Best Practices

1. **Gradual Rollout**: Deploy new capabilities gradually, starting with simple tasks and expanding to complex behaviors.

2. **Continuous Monitoring**: Implement continuous monitoring systems to detect performance degradation and safety violations.

3. **Human-in-the-Loop**: Maintain human oversight capabilities, especially for safety-critical operations.

4. **Regular Updates**: Plan for regular model updates based on real-world experience and performance feedback.

## Summary

Chapter 16 has provided comprehensive coverage of Isaac Foundation Models for autonomous mobile manipulation. You've learned about the integration of navigation and manipulation capabilities, coordinated control strategies, reinforcement learning approaches for mobile manipulation, and real-world deployment considerations. The examples demonstrate practical implementations for creating robust mobile manipulation systems that can operate effectively in complex environments. The safety considerations and best practices ensure reliable and safe operation of these systems in real-world applications.

## Exercises

1. Implement a custom mobile manipulation task planner that optimizes for both navigation efficiency and manipulation success.
2. Design a coordinated control system that dynamically adjusts the priority between navigation and manipulation based on task requirements.
3. Create a reinforcement learning environment specifically for mobile manipulation tasks with realistic dynamics.
4. Develop a safety system that prevents collisions while maintaining task efficiency in cluttered environments.
5. Build a simulation-to-reality transfer system for mobile manipulation with domain randomization.

## References

- Siciliano, B., & Khatib, O. (2016). "Springer Handbook of Robotics." Springer.
- Patil, S., et al. (2014). "Scaling Up Gaussian Belief Space Planning Through Covariance-Free Trajectory Optimization." Robotics: Science and Systems.
- Mason, S., et al. (2020). "Learning Latent Representations to Influence Multi-Agent Interaction." Conference on Robot Learning.
- Isaac Foundation Models Documentation: Mobile Manipulation Implementation Guide
- NVIDIA. (2023). "Isaac Sim Mobile Manipulation Tutorials." NVIDIA Corporation.