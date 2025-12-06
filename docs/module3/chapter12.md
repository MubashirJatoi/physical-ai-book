---
sidebar_position: 4
title: Chapter 12 - Sim-to-Real Transfer Techniques
---

# Chapter 12 - Sim-to-Real Transfer Techniques

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the sim-to-real transfer problem and its challenges
- Implement domain randomization techniques for improved transfer
- Apply system identification methods to bridge sim-to-real gaps
- Design reality gaps characterization and modeling approaches
- Implement adaptive control methods for real-world deployment
- Evaluate and validate sim-to-real transfer performance
- Apply transfer learning techniques for robotics applications
- Optimize simulation environments for better transferability
- Implement safety mechanisms for safe real-world deployment

## Prerequisites

Before starting this chapter, you should have:
- Understanding of reinforcement learning fundamentals from Chapter 11
- Knowledge of robot dynamics and control systems
- Experience with simulation environments (Isaac Sim, Gazebo, PyBullet)
- Basic understanding of system identification and parameter estimation
- Completion of Module 1 and Module 3 Chapter 11 content
- Familiarity with robot hardware and control interfaces

## Introduction to Sim-to-Real Transfer

### The Reality Gap Problem

The "reality gap" refers to the fundamental challenge of transferring policies trained in simulation to real robotic systems. Despite advances in simulation fidelity, significant differences exist between simulated and real environments that can severely impact the performance of learned policies.

**Sources of Reality Gap**:
- **Dynamics Mismatch**: Differences in friction, damping, and mass properties
- **Actuator Imperfections**: Motor delays, backlash, and saturation effects
- **Sensor Noise**: Different noise characteristics and sensor imperfections
- **Environmental Conditions**: Lighting, textures, and surface properties
- **Modeling Errors**: Inaccuracies in physical models and simplifications
- **Temporal Discrepancies**: Timing differences in control loops and processing

### Transfer Learning in Robotics

Transfer learning in robotics aims to leverage knowledge acquired in simulation to accelerate learning or improve performance in the real world. The goal is to reduce the need for extensive real-world training while maintaining safety and performance.

**Types of Transfer**:
- **Direct Transfer**: Applying simulation-trained policies directly to real robots
- **Fine-tuning**: Adapting simulation policies with limited real-world data
- **Domain Adaptation**: Modifying policies to account for domain differences
- **Meta-learning**: Learning to adapt quickly to new environments

### Key Challenges in Sim-to-Real Transfer

**Fidelity vs. Transferability Trade-off**:
- High-fidelity simulations may not always provide the best transfer
- Oversimplified models might actually transfer better in some cases
- Finding the right level of detail is crucial

**Safety Considerations**:
- Direct transfer can be dangerous without proper safeguards
- Real robots have physical constraints and safety requirements
- Robustness to modeling errors is essential

**Sample Efficiency**:
- Real-world interactions are expensive and time-consuming
- Minimizing required real-world samples is critical
- Leveraging simulation data effectively

## Domain Randomization Techniques

### Concept and Theory

Domain randomization is a technique that randomizes simulation parameters during training to improve the robustness of learned policies to domain shifts. The idea is to expose the policy to a wide range of conditions during training, making it more adaptable to real-world variations.

**Mathematical Foundation**:
The domain randomization approach can be formalized as training on a distribution of environments:

$$P_{sim}(s_{t+1}|s_t, a_t) = \int P(s_{t+1}|s_t, a_t, \theta) P(\theta)d\theta$$

Where $\theta$ represents the randomized parameters.

### Implementation of Domain Randomization

```python
# domain_randomization.py
import numpy as np
import torch
import torch.nn as nn
import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class DomainRandomizationParams:
    """
    Parameters for domain randomization.
    """
    # Physical properties
    mass_range: Tuple[float, float] = (0.8, 1.2)
    friction_range: Tuple[float, float] = (0.5, 1.5)
    damping_range: Tuple[float, float] = (0.8, 1.2)
    inertia_range: Tuple[float, float] = (0.9, 1.1)

    # Actuator properties
    motor_gain_range: Tuple[float, float] = (0.9, 1.1)
    actuator_delay_range: Tuple[float, float] = (0.0, 0.02)  # seconds
    actuator_noise_range: Tuple[float, float] = (0.0, 0.05)

    # Sensor properties
    sensor_noise_range: Tuple[float, float] = (0.0, 0.1)
    sensor_bias_range: Tuple[float, float] = (-0.05, 0.05)
    sensor_delay_range: Tuple[float, float] = (0.0, 0.01)

    # Environmental properties
    gravity_range: Tuple[float, float] = (9.78, 9.83)  # m/s^2
    lighting_range: Tuple[float, float] = (0.5, 1.5)
    texture_options: List[str] = None
    surface_roughness_range: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        if self.texture_options is None:
            self.texture_options = ['metal', 'wood', 'plastic', 'fabric', 'concrete']

class DomainRandomizer:
    """
    Domain randomization system for sim-to-real transfer.
    """
    def __init__(self, params: DomainRandomizationParams):
        self.params = params
        self.current_params = {}
        self.param_history = []

    def randomize_dynamics(self) -> Dict[str, float]:
        """Randomize dynamic properties."""
        randomized_params = {
            'mass_multiplier': np.random.uniform(
                self.params.mass_range[0], self.params.mass_range[1]
            ),
            'friction_multiplier': np.random.uniform(
                self.params.friction_range[0], self.params.friction_range[1]
            ),
            'damping_multiplier': np.random.uniform(
                self.params.damping_range[0], self.params.damping_range[1]
            ),
            'inertia_multiplier': np.random.uniform(
                self.params.inertia_range[0], self.params.inertia_range[1]
            )
        }
        return randomized_params

    def randomize_actuators(self) -> Dict[str, float]:
        """Randomize actuator properties."""
        randomized_params = {
            'motor_gain_multiplier': np.random.uniform(
                self.params.motor_gain_range[0], self.params.motor_gain_range[1]
            ),
            'actuator_delay': np.random.uniform(
                self.params.actuator_delay_range[0], self.params.actuator_delay_range[1]
            ),
            'actuator_noise_std': np.random.uniform(
                self.params.actuator_noise_range[0], self.params.actuator_noise_range[1]
            )
        }
        return randomized_params

    def randomize_sensors(self) -> Dict[str, float]:
        """Randomize sensor properties."""
        randomized_params = {
            'sensor_noise_std': np.random.uniform(
                self.params.sensor_noise_range[0], self.params.sensor_noise_range[1]
            ),
            'sensor_bias': np.random.uniform(
                self.params.sensor_bias_range[0], self.params.sensor_bias_range[1]
            ),
            'sensor_delay': np.random.uniform(
                self.params.sensor_delay_range[0], self.params.sensor_delay_range[1]
            )
        }
        return randomized_params

    def randomize_environment(self) -> Dict[str, Any]:
        """Randomize environmental properties."""
        randomized_params = {
            'gravity': np.random.uniform(
                self.params.gravity_range[0], self.params.gravity_range[1]
            ),
            'lighting_intensity': np.random.uniform(
                self.params.lighting_range[0], self.params.lighting_range[1]
            ),
            'surface_roughness': np.random.uniform(
                self.params.surface_roughness_range[0], self.params.surface_roughness_range[1]
            ),
            'texture_type': random.choice(self.params.texture_options)
        }
        return randomized_params

    def apply_randomization(self, sim_env, randomize_per_episode: bool = True):
        """
        Apply domain randomization to simulation environment.

        Args:
            sim_env: Simulation environment to apply randomization to
            randomize_per_episode: Whether to randomize per episode or per step
        """
        # Generate new random parameters
        dynamics_params = self.randomize_dynamics()
        actuator_params = self.randomize_actuators()
        sensor_params = self.randomize_sensors()
        env_params = self.randomize_environment()

        # Store current parameters
        self.current_params = {
            **dynamics_params,
            **actuator_params,
            **sensor_params,
            **env_params
        }

        # Apply to simulation environment (conceptual implementation)
        # In practice, this would interface with the simulation engine
        self._apply_to_simulation(sim_env, self.current_params)

        # Store in history
        self.param_history.append(self.current_params.copy())

    def _apply_to_simulation(self, sim_env, params: Dict[str, Any]):
        """
        Apply parameters to simulation environment.
        This is a conceptual implementation - actual implementation
        depends on the simulation engine used.
        """
        # Example for Isaac Sim or similar engines
        if hasattr(sim_env, 'set_dynamics_params'):
            sim_env.set_dynamics_params(
                mass_mult=params['mass_multiplier'],
                friction_mult=params['friction_multiplier'],
                damping_mult=params['damping_multiplier']
            )

        if hasattr(sim_env, 'set_actuator_params'):
            sim_env.set_actuator_params(
                gain_mult=params['motor_gain_multiplier'],
                delay=params['actuator_delay'],
                noise_std=params['actuator_noise_std']
            )

        if hasattr(sim_env, 'set_sensor_params'):
            sim_env.set_sensor_params(
                noise_std=params['sensor_noise_std'],
                bias=params['sensor_bias'],
                delay=params['sensor_delay']
            )

        if hasattr(sim_env, 'set_environment_params'):
            sim_env.set_environment_params(
                gravity=params['gravity'],
                lighting=params['lighting_intensity'],
                surface_roughness=params['surface_roughness']
            )

    def get_current_params(self) -> Dict[str, Any]:
        """Get current randomization parameters."""
        return self.current_params.copy()

    def reset(self):
        """Reset randomization state."""
        self.current_params = {}
        self.param_history = []

class CurriculumDomainRandomization(DomainRandomizer):
    """
    Curriculum-based domain randomization that gradually increases
    randomization range as policy improves.
    """
    def __init__(self, params: DomainRandomizationParams):
        super().__init__(params)
        self.progress = 0.0  # Training progress (0.0 to 1.0)
        self.min_progress_threshold = 0.1  # Minimum progress before increasing difficulty

    def update_progress(self, performance_metric: float, threshold: float = 0.8):
        """
        Update progress based on performance.

        Args:
            performance_metric: Current performance metric (higher is better)
            threshold: Performance threshold for progress calculation
        """
        self.progress = min(1.0, performance_metric / threshold)

    def randomize_dynamics(self) -> Dict[str, float]:
        """Randomize dynamics with curriculum-based ranges."""
        # Scale randomization range based on progress
        current_range = self._scale_range(self.params.mass_range, self.progress)
        mass_multiplier = np.random.uniform(current_range[0], current_range[1])

        current_range = self._scale_range(self.params.friction_range, self.progress)
        friction_multiplier = np.random.uniform(current_range[0], current_range[1])

        current_range = self._scale_range(self.params.damping_range, self.progress)
        damping_multiplier = np.random.uniform(current_range[0], current_range[1])

        current_range = self._scale_range(self.params.inertia_range, self.progress)
        inertia_multiplier = np.random.uniform(current_range[0], current_range[1])

        return {
            'mass_multiplier': mass_multiplier,
            'friction_multiplier': friction_multiplier,
            'damping_multiplier': damping_multiplier,
            'inertia_multiplier': inertia_multiplier
        }

    def _scale_range(self, original_range: Tuple[float, float],
                     progress: float) -> Tuple[float, float]:
        """
        Scale a range based on training progress.

        Args:
            original_range: Original (min, max) range
            progress: Training progress (0.0 to 1.0)

        Returns:
            Scaled range tuple
        """
        if progress < self.min_progress_threshold:
            # Start with narrow range when performance is low
            mid = (original_range[0] + original_range[1]) / 2
            narrow_range = 0.1 + progress * 0.4  # From 0.1 to 0.5 of original range
            range_size = (original_range[1] - original_range[0]) * narrow_range
            new_min = mid - range_size / 2
            new_max = mid + range_size / 2
            return (new_min, new_max)
        else:
            # Gradually expand to full range
            expansion_factor = min(1.0, (progress - self.min_progress_threshold) / (1.0 - self.min_progress_threshold))
            original_size = original_range[1] - original_range[0]
            mid = (original_range[0] + original_range[1]) / 2

            new_size = original_size * expansion_factor
            new_min = mid - new_size / 2
            new_max = mid + new_size / 2

            return (max(original_range[0], new_min), min(original_range[1], new_max))

class CorrelatedDomainRandomization:
    """
    Domain randomization with correlated parameters to maintain
    physical plausibility of randomizations.
    """
    def __init__(self, correlation_matrix: np.ndarray = None):
        """
        Initialize with correlation matrix for parameter dependencies.

        Args:
            correlation_matrix: Correlation matrix for parameters
        """
        if correlation_matrix is None:
            # Default weak correlations
            self.correlation_matrix = np.eye(5)  # 5 parameters
            # Add some weak correlations
            self.correlation_matrix[0, 1] = 0.3  # mass-friction correlation
            self.correlation_matrix[1, 2] = 0.2  # friction-damping correlation
            self.correlation_matrix[2, 3] = 0.1  # damping-inertia correlation
        else:
            self.correlation_matrix = correlation_matrix

        self.cholesky_decomposition = np.linalg.cholesky(self.correlation_matrix)

    def sample_correlated_params(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample correlated parameters using Cholesky decomposition.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of correlated parameter samples
        """
        # Generate independent standard normal samples
        independent_samples = np.random.standard_normal((n_samples, self.correlation_matrix.shape[0]))

        # Apply correlation using Cholesky decomposition
        correlated_samples = independent_samples @ self.cholesky_decomposition.T

        return correlated_samples
```

### Advanced Domain Randomization Techniques

```python
# advanced_domain_randomization.py
import numpy as np
import torch
from scipy import stats
from typing import Dict, Any, List
import cv2

class PerceptualDomainRandomization:
    """
    Domain randomization focused on visual perception differences.
    """
    def __init__(self):
        self.color_jitter_params = {
            'brightness': (0.6, 1.4),
            'contrast': (0.6, 1.4),
            'saturation': (0.6, 1.4),
            'hue': (-0.1, 0.1)
        }

        self.noise_params = {
            'gaussian_std': (0.0, 0.1),
            'poisson_lambda': (0.8, 1.2),
            'speckle_std': (0.0, 0.05)
        }

        self.distortion_params = {
            'blur_sigma': (0.5, 2.0),
            'motion_blur_kernel': (3, 7),
            'lens_distortion': (0.0, 0.1)
        }

    def randomize_visual_properties(self) -> Dict[str, Any]:
        """Randomize visual properties for perception tasks."""
        visual_params = {}

        # Color jittering
        visual_params['color_jitter'] = {
            'brightness': np.random.uniform(*self.color_jitter_params['brightness']),
            'contrast': np.random.uniform(*self.color_jitter_params['contrast']),
            'saturation': np.random.uniform(*self.color_jitter_params['saturation']),
            'hue': np.random.uniform(*self.color_jitter_params['hue'])
        }

        # Noise properties
        visual_params['noise'] = {
            'gaussian_std': np.random.uniform(*self.noise_params['gaussian_std']),
            'poisson_lambda': np.random.uniform(*self.noise_params['poisson_lambda']),
            'speckle_std': np.random.uniform(*self.noise_params['speckle_std'])
        }

        # Distortion properties
        visual_params['distortion'] = {
            'blur_sigma': np.random.uniform(*self.distortion_params['blur_sigma']),
            'motion_blur_kernel': int(np.random.uniform(*self.distortion_params['motion_blur_kernel'])),
            'lens_distortion': np.random.uniform(*self.distortion_params['lens_distortion'])
        }

        return visual_params

    def apply_visual_randomization(self, image: np.ndarray) -> np.ndarray:
        """Apply visual randomization to an image."""
        params = self.randomize_visual_properties()

        # Apply color jittering
        image = self._apply_color_jitter(image, params['color_jitter'])

        # Apply noise
        image = self._apply_noise(image, params['noise'])

        # Apply distortion
        image = self._apply_distortion(image, params['distortion'])

        return image

    def _apply_color_jitter(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply color jittering to image."""
        # Convert to HSV for easier manipulation
        image_hsv = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)

        # Apply hue shift
        image_hsv[:, :, 0] = np.clip(image_hsv[:, :, 0] + params['hue'], 0, 1)

        # Apply saturation scaling
        image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * params['saturation'], 0, 1)

        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Apply brightness and contrast
        image_rgb = np.clip(image_rgb * params['brightness'], 0, 1)
        mean_val = np.mean(image_rgb)
        image_rgb = np.clip((image_rgb - mean_val) * params['contrast'] + mean_val, 0, 1)

        return (image_rgb * 255).astype(np.uint8)

    def _apply_noise(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply various types of noise to image."""
        image_float = image.astype(np.float32) / 255.0

        # Gaussian noise
        gaussian_noise = np.random.normal(0, params['gaussian_std'], image_float.shape)
        image_noisy = np.clip(image_float + gaussian_noise, 0, 1)

        # Poisson noise (approximated)
        poisson_noise = np.random.poisson(image_noisy * params['poisson_lambda']) / params['poisson_lambda']
        image_noisy = np.clip(image_noisy + (poisson_noise - image_noisy) * 0.1, 0, 1)  # Scale down effect

        # Speckle noise
        speckle_noise = image_noisy * np.random.normal(0, params['speckle_std'], image_noisy.shape)
        image_noisy = np.clip(image_noisy + speckle_noise, 0, 1)

        return (image_noisy * 255).astype(np.uint8)

    def _apply_distortion(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply various distortions to image."""
        # Gaussian blur
        kernel_size = int(params['blur_sigma'] * 6 + 1)  # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), params['blur_sigma'])

        # Motion blur
        kernel_size = params['motion_blur_kernel']
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        motion_blurred = cv2.filter2D(blurred, -1, kernel)

        return motion_blurred

class AdaptiveDomainRandomization:
    """
    Adaptive domain randomization that adjusts based on policy performance.
    """
    def __init__(self, initial_params: DomainRandomizationParams):
        self.initial_params = initial_params
        self.current_params = initial_params
        self.performance_history = []
        self.randomization_strength = 0.1  # Initial strength

    def update_randomization(self, episode_rewards: List[float],
                           success_rate: float = None):
        """
        Update randomization parameters based on recent performance.

        Args:
            episode_rewards: Recent episode rewards
            success_rate: Success rate for discrete tasks
        """
        if len(episode_rewards) < 10:
            return  # Need sufficient history

        recent_avg = np.mean(episode_rewards[-10:])
        historical_avg = np.mean(episode_rewards[:-10]) if len(episode_rewards) > 10 else recent_avg

        # If performance is degrading, increase randomization
        if recent_avg < historical_avg * 0.95:
            self.randomization_strength = min(1.0, self.randomization_strength * 1.1)
        # If performance is improving, decrease randomization slightly
        elif recent_avg > historical_avg * 1.05:
            self.randomization_strength = max(0.1, self.randomization_strength * 0.95)

        # Adjust parameter ranges based on strength
        self._adjust_parameter_ranges()

    def _adjust_parameter_ranges(self):
        """Adjust parameter ranges based on current randomization strength."""
        # Scale all ranges by current strength
        adjusted_params = DomainRandomizationParams()

        # Adjust based on current strength
        scale_factor = 0.5 + 0.5 * self.randomization_strength  # Scale from 0.5 to 1.0

        # Adjust mass range
        mid_mass = (self.initial_params.mass_range[0] + self.initial_params.mass_range[1]) / 2
        range_size = (self.initial_params.mass_range[1] - self.initial_params.mass_range[0]) * scale_factor
        adjusted_params.mass_range = (mid_mass - range_size/2, mid_mass + range_size/2)

        # Adjust friction range
        mid_friction = (self.initial_params.friction_range[0] + self.initial_params.friction_range[1]) / 2
        range_size = (self.initial_params.friction_range[1] - self.initial_params.friction_range[0]) * scale_factor
        adjusted_params.friction_range = (mid_friction - range_size/2, mid_friction + range_size/2)

        # Apply similar scaling to other parameters
        # (This would be repeated for all parameters)

        self.current_params = adjusted_params
```

## System Identification for Reality Gap Bridging

### Physics Parameter Estimation

System identification helps bridge the reality gap by estimating real-world parameters that differ from simulation:

```python
# system_identification.py
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, List
import torch

class RobotSystemIdentifier:
    """
    System identification for robotic systems to estimate real-world parameters.
    """
    def __init__(self, robot_model: str = 'manipulator'):
        self.robot_model = robot_model
        self.identified_params = {}
        self.param_bounds = {}
        self.identification_data = {
            'inputs': [],
            'outputs': [],
            'timestamps': []
        }
        self.model_order = 2  # Default model order for polynomial fitting

    def collect_identification_data(self,
                                  robot_interface,
                                  excitation_signals: List[np.ndarray],
                                  sample_rate: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Collect data for system identification.

        Args:
            robot_interface: Interface to real robot
            excitation_signals: List of excitation signals to apply
            sample_rate: Sampling rate for data collection

        Returns:
            Dictionary containing collected data
        """
        collected_data = {
            'joint_positions': [],
            'joint_velocities': [],
            'joint_torques': [],
            'timestamps': [],
            'commanded_positions': [],
            'commanded_velocities': []
        }

        dt = 1.0 / sample_rate

        for signal in excitation_signals:
            # Reset robot to initial position
            robot_interface.move_to_initial_position()
            time.sleep(1.0)  # Wait for settling

            # Apply excitation signal
            start_time = robot_interface.get_current_time()

            for t_idx, commanded_pos in enumerate(signal):
                current_time = robot_interface.get_current_time()

                # Apply position command
                robot_interface.set_joint_position(commanded_pos)

                # Collect measurements
                actual_pos = robot_interface.get_joint_position()
                actual_vel = robot_interface.get_joint_velocity()
                applied_torque = robot_interface.get_joint_torque()

                collected_data['joint_positions'].append(actual_pos.copy())
                collected_data['joint_velocities'].append(actual_vel.copy())
                collected_data['joint_torques'].append(applied_torque.copy())
                collected_data['timestamps'].append(current_time)
                collected_data['commanded_positions'].append(commanded_pos.copy())

                # Sleep to maintain sample rate
                time.sleep(max(0, dt - (robot_interface.get_current_time() - current_time)))

        # Convert to numpy arrays
        for key in collected_data:
            collected_data[key] = np.array(collected_data[key])

        return collected_data

    def estimate_dynamic_parameters(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Estimate dynamic parameters using inverse dynamics.

        Args:
            data: Dictionary containing collected data

        Returns:
            Dictionary of estimated parameters
        """
        # Extract data
        q = data['joint_positions']  # Joint positions
        q_dot = data['joint_velocities']  # Joint velocities
        tau = data['joint_torques']  # Applied torques

        # Estimate joint accelerations using numerical differentiation
        q_ddot = self._estimate_accelerations(q_dot, 1.0/100.0)  # Assuming 100Hz

        # Use inverse dynamics to estimate parameters
        # For a simple 2-DOF manipulator, we can estimate masses, lengths, etc.
        params = self._inverse_dynamics_identification(q, q_dot, q_ddot, tau)

        return params

    def _estimate_accelerations(self, velocities: np.ndarray, dt: float) -> np.ndarray:
        """Estimate accelerations from velocities using numerical differentiation."""
        # Use central difference for interior points, forward/backward for edges
        accelerations = np.zeros_like(velocities)

        # Interior points: central difference
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)

        # Edge points: forward/backward difference
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt

        return accelerations

    def _inverse_dynamics_identification(self,
                                       q: np.ndarray,
                                       q_dot: np.ndarray,
                                       q_ddot: np.ndarray,
                                       tau: np.ndarray) -> Dict[str, float]:
        """
        Perform inverse dynamics identification to estimate parameters.

        This is a simplified example for a 2-DOF planar manipulator.
        """
        n_samples = len(q)
        n_joints = q.shape[1] if len(q.shape) > 1 else 1

        # Prepare regression matrices for each joint
        # This is a simplified example - full implementation would be more complex
        Y_regression = np.zeros((n_samples * n_joints, 10))  # 10 parameters to estimate
        tau_regression = np.zeros(n_samples * n_joints)  # Combined torque vector

        for i in range(n_samples):
            for j in range(n_joints):
                # Extract joint values for this timestep
                q1, q2 = q[i, 0], q[i, 1] if n_joints > 1 else 0
                q1_dot, q2_dot = q_dot[i, 0], q_dot[i, 1] if n_joints > 1 else 0
                q1_ddot, q2_ddot = q_ddot[i, 0], q_ddot[i, 1] if n_joints > 1 else 0
                tau1, tau2 = tau[i, 0], tau[i, 1] if n_joints > 1 else 0

                # Row index in regression matrix
                row_idx = i * n_joints + j

                if j == 0:  # Joint 1 regression row
                    Y_regression[row_idx, :] = [
                        q1_ddot,                    # Coefficient for q1_ddot
                        q2_ddot,                    # Coefficient for q2_ddot
                        q1_ddot*q2,                 # Coupling term
                        q1_dot**2,                  # Centrifugal term
                        q2_dot**2,                  # Centrifugal term
                        q1_dot*q2_dot,              # Coriolis term
                        np.cos(q1),                 # Gravity term
                        np.cos(q1 + q2),            # Gravity coupling
                        q1_dot,                     # Friction approximation
                        1.0                         # Constant term
                    ]
                    tau_regression[row_idx] = tau1
                else:  # Joint 2 regression row
                    Y_regression[row_idx, :] = [
                        q2_ddot,                    # Coefficient for q2_ddot
                        q1_ddot,                    # Coupling term
                        q1_ddot*q2,                 # Coupling term
                        q1_dot**2,                  # Centrifugal term
                        q2_dot**2,                  # Centrifugal term
                        q1_dot*q2_dot,              # Coriolis term
                        np.cos(q1 + q2),            # Gravity term
                        np.cos(q2),                 # Gravity term
                        q2_dot,                     # Friction approximation
                        1.0                         # Constant term
                    ]
                    tau_regression[row_idx] = tau2

        # Solve the linear system Y * theta = tau for parameters theta
        try:
            # Use least squares to estimate parameters
            estimated_params, residuals, rank, s = np.linalg.lstsq(Y_regression, tau_regression, rcond=None)

            # Map estimated parameters to meaningful robot parameters
            identified_params = {
                'mass_link1': estimated_params[0],  # Simplified mapping
                'mass_link2': estimated_params[1],
                'length_link1': estimated_params[6],  # Related to gravity term
                'length_link2': estimated_params[7],
                'inertia_link1': estimated_params[0],  # Related to acceleration term
                'inertia_link2': estimated_params[1],
                'friction_link1': estimated_params[8],
                'friction_link2': estimated_params[9],
                'gravity_effect': estimated_params[6],  # Dominant gravity parameter
                'coupling_coeff': estimated_params[2]   # Dynamic coupling
            }

            return identified_params

        except np.linalg.LinAlgError:
            print("Linear system is singular, using regularization")
            # Use regularized solution
            reg_matrix = Y_regression.T @ Y_regression + 0.01 * np.eye(Y_regression.shape[1])
            reg_vector = Y_regression.T @ tau_regression
            estimated_params = np.linalg.solve(reg_matrix, reg_vector)

            # Same mapping as above
            identified_params = {
                'mass_link1': estimated_params[0],
                'mass_link2': estimated_params[1],
                'length_link1': estimated_params[6],
                'length_link2': estimated_params[7],
                'inertia_link1': estimated_params[0],
                'inertia_link2': estimated_params[1],
                'friction_link1': estimated_params[8],
                'friction_link2': estimated_params[9],
                'gravity_effect': estimated_params[6],
                'coupling_coeff': estimated_params[2]
            }

            return identified_params

    def estimate_friction_parameters(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Estimate friction parameters using the Coulomb + viscous friction model.

        tau_friction = Fc * sign(q_dot) + Fv * q_dot
        """
        q_dot = data['joint_velocities']
        tau = data['joint_torques']

        # Separate positive and negative velocities for each joint
        friction_params = {}

        for joint_idx in range(q_dot.shape[1]):
            vel = q_dot[:, joint_idx]
            torque = tau[:, joint_idx]

            # Use only quasi-static movements (low velocity) to estimate Coulomb friction
            low_vel_mask = np.abs(vel) < 0.1  # rad/s threshold

            if np.sum(low_vel_mask) > 10:  # Need sufficient data
                # Coulomb friction estimation (magnitude of torque at low speeds)
                coulomb_friction = np.mean(np.abs(torque[low_vel_mask]))
            else:
                # Estimate from all data using robust method
                coulomb_friction = np.median(np.abs(torque))

            # Estimate viscous friction from linear relationship
            # Remove low-velocity data to avoid Coulomb friction effects
            high_vel_mask = np.abs(vel) > 0.2
            if np.sum(high_vel_mask) > 10:
                # Use linear regression to estimate viscous friction coefficient
                from sklearn.linear_model import LinearRegression

                X = vel[high_vel_mask].reshape(-1, 1)
                y = torque[high_vel_mask] - coulomb_friction * np.sign(vel[high_vel_mask])

                reg = LinearRegression().fit(X, y)
                viscous_friction = reg.coef_[0]
            else:
                viscous_friction = 0.1  # Default estimate

            friction_params[f'joint_{joint_idx}_coulomb'] = coulomb_friction
            friction_params[f'joint_{joint_idx}_viscous'] = viscous_friction

        return friction_params

    def update_simulation_model(self, sim_env, identified_params: Dict[str, float]):
        """
        Update simulation model with identified real-world parameters.

        Args:
            sim_env: Simulation environment to update
            identified_params: Identified real-world parameters
        """
        # Update simulation with identified parameters
        for param_name, param_value in identified_params.items():
            if hasattr(sim_env, f'set_{param_name.replace(" ", "_")}'):
                setter_method = getattr(sim_env, f'set_{param_name.replace(" ", "_")}')
                setter_method(param_value)
            elif hasattr(sim_env, 'set_parameter'):
                # Generic parameter setting method
                sim_env.set_parameter(param_name, param_value)

    def validate_identification(self,
                              sim_env,
                              real_env,
                              test_trajectory: np.ndarray) -> Dict[str, float]:
        """
        Validate system identification by comparing sim and real responses.

        Args:
            sim_env: Simulation environment with updated parameters
            real_env: Real environment (or interface)
            test_trajectory: Test trajectory to validate with

        Returns:
            Dictionary of validation metrics
        """
        sim_responses = []
        real_responses = []

        # Execute trajectory in simulation
        sim_state = sim_env.reset()
        for cmd in test_trajectory:
            sim_next_state, sim_reward, _, _ = sim_env.step(cmd)
            sim_responses.append(sim_next_state.copy())
            sim_state = sim_next_state

        # Execute trajectory in real environment (or use stored data)
        real_state = real_env.reset()
        for cmd in test_trajectory:
            real_next_state, real_reward, _, _ = real_env.step(cmd)
            real_responses.append(real_next_state.copy())
            real_state = real_next_state

        # Calculate validation metrics
        sim_array = np.array(sim_responses)
        real_array = np.array(real_responses)

        # RMSE between sim and real responses
        rmse = np.sqrt(np.mean((sim_array - real_array)**2))

        # Correlation coefficient
        correlation = np.corrcoef(sim_array.flatten(), real_array.flatten())[0, 1]

        # Maximum absolute error
        max_error = np.max(np.abs(sim_array - real_array))

        validation_metrics = {
            'rmse': rmse,
            'correlation': correlation,
            'max_error': max_error,
            'relative_error': rmse / (np.std(real_array) + 1e-8)  # Normalize by data variance
        }

        return validation_metrics

class NeuralNetworkSystemIdentifier(nn.Module):
    """
    Neural network-based system identification for complex nonlinear dynamics.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = None,
                 model_order: int = 3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.model_order = model_order  # How many past steps to consider

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def prepare_sequences(self,
                         states: np.ndarray,
                         actions: np.ndarray,
                         seq_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for neural network training.

        Args:
            states: State trajectory [T, state_dim]
            actions: Action trajectory [T, action_dim]
            seq_length: Length of input sequences

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Create input-output pairs using sliding window
        input_seqs = []
        target_seqs = []

        for i in range(len(states) - seq_length):
            # Input: sequence of past states and actions
            past_states = states[i:i+seq_length]
            past_actions = actions[i:i+seq_length]

            # Combine into single feature vector
            input_features = np.concatenate([past_states.flatten(), past_actions.flatten()])

            # Target: next state
            target_state = states[i + seq_length]

            input_seqs.append(input_features)
            target_seqs.append(target_state)

        return torch.FloatTensor(input_seqs), torch.FloatTensor(target_seqs)

    def train_identifier(self,
                        training_data: Dict[str, np.ndarray],
                        learning_rate: float = 1e-3,
                        epochs: int = 1000,
                        batch_size: int = 32):
        """
        Train the neural network identifier.

        Args:
            training_data: Dictionary with 'states' and 'actions' trajectories
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare sequences
        X, y = self.prepare_sequences(
            training_data['states'],
            training_data['actions']
        )

        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X.size(0))

            for i in range(0, X.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = X[indices], y[indices]

                optimizer.zero_grad()

                # Forward pass
                outputs = self(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

    def predict_next_state(self,
                          current_state: torch.Tensor,
                          action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict the next state given current state and action sequence.

        Args:
            current_state: Current state [state_dim]
            action_sequence: Sequence of actions [seq_len, action_dim]

        Returns:
            Predicted next state [state_dim]
        """
        # Combine state and action sequence into single input
        input_features = torch.cat([
            current_state.flatten(),
            action_sequence.flatten()
        ])

        # Get prediction
        next_state = self(input_features)

        return next_state

    def correct_simulator(self,
                         sim_env,
                         current_state: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        """
        Correct simulator prediction using neural network identifier.

        Args:
            sim_env: Simulation environment
            current_state: Current state
            action: Action to execute

        Returns:
            Corrected next state
        """
        # Get simulator prediction
        sim_next_state = sim_env.predict_next_state(current_state, action)

        # Get neural network correction
        nn_correction = self.predict_next_state(current_state, action.unsqueeze(0))

        # Combine simulator prediction with NN correction
        corrected_state = sim_next_state + nn_correction

        return corrected_state
```

## Reality Gap Characterization and Modeling

### Gap Analysis Framework

```python
# gap_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics.pairwise import pairwise_distances

class RealityGapAnalyzer:
    """
    Framework for characterizing and analyzing the reality gap.
    """
    def __init__(self):
        self.sim_data = []
        self.real_data = []
        self.gap_metrics = {}
        self.gap_models = {}

    def collect_data(self,
                    sim_env,
                    real_env,
                    policy,
                    num_episodes: int = 10) -> Tuple[List, List]:
        """
        Collect data from both simulation and real environments.

        Args:
            sim_env: Simulation environment
            real_env: Real environment
            policy: Policy to evaluate
            num_episodes: Number of episodes to collect data from

        Returns:
            Tuple of (sim_data, real_data)
        """
        sim_episode_data = []
        real_episode_data = []

        # Collect simulation data
        for episode in range(num_episodes):
            sim_trajectory = self._collect_trajectory(sim_env, policy, 'simulation')
            sim_episode_data.append(sim_trajectory)

        # Collect real data
        for episode in range(num_episodes):
            real_trajectory = self._collect_trajectory(real_env, policy, 'real')
            real_episode_data.append(real_trajectory)

        self.sim_data = sim_episode_data
        self.real_data = real_episode_data

        return sim_episode_data, real_episode_data

    def _collect_trajectory(self, env, policy, env_type: str) -> Dict[str, np.ndarray]:
        """Collect a single trajectory from environment."""
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'observations': [],
            'env_type': env_type
        }

        state = env.reset()
        done = False

        while not done:
            # Get action from policy
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state)
            else:
                # Assume policy is a function
                action = policy(state)

            # Store data
            trajectory['states'].append(state.copy())
            trajectory['actions'].append(action.copy())

            # Take step in environment
            state, reward, done, info = env.step(action)

            # Store additional information
            trajectory['rewards'].append(reward)
            trajectory['observations'].append(state.copy())  # Simplified

        # Convert to numpy arrays
        for key in ['states', 'actions', 'rewards', 'observations']:
            trajectory[key] = np.array(trajectory[key])

        return trajectory

    def compute_gap_metrics(self) -> Dict[str, float]:
        """
        Compute various metrics to quantify the reality gap.

        Returns:
            Dictionary of gap metrics
        """
        metrics = {}

        # Flatten all trajectories for analysis
        all_sim_states = np.vstack([ep['states'] for ep in self.sim_data])
        all_real_states = np.vstack([ep['states'] for ep in self.real_data])

        all_sim_actions = np.vstack([ep['actions'] for ep in self.sim_data])
        all_real_actions = np.vstack([ep['actions'] for ep in self.real_data])

        # 1. Wasserstein Distance (Earth Mover's Distance)
        try:
            # Compute Wasserstein distance for each state dimension
            state_wasserstein = []
            for dim in range(all_sim_states.shape[1]):
                w_dist = wasserstein_distance(
                    all_sim_states[:, dim],
                    all_real_states[:, dim]
                )
                state_wasserstein.append(w_dist)
            metrics['state_wasserstein_mean'] = np.mean(state_wasserstein)
            metrics['state_wasserstein_max'] = np.max(state_wasserstein)
        except:
            metrics['state_wasserstein_mean'] = float('inf')
            metrics['state_wasserstein_max'] = float('inf')

        # 2. Kolmogorov-Smirnov Test
        try:
            state_ks_stats = []
            for dim in range(all_sim_states.shape[1]):
                ks_stat, p_value = ks_2samp(
                    all_sim_states[:, dim],
                    all_real_states[:, dim]
                )
                state_ks_stats.append(ks_stat)
            metrics['state_ks_mean'] = np.mean(state_ks_stats)
            metrics['state_ks_max'] = np.max(state_ks_stats)
        except:
            metrics['state_ks_mean'] = 1.0  # KS statistic upper bound
            metrics['state_ks_max'] = 1.0

        # 3. Maximum Mean Discrepancy (simplified)
        try:
            mmd_approx = self._compute_mmd_approx(all_sim_states, all_real_states)
            metrics['state_mmd_approx'] = mmd_approx
        except:
            metrics['state_mmd_approx'] = float('inf')

        # 4. Action distribution similarity
        try:
            action_wasserstein = []
            for dim in range(all_sim_actions.shape[1]):
                w_dist = wasserstein_distance(
                    all_sim_actions[:, dim],
                    all_real_actions[:, dim]
                )
                action_wasserstein.append(w_dist)
            metrics['action_wasserstein_mean'] = np.mean(action_wasserstein)
        except:
            metrics['action_wasserstein_mean'] = float('inf')

        # 5. Behavioral similarity (based on trajectories)
        try:
            behavioral_similarity = self._compute_behavioral_similarity()
            metrics['behavioral_similarity'] = behavioral_similarity
        except:
            metrics['behavioral_similarity'] = 0.0

        # Store metrics
        self.gap_metrics = metrics

        return metrics

    def _compute_mmd_approx(self, X: np.ndarray, Y: np.ndarray,
                           sigma: float = 1.0) -> float:
        """
        Compute approximate Maximum Mean Discrepancy between two distributions.
        """
        # Simplified MMD computation
        # MMD^2 = ||μ_X - μ_Y||^2 where μ are mean embeddings

        # Compute squared distances between samples
        XX = pairwise_distances(X, metric='sqeuclidean')
        YY = pairwise_distances(Y, metric='sqeuclidean')
        XY = pairwise_distances(X, Y, metric='sqeuclidean')

        # Gaussian kernel matrix elements
        K_XX = np.exp(-XX / (2 * sigma**2))
        K_YY = np.exp(-YY / (2 * sigma**2))
        K_XY = np.exp(-XY / (2 * sigma**2))

        # Compute MMD estimate
        m, n = X.shape[0], Y.shape[0]
        mmd_squared = (K_XX.sum() / (m * m) + K_YY.sum() / (n * n) -
                      2 * K_XY.sum() / (m * n))

        return np.sqrt(max(0, mmd_squared))  # Ensure non-negative

    def _compute_behavioral_similarity(self) -> float:
        """
        Compute behavioral similarity based on trajectory comparisons.
        """
        # Compare trajectory shapes and patterns
        total_similarity = 0.0
        comparison_count = 0

        for sim_traj in self.sim_data:
            for real_traj in self.real_data:
                # Compute trajectory similarity using dynamic time warping
                # or other sequence similarity measures
                similarity = self._trajectory_similarity(
                    sim_traj['states'], real_traj['states']
                )
                total_similarity += similarity
                comparison_count += 1

        return total_similarity / comparison_count if comparison_count > 0 else 0.0

    def _trajectory_similarity(self, traj1: np.ndarray,
                             traj2: np.ndarray) -> float:
        """
        Compute similarity between two trajectories.
        """
        # Use Dynamic Time Warping or other sequence alignment method
        # For simplicity, use a basic distance measure
        min_len = min(len(traj1), len(traj2))

        if min_len == 0:
            return 0.0

        # Compute average distance between corresponding points
        distances = []
        for i in range(min_len):
            dist = np.linalg.norm(traj1[i] - traj2[i])
            distances.append(dist)

        avg_distance = np.mean(distances)

        # Convert to similarity (inverse relationship)
        # Use a sigmoid-like function to bound between 0 and 1
        similarity = 1.0 / (1.0 + avg_distance)

        return similarity

    def model_gap_dynamics(self) -> Dict[str, Any]:
        """
        Model the gap dynamics using various approaches.

        Returns:
            Dictionary of gap models
        """
        gap_models = {}

        # 1. Simple offset model: real = sim + offset
        offset_model = self._fit_offset_model()
        gap_models['offset'] = offset_model

        # 2. Linear scaling model: real = scale * sim + offset
        linear_model = self._fit_linear_model()
        gap_models['linear'] = linear_model

        # 3. Nonlinear neural network model
        nn_model = self._fit_neural_network_model()
        gap_models['neural_network'] = nn_model

        # Store models
        self.gap_models = gap_models

        return gap_models

    def _fit_offset_model(self) -> Dict[str, np.ndarray]:
        """Fit a simple offset model."""
        # Compute average difference between sim and real
        all_sim_states = np.vstack([ep['states'] for ep in self.sim_data])
        all_real_states = np.vstack([ep['states'] for ep in self.real_data])

        offset = np.mean(all_real_states - all_sim_states, axis=0)

        return {
            'offset': offset,
            'type': 'offset',
            'mse': np.mean((all_real_states - (all_sim_states + offset))**2)
        }

    def _fit_linear_model(self) -> Dict[str, np.ndarray]:
        """Fit a linear scaling model: real = A * sim + b."""
        from sklearn.linear_model import LinearRegression

        all_sim_states = np.vstack([ep['states'] for ep in self.sim_data])
        all_real_states = np.vstack([ep['states'] for ep in self.real_data])

        # Fit linear model: real = A * sim + b
        model = LinearRegression().fit(all_sim_states, all_real_states)

        predictions = model.predict(all_sim_states)
        mse = np.mean((all_real_states - predictions)**2)

        return {
            'A': model.coef_,  # Transformation matrix
            'b': model.intercept_,  # Offset vector
            'type': 'linear',
            'mse': mse,
            'model': model
        }

    def _fit_neural_network_model(self) -> Dict[str, Any]:
        """Fit a neural network model for complex gap dynamics."""
        # Create and train neural network model
        nn_identifier = NeuralNetworkSystemIdentifier(
            input_dim=self.sim_data[0]['states'].shape[1],  # State dimension
            output_dim=self.sim_data[0]['states'].shape[1]   # Same output dimension
        )

        # Prepare training data: sim_state -> (real_state - sim_state) offset
        training_data = {
            'states': [],
            'actions': []  # Placeholder for action-based gap modeling
        }

        for sim_ep, real_ep in zip(self.sim_data, self.real_data):
            state_offsets = real_ep['states'] - sim_ep['states']
            training_data['states'].extend(state_offsets)
            training_data['actions'].extend(sim_ep['actions'])

        training_data['states'] = np.array(training_data['states'])
        training_data['actions'] = np.array(training_data['actions'])

        # Train the model
        nn_identifier.train_identifier(training_data)

        return {
            'model': nn_identifier,
            'type': 'neural_network',
            'mse': self._evaluate_nn_model(nn_identifier)
        }

    def _evaluate_nn_model(self, model: NeuralNetworkSystemIdentifier) -> float:
        """Evaluate the neural network model."""
        all_sim_states = np.vstack([ep['states'] for ep in self.sim_data])
        all_real_states = np.vstack([ep['states'] for ep in self.real_data])

        # Predict offsets
        state_tensor = torch.FloatTensor(all_sim_states)
        predicted_offsets = model(state_tensor)

        # Calculate MSE
        real_offsets = torch.FloatTensor(all_real_states - all_sim_states)
        mse = nn.MSELoss()(predicted_offsets, real_offsets).item()

        return mse

    def visualize_gap_analysis(self):
        """Create visualizations for gap analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. State distribution comparison
        if len(self.sim_data) > 0 and len(self.real_data) > 0:
            all_sim_states = np.vstack([ep['states'] for ep in self.sim_data])
            all_real_states = np.vstack([ep['states'] for ep in self.real_data])

            # Plot first two state dimensions
            if all_sim_states.shape[1] >= 2:
                axes[0, 0].scatter(all_sim_states[:, 0], all_sim_states[:, 1],
                                 alpha=0.5, label='Simulation', s=5)
                axes[0, 0].scatter(all_real_states[:, 0], all_real_states[:, 1],
                                 alpha=0.5, label='Real', s=5)
                axes[0, 0].set_title('State Distribution Comparison')
                axes[0, 0].legend()

        # 2. Action distribution comparison
        if len(self.sim_data) > 0 and len(self.real_data) > 0:
            all_sim_actions = np.vstack([ep['actions'] for ep in self.sim_data])
            all_real_actions = np.vstack([ep['actions'] for ep in self.real_data])

            if all_sim_actions.shape[1] >= 2:
                axes[0, 1].scatter(all_sim_actions[:, 0], all_sim_actions[:, 1],
                                 alpha=0.5, label='Simulation', s=5)
                axes[0, 1].scatter(all_real_actions[:, 0], all_real_actions[:, 1],
                                 alpha=0.5, label='Real', s=5)
                axes[0, 1].set_title('Action Distribution Comparison')
                axes[0, 1].legend()

        # 3. Gap metrics bar chart
        if self.gap_metrics:
            metrics_names = list(self.gap_metrics.keys())[:6]  # Limit to first 6 metrics
            metrics_values = [self.gap_metrics[name] for name in metrics_names]

            axes[1, 0].bar(metrics_names, metrics_values)
            axes[1, 0].set_title('Reality Gap Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Reward comparison
        if len(self.sim_data) > 0 and len(self.real_data) > 0:
            sim_rewards = [np.sum(ep['rewards']) for ep in self.sim_data]
            real_rewards = [np.sum(ep['rewards']) for ep in self.real_data]

            axes[1, 1].hist(sim_rewards, alpha=0.5, label='Simulation', bins=10)
            axes[1, 1].hist(real_rewards, alpha=0.5, label='Real', bins=10)
            axes[1, 1].set_title('Reward Distribution Comparison')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def generate_gap_report(self) -> str:
        """Generate a textual report of the gap analysis."""
        report = []
        report.append("Reality Gap Analysis Report")
        report.append("=" * 30)
        report.append(f"Simulation episodes: {len(self.sim_data)}")
        report.append(f"Real episodes: {len(self.real_data)}")
        report.append("")

        if self.gap_metrics:
            report.append("Gap Metrics:")
            for metric_name, value in self.gap_metrics.items():
                report.append(f"  {metric_name}: {value:.4f}")
            report.append("")

        # Interpretation
        gap_severity = self._interpret_gap_severity()
        report.append("Gap Severity Assessment:")
        report.append(f"  {gap_severity}")
        report.append("")

        # Recommendations
        recommendations = self._generate_recommendations()
        report.append("Recommendations:")
        for rec in recommendations:
            report.append(f"  • {rec}")

        return "\n".join(report)

    def _interpret_gap_severity(self) -> str:
        """Interpret the severity of the reality gap."""
        if not self.gap_metrics:
            return "No gap metrics computed."

        # Use Wasserstein distance as primary indicator
        wasserstein_mean = self.gap_metrics.get('state_wasserstein_mean', float('inf'))

        if wasserstein_mean < 0.1:
            return "Low gap severity - direct transfer likely to work well."
        elif wasserstein_mean < 0.5:
            return "Medium gap severity - domain randomization recommended."
        elif wasserstein_mean < 1.0:
            return "High gap severity - significant adaptation needed."
        else:
            return "Critical gap severity - major differences between sim and real."

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on gap analysis."""
        recommendations = []

        if not self.gap_metrics:
            return ["Run gap analysis first to generate recommendations."]

        # Check specific metrics
        wasserstein_mean = self.gap_metrics.get('state_wasserstein_mean', float('inf'))
        ks_mean = self.gap_metrics.get('state_ks_mean', 1.0)
        behavioral_sim = self.gap_metrics.get('behavioral_similarity', 0.0)

        if wasserstein_mean > 0.5:
            recommendations.append("Increase domain randomization range for physical parameters.")

        if ks_mean > 0.5:
            recommendations.append("Focus on sensor noise and actuator imperfection modeling.")

        if behavioral_sim < 0.3:
            recommendations.append("Consider fine-tuning policy with real-world data.")

        if not recommendations:
            recommendations.append("Reality gap appears manageable. Proceed with cautious deployment.")

        return recommendations
```

## Adaptive Control for Real-World Deployment

### Model Reference Adaptive Control (MRAC)

```python
# adaptive_control.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from scipy.linalg import solve_continuous_are

class ModelReferenceAdaptiveControl:
    """
    Model Reference Adaptive Control (MRAC) for real-world deployment.
    """
    def __init__(self,
                 reference_model: nn.Module,
                 plant_model: nn.Module,
                 adaptation_gain: float = 1.0,
                 control_gain: float = 1.0):
        self.reference_model = reference_model
        self.plant_model = plant_model
        self.adaptation_gain = adaptation_gain
        self.control_gain = control_gain

        # Adaptive parameters (theta in MRAC terminology)
        self.theta = np.zeros(plant_model.num_params)  # Initialize to zeros
        self.P = np.eye(plant_model.num_params) * 1000  # Large initial uncertainty

        # For parameter projection to keep them bounded
        self.theta_min = -10 * np.ones_like(self.theta)
        self.theta_max = 10 * np.ones_like(self.theta)

    def compute_control(self,
                       state: np.ndarray,
                       reference_state: np.ndarray,
                       disturbance_estimate: np.ndarray = None) -> np.ndarray:
        """
        Compute control using MRAC law.

        u = u_ref + theta^T * phi
        where phi is the regressor vector
        """
        # Get reference control
        u_ref = self._compute_reference_control(state, reference_state)

        # Compute regressor vector (depends on state and control)
        phi = self._compute_regressor(state)

        # Compute adaptive control term
        u_adaptive = self.theta.T @ phi

        # Total control
        u = u_ref + u_adaptive

        # Add disturbance compensation if available
        if disturbance_estimate is not None:
            u = u - disturbance_estimate

        return u

    def update_parameters(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         next_state: np.ndarray,
                         reference_next_state: np.ndarray):
        """
        Update adaptive parameters using Lyapunov-based adaptation law.
        """
        # Compute tracking error
        error = reference_next_state - next_state

        # Compute regressor
        phi = self._compute_regressor(state)

        # MRAC adaptation law: theta_dot = Gamma * phi * error^T * P
        # where Gamma is adaptation gain and P is the solution to the ARE
        gamma = self.adaptation_gain

        # Simplified adaptation: theta_dot = gamma * phi * error
        # In practice, you'd use the full MRAC formulation
        adaptation_term = gamma * phi * error  # Broadcasting handles multiple outputs

        # Update parameters
        self.theta += adaptation_term * 0.01  # Small time step

        # Project parameters to keep them bounded
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)

    def _compute_reference_control(self,
                                  state: np.ndarray,
                                  reference_state: np.ndarray) -> np.ndarray:
        """Compute reference model control."""
        # This would use the reference model to compute desired control
        # For now, return a simple proportional control
        error = reference_state - state
        return self.control_gain * error

    def _compute_regressor(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the regressor vector phi.
        This depends on the specific plant model structure.
        """
        # Example for a simple linear system: phi = [state, action]
        # In practice, this would be derived from the system's parametric model
        return np.concatenate([state, np.ones(1)])  # Add bias term

    def get_parameter_estimate(self) -> np.ndarray:
        """Get current parameter estimates."""
        return self.theta.copy()

class NeuralNetworkAdaptiveController(nn.Module):
    """
    Neural network-based adaptive controller for complex systems.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 2):
        super().__init__()

        layers = []
        prev_dim = state_dim + action_dim  # State + reference action

        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer for parameter updates
        layers.append(nn.Linear(prev_dim, state_dim * action_dim))  # For gain matrix
        self.adaptation_network = nn.Sequential(*layers)

        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Initialize adaptive gain matrix
        self.adaptive_gain = nn.Parameter(torch.eye(action_dim))

    def forward(self,
                state: torch.Tensor,
                reference_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute adaptive control correction.

        Args:
            state: Current system state
            reference_action: Reference action from policy

        Returns:
            Adaptive control correction
        """
        # Combine state and reference action
        combined_input = torch.cat([state, reference_action], dim=-1)

        # Compute adaptive correction
        correction = self.adaptation_network(combined_input)

        # Reshape to appropriate dimensions
        correction = correction.view(-1, self.action_dim)

        # Apply adaptive gain
        adaptive_correction = torch.matmul(correction, self.adaptive_gain)

        return adaptive_correction

    def update_adaptation(self,
                         state: torch.Tensor,
                         action: torch.Tensor,
                         next_state: torch.Tensor,
                         reference_next_state: torch.Tensor,
                         learning_rate: float = 1e-3):
        """
        Update the adaptation network based on prediction error.
        """
        # Compute prediction error
        error = reference_next_state - next_state

        # Use error as target for adaptation network
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        target = error

        # Compute loss
        predicted_correction = self(state, action)
        loss = nn.MSELoss()(predicted_correction, target)

        # Backward pass
        self.zero_grad()
        loss.backward()
        torch.optim.SGD(self.parameters(), lr=learning_rate).step()

class SafeAdaptiveController:
    """
    Safe adaptive controller with constraint enforcement.
    """
    def __init__(self,
                 base_controller: nn.Module,
                 safety_constraints: Dict[str, Any],
                 adaptation_rate: float = 0.1):
        self.base_controller = base_controller
        self.safety_constraints = safety_constraints
        self.adaptation_rate = adaptation_rate

        # Initialize safety filters
        self.safety_filter = self._create_safety_filter()

    def _create_safety_filter(self) -> nn.Module:
        """Create safety filter to enforce constraints."""
        # This would be a learned safety filter or analytical safety barrier
        class SafetyFilter(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.state_dim = state_dim
                self.action_dim = action_dim

                # Learnable safety parameters
                self.safety_weights = nn.Parameter(torch.eye(action_dim))
                self.safety_bias = nn.Parameter(torch.zeros(action_dim))

            def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                """Apply safety filter to action."""
                # Check if action violates safety constraints
                safe_action = self._enforce_constraints(state, action)
                return safe_action

            def _enforce_constraints(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                """Enforce safety constraints on action."""
                # Apply soft constraints using quadratic programming or other methods
                # For now, use simple clamping based on safety regions
                action_min = torch.tensor(self.safety_constraints.get('action_min', -1.0))
                action_max = torch.tensor(self.safety_constraints.get('action_max', 1.0))

                return torch.clamp(action, action_min, action_max)

        return SafetyFilter(
            state_dim=self.safety_constraints['state_dim'],
            action_dim=self.safety_constraints['action_dim']
        )

    def compute_safe_control(self,
                           state: torch.Tensor,
                           reference_action: torch.Tensor) -> torch.Tensor:
        """
        Compute safe adaptive control action.

        Args:
            state: Current system state
            reference_action: Reference action from base controller

        Returns:
            Safe adaptive control action
        """
        # Get base control action
        base_action = self.base_controller(state)

        # Apply adaptation
        adaptive_correction = self._compute_adaptation(state, reference_action)
        adapted_action = base_action + adaptive_correction * self.adaptation_rate

        # Apply safety filtering
        safe_action = self.safety_filter(state, adapted_action)

        return safe_action

    def _compute_adaptation(self, state: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
        """Compute adaptation based on system identification."""
        # This would use system identification to compute appropriate adaptation
        # For now, return a placeholder
        return torch.randn_like(reference_action) * 0.1  # Small random adaptation

    def update_safety_model(self,
                           safe_data: Dict[str, torch.Tensor],
                           learning_rate: float = 1e-4):
        """
        Update safety model based on safe interaction data.
        """
        # Update safety filter parameters based on safe experiences
        # This would involve learning safety barriers or constraints
        pass

class AdaptiveTrajectoryFollowing:
    """
    Adaptive controller for trajectory following with sim-to-real transfer.
    """
    def __init__(self,
                 trajectory_planner: Any,
                 control_model: nn.Module,
                 adaptation_params: Dict[str, float] = None):
        self.trajectory_planner = trajectory_planner
        self.control_model = control_model
        self.adaptation_params = adaptation_params or {
            'tracking_gain': 1.0,
            'adaptation_rate': 0.1,
            'disturbance_rejection': 0.5
        }

        # Adaptive disturbance estimator
        self.disturbance_estimator = self._initialize_disturbance_estimator()

    def follow_trajectory(self,
                         current_state: torch.Tensor,
                         trajectory: List[torch.Tensor],
                         time_step: int) -> torch.Tensor:
        """
        Follow reference trajectory with adaptive control.

        Args:
            current_state: Current robot state
            trajectory: Reference trajectory points
            time_step: Current time step in trajectory

        Returns:
            Control action to follow trajectory
        """
        if time_step >= len(trajectory):
            return torch.zeros_like(current_state[:self.control_model.action_dim])

        # Get reference state for current time step
        reference_state = trajectory[time_step]

        # Compute tracking error
        tracking_error = reference_state - current_state[:len(reference_state)]

        # Compute reference control
        reference_control = self._compute_reference_control(tracking_error)

        # Estimate disturbances
        estimated_disturbance = self.disturbance_estimator.estimate(
            current_state, reference_control
        )

        # Apply adaptive compensation
        adaptive_compensation = self._compute_adaptive_compensation(
            current_state, tracking_error
        )

        # Combine all control components
        final_control = (reference_control +
                        adaptive_compensation * self.adaptation_params['adaptation_rate'] -
                        estimated_disturbance * self.adaptation_params['disturbance_rejection'])

        return final_control

    def _compute_reference_control(self, tracking_error: torch.Tensor) -> torch.Tensor:
        """Compute reference control based on tracking error."""
        # Simple PD control for trajectory following
        proportional_gain = self.adaptation_params['tracking_gain']
        derivative_gain = self.adaptation_params['tracking_gain'] * 0.5

        # In practice, would use more sophisticated trajectory following control
        return proportional_gain * tracking_error

    def _compute_adaptive_compensation(self,
                                     state: torch.Tensor,
                                     error: torch.Tensor) -> torch.Tensor:
        """Compute adaptive compensation term."""
        # This would use learned adaptive parameters
        # For now, use a simple function of state and error
        return torch.tanh(torch.cat([state, error]))[:self.control_model.action_dim]

    def update_disturbance_model(self,
                               state_sequence: List[torch.Tensor],
                               action_sequence: List[torch.Tensor],
                               prediction_errors: List[torch.Tensor]):
        """
        Update disturbance model based on prediction errors.
        """
        # Update disturbance estimator using prediction errors
        # This would involve learning the disturbance dynamics
        pass
```

## Sim-to-Real Transfer Validation

### Performance Evaluation Framework

```python
# transfer_validation.py
import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

class SimToRealEvaluator:
    """
    Comprehensive evaluation framework for sim-to-real transfer.
    """
    def __init__(self):
        self.sim_performance = []
        self.real_performance = []
        self.transfer_metrics = {}
        self.safety_metrics = []

    def evaluate_transfer_performance(self,
                                    sim_env,
                                    real_env,
                                    policy,
                                    num_evaluations: int = 10) -> Dict[str, float]:
        """
        Evaluate transfer performance across sim and real environments.

        Args:
            sim_env: Simulation environment
            real_env: Real environment
            policy: Policy to evaluate
            num_evaluations: Number of evaluation episodes

        Returns:
            Dictionary of transfer performance metrics
        """
        sim_rewards = []
        real_rewards = []
        sim_success_rates = []
        real_success_rates = []

        # Evaluate in simulation
        for _ in range(num_evaluations):
            sim_reward, sim_success = self._evaluate_episode(sim_env, policy)
            sim_rewards.append(sim_reward)
            sim_success_rates.append(sim_success)

        # Evaluate in real environment
        for _ in range(num_evaluations):
            real_reward, real_success = self._evaluate_episode(real_env, policy)
            real_rewards.append(real_reward)
            real_success_rates.append(real_success)

        # Calculate metrics
        metrics = {
            'sim_avg_reward': np.mean(sim_rewards),
            'sim_std_reward': np.std(sim_rewards),
            'sim_avg_success': np.mean(sim_success_rates),
            'real_avg_reward': np.mean(real_rewards),
            'real_std_reward': np.std(real_rewards),
            'real_avg_success': np.mean(real_success_rates),
            'transfer_gap_reward': np.mean(sim_rewards) - np.mean(real_rewards),
            'transfer_gap_success': np.mean(sim_success_rates) - np.mean(real_success_rates),
            'transfer_efficiency': np.mean(real_rewards) / (np.mean(sim_rewards) + 1e-8),
            'num_evaluations': num_evaluations
        }

        self.sim_performance = sim_rewards
        self.real_performance = real_rewards
        self.transfer_metrics = metrics

        return metrics

    def _evaluate_episode(self, env, policy) -> Tuple[float, bool]:
        """Evaluate a single episode."""
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state)
            else:
                # Assume policy is a function
                action = policy(state)

            state, reward, done, info = env.step(action)
            total_reward += reward

        # Determine success based on environment-specific criteria
        success = self._check_success(info)

        return total_reward, success

    def _check_success(self, info: Dict[str, Any]) -> bool:
        """Check if episode was successful."""
        # This would be environment-specific
        # For now, return True if final reward is positive
        return info.get('final_reward', 0) > 0

    def evaluate_safety_metrics(self,
                              real_env,
                              policy,
                              safety_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate safety metrics during real-world deployment.

        Args:
            real_env: Real environment
            policy: Policy to evaluate
            safety_thresholds: Dictionary of safety thresholds

        Returns:
            Dictionary of safety metrics
        """
        safety_violations = {
            'joint_limits_violated': 0,
            'collision_occurred': 0,
            'velocity_exceeded': 0,
            'torque_exceeded': 0
        }

        total_episodes = 0
        total_steps = 0

        for episode in range(10):  # Evaluate 10 episodes
            state = real_env.reset()
            done = False

            while not done:
                if hasattr(policy, 'select_action'):
                    action = policy.select_action(state)
                else:
                    action = policy(state)

                next_state, reward, done, info = real_env.step(action)

                # Check for safety violations
                if info.get('joint_limit_violation', False):
                    safety_violations['joint_limits_violated'] += 1

                if info.get('collision', False):
                    safety_violations['collision_occurred'] += 1

                if info.get('velocity_exceeded', False):
                    safety_violations['velocity_exceeded'] += 1

                if info.get('torque_exceeded', False):
                    safety_violations['torque_exceeded'] += 1

                state = next_state
                total_steps += 1

            total_episodes += 1

        # Calculate safety rates
        safety_rates = {}
        for violation_type, count in safety_violations.items():
            safety_rates[f'{violation_type}_rate'] = count / total_steps if total_steps > 0 else 0.0

        # Overall safety score (lower is better)
        total_violations = sum(safety_violations.values())
        safety_score = total_violations / total_steps if total_steps > 0 else 0.0

        safety_rates['overall_safety_score'] = safety_score
        safety_rates['total_violations'] = total_violations
        safety_rates['total_steps'] = total_steps

        self.safety_metrics.append(safety_rates)

        return safety_rates

    def evaluate_robustness(self,
                          real_env,
                          policy,
                          perturbation_levels: List[float] = [0.0, 0.1, 0.2, 0.3]) -> Dict[str, List[float]]:
        """
        Evaluate policy robustness to environmental perturbations.

        Args:
            real_env: Real environment
            policy: Policy to evaluate
            perturbation_levels: Levels of environmental perturbations to test

        Returns:
            Dictionary of performance under different perturbation levels
        """
        robustness_results = {
            'perturbation_levels': [],
            'performance': [],
            'success_rates': []
        }

        for level in perturbation_levels:
            # Apply perturbation to environment
            real_env.apply_perturbation(level)

            episode_rewards = []
            success_counts = 0

            for _ in range(5):  # 5 episodes per level
                state = real_env.reset()
                total_reward = 0
                done = False

                while not done:
                    if hasattr(policy, 'select_action'):
                        action = policy.select_action(state)
                    else:
                        action = policy(state)

                    state, reward, done, info = real_env.step(action)
                    total_reward += reward

                episode_rewards.append(total_reward)

                if self._check_success(info):
                    success_counts += 1

            avg_reward = np.mean(episode_rewards)
            success_rate = success_counts / 5

            robustness_results['perturbation_levels'].append(level)
            robustness_results['performance'].append(avg_reward)
            robustness_results['success_rates'].append(success_rate)

        return robustness_results

    def generate_transfer_report(self) -> str:
        """Generate comprehensive transfer evaluation report."""
        report = []
        report.append("Sim-to-Real Transfer Evaluation Report")
        report.append("=" * 45)
        report.append("")

        if self.transfer_metrics:
            report.append("Transfer Performance Metrics:")
            for key, value in self.transfer_metrics.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        if self.safety_metrics:
            report.append("Safety Metrics:")
            latest_safety = self.safety_metrics[-1]
            for key, value in latest_safety.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Analysis
        if self.transfer_metrics:
            transfer_gap = self.transfer_metrics.get('transfer_gap_reward', 0)
            if transfer_gap > 0.5:
                analysis = "SIGNIFICANT REALITY GAP: Performance degradation substantial, requires significant adaptation."
            elif transfer_gap > 0.1:
                analysis = "MODERATE REALITY GAP: Some performance degradation, adaptation recommended."
            else:
                analysis = "MINIMAL REALITY GAP: Transfer performance is good, minimal adaptation needed."

            report.append(f"Analysis: {analysis}")
            report.append("")

        # Recommendations
        report.append("Recommendations:")
        if transfer_gap > 0.3:
            report.append("  • Implement additional domain randomization in simulation")
            report.append("  • Conduct system identification to bridge reality gap")
            report.append("  • Use adaptive control techniques for real-world deployment")
        else:
            report.append("  • Transfer performance is acceptable for deployment")
            report.append("  • Monitor performance during initial real-world operation")

        return "\n".join(report)

    def visualize_transfer_results(self):
        """Create visualizations for transfer evaluation results."""
        if not self.sim_performance or not self.real_performance:
            print("No performance data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Performance comparison
        axes[0, 0].boxplot([self.sim_performance, self.real_performance],
                          labels=['Simulation', 'Real'])
        axes[0, 0].set_title('Performance Comparison: Sim vs Real')
        axes[0, 0].set_ylabel('Reward')

        # 2. Learning curves (if available)
        if len(self.sim_performance) > 10:
            window_size = max(1, len(self.sim_performance) // 10)
            sim_smoothed = [np.mean(self.sim_performance[i:i+window_size])
                           for i in range(0, len(self.sim_performance) - window_size + 1, window_size)]
            real_smoothed = [np.mean(self.real_performance[i:i+window_size])
                            for i in range(0, len(self.real_performance) - window_size + 1, window_size)]

            x_sim = range(len(sim_smoothed))
            x_real = range(len(real_smoothed))

            axes[0, 1].plot(x_sim, sim_smoothed, label='Simulation', alpha=0.7)
            axes[0, 1].plot(x_real, real_smoothed, label='Real', alpha=0.7)
            axes[0, 1].set_title('Performance Over Time')
            axes[0, 1].set_xlabel('Evaluation Episodes')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].legend()

        # 3. Transfer gap analysis
        if self.transfer_metrics:
            gap_metrics = ['transfer_gap_reward', 'transfer_gap_success']
            gap_values = [self.transfer_metrics.get(metric, 0) for metric in gap_metrics]
            axes[1, 0].bar(gap_metrics, gap_values)
            axes[1, 0].set_title('Transfer Gap Analysis')
            axes[1, 0].set_ylabel('Gap (Sim - Real)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Success rate comparison
        success_sim = [1 if r > 0 else 0 for r in self.sim_performance]
        success_real = [1 if r > 0 else 0 for r in self.real_performance]

        success_rates = [np.mean(success_sim), np.mean(success_real)]
        axes[1, 1].bar(['Simulation', 'Real'], success_rates)
        axes[1, 1].set_title('Success Rate Comparison')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

class ContinualTransferLearning:
    """
    Framework for continual learning during sim-to-real transfer.
    """
    def __init__(self, base_policy: nn.Module, buffer_size: int = 10000):
        self.base_policy = base_policy
        self.buffer_size = buffer_size
        self.experience_buffer = []
        self.transfer_history = []
        self.performance_threshold = 0.8  # Minimum performance to continue transfer

    def add_real_experience(self, state: np.ndarray, action: np.ndarray,
                          reward: float, next_state: np.ndarray, done: bool):
        """Add real-world experience to buffer for continual learning."""
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }

        self.experience_buffer.append(experience)

        # Keep buffer size within limits
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def adapt_policy(self, learning_rate: float = 1e-4, batch_size: int = 32):
        """Adapt policy using real-world experiences."""
        if len(self.experience_buffer) < batch_size:
            return  # Not enough data yet

        # Sample batch from experience buffer
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        # Extract batch components
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])

        # Update policy using real experiences
        self._update_policy_batch(states, actions, rewards, next_states, dones, learning_rate)

    def _update_policy_batch(self,
                           states: torch.Tensor,
                           actions: torch.Tensor,
                           rewards: torch.Tensor,
                           next_states: torch.Tensor,
                           dones: torch.Tensor,
                           learning_rate: float):
        """Update policy with batch of real experiences."""
        # This would implement policy gradient or other RL updates
        # For now, it's a placeholder
        pass

    def evaluate_adaptation(self, real_env, num_episodes: int = 5) -> float:
        """Evaluate current adaptation level."""
        episode_rewards = []

        for _ in range(num_episodes):
            state = real_env.reset()
            total_reward = 0
            done = False

            while not done:
                if hasattr(self.base_policy, 'select_action'):
                    action = self.base_policy.select_action(state)
                else:
                    action = self.base_policy(state)

                state, reward, done, info = real_env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        self.transfer_history.append(avg_reward)

        return avg_reward

    def should_continue_transfer(self, current_performance: float) -> bool:
        """Determine if transfer learning should continue."""
        if current_performance < self.performance_threshold:
            # If performance is too low, stop transfer and reconsider approach
            return False

        # Check for improvement trends
        if len(self.transfer_history) > 5:
            recent_performance = self.transfer_history[-5:]
            if np.mean(recent_performance) < np.mean(self.transfer_history[:-5]):
                # Performance is degrading, stop adaptation
                return False

        return True
```

## Code Examples with Explanations

### Complete Sim-to-Real Transfer System

```python
# complete_sim_to_real_system.py
import torch
import numpy as np
from typing import Dict, Any, List
import time

class CompleteSimToRealSystem:
    """
    Complete system integrating all sim-to-real transfer techniques.
    """
    def __init__(self, config: IsaacFoundationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.domain_randomizer = DomainRandomizer(config.domain_randomization_params)
        self.system_identifier = RobotSystemIdentifier()
        self.reality_analyzer = RealityGapAnalyzer()
        self.adaptive_controller = SafeAdaptiveController(
            base_controller=None,  # Will be set later
            safety_constraints={
                'state_dim': config.state_dim,
                'action_dim': config.action_dim,
                'action_min': -1.0,
                'action_max': 1.0
            }
        )
        self.transfer_evaluator = SimToRealEvaluator()
        self.continual_learner = ContinualTransferLearning(None)

        # Performance tracking
        self.sim_rewards_history = []
        self.real_rewards_history = []
        self.transfer_efficiency_history = []

        print("Complete Sim-to-Real Transfer System initialized")
        print(f"Domain randomization: {config.domain_randomization}")
        print(f"Continual learning: {config.continual_learning}")
        print(f"Target transfer efficiency: {config.target_transfer_efficiency}")

    def train_with_domain_randomization(self,
                                      sim_env,
                                      policy,
                                      total_steps: int = 50000,
                                      eval_freq: int = 5000):
        """
        Train policy with domain randomization in simulation.

        Args:
            sim_env: Simulation environment
            policy: Policy to train
            total_steps: Total training steps
            eval_freq: Evaluation frequency
        """
        print("Starting training with domain randomization...")

        state = sim_env.reset()
        episode_reward = 0
        step_count = 0

        for step in range(total_steps):
            # Apply domain randomization every N steps
            if step % self.config.randomization_frequency == 0:
                self.domain_randomizer.apply_randomization(sim_env)

            # Select action from policy
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state)
            else:
                action = policy(state)

            # Take step in environment
            next_state, reward, done, info = sim_env.step(action)

            # Store experience
            policy.store_experience(state, action, reward, next_state, done)

            # Train policy
            if step > 1000:  # Start training after some random exploration
                policy.train()

            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1

            if done:
                # Log episode
                self.sim_rewards_history.append(episode_reward)

                if len(self.sim_rewards_history) % 10 == 0:
                    avg_reward = np.mean(self.sim_rewards_history[-10:])
                    print(f"Step {step}, Average Sim Reward: {avg_reward:.2f}")

                # Reset environment
                state = sim_env.reset()
                episode_reward = 0

            # Periodic evaluation
            if step > 0 and step % eval_freq == 0:
                avg_sim_reward = np.mean(self.sim_rewards_history[-eval_freq//1000:])
                print(f"Step {step}: Sim Average Reward: {avg_sim_reward:.2f}")

        print(f"Training completed after {total_steps} steps")

    def perform_system_identification(self, real_env, sim_env):
        """
        Perform system identification to bridge sim-to-real gap.
        """
        print("Performing system identification...")

        # Collect data from real robot
        excitation_signals = self._generate_excitation_signals(duration=10.0, freq=1.0)
        real_data = self.system_identifier.collect_identification_data(
            real_env, excitation_signals, sample_rate=100.0
        )

        # Estimate real-world parameters
        real_params = self.system_identifier.estimate_dynamic_parameters(real_data)
        friction_params = self.system_identifier.estimate_friction_parameters(real_data)

        print("Real-world parameters identified:")
        for param_name, value in real_params.items():
            print(f"  {param_name}: {value:.4f}")

        # Update simulation with identified parameters
        self.system_identifier.update_simulation_model(sim_env, {**real_params, **friction_params})

        # Validate identification
        validation_metrics = self.system_identifier.validate_identification(
            sim_env, real_env, excitation_signals[0]  # Use first trajectory for validation
        )

        print("Validation metrics:")
        for metric_name, value in validation_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        return {
            'identified_params': {**real_params, **friction_params},
            'validation_metrics': validation_metrics
        }

    def _generate_excitation_signals(self, duration: float, freq: float, amplitude: float = 1.0) -> List[np.ndarray]:
        """
        Generate excitation signals for system identification.
        """
        num_signals = 5  # Different excitation patterns
        signals = []

        dt = 0.01  # 100 Hz
        steps = int(duration / dt)

        for i in range(num_signals):
            # Different types of excitation signals
            if i == 0:
                # Multi-sine signal
                t = np.linspace(0, duration, steps)
                frequencies = [freq, freq*2, freq*3]
                signal = np.sum([amplitude/len(frequencies) * np.sin(2*np.pi*f*t) for f in frequencies], axis=0)
            elif i == 1:
                # Pseudo-random binary sequence
                signal = np.random.choice([-amplitude, amplitude], size=steps)
            elif i == 2:
                # Chirp signal (frequency sweep)
                t = np.linspace(0, duration, steps)
                signal = amplitude * np.sin(2 * np.pi * freq * t * t / (2 * duration))
            elif i == 3:
                # Step signal
                signal = np.zeros(steps)
                signal[steps//4:3*steps//4] = amplitude
            else:  # i == 4
                # Random walk
                signal = np.cumsum(np.random.normal(0, amplitude*0.1, steps))
                signal = np.clip(signal, -amplitude, amplitude)

            signals.append(signal)

        return signals

    def execute_sim_to_real_transfer(self,
                                   sim_env,
                                   real_env,
                                   trained_policy,
                                   adaptation_episodes: int = 20):
        """
        Execute complete sim-to-real transfer process.

        Args:
            sim_env: Simulation environment
            real_env: Real environment
            trained_policy: Policy trained in simulation
            adaptation_episodes: Number of adaptation episodes

        Returns:
            Transfer evaluation results
        """
        print("Starting sim-to-real transfer process...")

        # Step 1: Evaluate initial transfer performance
        print("\n1. Evaluating initial transfer performance...")
        initial_metrics = self.transfer_evaluator.evaluate_transfer_performance(
            sim_env, real_env, trained_policy, num_evaluations=10
        )

        print(f"Initial transfer gap (reward): {initial_metrics['transfer_gap_reward']:.3f}")
        print(f"Initial transfer efficiency: {initial_metrics['transfer_efficiency']:.3f}")

        # Step 2: Perform system identification if needed
        if initial_metrics['transfer_gap_reward'] > 0.5:  # Significant gap
            print("\n2. Performing system identification to reduce gap...")
            ident_results = self.perform_system_identification(real_env, sim_env)

        # Step 3: Apply adaptive control
        print("\n3. Applying adaptive control for real-world deployment...")
        self.adaptive_controller.base_controller = trained_policy

        # Step 4: Fine-tune with real-world data (continual learning)
        print(f"\n4. Fine-tuning with {adaptation_episodes} real-world episodes...")

        for episode in range(adaptation_episodes):
            state = real_env.reset()
            total_reward = 0
            done = False

            while not done:
                # Use adaptive controller
                action = self.adaptive_controller.compute_safe_control(state, state)  # Simplified

                state, reward, done, info = real_env.step(action)
                total_reward += reward

                # Add to continual learning buffer
                self.continual_learner.add_real_experience(
                    state, action, reward, state, done  # Using state as next_state temporarily
                )

            # Adapt policy with real experiences
            if episode % 5 == 0:  # Adapt every 5 episodes
                self.continual_learner.adapt_policy()

            # Evaluate current adaptation level
            current_performance = self.continual_learner.evaluate_adaptation(real_env)

            print(f"  Episode {episode + 1}/{adaptation_episodes}, Reward: {total_reward:.2f}, Adapted Perf: {current_performance:.2f}")

            # Check if we should continue
            if not self.continual_learner.should_continue_transfer(current_performance):
                print("  Stopping adaptation - performance plateau reached")
                break

        # Step 5: Final evaluation
        print("\n5. Evaluating final transfer performance...")
        final_metrics = self.transfer_evaluator.evaluate_transfer_performance(
            sim_env, real_env, trained_policy, num_evaluations=10
        )

        # Calculate improvement
        improvement = final_metrics['transfer_efficiency'] - initial_metrics['transfer_efficiency']
        print(f"Final transfer efficiency: {final_metrics['transfer_efficiency']:.3f}")
        print(f"Improvement achieved: {improvement:.3f}")

        # Step 6: Safety evaluation
        print("\n6. Evaluating safety metrics...")
        safety_metrics = self.transfer_evaluator.evaluate_safety_metrics(
            real_env, trained_policy, {
                'joint_limit_threshold': 0.95,
                'collision_distance': 0.1,
                'velocity_limit': 1.0,
                'torque_limit': 100.0
            }
        )

        print("Safety metrics:")
        for metric, value in safety_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Compile final results
        transfer_results = {
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': improvement,
            'safety_metrics': safety_metrics,
            'identification_results': ident_results if 'ident_results' in locals() else None
        }

        return transfer_results

    def generate_transfer_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive transfer report."""
        report = []
        report.append("SIM-TO-REAL TRANSFER COMPLETION REPORT")
        report.append("=" * 50)
        report.append("")

        # Performance metrics
        report.append("Performance Metrics:")
        initial_eff = results['initial_metrics']['transfer_efficiency']
        final_eff = results['final_metrics']['transfer_efficiency']
        report.append(f"  Initial Transfer Efficiency: {initial_eff:.3f}")
        report.append(f"  Final Transfer Efficiency: {final_eff:.3f}")
        report.append(f"  Improvement: {results['improvement']:.3f}")
        report.append("")

        # Safety metrics
        report.append("Safety Metrics:")
        for metric, value in results['safety_metrics'].items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.4f}")
        report.append("")

        # Recommendations
        report.append("Recommendations:")
        if final_eff > 0.9:
            report.append("  ✓ Transfer successful - ready for deployment")
        elif final_eff > 0.7:
            report.append("  ○ Transfer adequate - monitor during deployment")
        else:
            report.append("  ✗ Transfer needs improvement - consider redesign")

        if results['safety_metrics']['overall_safety_score'] < 0.05:
            report.append("  ✓ Safety metrics acceptable")
        else:
            report.append("  ⚠ Safety metrics concerning - review before deployment")

        return "\n".join(report)

def main():
    """Demonstrate complete sim-to-real transfer system."""
    print("Initializing Complete Sim-to-Real Transfer System...")

    # Create configuration
    config = IsaacFoundationConfig(
        hidden_dim=512,
        action_dim=7,
        sensor_dim=360,
        domain_randomization=True,
        continual_learning=True,
        target_transfer_efficiency=0.8
    )

    # Create transfer system
    transfer_system = CompleteSimToRealSystem(config)

    print("\nSim-to-Real Transfer System Components:")
    print(f"- Domain randomization: {config.domain_randomization}")
    print(f"- Continual learning: {config.continual_learning}")
    print(f-" Target efficiency: {config.target_transfer_efficiency}")

    print("\nThe system is now ready for sim-to-real transfer experiments.")
    print("Next steps would include:")
    print("1. Training policy in simulation with domain randomization")
    print("2. Performing system identification with real robot")
    print("3. Executing transfer with adaptive control")
    print("4. Evaluating and validating transfer performance")

if __name__ == "__main__":
    main()