---
sidebar_position: 3
title: Hardware Requirements
---

import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Hardware Requirements

<ChatbotWidget />
<ProgressTracker chapterId="hardware-requirements" />
<UrduTranslationToggle />
<PersonalizationButton />

This textbook covers both theoretical concepts and practical implementations. This page outlines the hardware requirements for different aspects of the course.

## Minimum System Requirements

### Development Environment
- **Operating System**: Ubuntu 20.04 LTS or later, Windows 10/11 (with WSL2), or macOS 10.15+
- **Processor**: Intel i5 or equivalent AMD processor (4 cores, 2.5 GHz or faster)
- **Memory**: 8 GB RAM (16 GB recommended)
- **Storage**: 50 GB available space
- **Graphics**: Integrated graphics or dedicated GPU (NVIDIA preferred for CUDA operations)

### ROS 2 Development (Module 1)
- **OS**: Ubuntu 22.04 LTS recommended
- **RAM**: 16 GB minimum (32 GB recommended for simulation)
- **Processor**: Multi-core processor (Intel i7 or equivalent AMD)
- **Network**: Stable internet connection for package installation

## Recommended Hardware for Practical Applications

### Physical Robot Platforms (Optional)
While not required for the course, these platforms can be used for hands-on experience:

- **TurtleBot3** - Ideal for beginners, supports ROS 2
- **NVIDIA JetBot** - AI-powered robot for perception and navigation
- **Unitree Go1/Unitree A1** - Quadruped robots for advanced locomotion
- **Any ROS-compatible platform** with basic sensors

### Simulation Hardware Requirements
For running Gazebo and Unity simulations:

- **CPU**: Multi-core processor (Intel i7-10700K or AMD Ryzen 7 5800X)
- **GPU**: NVIDIA GTX 1060 6GB or better (RTX series recommended)
- **RAM**: 32 GB for complex simulations
- **Storage**: SSD with 100+ GB free space for simulation assets

## NVIDIA Isaac Platform Requirements

For Module 3 (AI-Robot Brain):

- **GPU**: NVIDIA RTX 3080 or better (RTX 4090 recommended)
- **VRAM**: 10 GB minimum (24 GB recommended)
- **CUDA**: CUDA 11.8 or later
- **Tensor Cores**: Required for optimal performance

## Specialized Equipment (Optional)

### Sensors for Practical Projects
- RGB-D Camera (Intel RealSense D435, Orbbec Astra Pro)
- IMU (Inertial Measurement Unit) for balance control
- LiDAR (RPLIDAR A2/A3, Hokuyo URG-04LX-UG01)
- Force/Torque sensors for manipulation

### Development Tools
- USB-to-Serial adapters for hardware communication
- Oscilloscope for electronics debugging
- Multimeter for basic electrical measurements

## Software Prerequisites

### Required Software
- **Docker**: For containerized development environments
- **Git**: For version control
- **Visual Studio Code**: Recommended IDE with ROS extensions
- **Python 3.8+**: For ROS 2 and AI development

### Module-Specific Software
- **ROS 2 Humble Hawksbill**: For Module 1
- **Gazebo Garden**: For Module 2 simulation
- **Unity 2022.3 LTS**: For Unity-based simulation
- **NVIDIA Isaac Sim**: For Module 3
- **NVIDIA Isaac ROS**: For perception packages

## Cloud Computing Alternative

For students without access to high-end hardware:

- **NVIDIA GPU Cloud (NGC)**: Containerized Isaac SDK
- **AWS RoboMaker**: Cloud-based ROS development
- **Google Colab Pro**: For AI model training (limited)
- **Azure Percept**: For edge AI development

## Budget Considerations

### Budget-Friendly Setup
- Personal laptop meeting minimum requirements
- Software-only development with simulation
- Access to university lab equipment (if available)

### Recommended Setup
- Dedicated development workstation
- Access to physical robots (shared lab environment)
- High-end GPU for AI training

### Advanced Setup
- Multiple robot platforms
- Complete sensor suite
- High-performance computing cluster access

## Getting Started

Most of the course content can be completed using simulation environments, which have lower hardware requirements than physical robot control. Start with the software installation and basic tutorials before investing in hardware.

For students in academic settings, check with your institution for access to robotics labs and specialized equipment.