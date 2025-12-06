---
sidebar_position: 2
title: Chapter 13 - NVIDIA Isaac Foundation Models
---

import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Chapter 13 - NVIDIA Isaac Foundation Models

<ChatbotWidget />
<ProgressTracker chapterId="module4-chapter13" />
<UrduTranslationToggle />
<PersonalizationButton />

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the NVIDIA Isaac Foundation Model ecosystem and its components
- Implement Isaac Foundation Models for robotic perception and control
- Integrate Isaac Foundation Models with robotic systems and sensors
- Fine-tune Isaac Foundation Models for specific robotic tasks
- Deploy Isaac Foundation Models to NVIDIA hardware platforms
- Optimize Isaac Foundation Models for real-time robotic applications
- Evaluate performance of Isaac Foundation Models in robotics tasks
- Create custom applications using Isaac Foundation Model APIs

## Prerequisites

Before starting this chapter, you should have:
- Understanding of deep learning fundamentals and transformer architectures
- Experience with robotics simulation and control systems
- Knowledge of computer vision and natural language processing concepts
- Experience with NVIDIA hardware and CUDA programming
- Completion of Module 1-3 content
- Familiarity with Isaac Sim and VLA models from Chapter 12

## Introduction to Isaac Foundation Models

### Overview of Isaac Foundation Models

NVIDIA Isaac Foundation Models represent a new paradigm in robotics AI, providing pre-trained, general-purpose models that can be adapted to various robotic tasks. These models leverage large-scale training on diverse datasets to achieve robust performance across different robotic applications.

**Key Components**:
- **Isaac Foundation Model API**: High-level interface for model interaction
- **Isaac Foundation Model Runtime**: Optimized inference engine for NVIDIA hardware
- **Isaac Foundation Model Training Toolkit**: Tools for fine-tuning and customization
- **Isaac Foundation Model Gallery**: Pre-trained models for common robotic tasks

### Isaac Foundation Model Architecture

The Isaac Foundation Model architecture is built on several key principles:

**Modularity**: Components are designed to be modular and composable, allowing for flexible system design.

**Scalability**: Models can scale from edge devices to data center GPUs.

**Efficiency**: Optimized for real-time performance on NVIDIA hardware.

**Generalization**: Pre-trained on diverse datasets for broad applicability.

### Available Isaac Foundation Models

**Perception Models**:
- **Isaac Perceptor**: Multi-modal perception for objects, people, and scenes
- **Isaac Segmentor**: Semantic and instance segmentation
- **Isaac Detector**: Object detection and tracking
- **Isaac Depth Estimator**: Monocular and stereo depth estimation

**Control Models**:
- **Isaac Controller**: Vision-language-action control policies
- **Isaac Manipulator**: Dexterous manipulation policies
- **Isaac Navigator**: Navigation and path planning
- **Isaac Grasper**: Grasping and manipulation strategies

**Foundation Models**:
- **Isaac GEMINI**: Generative models for robotic environments
- **Isaac ORION**: Multi-task learning foundation model
- **Isaac AEGIS**: Safety and validation models

## Setting Up Isaac Foundation Models

### Prerequisites and Dependencies

Before using Isaac Foundation Models, ensure your system meets the requirements:

**Hardware Requirements**:
- NVIDIA GPU with compute capability 7.0 or higher (Volta or newer)
- At least 8GB VRAM (16GB+ recommended)
- Multi-core CPU (Intel i7 or AMD Ryzen 7+)
- 32GB+ system RAM recommended

**Software Requirements**:
- Ubuntu 20.04 LTS or 22.04 LTS
- NVIDIA Driver 525 or later
- CUDA 12.0 or later
- Docker with NVIDIA Container Toolkit
- Isaac Sim 2023.1 or later

**Installation Prerequisites**:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install Docker
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Installing Isaac Foundation Models

```bash
# Pull Isaac Foundation Model containers
docker pull nvcr.io/nvidia/isaac/foundation_models:latest
docker pull nvcr.io/nvidia/isaac/isaac_sim:latest

# Create working directory
mkdir -p ~/isaac_foundation_models
cd ~/isaac_foundation_models

# Set up environment variables
export ISAAC_FOUNDMODELS_PATH=~/isaac_foundation_models
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Basic Setup and Verification

```python
# isaac_foundation_setup.py
import os
import sys
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import requests
import json

class IsaacFoundationSetup:
    """
    Setup and verification for Isaac Foundation Models.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.isaac_api_url = "https://api.nvidia.com/isaac/foundation_models"
        self.available_models = []
        self.model_info_cache = {}

        # Check NVIDIA GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_count > 0 else "Unknown"
            print(f"GPU Available: {self.gpu_name} ({self.gpu_count} devices)")
        else:
            print("WARNING: No GPU detected. Performance will be limited.")

        # Verify Isaac Foundation Models access
        self._verify_access()

    def _verify_access(self):
        """Verify access to Isaac Foundation Models API."""
        try:
            if self.api_key:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                response = requests.get(
                    f"{self.isaac_api_url}/models",
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    self.available_models = response.json().get('models', [])
                    print(f"Access verified. Available models: {len(self.available_models)}")
                elif response.status_code == 401:
                    print(f"API authentication failed with status {response.status_code} - Invalid API key")
                    print("Please check your API key and ensure it has proper permissions.")
                    print("Using local models only.")
                elif response.status_code == 403:
                    print(f"API access forbidden with status {response.status_code} - Insufficient permissions")
                    print("Please check your API key permissions.")
                    print("Using local models only.")
                else:
                    print(f"API access failed with status {response.status_code}")
                    print(f"Response: {response.text}")
                    print("Using local models only.")
            else:
                print("No API key provided. Using local models only.")
        except requests.exceptions.ConnectionError:
            print("Could not connect to Isaac Foundation Models API. Check your internet connection.")
            print("Using local models only.")
        except requests.exceptions.Timeout:
            print("Request to Isaac Foundation Models API timed out.")
            print("Using local models only.")
        except Exception as e:
            print(f"Verification failed: {e}")
            print("Using local models only.")

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available Isaac Foundation Models."""
        if self.available_models:
            return self.available_models
        else:
            # Return local models
            local_models = [
                {
                    'name': 'perceptor_v1.0',
                    'type': 'perception',
                    'description': 'Multi-modal perception model',
                    'size': 'large',
                    'input_modalities': ['rgb', 'depth', 'language'],
                    'output_types': ['detections', 'segmentation', 'attributes']
                },
                {
                    'name': 'controller_v1.0',
                    'type': 'control',
                    'description': 'Vision-language-action control model',
                    'size': 'large',
                    'input_modalities': ['rgb', 'language', 'state'],
                    'output_types': ['actions', 'trajectories', 'plans']
                },
                {
                    'name': 'manipulator_v1.0',
                    'type': 'manipulation',
                    'description': 'Dexterous manipulation model',
                    'size': 'medium',
                    'input_modalities': ['rgb', 'depth', 'language', 'joint_state'],
                    'output_types': ['joint_commands', 'grasp_poses', 'manipulation_plans']
                }
            ]
            return local_models

    def download_model(self, model_name: str, download_path: str = "./models") -> str:
        """Download a specific Isaac Foundation Model."""
        os.makedirs(download_path, exist_ok=True)
        model_path = os.path.join(download_path, model_name)

        # In a real implementation, this would download from NVIDIA NGC
        # For demonstration, we'll create a placeholder
        print(f"Downloading {model_name} to {model_path}...")

        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists at {model_path}")
            return model_path

        # Create model directory structure
        os.makedirs(model_path, exist_ok=True)

        # Create model metadata
        metadata = {
            'model_name': model_name,
            'downloaded_at': time.time(),
            'version': '1.0.0',
            'input_modalities': ['rgb', 'depth', 'language'],
            'output_types': ['actions', 'predictions'],
            'recommended_hardware': 'RTX 3080 or better'
        }

        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model {model_name} downloaded successfully")
        return model_path

    def verify_model_integrity(self, model_path: str) -> bool:
        """Verify the integrity of a downloaded model."""
        metadata_path = os.path.join(model_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Model metadata not found at {metadata_path}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check for required fields
            required_fields = ['model_name', 'downloaded_at', 'version']
            for field in required_fields:
                if field not in metadata:
                    print(f"Missing required field in metadata: {field}")
                    return False

            print(f"Model integrity verified for {metadata['model_name']}")
            return True
        except Exception as e:
            print(f"Error verifying model integrity: {e}")
            return False

class IsaacFoundationModel:
    """
    Base class for Isaac Foundation Models.
    """
    def __init__(self,
                 model_name: str,
                 model_path: str,
                 device: str = 'cuda',
                 precision: str = 'fp16'):
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.precision = precision

        # Load model configuration
        self.config = self._load_model_config()
        self.model = self._load_model()

        # Initialize preprocessing and postprocessing
        self.preprocessor = self._initialize_preprocessor()
        self.postprocessor = self._initialize_postprocessor()

        print(f"Isaac Foundation Model {model_name} loaded successfully on {self.device}")

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default config
            return {
                'input_resolution': [224, 224],
                'output_dimension': 512,
                'model_architecture': 'transformer',
                'max_sequence_length': 512
            }

    def _load_model(self) -> torch.nn.Module:
        """Load the actual model."""
        # This would load the specific model based on type
        # For now, we'll create a placeholder
        model = self._create_placeholder_model()
        model_path = os.path.join(self.model_path, 'model.pth')

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model weights loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model weights: {e}. Using random initialization.")
        else:
            print(f"Model weights not found at {model_path}. Using random initialization.")

        model = model.to(self.device)
        if self.precision == 'fp16':
            model = model.half()

        return model

    def _create_placeholder_model(self) -> torch.nn.Module:
        """Create a placeholder model for demonstration."""
        # This is a simplified placeholder - real models would be more complex
        class PlaceholderModel(torch.nn.Module):
            def __init__(self, input_dim=512, output_dim=7):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU()
                )
                self.output_head = torch.nn.Linear(256, output_dim)

            def forward(self, x):
                features = self.backbone(x)
                output = self.output_head(features)
                return output

        return PlaceholderModel()

    def _initialize_preprocessor(self):
        """Initialize input preprocessing."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(self.config.get('input_resolution', [224, 224])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _initialize_postprocessor(self):
        """Initialize output postprocessing."""
        # This would contain model-specific postprocessing
        return lambda x: x  # Identity function for now

    def preprocess_input(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Preprocess input data for the model."""
        # This would handle different input modalities
        # For now, assume single image input
        if 'image' in input_data:
            image = input_data['image']
            if isinstance(image, np.ndarray):
                image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            elif isinstance(image, torch.Tensor):
                pass  # Already tensor
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Apply preprocessing
            processed_image = self.preprocessor(image)
            return processed_image.unsqueeze(0)  # Add batch dimension

        elif 'features' in input_data:
            # Direct feature input
            features = input_data['features']
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            return features.unsqueeze(0)  # Add batch dimension

        else:
            raise ValueError("Input must contain 'image' or 'features'")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on input data."""
        self.model.eval()

        with torch.no_grad():
            # Preprocess input
            processed_input = self.preprocess_input(input_data)

            # Move to device
            processed_input = processed_input.to(self.device)

            # Run inference
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    output = self.model(processed_input)
            else:
                output = self.model(processed_input)

            # Postprocess output
            processed_output = self.postprocessor(output)

        return {
            'predictions': processed_output.cpu().numpy(),
            'raw_output': output.cpu().numpy(),
            'model_name': self.model_name,
            'timestamp': time.time()
        }

    def batch_predict(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch inference."""
        results = []
        for data in batch_data:
            result = self.predict(data)
            results.append(result)
        return results
```

## Isaac Perceptor: Multi-Modal Perception

### Isaac Perceptor Overview

Isaac Perceptor is a foundation model for multi-modal robotic perception, combining visual, depth, and language understanding to provide comprehensive scene interpretation.

```python
# isaac_perceptor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
from typing import Dict, Any, List, Tuple
import numpy as np

class IsaacPerceptor(nn.Module):
    """
    Isaac Perceptor: Multi-modal perception foundation model.
    Combines visual, depth, and language understanding for comprehensive scene interpretation.
    """
    def __init__(self,
                 vision_model_name: str = "openai/clip-vit-large-patch14",
                 text_model_name: str = "openai/clip-vit-large-patch14",
                 hidden_dim: int = 768,
                 output_dim: int = 512):
        super().__init__()

        # Load pre-trained vision and text encoders
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)

        # Projection layers to common space
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # Multi-modal fusion
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Task-specific heads
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4 + 1000)  # 4 for bbox + 1000 for classes
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 50, 1)  # 50 semantic classes
        )

        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 200)  # 200 attribute classes
        )

        # Depth estimation head
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Single channel for depth
        )

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self,
                images: torch.Tensor,
                text_inputs: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Isaac Perceptor.

        Args:
            images: Input images (B, C, H, W)
            text_inputs: Optional text tokens (B, T)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing all perception outputs
        """
        batch_size = images.size(0)

        # Process vision
        vision_outputs = self.vision_encoder(images)
        vision_features = vision_outputs.pooler_output  # (B, vision_hidden)
        vision_features = self.vision_proj(vision_features)  # (B, hidden_dim)

        # Process text if provided
        if text_inputs is not None:
            text_outputs = self.text_encoder(text_inputs)
            text_features = text_outputs.pooler_output  # (B, text_hidden)
            text_features = self.text_proj(text_features)  # (B, hidden_dim)

            # Fuse vision and text
            fused_features = vision_features + text_features
        else:
            fused_features = vision_features

        # Apply fusion transformer for complex reasoning
        fused_features = fused_features.unsqueeze(1)  # (B, 1, hidden_dim)
        fused_features = self.fusion_transformer(fused_features)
        fused_features = fused_features.squeeze(1)  # (B, hidden_dim)

        # Generate task-specific outputs
        outputs = {}

        # Object detection
        detection_output = self.detection_head(fused_features)
        outputs['detection'] = {
            'bbox': detection_output[:, :4],
            'class_logits': detection_output[:, 4:]
        }

        # Attributes
        attribute_logits = self.attribute_head(fused_features)
        outputs['attributes'] = attribute_logits

        # Return features if requested
        if return_features:
            outputs['features'] = fused_features

        return outputs

    def detect_objects(self,
                      image: torch.Tensor,
                      text_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in image with optional text query.

        Args:
            image: Input image (C, H, W)
            text_query: Optional text query for specific objects

        Returns:
            List of detected objects with bounding boxes and confidence
        """
        self.eval()

        with torch.no_grad():
            # Prepare inputs
            image = image.unsqueeze(0)  # Add batch dimension

            if text_query:
                text_tokens = self.processor(text=text_query, return_tensors="pt", padding=True)
                text_input_ids = text_tokens['input_ids']
            else:
                text_input_ids = None

            # Get outputs
            outputs = self.forward(image, text_input_ids)

            # Process detection results
            bboxes = outputs['detection']['bbox'].squeeze(0)
            class_logits = outputs['detection']['class_logits'].squeeze(0)

            # Convert to detection format
            detections = []
            for i in range(bboxes.size(0)):
                bbox = bboxes[i].cpu().numpy()
                class_probs = torch.softmax(class_logits[i], dim=0)
                confidence, class_idx = torch.max(class_probs, dim=0)

                detection = {
                    'bbox': bbox,
                    'confidence': confidence.item(),
                    'class_id': class_idx.item(),
                    'class_name': self._get_class_name(class_idx.item())
                }
                detections.append(detection)

        return detections

    def segment_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform semantic segmentation on image.

        Args:
            image: Input image (C, H, W)

        Returns:
            Segmentation mask (H, W) with class indices
        """
        self.eval()

        with torch.no_grad():
            image = image.unsqueeze(0)  # Add batch dimension

            # Get vision features for spatial processing
            vision_outputs = self.vision_encoder(image)
            spatial_features = vision_outputs.last_hidden_state  # (B, patches, hidden_dim)

            # Reshape for segmentation head
            batch_size, seq_len, hidden_dim = spatial_features.shape
            h = w = int(seq_len ** 0.5)  # Assuming square patches
            spatial_features = spatial_features.view(batch_size, h, w, hidden_dim)
            spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, hidden_dim, h, w)

            # Apply segmentation head
            segmentation_logits = self.segmentation_head(spatial_features)
            segmentation_mask = torch.argmax(segmentation_logits, dim=1)  # (B, h, w)

        return segmentation_mask.squeeze(0)  # (H, W)

    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from single image.

        Args:
            image: Input image (C, H, W)

        Returns:
            Depth map (H, W)
        """
        self.eval()

        with torch.no_grad():
            image = image.unsqueeze(0)  # Add batch dimension

            # Get vision features
            vision_outputs = self.vision_encoder(image)
            spatial_features = vision_outputs.last_hidden_state  # (B, patches, hidden_dim)

            # Reshape for depth estimation head
            batch_size, seq_len, hidden_dim = spatial_features.shape
            h = w = int(seq_len ** 0.5)
            spatial_features = spatial_features.view(batch_size, h, w, hidden_dim)
            spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, hidden_dim, h, w)

            # Apply depth head
            depth_map = self.depth_head(spatial_features)
            depth_map = torch.sigmoid(depth_map)  # Normalize to [0, 1]

        return depth_map.squeeze(0).squeeze(0)  # (H, W)

    def _get_class_name(self, class_idx: int) -> str:
        """
        Get class name for COCO dataset (simplified).
        In practice, this would use proper class mapping.
        """
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if class_idx < len(coco_classes):
            return coco_classes[class_idx]
        else:
            return f"unknown_{class_idx}"

class IsaacPerceptorTrainer:
    """
    Trainer for Isaac Perceptor model.
    """
    def __init__(self,
                 model: IsaacPerceptor,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.detection_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.SmoothL1Loss()

        # Training metrics
        self.training_losses = []
        self.validation_metrics = []

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   task_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch with images, texts, targets
            task_weights: Weights for different tasks

        Returns:
            Dictionary of loss values
        """
        if task_weights is None:
            task_weights = {
                'detection': 1.0,
                'segmentation': 1.0,
                'depth': 1.0,
                'attributes': 1.0
            }

        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images'].to(self.device)
        texts = batch.get('texts', None)
        if texts is not None:
            texts = texts.to(self.device)

        # Forward pass
        outputs = self.model(images, texts, return_features=True)

        # Compute losses for each task
        total_loss = 0.0
        losses = {}

        # Detection loss
        if 'detection_targets' in batch:
            det_targets = batch['detection_targets'].to(self.device)
            det_pred = outputs['detection']['bbox']
            det_loss = self.detection_loss(det_pred, det_targets[:, :4])  # bbox only
            losses['detection_loss'] = det_loss.item()
            total_loss += task_weights['detection'] * det_loss

        # Classification loss
        if 'classification_targets' in batch:
            cls_targets = batch['classification_targets'].to(self.device)
            cls_pred = outputs['detection']['class_logits']
            cls_loss = self.classification_loss(cls_pred, cls_targets)
            losses['classification_loss'] = cls_loss.item()
            total_loss += task_weights['detection'] * cls_loss

        # Segmentation loss
        if 'segmentation_targets' in batch:
            seg_targets = batch['segmentation_targets'].to(self.device)
            # Apply segmentation head separately for spatial features
            seg_pred = self._get_segmentation_prediction(images)
            seg_loss = self.segmentation_loss(seg_pred, seg_targets)
            losses['segmentation_loss'] = seg_loss.item()
            total_loss += task_weights['segmentation'] * seg_loss

        # Depth loss
        if 'depth_targets' in batch:
            depth_targets = batch['depth_targets'].to(self.device)
            # Apply depth head separately for spatial features
            depth_pred = self._get_depth_prediction(images)
            depth_loss = self.depth_loss(depth_pred, depth_targets)
            losses['depth_loss'] = depth_loss.item()
            total_loss += task_weights['depth'] * depth_loss

        # Attribute prediction loss
        if 'attribute_targets' in batch:
            attr_targets = batch['attribute_targets'].to(self.device)
            attr_pred = outputs['attributes']
            attr_loss = self.classification_loss(attr_pred, attr_targets)
            losses['attribute_loss'] = attr_loss.item()
            total_loss += task_weights['attributes'] * attr_loss

        losses['total_loss'] = total_loss.item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return losses

    def _get_segmentation_prediction(self, images: torch.Tensor) -> torch.Tensor:
        """Get segmentation prediction (separate forward pass)."""
        with torch.no_grad():
            vision_outputs = self.model.vision_encoder(images)
            spatial_features = vision_outputs.last_hidden_state
            batch_size, seq_len, hidden_dim = spatial_features.shape
            h = w = int(seq_len ** 0.5)
            spatial_features = spatial_features.view(batch_size, h, w, hidden_dim)
            spatial_features = spatial_features.permute(0, 3, 1, 2)

            segmentation_logits = self.model.segmentation_head(spatial_features)
            return segmentation_logits

    def _get_depth_prediction(self, images: torch.Tensor) -> torch.Tensor:
        """Get depth prediction (separate forward pass)."""
        with torch.no_grad():
            vision_outputs = self.model.vision_encoder(images)
            spatial_features = vision_outputs.last_hidden_state
            batch_size, seq_len, hidden_dim = spatial_features.shape
            h = w = int(seq_len ** 0.5)
            spatial_features = spatial_features.view(batch_size, h, w, hidden_dim)
            spatial_features = spatial_features.permute(0, 3, 1, 2)

            depth_map = self.model.depth_head(spatial_features)
            return depth_map

    def evaluate(self,
                 eval_loader: torch.utils.data.DataLoader,
                 tasks: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model on validation set.

        Args:
            eval_loader: Evaluation data loader
            tasks: Specific tasks to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        if tasks is None:
            tasks = ['detection', 'segmentation', 'depth', 'attributes']

        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for batch in eval_loader:
                images = batch['images'].to(self.device)
                texts = batch.get('texts', None)
                if texts is not None:
                    texts = texts.to(self.device)

                outputs = self.model(images, texts)

                # Calculate metrics for each task
                if 'detection' in tasks and 'detection_targets' in batch:
                    det_targets = batch['detection_targets'].to(self.device)
                    det_pred = outputs['detection']['bbox']

                    # Calculate mAP, IoU, etc.
                    det_metrics = self._calculate_detection_metrics(det_pred, det_targets)
                    for key, value in det_metrics.items():
                        metrics[key] = metrics.get(key, []) + [value]

                if 'segmentation' in tasks and 'segmentation_targets' in batch:
                    seg_targets = batch['segmentation_targets'].to(self.device)
                    seg_pred = self._get_segmentation_prediction(images)
                    seg_pred = torch.argmax(seg_pred, dim=1)

                    # Calculate IoU, accuracy, etc.
                    seg_metrics = self._calculate_segmentation_metrics(seg_pred, seg_targets)
                    for key, value in seg_metrics.items():
                        metrics[key] = metrics.get(key, []) + [value]

        # Average metrics
        averaged_metrics = {}
        for key, values in metrics.items():
            if isinstance(values[0], torch.Tensor):
                averaged_metrics[key] = torch.mean(torch.stack(values)).item()
            else:
                averaged_metrics[key] = np.mean(values)

        return averaged_metrics

    def _calculate_detection_metrics(self,
                                   predictions: torch.Tensor,
                                   targets: torch.Tensor) -> Dict[str, float]:
        """Calculate detection metrics."""
        # Calculate IoU between predicted and target bounding boxes
        ious = self._calculate_bbox_iou(predictions[:, :4], targets[:, :4])
        mean_iou = torch.mean(ious).item()

        # Calculate classification accuracy
        pred_classes = torch.argmax(predictions[:, 4:], dim=1)
        target_classes = targets[:, 4].long()
        class_accuracy = (pred_classes == target_classes).float().mean().item()

        return {
            'mean_iou': mean_iou,
            'class_accuracy': class_accuracy,
            'bbox_accuracy': torch.mean((ious > 0.5).float()).item()  # IoU > 0.5
        }

    def _calculate_segmentation_metrics(self,
                                      predictions: torch.Tensor,
                                      targets: torch.Tensor) -> Dict[str, float]:
        """Calculate segmentation metrics."""
        # Intersection over Union (IoU) for each class
        ious = []
        for class_idx in range(predictions.size(1)):  # Assuming class dimension is 1
            pred_mask = (predictions == class_idx).float()
            target_mask = (targets == class_idx).float()

            intersection = (pred_mask * target_mask).sum()
            union = (pred_mask + target_mask - pred_mask * target_mask).sum()

            if union > 0:
                iou = intersection / union
                ious.append(iou.item())

        mean_iou = np.mean(ious) if ious else 0.0
        pixel_accuracy = (predictions == targets).float().mean().item()

        return {
            'mean_iou': mean_iou,
            'pixel_accuracy': pixel_accuracy
        }

    def _calculate_bbox_iou(self, boxes1: torch.Tensor,
                           boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of bounding boxes."""
        # boxes format: [x1, y1, x2, y2]
        # Calculate intersection
        x1_inter = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1_inter = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2_inter = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2_inter = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        intersection = inter_width * inter_height

        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

        return iou
```

## Isaac Controller: Vision-Language-Action Control

### Isaac Controller Architecture

The Isaac Controller is a foundation model designed for vision-language-action tasks, enabling robots to execute complex instructions based on visual and linguistic inputs.

```python
# isaac_controller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
import einops

class IsaacController(nn.Module):
    """
    Isaac Controller: Vision-Language-Action foundation model for robotic control.
    """
    def __init__(self,
                 vision_model_name: str = "openai/clip-vit-large-patch14",
                 text_model_name: str = "openai/clip-vit-large-patch14",
                 action_space_dim: int = 7,  # 7-DOF robot arm
                 hidden_dim: int = 768,
                 max_seq_len: int = 512):
        super().__init__()

        # Load pre-trained vision and text encoders
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)

        # Projection layers
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # Action encoder/decoder
        self.action_encoder = nn.Linear(action_space_dim, hidden_dim)
        self.action_decoder = nn.Linear(hidden_dim, action_space_dim)

        # Temporal transformer for sequence modeling
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Task-specific policy heads
        self.manipulation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_space_dim)
        )

        self.navigation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # x, y, theta for 2D navigation
        )

        self.grasping_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # x, y, z, gripper angle
        )

        # Task classifier to determine appropriate action head
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # manipulation, navigation, grasping
        )

        # State estimation head for robot state prediction
        self.state_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_space_dim * 2)  # position + velocity
        )

        self.hidden_dim = hidden_dim
        self.action_space_dim = action_space_dim
        self.max_seq_len = max_seq_len

    def forward(self,
                images: torch.Tensor,
                text_inputs: torch.Tensor,
                robot_state: Optional[torch.Tensor] = None,
                previous_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Isaac Controller.

        Args:
            images: Input images (B, T, C, H, W) or (B, C, H, W)
            text_inputs: Text tokens (B, T)
            robot_state: Current robot state (B, state_dim)
            previous_actions: Previous actions (B, T_prev, action_dim)

        Returns:
            Dictionary containing action predictions and task classifications
        """
        batch_size = images.size(0)

        # Process vision
        if len(images.shape) == 5:  # (B, T, C, H, W)
            # Process sequence of images
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            vision_outputs = self.vision_encoder(images_flat)
            vision_features = vision_outputs.pooler_output
            vision_features = vision_features.view(B, T, -1)  # (B, T, vision_hidden)
            vision_features = self.vision_proj(vision_features)  # (B, T, hidden_dim)
        else:  # (B, C, H, W)
            vision_outputs = self.vision_encoder(images)
            vision_features = vision_outputs.pooler_output  # (B, vision_hidden)
            vision_features = self.vision_proj(vision_features)  # (B, hidden_dim)
            vision_features = vision_features.unsqueeze(1)  # (B, 1, hidden_dim)

        # Process text
        text_outputs = self.text_encoder(text_inputs)
        text_features = text_outputs.pooler_output  # (B, text_hidden)
        text_features = self.text_proj(text_features)  # (B, hidden_dim)
        text_features = text_features.unsqueeze(1)  # (B, 1, hidden_dim)

        # Process robot state if provided
        state_features = None
        if robot_state is not None:
            state_features = robot_state.unsqueeze(1)  # (B, 1, state_dim)
            state_features = self.state_proj(state_features)  # (B, 1, hidden_dim)

        # Process previous actions if provided
        action_features = None
        if previous_actions is not None:
            B_prev, T_prev, A = previous_actions.shape
            action_features_flat = previous_actions.view(B_prev * T_prev, A)
            action_features = self.action_encoder(action_features_flat)  # (B*T, hidden_dim)
            action_features = action_features.view(B_prev, T_prev, -1)  # (B, T, hidden_dim)

        # Concatenate all features
        if state_features is not None and action_features is not None:
            # Concatenate vision, text, state, and actions
            all_features = torch.cat([
                vision_features,
                text_features,
                state_features,
                action_features
            ], dim=1)  # (B, 2+1+state_seq+action_seq, hidden_dim)
        elif state_features is not None:
            all_features = torch.cat([vision_features, text_features, state_features], dim=1)
        else:
            all_features = torch.cat([vision_features, text_features], dim=1)

        # Apply temporal transformer for sequence modeling
        sequence_features = self.temporal_transformer(all_features)

        # Get fused representation (use first token or mean pooling)
        fused_features = sequence_features[:, 0, :]  # Use first token as global representation

        # Predict task type
        task_logits = self.task_classifier(fused_features)
        task_probs = F.softmax(task_logits, dim=1)

        # Generate task-specific actions
        actions = {}

        # Manipulation actions
        manipulation_actions = self.manipulation_head(fused_features)
        actions['manipulation'] = manipulation_actions

        # Navigation actions
        navigation_actions = self.navigation_head(fused_features)
        actions['navigation'] = navigation_actions

        # Grasping actions
        grasping_actions = self.grasping_head(fused_features)
        actions['grasping'] = grasping_actions

        # Estimate current state
        estimated_state = self.state_estimator(fused_features)

        return {
            'actions': actions,
            'task_probs': task_probs,
            'task_logits': task_logits,
            'fused_features': fused_features,
            'estimated_state': estimated_state
        }

    def plan_trajectory(self,
                       initial_state: torch.Tensor,
                       goal_description: str,
                       image_sequence: torch.Tensor,
                       max_steps: int = 100) -> List[torch.Tensor]:
        """
        Plan a trajectory using the Isaac Controller.

        Args:
            initial_state: Initial robot state
            goal_description: Natural language goal description
            image_sequence: Sequence of images for planning
            max_steps: Maximum planning steps

        Returns:
            List of planned actions
        """
        self.eval()
        trajectory = []
        current_state = initial_state.clone()

        # Tokenize goal description
        text_tokens = self.processor(text=goal_description, return_tensors="pt", padding=True)
        text_input_ids = text_tokens['input_ids'].to(self.device)

        with torch.no_grad():
            for step in range(max_steps):
                # Get action from controller
                outputs = self.forward(
                    images=image_sequence[step % len(image_sequence)].unsqueeze(0),
                    text_inputs=text_input_ids,
                    robot_state=current_state.unsqueeze(0)
                )

                # Select appropriate action based on predicted task
                task_idx = torch.argmax(outputs['task_probs'], dim=1).item()
                task_types = ['manipulation', 'navigation', 'grasping']
                selected_task = task_types[task_idx]

                action = outputs['actions'][selected_task]

                # Apply action to current state (simplified)
                next_state = self._apply_action_to_state(current_state, action)

                trajectory.append(action.squeeze(0))
                current_state = next_state

                # Check termination condition (simplified)
                if self._check_goal_reached(current_state, goal_description):
                    break

        return trajectory

    def _apply_action_to_state(self, state: torch.Tensor,
                              action: torch.Tensor) -> torch.Tensor:
        """Apply action to state (simplified dynamics model)."""
        # This is a simplified example - in reality, this would depend on robot dynamics
        next_state = state + action * 0.1  # Simple integration
        return next_state

    def _check_goal_reached(self, state: torch.Tensor,
                           goal_description: str) -> bool:
        """Check if goal has been reached (simplified)."""
        # This would be more sophisticated in practice
        # For now, just return False to continue planning
        return False

    def execute_instruction(self,
                          image: torch.Tensor,
                          instruction: str,
                          robot_state: torch.Tensor) -> Dict[str, Any]:
        """
        Execute a natural language instruction.

        Args:
            image: Current image observation
            instruction: Natural language instruction
            robot_state: Current robot state

        Returns:
            Dictionary containing action and confidence
        """
        self.eval()

        with torch.no_grad():
            # Tokenize instruction
            text_tokens = self.processor(text=instruction, return_tensors="pt", padding=True)
            text_input_ids = text_tokens['input_ids'].to(self.device)

            # Get outputs
            outputs = self.forward(
                images=image.unsqueeze(0),
                text_inputs=text_input_ids,
                robot_state=robot_state.unsqueeze(0)
            )

            # Get most probable task
            task_idx = torch.argmax(outputs['task_probs'], dim=1).item()
            task_types = ['manipulation', 'navigation', 'grasping']
            selected_task = task_types[task_idx]
            confidence = torch.max(outputs['task_probs'], dim=1).values.item()

            action = outputs['actions'][selected_task].squeeze(0)

            return {
                'action': action,
                'task_type': selected_task,
                'confidence': confidence,
                'task_probs': outputs['task_probs'].squeeze(0).cpu().numpy(),
                'estimated_state': outputs['estimated_state'].squeeze(0)
            }

class IsaacControllerTrainer:
    """
    Trainer for Isaac Controller model.
    """
    def __init__(self,
                 model: IsaacController,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.action_loss = nn.MSELoss()
        self.task_loss = nn.CrossEntropyLoss()
        self.state_loss = nn.MSELoss()

        # Training metrics
        self.training_losses = []
        self.task_accuracies = []

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   task_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch with images, texts, actions, states
            task_weights: Weights for different components

        Returns:
            Dictionary of loss values
        """
        if task_weights is None:
            task_weights = {
                'action': 1.0,
                'task': 1.0,
                'state': 0.5
            }

        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images'].to(self.device)
        texts = batch['texts'].to(self.device)
        actions = batch['actions'].to(self.device)
        states = batch['states'].to(self.device)
        task_labels = batch.get('task_labels', None)
        if task_labels is not None:
            task_labels = task_labels.to(self.device)

        # Forward pass
        outputs = self.model(images, texts, states)

        # Compute losses
        total_loss = 0.0
        losses = {}

        # Action prediction loss (weighted by task probability)
        if 'task_probs' in outputs and task_labels is not None:
            # Select actions for the correct task
            task_mask = F.one_hot(task_labels, num_classes=3).float()  # (B, 3)
            predicted_actions = torch.stack([
                outputs['actions']['manipulation'],
                outputs['actions']['navigation'],
                outputs['actions']['grasping']
            ], dim=1)  # (B, 3, action_dim)

            # Weighted action loss
            weighted_predicted = (predicted_actions * task_mask.unsqueeze(-1)).sum(dim=1)
            action_loss = self.action_loss(weighted_predicted, actions)
        else:
            # Use all tasks equally
            action_loss = 0
            for task_name, task_actions in outputs['actions'].items():
                action_loss += self.action_loss(task_actions, actions)
            action_loss = action_loss / len(outputs['actions'])

        losses['action_loss'] = action_loss.item()
        total_loss += task_weights['action'] * action_loss

        # Task classification loss
        if task_labels is not None:
            task_loss = self.task_loss(outputs['task_logits'], task_labels)
            losses['task_loss'] = task_loss.item()
            total_loss += task_weights['task'] * task_loss

        # State estimation loss
        state_loss = self.state_loss(outputs['estimated_state'], states)
        losses['state_loss'] = state_loss.item()
        total_loss += task_weights['state'] * state_loss

        losses['total_loss'] = total_loss.item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return losses

    def evaluate_policy(self,
                       eval_loader: torch.utils.data.DataLoader,
                       max_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the policy in simulation.

        Args:
            eval_loader: Evaluation data loader
            max_episodes: Maximum episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        episode_rewards = []
        success_rates = []
        task_accuracies = []

        with torch.no_grad():
            for episode_idx, batch in enumerate(eval_loader):
                if episode_idx >= max_episodes:
                    break

                images = batch['images'].to(self.device)
                texts = batch['texts'].to(self.device)
                target_actions = batch['actions'].to(self.device)
                task_labels = batch.get('task_labels', None)
                if task_labels is not None:
                    task_labels = task_labels.to(self.device)

                # Get predictions
                outputs = self.model(images, texts)

                # Calculate metrics
                predicted_actions = torch.stack([
                    outputs['actions']['manipulation'],
                    outputs['actions']['navigation'],
                    outputs['actions']['grasping']
                ], dim=1)  # (B, 3, action_dim)

                # Calculate action accuracy (simplified as MSE)
                action_errors = F.mse_loss(predicted_actions, target_actions.unsqueeze(1).expand_as(predicted_actions), reduction='none').mean(dim=2)
                min_action_errors, best_task_idx = torch.min(action_errors, dim=1)
                avg_action_error = min_action_errors.mean().item()

                episode_rewards.append(-avg_action_error)  # Negative because lower error is better

                # Calculate task accuracy if labels provided
                if task_labels is not None:
                    predicted_task_idx = torch.argmax(outputs['task_probs'], dim=1)
                    task_acc = (predicted_task_idx == task_labels).float().mean().item()
                    task_accuracies.append(task_acc)

        metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'num_episodes': len(episode_rewards)
        }

        if task_accuracies:
            metrics['task_accuracy'] = np.mean(task_accuracies)

        return metrics
```

## Isaac Manipulator: Dexterous Manipulation Foundation Model

### Isaac Manipulator Architecture

```python
# isaac_manipulator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math

class ManipulationTransformer(nn.Module):
    """
    Transformer architecture for robotic manipulation tasks.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_seq_len: int = 100):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Task embedding (for multi-task learning)
        self.task_embedding = nn.Embedding(10, hidden_dim)  # 10 different tasks

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.grasp_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Grasp quality between 0 and 1
        )

    def forward(self,
                states: torch.Tensor,
                actions: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the manipulation transformer.

        Args:
            states: Sequence of robot states (B, T, state_dim)
            actions: Sequence of actions (B, T, action_dim) - optional for conditioning
            task_id: Task ID for multi-task learning (B,)
            attention_mask: Attention mask (B, T)

        Returns:
            Dictionary of outputs
        """
        batch_size, seq_len = states.size(0), states.size(1)

        # Encode states
        state_embeddings = self.state_encoder(states)  # (B, T, hidden_dim)

        # Add action embeddings if provided
        if actions is not None:
            action_embeddings = self.action_encoder(actions)  # (B, T, hidden_dim)
            # Concatenate or add state and action embeddings
            embeddings = state_embeddings + action_embeddings
        else:
            embeddings = state_embeddings

        # Add task embedding if provided
        if task_id is not None:
            task_embed = self.task_embedding(task_id).unsqueeze(1)  # (B, 1, hidden_dim)
            embeddings = embeddings + task_embed

        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :seq_len, :]  # (1, T, hidden_dim)
        embeddings = embeddings + pos_encoding

        # Apply dropout
        embeddings = F.dropout(embeddings, p=0.1, training=self.training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask)

        # Generate outputs
        action_pred = self.action_head(embeddings)  # (B, T, action_dim)
        value_pred = self.value_head(embeddings)  # (B, T, 1)
        grasp_quality = self.grasp_quality_head(embeddings)  # (B, T, 1)

        return {
            'actions': action_pred,
            'values': value_pred,
            'grasp_quality': grasp_quality,
            'embeddings': embeddings
        }

class IsaacManipulator(nn.Module):
    """
    Isaac Manipulator: Foundation model for dexterous robotic manipulation.
    """
    def __init__(self,
                 robot_state_dim: int = 14,  # 7 joint positions + 7 joint velocities
                 action_dim: int = 7,  # 7-DOF robot arm
                 hidden_dim: int = 512,
                 max_episode_length: int = 100):
        super().__init__()

        # Vision processing
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU()
        )

        # Language processing
        self.language_encoder = nn.Sequential(
            nn.Linear(512, hidden_dim),  # Assuming 512-dim language embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Manipulation transformer
        self.manipulation_transformer = ManipulationTransformer(
            state_dim=hidden_dim * 3,  # Combined vision, language, state
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )

        # Grasp planning head
        self.grasp_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 3D position + 3D orientation
        )

        # Skill decomposition network
        self.skill_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 10 different manipulation skills
        )

        self.robot_state_dim = robot_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                image: torch.Tensor,
                language_embedding: torch.Tensor,
                robot_state: torch.Tensor,
                task_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Isaac Manipulator.

        Args:
            image: Input image (B, C, H, W)
            language_embedding: Language embedding (B, lang_dim)
            robot_state: Robot state (B, state_dim)
            task_history: Task history (B, T, hidden_dim) - optional

        Returns:
            Dictionary of manipulation outputs
        """
        batch_size = image.size(0)

        # Process vision
        vision_features = self.vision_backbone(image)  # (B, hidden_dim)

        # Process language
        language_features = self.language_encoder(language_embedding)  # (B, hidden_dim)

        # Process robot state
        state_features = self.state_processor(robot_state)  # (B, hidden_dim)

        # Combine features
        combined_features = torch.cat([vision_features, language_features, state_features], dim=1)  # (B, 3*hidden_dim)

        # Process through transformer
        transformer_outputs = self.manipulation_transformer(
            states=combined_features.unsqueeze(1),  # Add sequence dimension
            task_id=None
        )

        # Generate manipulation outputs
        outputs = {}

        # Action prediction
        action_pred = transformer_outputs['actions'].squeeze(1)  # Remove sequence dim
        outputs['actions'] = action_pred

        # Grasp planning
        grasp_pose = self.grasp_planner(combined_features)
        outputs['grasp_pose'] = grasp_pose

        # Skill prediction
        skill_logits = self.skill_decomposer(combined_features)
        skill_probs = F.softmax(skill_logits, dim=1)
        outputs['skill_probs'] = skill_probs
        outputs['skill_logits'] = skill_logits

        # Grasp quality prediction
        outputs['grasp_quality'] = transformer_outputs['grasp_quality'].squeeze(1)

        # Value prediction
        outputs['value'] = transformer_outputs['values'].squeeze(1)

        return outputs

    def plan_manipulation_sequence(self,
                                 image: torch.Tensor,
                                 instruction: str,
                                 initial_state: torch.Tensor,
                                 language_model) -> List[Dict[str, torch.Tensor]]:
        """
        Plan a manipulation sequence for a given instruction.

        Args:
            image: Input image
            instruction: Natural language instruction
            initial_state: Initial robot state
            language_model: Language model for encoding instructions

        Returns:
            List of planned actions with associated information
        """
        self.eval()

        # Encode instruction
        with torch.no_grad():
            lang_embedding = language_model.encode(instruction).to(self.device)

        # Get initial manipulation plan
        with torch.no_grad():
            outputs = self.forward(image, lang_embedding, initial_state)

        # Decompose into sub-skills based on predicted skill probabilities
        skill_probs = outputs['skill_probs'].squeeze(0).cpu().numpy()
        predicted_skill = np.argmax(skill_probs)

        # Generate skill-specific sub-actions
        sub_actions = self._decompose_skill(predicted_skill, outputs)

        # Plan trajectory
        plan = []
        current_state = initial_state.clone()

        for sub_action in sub_actions:
            # Apply action to get next state (simplified dynamics)
            next_state = self._apply_action_dynamics(current_state, sub_action['action'])

            plan.append({
                'action': sub_action['action'],
                'skill_type': sub_action['skill_type'],
                'confidence': sub_action['confidence'],
                'next_state': next_state,
                'grasp_quality': sub_action.get('grasp_quality', 0.0)
            })

            current_state = next_state

        return plan

    def _decompose_skill(self, skill_idx: int,
                        outputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Decompose a skill into sub-actions.

        Args:
            skill_idx: Predicted skill index
            outputs: Model outputs

        Returns:
            List of sub-actions
        """
        skill_names = [
            'reach', 'grasp', 'lift', 'transport', 'place',
            'push', 'pull', 'rotate', 'align', 'release'
        ]

        if skill_idx >= len(skill_names):
            skill_idx = 0  # Default to reach

        skill_name = skill_names[skill_idx]

        # Generate sub-actions based on skill type
        sub_actions = []

        if skill_name == 'reach':
            # Generate reaching trajectory
            target_pos = outputs['grasp_pose'][:, :3]  # Position part of grasp pose
            current_pos = outputs['robot_state'][:, :3]  # Current position
            delta = target_pos - current_pos

            # Create intermediate waypoints
            num_waypoints = 5
            for i in range(1, num_waypoints + 1):
                frac = i / num_waypoints
                waypoint = current_pos + delta * frac
                action = torch.cat([waypoint, outputs['robot_state'][:, 3:7], torch.zeros(1, 1)], dim=1)  # Include orientation and gripper
                sub_actions.append({
                    'action': action,
                    'skill_type': 'reach',
                    'confidence': outputs['skill_probs'][0, skill_idx].item(),
                    'waypoint_fraction': frac
                })

        elif skill_name == 'grasp':
            # Generate grasping action
            grasp_pose = outputs['grasp_pose']
            grasp_action = torch.cat([
                grasp_pose[:, :3],  # Position
                grasp_pose[:, 3:6],  # Orientation (simplified)
                torch.tensor([[0.0]])  # Close gripper
            ], dim=1)

            sub_actions.append({
                'action': grasp_action,
                'skill_type': 'grasp',
                'confidence': outputs['skill_probs'][0, skill_idx].item(),
                'grasp_quality': outputs['grasp_quality'][0].item()
            })

        elif skill_name == 'lift':
            # Generate lifting action (move up from grasp position)
            current_pos = outputs['robot_state'][:, :3]
            lift_offset = torch.tensor([[0.0, 0.0, 0.1]])  # Lift 10cm up
            lift_action = torch.cat([
                current_pos + lift_offset,
                outputs['robot_state'][:, 3:7],  # Keep orientation
                outputs['robot_state'][:, 7:8]   # Keep gripper state
            ], dim=1)

            sub_actions.append({
                'action': lift_action,
                'skill_type': 'lift',
                'confidence': outputs['skill_probs'][0, skill_idx].item()
            })

        else:
            # Default action based on skill prediction
            action = outputs['actions']
            sub_actions.append({
                'action': action,
                'skill_type': skill_name,
                'confidence': outputs['skill_probs'][0, skill_idx].item()
            })

        return sub_actions

    def _apply_action_dynamics(self, state: torch.Tensor,
                              action: torch.Tensor) -> torch.Tensor:
        """
        Apply action to state using simplified dynamics model.

        Args:
            state: Current state
            action: Action to apply

        Returns:
            Next state
        """
        # Simplified dynamics - in practice, this would use robot kinematics/dynamics
        next_state = state.clone()
        next_state[:, :self.action_dim] = action[:, :self.action_dim]  # Update positions
        # Update velocities based on position change
        next_state[:, self.action_dim:2*self.action_dim] = (action[:, :self.action_dim] - state[:, :self.action_dim]) / 0.1  # Assuming 0.1s time step

        return next_state

class ManipulationSkillLibrary:
    """
    Library of manipulation skills learned by Isaac Manipulator.
    """
    def __init__(self, manipulator_model: IsaacManipulator):
        self.manipulator = manipulator_model
        self.skills = {}
        self.skill_embeddings = {}

    def register_skill(self, skill_name: str, demonstration_data: List[Dict]):
        """
        Register a new skill with demonstration data.

        Args:
            skill_name: Name of the skill
            demonstration_data: List of (state, action) pairs demonstrating the skill
        """
        # Encode the skill using the manipulator's representation
        skill_embedding = self._encode_skill(demonstration_data)

        self.skills[skill_name] = {
            'demonstrations': demonstration_data,
            'embedding': skill_embedding,
            'success_rate': 0.0,
            'usage_count': 0
        }

        self.skill_embeddings[skill_name] = skill_embedding

    def _encode_skill(self, demonstration_data: List[Dict]) -> torch.Tensor:
        """
        Encode a skill using the manipulator's internal representation.
        """
        # This would use the manipulator model to extract skill features
        # For now, return a placeholder
        return torch.randn(self.manipulator.hidden_dim)

    def retrieve_similar_skills(self, query_embedding: torch.Tensor,
                               top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve skills similar to the query embedding.

        Args:
            query_embedding: Query embedding to match against
            top_k: Number of top skills to return

        Returns:
            List of (skill_name, similarity_score) tuples
        """
        similarities = []
        for skill_name, skill_data in self.skills.items():
            skill_embedding = skill_data['embedding']
            similarity = F.cosine_similarity(query_embedding, skill_embedding, dim=0)
            similarities.append((skill_name, similarity.item()))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def execute_skill(self, skill_name: str, state: torch.Tensor,
                     image: torch.Tensor, language_embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Execute a registered skill.

        Args:
            skill_name: Name of the skill to execute
            state: Current robot state
            image: Current image observation
            language_embedding: Current language embedding

        Returns:
            List of actions to execute the skill
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not found in library")

        # Use the manipulator to adapt the skill to current context
        outputs = self.manipulator(image, language_embedding, state)

        # Generate skill-specific actions
        skill_actions = self._adapt_skill_to_context(
            skill_name, state, outputs
        )

        return skill_actions

    def _adapt_skill_to_context(self, skill_name: str,
                               current_state: torch.Tensor,
                               model_outputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Adapt a skill to the current context.

        Args:
            skill_name: Name of the skill
            current_state: Current robot state
            model_outputs: Model outputs for current context

        Returns:
            List of adapted actions
        """
        # This would adapt the skill based on current context
        # For now, return a simple adaptation based on model outputs
        actions = []

        if skill_name in ['reach', 'grasp']:
            # Adapt based on predicted grasp pose
            grasp_pose = model_outputs['grasp_pose']
            action = torch.cat([
                grasp_pose[:, :3],  # Position
                current_state[:, 3:7],  # Keep current orientation
                current_state[:, 7:8]   # Keep current gripper state
            ], dim=1)
            actions.append(action)
        else:
            # Use predicted action from model
            action = model_outputs['actions']
            actions.append(action)

        return actions

class IsaacManipulatorTrainer:
    """
    Trainer for Isaac Manipulator model.
    """
    def __init__(self,
                 model: IsaacManipulator,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.action_loss = nn.MSELoss()
        self.grasp_loss = nn.MSELoss()
        self.skill_loss = nn.CrossEntropyLoss()
        self.quality_loss = nn.BCELoss()

        # Training metrics
        self.training_losses = []
        self.grasp_success_rates = []
        self.skill_accuracies = []

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   task_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch with images, language, states, actions, etc.
            task_weights: Weights for different loss components

        Returns:
            Dictionary of loss values
        """
        if task_weights is None:
            task_weights = {
                'action': 1.0,
                'grasp': 1.0,
                'skill': 1.0,
                'quality': 0.5
            }

        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images'].to(self.device)
        language_embeddings = batch['language_embeddings'].to(self.device)
        robot_states = batch['robot_states'].to(self.device)
        actions = batch['actions'].to(self.device)
        grasp_poses = batch.get('grasp_poses', None)
        if grasp_poses is not None:
            grasp_poses = grasp_poses.to(self.device)
        skill_labels = batch.get('skill_labels', None)
        if skill_labels is not None:
            skill_labels = skill_labels.to(self.device)
        grasp_success = batch.get('grasp_success', None)
        if grasp_success is not None:
            grasp_success = grasp_success.to(self.device)

        # Forward pass
        outputs = self.model(images, language_embeddings, robot_states)

        # Compute losses
        total_loss = 0.0
        losses = {}

        # Action prediction loss
        action_loss = self.action_loss(outputs['actions'], actions)
        losses['action_loss'] = action_loss.item()
        total_loss += task_weights['action'] * action_loss

        # Grasp prediction loss
        if grasp_poses is not None:
            grasp_loss = self.grasp_loss(outputs['grasp_pose'], grasp_poses)
            losses['grasp_loss'] = grasp_loss.item()
            total_loss += task_weights['grasp'] * grasp_loss

        # Skill prediction loss
        if skill_labels is not None:
            skill_loss = self.skill_loss(outputs['skill_logits'], skill_labels)
            losses['skill_loss'] = skill_loss.item()
            total_loss += task_weights['skill'] * skill_loss

        # Grasp quality prediction loss
        if grasp_success is not None:
            quality_loss = self.quality_loss(outputs['grasp_quality'], grasp_success.float().unsqueeze(1))
            losses['quality_loss'] = quality_loss.item()
            total_loss += task_weights['quality'] * quality_loss

        losses['total_loss'] = total_loss.item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return losses

    def evaluate_manipulation(self,
                            eval_loader: torch.utils.data.DataLoader,
                            max_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate manipulation performance.

        Args:
            eval_loader: Evaluation data loader
            max_episodes: Maximum episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        episode_rewards = []
        grasp_success_rates = []
        skill_accuracies = []

        with torch.no_grad():
            for episode_idx, batch in enumerate(eval_loader):
                if episode_idx >= max_episodes:
                    break

                images = batch['images'].to(self.device)
                language_embeddings = batch['language_embeddings'].to(self.device)
                robot_states = batch['robot_states'].to(self.device)
                target_actions = batch['actions'].to(self.device)
                target_grasps = batch.get('grasp_poses', None)
                if target_grasps is not None:
                    target_grasps = target_grasps.to(self.device)
                target_skills = batch.get('skill_labels', None)
                if target_skills is not None:
                    target_skills = target_skills.to(self.device)

                # Get predictions
                outputs = self.model(images, language_embeddings, robot_states)

                # Calculate metrics
                action_error = F.mse_loss(outputs['actions'], target_actions).item()
                episode_rewards.append(-action_error)  # Negative because lower error is better

                if target_grasps is not None:
                    grasp_error = F.mse_loss(outputs['grasp_pose'], target_grasps).item()
                    grasp_success = grasp_error < 0.05  # 5cm threshold
                    grasp_success_rates.append(float(grasp_success))

                if target_skills is not None:
                    predicted_skills = torch.argmax(outputs['skill_logits'], dim=1)
                    skill_acc = (predicted_skills == target_skills).float().mean().item()
                    skill_accuracies.append(skill_acc)

        metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'action_error_mean': -np.mean(episode_rewards),  # Positive error value
            'num_episodes': len(episode_rewards)
        }

        if grasp_success_rates:
            metrics['grasp_success_rate'] = np.mean(grasp_success_rates)
            metrics['grasp_error_mean'] = np.mean(grasp_success_rates) * 0.05 + np.mean([e for e, s in zip(episode_rewards, grasp_success_rates) if s]) * 0.95

        if skill_accuracies:
            metrics['skill_accuracy'] = np.mean(skill_accuracies)

        return metrics
```

## Isaac Navigator: Navigation Foundation Model

### Isaac Navigator Architecture

```python
# isaac_navigator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math

class NavigationTransformer(nn.Module):
    """
    Transformer architecture for robotic navigation tasks.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 2,  # x, y velocity or angular velocities
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_seq_len: int = 200):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Sensor encoder (handles multiple sensor modalities)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Map encoder (for known environment information)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Occupancy grid
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3D goal position
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Positional encoding for temporal sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output heads
        self.navigation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.path_planning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 50 * 2)  # 50 waypoints * 2D coordinates
        )

        self.collision_avoidance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Collision probability
        )

        self.localization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # x, y, theta position
        )

    def forward(self,
                sensor_data: torch.Tensor,
                map_data: torch.Tensor,
                goal: torch.Tensor,
                sequence_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the navigation transformer.

        Args:
            sensor_data: Sensor data (B, T, sensor_dim)
            map_data: Map data (B, 1, H, W) - occupancy grid
            goal: Goal position (B, 3) - x, y, theta
            sequence_mask: Mask for sequence processing (B, T)

        Returns:
            Dictionary of navigation outputs
        """
        batch_size, seq_len = sensor_data.size(0), sensor_data.size(1)

        # Encode sensor data
        sensor_features = self.sensor_encoder(sensor_data)  # (B, T, hidden_dim)

        # Encode map data
        map_features = self.map_encoder(map_data)  # (B, hidden_dim)
        map_features = map_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, hidden_dim)

        # Encode goal
        goal_features = self.goal_encoder(goal)  # (B, hidden_dim)
        goal_features = goal_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, hidden_dim)

        # Combine all features
        combined_features = sensor_features + map_features + goal_features

        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :seq_len, :]  # (1, T, hidden_dim)
        combined_features = combined_features + pos_encoding

        # Apply dropout
        combined_features = F.dropout(combined_features, p=0.1, training=self.training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            combined_features = layer(combined_features, src_key_padding_mask=sequence_mask)

        # Generate outputs
        navigation_actions = self.navigation_head(combined_features)  # (B, T, action_dim)
        path_predictions = self.path_planning_head(combined_features).view(batch_size, seq_len, 50, 2)  # (B, T, 50, 2)
        collision_prob = self.collision_avoidance_head(combined_features)  # (B, T, 1)
        localization_estimates = self.localization_head(combined_features)  # (B, T, 3)

        return {
            'navigation_actions': navigation_actions,
            'path_predictions': path_predictions,
            'collision_probabilities': collision_prob,
            'localization_estimates': localization_estimates,
            'features': combined_features
        }

class IsaacNavigator(nn.Module):
    """
    Isaac Navigator: Foundation model for robotic navigation and path planning.
    """
    def __init__(self,
                 sensor_dim: int = 360,  # LIDAR scan dimension
                 action_dim: int = 2,    # x, y velocity
                 hidden_dim: int = 512,
                 max_map_size: Tuple[int, int] = (100, 100)):
        super().__init__()

        # Vision processing for visual navigation
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        # LIDAR processing
        self.lidar_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # IMU processing
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 3D acceleration + 3D angular velocity
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Navigation transformer
        self.navigation_transformer = NavigationTransformer(
            state_dim=hidden_dim * 3,  # Combined vision, lidar, imu
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )

        # Global path planner
        self.global_planner = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for goal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 100)  # 100 waypoints
        )

        # Local obstacle avoidance
        self.obstacle_avoidance = nn.Sequential(
            nn.Linear(hidden_dim + sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Dynamic obstacle prediction
        self.dynamic_obstacle_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5 * 2)  # 5 future positions * 2D coordinates
        )

        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_map_size = max_map_size

    def forward(self,
                vision_data: torch.Tensor,
                lidar_data: torch.Tensor,
                imu_data: torch.Tensor,
                map_data: torch.Tensor,
                goal: torch.Tensor,
                sequence_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Isaac Navigator.

        Args:
            vision_data: Visual input (B, C, H, W)
            lidar_data: LIDAR scan data (B, sensor_dim)
            imu_data: IMU data (B, 6)
            map_data: Map data (B, 1, H, W)
            goal: Goal position (B, 3)
            sequence_mask: Sequence mask (B, T)

        Returns:
            Dictionary of navigation outputs
        """
        batch_size = vision_data.size(0)

        # Process different sensor modalities
        vision_features = self.vision_encoder(vision_data)  # (B, hidden_dim)
        lidar_features = self.lidar_encoder(lidar_data)     # (B, hidden_dim)
        imu_features = self.imu_encoder(imu_data)          # (B, hidden_dim)

        # Combine sensor features
        combined_sensor_features = torch.cat([vision_features, lidar_features, imu_features], dim=1)  # (B, 3*hidden_dim)

        # Process through navigation transformer
        transformer_outputs = self.navigation_transformer(
            sensor_data=combined_sensor_features.unsqueeze(1),  # Add sequence dimension
            map_data=map_data,
            goal=goal,
            sequence_mask=sequence_mask
        )

        # Generate navigation outputs
        outputs = {}

        # Navigation actions
        outputs['navigation_actions'] = transformer_outputs['navigation_actions'].squeeze(1)

        # Path planning
        outputs['global_path'] = self._plan_global_path(combined_sensor_features, goal)
        outputs['local_path'] = transformer_outputs['path_predictions'].squeeze(1)

        # Collision avoidance
        outputs['collision_probabilities'] = transformer_outputs['collision_probabilities'].squeeze(1)

        # Localization
        outputs['position_estimate'] = transformer_outputs['localization_estimates'].squeeze(1)

        # Dynamic obstacle prediction
        outputs['predicted_obstacles'] = self.dynamic_obstacle_predictor(combined_sensor_features)

        return outputs

    def _plan_global_path(self, features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Plan global path using learned planner.

        Args:
            features: Combined sensor features
            goal: Goal position (B, 3)

        Returns:
            Global path waypoints (B, 100, 2)
        """
        # Combine features with goal
        combined_input = torch.cat([features, goal], dim=1)  # (B, hidden_dim + 3)

        # Generate waypoints
        waypoints_flat = self.global_planner(combined_input)  # (B, 100)
        # Reshape to waypoints format
        waypoints = waypoints_flat.view(-1, 50, 2)  # (B, 50, 2) - 50 waypoints with 2D coordinates

        return waypoints

    def navigate_to_goal(self,
                        current_state: Dict[str, torch.Tensor],
                        goal: torch.Tensor,
                        map_data: torch.Tensor) -> Dict[str, Any]:
        """
        Navigate to goal using Isaac Navigator.

        Args:
            current_state: Current robot state with sensor data
            goal: Goal position (3,) - x, y, theta
            map_data: Current map data

        Returns:
            Navigation decision with action and confidence
        """
        self.eval()

        with torch.no_grad():
            # Prepare inputs
            vision_data = current_state['vision'].unsqueeze(0)  # Add batch dimension
            lidar_data = current_state['lidar'].unsqueeze(0)
            imu_data = current_state['imu'].unsqueeze(0)
            goal_tensor = goal.unsqueeze(0)

            # Get navigation outputs
            outputs = self.forward(
                vision_data=vision_data,
                lidar_data=lidar_data,
                imu_data=imu_data,
                map_data=map_data.unsqueeze(0),
                goal=goal_tensor
            )

            # Determine navigation action
            navigation_action = outputs['navigation_actions'].squeeze(0)
            collision_prob = outputs['collision_probabilities'].squeeze(0)
            predicted_path = outputs['local_path'].squeeze(0)

            # Apply obstacle avoidance if collision is likely
            if collision_prob > 0.5:
                # Use local obstacle avoidance
                obstacle_avoidance_input = torch.cat([
                    outputs['features'].squeeze(0),
                    current_state['lidar']
                ], dim=0)
                obstacle_action = self.obstacle_avoidance(obstacle_avoidance_input)
                navigation_action = 0.7 * navigation_action + 0.3 * obstacle_action

            # Ensure action is within bounds
            navigation_action = torch.clamp(navigation_action, -1.0, 1.0)

            return {
                'action': navigation_action,
                'collision_probability': collision_prob.item(),
                'predicted_path': predicted_path,
                'global_path': outputs['global_path'].squeeze(0),
                'position_estimate': outputs['position_estimate'].squeeze(0),
                'predicted_obstacles': outputs['predicted_obstacles'].squeeze(0)
            }

    def update_map_with_sensor_data(self,
                                  current_map: torch.Tensor,
                                  sensor_data: Dict[str, torch.Tensor],
                                  robot_pose: torch.Tensor) -> torch.Tensor:
        """
        Update occupancy map with new sensor data.

        Args:
            current_map: Current occupancy map
            sensor_data: New sensor observations
            robot_pose: Current robot pose (x, y, theta)

        Returns:
            Updated occupancy map
        """
        # This would implement sensor fusion to update the map
        # For now, return the current map
        return current_map

class NavigationSkillLibrary:
    """
    Library of navigation skills learned by Isaac Navigator.
    """
    def __init__(self, navigator_model: IsaacNavigator):
        self.navigator = navigator_model
        self.navigation_skills = {}
        self.skill_embeddings = {}

    def register_navigation_skill(self, skill_name: str, demonstration_data: List[Dict]):
        """
        Register a navigation skill with demonstration data.

        Args:
            skill_name: Name of the navigation skill
            demonstration_data: List of (state, action, goal) tuples
        """
        # Encode the navigation skill
        skill_embedding = self._encode_navigation_skill(demonstration_data)

        self.navigation_skills[skill_name] = {
            'demonstrations': demonstration_data,
            'embedding': skill_embedding,
            'success_rate': 0.0,
            'usage_count': 0
        }

        self.skill_embeddings[skill_name] = skill_embedding

    def _encode_navigation_skill(self, demonstration_data: List[Dict]) -> torch.Tensor:
        """
        Encode a navigation skill using the navigator's representation.
        """
        # This would use the navigator model to extract skill features
        # For now, return a placeholder
        return torch.randn(self.navigator.hidden_dim)

    def execute_navigation_skill(self, skill_name: str,
                               current_state: Dict[str, torch.Tensor],
                               goal: torch.Tensor,
                               map_data: torch.Tensor) -> Dict[str, Any]:
        """
        Execute a registered navigation skill.

        Args:
            skill_name: Name of the skill to execute
            current_state: Current robot state
            goal: Navigation goal
            map_data: Current map

        Returns:
            Navigation decision with skill-specific parameters
        """
        if skill_name not in self.navigation_skills:
            raise ValueError(f"Navigation skill '{skill_name}' not found")

        # Use the navigator to execute the skill
        navigation_decision = self.navigator.navigate_to_goal(
            current_state, goal, map_data
        )

        # Adapt the decision based on the specific skill
        adapted_decision = self._adapt_decision_to_skill(
            navigation_decision, skill_name
        )

        return adapted_decision

    def _adapt_decision_to_skill(self, base_decision: Dict[str, Any],
                                skill_name: str) -> Dict[str, Any]:
        """
        Adapt base navigation decision to specific skill requirements.
        """
        # Modify parameters based on skill type
        adapted_decision = base_decision.copy()

        if skill_name == 'exploration':
            # For exploration, prioritize visiting unknown areas
            adapted_decision['action'] *= 1.2  # Be more exploratory
        elif skill_name == 'efficient_navigation':
            # For efficient navigation, follow path more closely
            adapted_decision['action'] *= 0.8  # Be more conservative
        elif skill_name == 'cautious_navigation':
            # For cautious navigation, be more careful around obstacles
            adapted_decision['collision_probability'] *= 0.7  # Lower threshold

        return adapted_decision

class IsaacNavigatorTrainer:
    """
    Trainer for Isaac Navigator model.
    """
    def __init__(self,
                 model: IsaacNavigator,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.navigation_loss = nn.MSELoss()
        self.path_loss = nn.MSELoss()
        self.collision_loss = nn.BCELoss()
        self.localization_loss = nn.MSELoss()

        # Training metrics
        self.training_losses = []
        self.navigation_success_rates = []
        self.path_accuracy = []

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   task_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch with navigation data
            task_weights: Weights for different loss components

        Returns:
            Dictionary of loss values
        """
        if task_weights is None:
            task_weights = {
                'navigation': 1.0,
                'path': 1.0,
                'collision': 0.5,
                'localization': 1.0
            }

        self.model.train()
        self.optimizer.zero_grad()

        vision_data = batch['vision_data'].to(self.device)
        lidar_data = batch['lidar_data'].to(self.device)
        imu_data = batch['imu_data'].to(self.device)
        map_data = batch['map_data'].to(self.device)
        goal = batch['goal'].to(self.device)
        actions = batch['actions'].to(self.device)
        target_path = batch.get('target_path', None)
        if target_path is not None:
            target_path = target_path.to(self.device)
        collision_labels = batch.get('collision_labels', None)
        if collision_labels is not None:
            collision_labels = collision_labels.to(self.device).float()
        position_labels = batch.get('position_labels', None)
        if position_labels is not None:
            position_labels = position_labels.to(self.device)

        # Forward pass
        outputs = self.model(vision_data, lidar_data, imu_data, map_data, goal)

        # Compute losses
        total_loss = 0.0
        losses = {}

        # Navigation action loss
        navigation_loss = self.navigation_loss(outputs['navigation_actions'], actions)
        losses['navigation_loss'] = navigation_loss.item()
        total_loss += task_weights['navigation'] * navigation_loss

        # Path planning loss
        if target_path is not None:
            path_loss = self.path_loss(outputs['local_path'], target_path)
            losses['path_loss'] = path_loss.item()
            total_loss += task_weights['path'] * path_loss

        # Collision prediction loss
        if collision_labels is not None:
            collision_loss = self.collision_loss(outputs['collision_probabilities'], collision_labels)
            losses['collision_loss'] = collision_loss.item()
            total_loss += task_weights['collision'] * collision_loss

        # Localization loss
        if position_labels is not None:
            localization_loss = self.localization_loss(outputs['position_estimate'], position_labels)
            losses['localization_loss'] = localization_loss.item()
            total_loss += task_weights['localization'] * localization_loss

        losses['total_loss'] = total_loss.item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return losses

    def evaluate_navigation(self,
                          eval_loader: torch.utils.data.DataLoader,
                          max_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate navigation performance.

        Args:
            eval_loader: Evaluation data loader
            max_episodes: Maximum episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        episode_rewards = []
        success_rates = []
        path_efficiencies = []

        with torch.no_grad():
            for episode_idx, batch in enumerate(eval_loader):
                if episode_idx >= max_episodes:
                    break

                vision_data = batch['vision_data'].to(self.device)
                lidar_data = batch['lidar_data'].to(self.device)
                imu_data = batch['imu_data'].to(self.device)
                map_data = batch['map_data'].to(self.device)
                goal = batch['goal'].to(self.device)
                target_actions = batch['actions'].to(self.device)
                target_positions = batch.get('target_positions', None)
                if target_positions is not None:
                    target_positions = target_positions.to(self.device)

                # Get predictions
                outputs = self.model(vision_data, lidar_data, imu_data, map_data, goal)

                # Calculate navigation metrics
                action_error = F.mse_loss(outputs['navigation_actions'], target_actions).item()
                episode_rewards.append(-action_error)  # Negative because lower error is better

                # Calculate success based on reaching goal
                if target_positions is not None:
                    current_positions = outputs['position_estimate']
                    distances_to_goal = torch.norm(current_positions - target_positions, dim=1)
                    success = (distances_to_goal < 0.5).float().mean().item()  # 0.5m threshold
                    success_rates.append(success)

                    # Calculate path efficiency (ratio of straight-line to actual path length)
                    straight_line_distance = torch.norm(goal[:, :2] - batch['start_positions'][:, :2], dim=1)
                    actual_path_length = self._calculate_path_length(outputs['local_path'])
                    efficiency = (straight_line_distance / actual_path_length).mean().item()
                    path_efficiencies.append(efficiency)

        metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'action_error_mean': -np.mean(episode_rewards),
            'num_episodes': len(episode_rewards)
        }

        if success_rates:
            metrics['success_rate'] = np.mean(success_rates)
            metrics['success_std'] = np.std(success_rates)

        if path_efficiencies:
            metrics['path_efficiency'] = np.mean(path_efficiencies)
            metrics['path_efficiency_std'] = np.std(path_efficiencies)

        return metrics

    def _calculate_path_length(self, path: torch.Tensor) -> torch.Tensor:
        """Calculate the length of a path."""
        # Path shape: (B, num_waypoints, 2)
        diff = path[:, 1:, :] - path[:, :-1, :]  # Differences between consecutive points
        distances = torch.norm(diff, dim=2)  # Distance between consecutive points
        path_lengths = torch.sum(distances, dim=1)  # Total length for each path
        return path_lengths
```

## Isaac Foundation Model Integration

### Unified Isaac Foundation Model System

```python
# isaac_foundation_unified.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class IsaacFoundationConfig:
    """
    Configuration for Isaac Foundation Model system.
    """
    # Model dimensions
    hidden_dim: int = 512
    action_dim: int = 7
    sensor_dim: int = 360
    language_dim: int = 512

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_episode_steps: int = 1000

    # Foundation model weights
    perception_weight: float = 1.0
    control_weight: float = 1.0
    manipulation_weight: float = 1.0
    navigation_weight: float = 1.0

    # Domain randomization
    domain_randomization: bool = True
    randomization_frequency: int = 100

    # Continual learning
    continual_learning: bool = True
    memory_size: int = 10000

class IsaacFoundationModel(nn.Module):
    """
    Unified Isaac Foundation Model combining perception, control, manipulation, and navigation.
    """
    def __init__(self, config: IsaacFoundationConfig):
        super().__init__()

        self.config = config

        # Shared feature extractor
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, config.hidden_dim),
            nn.ReLU()
        )

        self.sensor_encoder = nn.Sequential(
            nn.Linear(config.sensor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.language_encoder = nn.Sequential(
            nn.Linear(config.language_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Task-specific encoders
        self.perception_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.control_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.manipulation_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.navigation_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Shared transformer for temporal processing
        self.shared_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Task-specific decoders
        self.perception_decoder = IsaacPerceptor(
            hidden_dim=config.hidden_dim,
            output_dim=1000  # For object detection, segmentation, etc.
        )

        self.control_decoder = IsaacController(
            action_space_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        self.manipulation_decoder = IsaacManipulator(
            robot_state_dim=config.action_dim * 2,  # Positions + velocities
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        self.navigation_decoder = IsaacNavigator(
            sensor_dim=config.sensor_dim,
            action_dim=2,  # x, y velocity
            hidden_dim=config.hidden_dim
        )

        # Task router for selecting appropriate decoder
        self.task_router = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4),  # 4 tasks: perception, control, manipulation, navigation
            nn.Softmax(dim=1)
        )

        # Memory for continual learning
        if config.continual_learning:
            self.memory_buffer = {
                'perception': [],
                'control': [],
                'manipulation': [],
                'navigation': []
            }
            self.memory_size = config.memory_size

    def forward(self,
                images: torch.Tensor,
                sensors: torch.Tensor,
                language: torch.Tensor,
                task_type: str = 'control',
                robot_state: Optional[torch.Tensor] = None,
                goal: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified Isaac Foundation Model.

        Args:
            images: Input images (B, C, H, W)
            sensors: Sensor data (B, sensor_dim)
            language: Language embeddings (B, language_dim)
            task_type: Type of task ('perception', 'control', 'manipulation', 'navigation')
            robot_state: Current robot state (B, state_dim) - for manipulation/navigation
            goal: Goal position (B, 3) - for navigation

        Returns:
            Dictionary of task-specific outputs
        """
        batch_size = images.size(0)

        # Extract features from different modalities
        vision_features = self.vision_encoder(images)  # (B, hidden_dim)
        sensor_features = self.sensor_encoder(sensors)  # (B, hidden_dim)
        language_features = self.language_encoder(language)  # (B, hidden_dim)

        # Combine features
        combined_features = torch.cat([vision_features, sensor_features, language_features], dim=1)  # (B, 3*hidden_dim)

        # Select appropriate encoder based on task
        if task_type == 'perception':
            task_features = self.perception_encoder(combined_features)
        elif task_type == 'control':
            task_features = self.control_encoder(combined_features)
        elif task_type == 'manipulation':
            task_features = self.manipulation_encoder(combined_features)
        elif task_type == 'navigation':
            task_features = self.navigation_encoder(combined_features)
        else:
            # Use task router to determine most appropriate task
            task_probs = self.task_router(combined_features)
            task_idx = torch.argmax(task_probs, dim=1)  # (B,)

            # For simplicity, use the first task type in batch
            task_type_mapping = ['perception', 'control', 'manipulation', 'navigation']
            task_type = task_type_mapping[task_idx[0].item()]

            # Get features for selected task
            if task_type == 'perception':
                task_features = self.perception_encoder(combined_features)
            elif task_type == 'control':
                task_features = self.control_encoder(combined_features)
            elif task_type == 'manipulation':
                task_features = self.manipulation_encoder(combined_features)
            else:  # navigation
                task_features = self.navigation_encoder(combined_features)

        # Apply shared transformer for temporal processing
        task_features = task_features.unsqueeze(1)  # (B, 1, hidden_dim)
        task_features = self.shared_transformer(task_features)
        task_features = task_features.squeeze(1)  # (B, hidden_dim)

        # Apply task-specific decoder
        if task_type == 'perception':
            outputs = self.perception_decoder(
                images=images,
                text_inputs=language,
                return_features=True
            )
        elif task_type == 'control':
            outputs = self.control_decoder(
                images=images,
                text_inputs=language,
                robot_state=robot_state if robot_state is not None else torch.zeros(batch_size, self.config.action_dim * 2)
            )
        elif task_type == 'manipulation':
            outputs = self.manipulation_decoder(
                image=images,
                language_embedding=language,
                robot_state=robot_state if robot_state is not None else torch.zeros(batch_size, self.config.action_dim * 2)
            )
        elif task_type == 'navigation':
            # Create dummy map data for navigation (in practice, this would come from SLAM)
            dummy_map = torch.zeros(batch_size, 1, 100, 100).to(images.device)
            outputs = self.navigation_decoder(
                vision_data=images,
                lidar_data=sensors,
                imu_data=torch.zeros(batch_size, 6).to(images.device),  # Dummy IMU
                map_data=dummy_map,
                goal=goal if goal is not None else torch.zeros(batch_size, 3).to(images.device)
            )

        # Add task probability information
        if 'task_probs' not in outputs:
            task_probs = self.task_router(combined_features)
            outputs['task_probs'] = task_probs
            outputs['selected_task'] = task_type

        return outputs

    def add_to_memory(self, task_type: str, experience: Dict[str, torch.Tensor]):
        """Add experience to memory buffer for continual learning."""
        if not self.config.continual_learning:
            return

        buffer = self.memory_buffer[task_type]
        buffer.append(experience)

        # Keep memory size within limits
        if len(buffer) > self.memory_size:
            buffer.pop(0)  # Remove oldest experience

    def get_memory_batch(self, task_type: str, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch from memory buffer."""
        if not self.config.continual_learning:
            return None

        buffer = self.memory_buffer[task_type]
        if len(buffer) < batch_size:
            return None

        # Sample random experiences
        indices = np.random.choice(len(buffer), batch_size, replace=False)
        batch = {key: [] for key in buffer[0].keys()}

        for idx in indices:
            experience = buffer[idx]
            for key, value in experience.items():
                batch[key].append(value)

        # Stack into tensors
        for key in batch:
            batch[key] = torch.stack(batch[key])

        return batch

class IsaacFoundationTrainer:
    """
    Trainer for Isaac Foundation Model system.
    """
    def __init__(self, model: IsaacFoundationModel, config: IsaacFoundationConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Separate optimizers for different components
        self.perception_optimizer = torch.optim.AdamW(
            list(self.model.perception_decoder.parameters()) +
            list(self.model.perception_encoder.parameters()),
            lr=config.learning_rate
        )

        self.control_optimizer = torch.optim.AdamW(
            list(self.model.control_decoder.parameters()) +
            list(self.model.control_encoder.parameters()),
            lr=config.learning_rate
        )

        self.manipulation_optimizer = torch.optim.AdamW(
            list(self.model.manipulation_decoder.parameters()) +
            list(self.model.manipulation_encoder.parameters()),
            lr=config.learning_rate
        )

        self.navigation_optimizer = torch.optim.AdamW(
            list(self.model.navigation_decoder.parameters()) +
            list(self.model.navigation_encoder.parameters()),
            lr=config.learning_rate
        )

        # Shared components optimizer
        self.shared_optimizer = torch.optim.AdamW(
            list(self.model.vision_encoder.parameters()) +
            list(self.model.sensor_encoder.parameters()) +
            list(self.model.language_encoder.parameters()) +
            list(self.model.shared_transformer.parameters()) +
            list(self.model.task_router.parameters()),
            lr=config.learning_rate * 0.1  # Lower LR for shared components
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

        # Training metrics
        self.training_losses = {
            'perception': [],
            'control': [],
            'manipulation': [],
            'navigation': []
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step for the unified model.

        Args:
            batch: Training batch with data for different tasks

        Returns:
            Dictionary of loss values for each component
        """
        task_type = batch['task_type']
        losses = {}

        if task_type == 'perception':
            losses.update(self._train_perception_step(batch))
        elif task_type == 'control':
            losses.update(self._train_control_step(batch))
        elif task_type == 'manipulation':
            losses.update(self._train_manipulation_step(batch))
        elif task_type == 'navigation':
            losses.update(self._train_navigation_step(batch))

        # Update shared components
        self._train_shared_components(batch, task_type)

        return losses

    def _train_perception_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train perception component."""
        self.model.train()

        images = batch['images'].to(self.device)
        language_tokens = batch['language_tokens'].to(self.device)
        targets = batch['perception_targets'].to(self.device)

        # Forward pass
        outputs = self.model(images, torch.zeros_like(batch['sensors']), language_tokens, 'perception')

        # Compute loss
        detection_loss = self.mse_loss(outputs['detection']['bbox'], targets['bbox'])
        classification_loss = self.ce_loss(outputs['detection']['class_logits'], targets['class_labels'])

        total_loss = detection_loss + classification_loss

        # Backward pass
        self.perception_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.perception_decoder.parameters(), max_norm=1.0)
        self.perception_optimizer.step()

        return {
            'perception_detection_loss': detection_loss.item(),
            'perception_classification_loss': classification_loss.item(),
            'perception_total_loss': total_loss.item()
        }

    def _train_control_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train control component."""
        self.model.train()

        images = batch['images'].to(self.device)
        language_tokens = batch['language_tokens'].to(self.device)
        robot_states = batch['robot_states'].to(self.device)
        actions = batch['actions'].to(self.device)

        # Forward pass
        outputs = self.model(images, torch.zeros_like(batch['sensors']), language_tokens, 'control', robot_states)

        # Compute loss
        action_loss = self.mse_loss(outputs['actions']['manipulation'], actions)

        # Backward pass
        self.control_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.control_decoder.parameters(), max_norm=1.0)
        self.control_optimizer.step()

        return {
            'control_action_loss': action_loss.item()
        }

    def _train_manipulation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train manipulation component."""
        self.model.train()

        images = batch['images'].to(self.device)
        language_tokens = batch['language_tokens'].to(self.device)
        robot_states = batch['robot_states'].to(self.device)
        actions = batch['manipulation_actions'].to(self.device)

        # Forward pass
        outputs = self.model(images, torch.zeros_like(batch['sensors']), language_tokens, 'manipulation', robot_states)

        # Compute loss
        action_loss = self.mse_loss(outputs['actions'], actions)

        # Backward pass
        self.manipulation_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.manipulation_decoder.parameters(), max_norm=1.0)
        self.manipulation_optimizer.step()

        return {
            'manipulation_action_loss': action_loss.item()
        }

    def _train_navigation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train navigation component."""
        self.model.train()

        images = batch['images'].to(self.device)
        sensors = batch['sensors'].to(self.device)
        language_tokens = batch['language_tokens'].to(self.device)
        goals = batch['goals'].to(self.device)
        actions = batch['navigation_actions'].to(self.device)

        # Forward pass
        outputs = self.model(images, sensors, language_tokens, 'navigation', goal=goals)

        # Compute loss
        action_loss = self.mse_loss(outputs['navigation_actions'], actions)

        # Backward pass
        self.navigation_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.navigation_decoder.parameters(), max_norm=1.0)
        self.navigation_optimizer.step()

        return {
            'navigation_action_loss': action_loss.item()
        }

    def _train_shared_components(self, batch: Dict[str, Any], task_type: str):
        """Train shared components using memory replay."""
        # Sample from memory for other tasks to prevent forgetting
        for other_task in ['perception', 'control', 'manipulation', 'navigation']:
            if other_task != task_type and len(self.model.memory_buffer[other_task]) > 0:
                memory_batch = self.model.get_memory_batch(other_task, self.config.batch_size // 4)
                if memory_batch is not None:
                    # Compute loss on memory batch to prevent forgetting
                    # This is a simplified approach - in practice, you'd implement proper rehearsal
                    pass

        # Update shared components with current batch
        self.shared_optimizer.zero_grad()
        # Shared loss would be computed based on task router accuracy or other metrics
        self.shared_optimizer.step()

    def evaluate_foundation_model(self,
                                eval_loaders: Dict[str, torch.utils.data.DataLoader],
                                max_episodes: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the unified foundation model across all tasks.

        Args:
            eval_loaders: Dictionary of evaluation loaders for each task
            max_episodes: Maximum episodes per task

        Returns:
            Dictionary of evaluation metrics for each task
        """
        self.model.eval()
        all_metrics = {}

        for task_type, eval_loader in eval_loaders.items():
            task_metrics = self._evaluate_task(eval_loader, task_type, max_episodes)
            all_metrics[task_type] = task_metrics

        # Compute overall metrics
        overall_success_rate = np.mean([metrics.get('success_rate', 0.0) for metrics in all_metrics.values()])
        all_metrics['overall'] = {'success_rate': overall_success_rate}

        return all_metrics

    def _evaluate_task(self,
                      eval_loader: torch.utils.data.DataLoader,
                      task_type: str,
                      max_episodes: int) -> Dict[str, float]:
        """Evaluate a specific task."""
        episode_rewards = []
        success_rates = []

        with torch.no_grad():
            for episode_idx, batch in enumerate(eval_loader):
                if episode_idx >= max_episodes:
                    break

                # Prepare inputs based on task type
                images = batch['images'].to(self.device)
                sensors = batch.get('sensors', torch.zeros_like(batch['images'][:, :1, :1, :360])).to(self.device)
                language_tokens = batch['language_tokens'].to(self.device)

                if task_type in ['control', 'manipulation']:
                    robot_states = batch['robot_states'].to(self.device)
                    targets = batch['actions'].to(self.device)

                    outputs = self.model(images, sensors, language_tokens, task_type, robot_states)
                elif task_type == 'navigation':
                    goals = batch['goals'].to(self.device)
                    targets = batch['navigation_actions'].to(self.device)

                    outputs = self.model(images, sensors, language_tokens, task_type, goal=goals)
                else:  # perception
                    targets = batch['perception_targets'].to(self.device)

                    outputs = self.model(images, sensors, language_tokens, task_type)

                # Calculate task-specific metrics
                if task_type == 'perception':
                    # Calculate detection accuracy, IoU, etc.
                    pred_boxes = outputs['detection']['bbox']
                    target_boxes = targets['bbox']
                    iou_scores = self._calculate_bbox_iou(pred_boxes, target_boxes)
                    success_rate = (iou_scores > 0.5).float().mean().item()
                else:
                    # Calculate action accuracy
                    action_error = F.mse_loss(outputs['actions'], targets).item()
                    success_rate = max(0.0, 1.0 - action_error)  # Convert error to success rate

                episode_rewards.append(-action_error if task_type != 'perception' else -iou_scores.mean().item())
                success_rates.append(success_rate)

        return {
            'avg_reward': np.mean(episode_rewards),
            'success_rate': np.mean(success_rates),
            'num_episodes': len(episode_rewards)
        }

    def save_model(self, filepath: str):
        """Save the complete foundation model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'perception_optimizer_state_dict': self.perception_optimizer.state_dict(),
            'control_optimizer_state_dict': self.control_optimizer.state_dict(),
            'manipulation_optimizer_state_dict': self.manipulation_optimizer.state_dict(),
            'navigation_optimizer_state_dict': self.navigation_optimizer.state_dict(),
            'shared_optimizer_state_dict': self.shared_optimizer.state_dict(),
            'config': self.config,
            'training_losses': self.training_losses
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the complete foundation model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.perception_optimizer.load_state_dict(checkpoint['perception_optimizer_state_dict'])
        self.control_optimizer.load_state_dict(checkpoint['control_optimizer_state_dict'])
        self.manipulation_optimizer.load_state_dict(checkpoint['manipulation_optimizer_state_dict'])
        self.navigation_optimizer.load_state_dict(checkpoint['navigation_optimizer_state_dict'])
        self.shared_optimizer.load_state_dict(checkpoint['shared_optimizer_state_dict'])
        self.config = checkpoint['config']
        self.training_losses = checkpoint['training_losses']

        print(f"Model loaded from {filepath}")

def deploy_isaac_foundation_model(model_path: str,
                                task_type: str,
                                device: str = 'cuda') -> IsaacFoundationModel:
    """
    Deploy Isaac Foundation Model for real-world use.

    Args:
        model_path: Path to saved model
        task_type: Type of task ('perception', 'control', 'manipulation', 'navigation')
        device: Device to deploy on

    Returns:
        Loaded and ready-to-use model
    """
    # Load configuration
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Create model
    model = IsaacFoundationModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Isaac Foundation Model deployed for {task_type} tasks")
    print(f"Model loaded on device: {device}")

    return model

# Example usage and demonstration
def main():
    """Demonstrate Isaac Foundation Model system."""
    print("Initializing Isaac Foundation Model System...")

    # Create configuration
    config = IsaacFoundationConfig(
        hidden_dim=512,
        action_dim=7,
        sensor_dim=360,
        language_dim=512,
        learning_rate=1e-4,
        domain_randomization=True,
        continual_learning=True
    )

    # Create model
    foundation_model = IsaacFoundationModel(config)

    # Create trainer
    trainer = IsaacFoundationTrainer(foundation_model, config)

    print(f"Isaac Foundation Model created with {config.hidden_dim} hidden dimensions")
    print(f"Supporting tasks: perception, control, manipulation, navigation")
    print(f"Domain randomization: {config.domain_randomization}")
    print(f"Continual learning: {config.continual_learning}")

    # Example of how to use the model
    print("\nExample usage:")
    print("# Train on different tasks sequentially")
    print("# Use domain randomization for sim-to-real transfer")
    print("# Apply continual learning to prevent forgetting")
    print("# Deploy to real robot with safety mechanisms")

    print("\nIsaac Foundation Model system initialized successfully!")

if __name__ == "__main__":
    main()