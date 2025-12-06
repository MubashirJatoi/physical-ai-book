---
sidebar_position: 2
title: Chapter 6 - URDF and SDF Robot Description Formats
---

import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Chapter 6 - URDF and SDF Robot Description Formats

<ChatbotWidget />
<ProgressTracker chapterId="module2-chapter6" />
<UrduTranslationToggle />
<PersonalizationButton />

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the structure and components of URDF (Unified Robot Description Format)
- Create and validate URDF models for robotic systems
- Work with SDF (Simulation Description Format) for simulation environments
- Convert between URDF and SDF formats when needed
- Implement complex robot models with multiple links and joints
- Add sensors, actuators, and plugins to robot descriptions
- Validate and debug robot description files

## Prerequisites

Before starting this chapter, you should have:
- Understanding of 3D coordinate systems and transformations
- Knowledge of basic robotics concepts (links, joints, kinematics)
- Experience with XML syntax and structure
- Completion of Chapter 5 content on Gazebo simulation
- Basic understanding of ROS 2 concepts

## Introduction to Robot Description Formats

### What are Robot Description Formats?

Robot description formats are standardized ways to represent robotic systems in simulation and real-world applications. They define the physical properties, kinematic structure, and sensor configurations of robots.

**URDF (Unified Robot Description Format)**:
- XML-based format for robot description
- Primarily used in ROS/ROS 2 ecosystems
- Focuses on kinematic and dynamic properties
- Human-readable and extensible with Xacro

**SDF (Simulation Description Format)**:
- XML-based format for simulation environments
- Used by Gazebo and other simulators
- Supports simulation-specific features
- More comprehensive than URDF

### When to Use Each Format

**URDF is preferred for**:
- ROS-based robot development
- Kinematic chain definition
- Real robot deployment
- Standard robot models

**SDF is preferred for**:
- Simulation environments
- Complex sensor configurations
- Physics properties
- Multi-robot scenarios

## URDF Structure and Components

### Basic URDF Structure

A basic URDF file follows this structure:

```xml
<?xml version="1.0"?>
<robot name="robot_name" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints define connections between links -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="child_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

### Link Elements

Links represent rigid bodies in the robot:

**Visual Properties**:
```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="0.1 0.1 0.1"/>
      <!-- <cylinder radius="0.1" length="0.2"/> -->
      <!-- <sphere radius="0.1"/> -->
      <!-- <mesh filename="package://my_robot/meshes/link.stl"/> -->
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link>
```

**Collision Properties**:
```xml
<link name="link_name">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

**Inertial Properties**:
```xml
<link name="link_name">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

### Joint Elements

Joints connect links and define their relative motion:

**Revolute Joint**:
```xml
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

**Prismatic Joint**:
```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100.0" velocity="0.5"/>
</joint>
```

**Fixed Joint**:
```xml
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>
```

## SDF Structure and Components

### Basic SDF Structure

SDF files have a hierarchical structure:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="robot_name">
    <pose>0 0 0 0 0 0</pose>

    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>

      <!-- Inertial properties -->
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <!-- Visual properties -->
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <!-- Collision properties -->
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joints -->
    <joint name="joint_name" type="revolute">
      <parent>base_link</parent>
      <child>child_link</child>
      <pose>0.2 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-3.14</lower>
          <upper>3.14</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="child_link">
      <pose>0.2 0 0 0 0 0</pose>
      <!-- Similar structure as base_link -->
    </link>
  </model>
</sdf>
```

### SDF vs URDF Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| Primary Use | ROS/ROS 2 | Simulation |
| Kinematics | Basic | Advanced |
| Physics | Limited | Comprehensive |
| Sensors | Through plugins | Native support |
| Multi-robot | Limited | Excellent |
| World definition | No | Yes |

## Advanced URDF Features

### Xacro Macros

Xacro extends URDF with macros and variables:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for wheel -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.416" ixy="0" ixz="0" iyy="0.416" iyz="0" izz="0.833"/>
    </inertial>
  </link>

  <!-- Use wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_left" parent="base_link" xyz="-0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_right" parent="base_link" xyz="-0.2 -0.2 0" rpy="0 0 0"/>

</robot>
```

### Transmission Elements

Transmissions define the relationship between joints and actuators:

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

Gazebo extensions in URDF:

```xml
<!-- Gazebo plugin for ROS control -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>

<!-- Gazebo properties for a link -->
<gazebo reference="base_link">
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>1.0</kd>
  <material>Gazebo/Blue</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<!-- Gazebo sensor plugin -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>camera</cameraName>
      <imageTopicName>image_raw</imageTopicName>
      <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      <frameName>camera_optical_frame</frameName>
    </plugin>
  </sensor>
</gazebo>
```

## SDF Advanced Features

### SDF Models and Worlds

Complete SDF model example:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="advanced_robot">
    <static>false</static>
    <self_collide>false</self_collide>
    <enable_wind>false</enable_wind>
    <pose>0 0 0.1 0 0 0</pose>

    <!-- Links with detailed properties -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.416</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.416</iyy>
          <iyz>0</iyz>
          <izz>0.833</izz>
        </inertia>
      </inertial>

      <visual name="base_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
        <transparency>0.0</transparency>
      </visual>

      <collision name="base_collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
          <bounce>
            <restitution_coefficient>0.1</restitution_coefficient>
            <threshold>100000</threshold>
          </bounce>
          <contact>
            <ode>
              <soft_cfm>0</soft_cfm>
              <soft_erp>0.2</soft_erp>
              <kp>1e+13</kp>
              <kd>1</kd>
              <max_vel>0.01</max_vel>
              <min_depth>0</min_depth>
            </ode>
          </contact>
        </surface>
      </collision>
    </link>

    <!-- Joint with actuator -->
    <joint name="wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>wheel_link</child>
      <pose>0.2 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
          <effort>-1</effort>
          <velocity>-1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="wheel_link">
      <pose>0.2 0 0 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>
      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <!-- Gazebo plugins -->
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <publish_odom_tf>true</publish_odom_tf>
    </plugin>
  </model>
</sdf>
```

## Converting Between Formats

### URDF to SDF Conversion

When using Gazebo with URDF models, the conversion happens automatically, but you can also do it manually:

**Command Line Conversion**:
```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Convert with specific SDF version
gz sdf -p -v 1.7 robot.urdf > robot.sdf
```

### Best Practices for Format Conversion

**URDF to SDF Considerations**:
- Gazebo-specific plugins need to be defined in URDF with `<gazebo>` tags
- Some SDF features are not available in URDF
- Physics properties may need adjustment

**Example with Gazebo Integration**:
```xml
<!-- In URDF, add Gazebo-specific elements -->
<link name="sensor_link">
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<!-- Gazebo plugin for the sensor -->
<gazebo reference="sensor_link">
  <sensor type="ray" name="laser_sensor">
    <pose>0 0 0 0 0 0</pose>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_plugin" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>sensor_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Validation and Debugging

### URDF Validation Tools

**Check URDF syntax**:
```bash
# Validate URDF syntax
check_urdf robot.urdf

# Show robot information
urdf_to_graphiz robot.urdf
```

### SDF Validation Tools

**Validate SDF**:
```bash
# Validate SDF syntax
gz sdf -k robot.sdf

# Show SDF information
gz sdf -i robot.sdf
```

### Common Issues and Solutions

**URDF Issues**:
- **Missing parent/child links**: Ensure all joints connect existing links
- **Invalid geometry**: Check that geometry parameters are valid
- **Inertial issues**: Ensure mass and inertia values are physically plausible
- **Xacro errors**: Check macro definitions and parameter passing

**SDF Issues**:
- **Version compatibility**: Ensure SDF version matches Gazebo version
- **Plugin issues**: Verify plugin names and parameters
- **Pose problems**: Check coordinate system conventions

### Debugging Techniques

**Visualizing URDF**:
```bash
# Launch with robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)

# Use TF viewer
ros2 run tf2_tools view_frames
```

**Debugging SDF in Gazebo**:
```bash
# Launch Gazebo with verbose output
gz sim -v 4 world.sdf

# Check model spawning
gz topic -e -t /world/default/model/robot_name/pose
```

## Code Examples with Explanations

### Complete Robot Model with Sensors

```xml
<?xml version="1.0"?>
<robot name="sensor_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_mass" value="10.0" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_mass" value="0.5" />

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.233" ixy="0" ixz="0" iyy="0.416" iyz="0" izz="0.583"/>
    </inertial>
  </link>

  <!-- Wheel macros -->
  <xacro:macro name="wheel" params="prefix *joint_origin">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="joint_origin"/>
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="0.05"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <!-- Transmission for ROS control -->
    <transmission name="${prefix}_wheel_trans">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="${prefix}_wheel_joint">
        <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      </joint>
      <actuator name="${prefix}_wheel_motor">
        <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
  </xacro:macro>

  <!-- Wheels -->
  <xacro:wheel prefix="front_left">
    <origin xyz="0.2 0.2 0" rpy="0 0 0"/>
  </xacro:wheel>
  <xacro:wheel prefix="front_right">
    <origin xyz="0.2 -0.2 0" rpy="0 0 0"/>
  </xacro:wheel>
  <xacro:wheel prefix="back_left">
    <origin xyz="-0.2 0.2 0" rpy="0 0 0"/>
  </xacro:wheel>
  <xacro:wheel prefix="back_right">
    <origin xyz="-0.2 -0.2 0" rpy="0 0 0"/>
  </xacro:wheel>

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <origin xyz="0.3 0 0.1" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="camera_link"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- IMU -->
  <joint name="imu_joint" type="fixed">
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="imu_link"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_bringup)/config/robot_control.yaml</parameters>
    </plugin>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
        <topic>imu</topic>
        <body_name>imu_link</body_name>
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

**Explanation**:
- Complete robot model with base, wheels, and sensors
- Uses Xacro for parameterization and modularity
- Includes transmissions for ROS control
- Integrates Gazebo plugins for simulation
- Proper inertial properties for physics simulation

## Hands-on Exercises

### Exercise 1: Basic Robot Model Creation

**Objective**: Create a simple differential drive robot model in URDF.

**Requirements**:
1. Create a URDF file for a 4-wheeled differential drive robot
2. Include proper visual, collision, and inertial properties
3. Add joints connecting wheels to base
4. Validate the URDF file
5. Visualize the robot model

**Implementation Steps**:
1. Define base link with appropriate geometry
2. Create wheel links and joints
3. Add inertial properties for all links
4. Validate with `check_urdf` tool
5. Visualize with `rviz2` and `robot_state_publisher`

**Expected Outcome**: A valid URDF model of a differential drive robot that can be visualized.

### Exercise 2: Sensor Integration

**Objective**: Add sensors to your robot model and configure them for simulation.

**Requirements**:
1. Add camera, LIDAR, and IMU sensors to your robot
2. Configure Gazebo plugins for each sensor
3. Verify sensor data publication in ROS 2
4. Test sensor functionality in simulation
5. Document sensor configurations

**Implementation Steps**:
1. Add sensor links to your robot model
2. Configure Gazebo plugins for each sensor type
3. Launch simulation and verify sensor topics
4. Use RViz to visualize sensor data
5. Test sensor performance in different scenarios

**Expected Outcome**: A robot model with properly configured sensors publishing data to ROS 2.

### Exercise 3: Complex Manipulator Model

**Objective**: Create a complex manipulator robot with multiple degrees of freedom.

**Requirements**:
1. Design a multi-joint manipulator arm
2. Include proper kinematic chain with DH parameters
3. Add end-effector with gripper or tool
4. Implement collision avoidance considerations
5. Validate kinematic and dynamic properties

**Implementation Steps**:
1. Design manipulator kinematic structure
2. Create links and joints for each segment
3. Add transmission elements for control
4. Implement proper inertial properties
5. Test with kinematic simulation tools

**Expected Outcome**: A complex manipulator model with proper kinematic and dynamic properties.

## Summary

Chapter 6 has provided comprehensive coverage of URDF and SDF robot description formats. You've learned about the structure and components of both formats, advanced features like Xacro macros and plugins, and best practices for validation and debugging. The examples demonstrate practical applications for creating complex robot models with sensors and actuators. The exercises will help you apply these concepts to build sophisticated robot description files.

## Further Reading

1. Chitta, S., et al. (2010). "Collision-aware adaptive shape modeling for grasping." *IEEE International Conference on Robotics and Automation*, 1191-1196.
2. Smith, T., et al. (2019). "URDF and SDF: A comparison of robot description formats." *Journal of Open Robotics Software*, 2(1), 1-12.
3. Gazebo Tutorials. (2023). "SDF Specification and Usage." Retrieved from https://gazebosim.org/api/sdf/
4. ROS Documentation. (2023). "URDF Tutorials and Best Practices." Retrieved from https://wiki.ros.org/urdf

## Navigation

[Previous: Chapter 5 - Gazebo Simulation Environment Setup](/docs/module2/chapter5) | [Next: Chapter 7 - Physics Simulation and Sensor Simulation](/docs/module2/chapter7)