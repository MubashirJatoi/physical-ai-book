---
sidebar_position: 3
title: Chapter 3 - Building ROS 2 Packages with Python
---

import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Chapter 3 - Building ROS 2 Packages with Python

<ProgressTracker chapterId="module1-chapter3" />
<UrduTranslationToggle />
<PersonalizationButton />

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and structure ROS 2 packages using Python
- Implement nodes, publishers, subscribers, services, and actions in Python
- Use ROS 2 client libraries (rclpy) effectively
- Create and use custom message and service types
- Package and distribute ROS 2 Python packages
- Debug and test Python-based ROS 2 nodes

## Prerequisites

Before starting this chapter, you should have:
- Understanding of ROS 2 communication patterns from Chapter 2
- Proficiency in Python programming
- Knowledge of Python packaging and modules
- Completion of Chapter 1 and 2 content

## ROS 2 Package Structure

### Package Organization

A ROS 2 package follows a specific directory structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package metadata
├── setup.py               # Python package configuration
├── setup.cfg              # Installation configuration
├── resource/              # Package resources
├── test/                  # Unit and integration tests
├── my_robot_package/      # Python module
│   ├── __init__.py       # Python package initialization
│   ├── main.py           # Main module entry point
│   ├── nodes/            # Node implementations
│   │   ├── __init__.py
│   │   ├── sensor_node.py
│   │   └── controller_node.py
│   ├── msgs/             # Custom message definitions
│   │   ├── __init__.py
│   │   └── custom_msgs.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── helpers.py
└── launch/                # Launch files
    ├── robot.launch.py
    └── simulation.launch.py
```

### Package.xml Configuration

The `package.xml` file contains package metadata:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py Configuration

The `setup.py` file configures the Python package:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package for my robot functionality',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_node = my_robot_package.nodes.sensor_node:main',
            'controller_node = my_robot_package.nodes.controller_node:main',
        ],
    },
)
```

## Creating ROS 2 Nodes in Python

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class BasicNode(Node):
    def __init__(self):
        super().__init__('basic_node')
        self.get_logger().info('BasicNode initialized')

def main(args=None):
    rclpy.init(args=args)
    node = BasicNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node with Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_threshold', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_threshold = self.get_parameter('safety_threshold').value

        # Create parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(
            f'Robot: {self.robot_name}, Max Vel: {self.max_velocity}, '
            f'Safety Threshold: {self.safety_threshold}'
        )

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.value > 5.0:
                return SetParametersResult(successful=False, reason='Max velocity too high')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publisher and Subscriber Implementation

### Publisher Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import LaserScan
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different message types
        self.string_publisher = self.create_publisher(String, 'robot_status', 10)
        self.float_publisher = self.create_publisher(Float64, 'battery_level', 10)
        self.scan_publisher = self.create_publisher(LaserScan, 'laser_scan', 10)

        # Create timer to publish data periodically
        self.timer = self.create_timer(0.5, self.publish_sensor_data)
        self.counter = 0

        self.get_logger().info('Sensor publisher started')

    def publish_sensor_data(self):
        # Publish string status
        status_msg = String()
        status_msg.data = f'Operating normally - {self.counter}'
        self.string_publisher.publish(status_msg)

        # Publish battery level
        battery_msg = Float64()
        battery_msg.data = 100.0 - (self.counter * 0.1)  # Simulated battery drain
        self.float_publisher.publish(battery_msg)

        # Publish laser scan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -1.57  # -90 degrees
        scan_msg.angle_max = 1.57   # 90 degrees
        scan_msg.angle_increment = 0.0174  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Generate random ranges for simulation
        num_readings = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment) + 1
        scan_msg.ranges = [random.uniform(0.5, 5.0) for _ in range(num_readings)]

        self.scan_publisher.publish(scan_msg)
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = SensorPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Publisher interrupted')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import LaserScan
import statistics

class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')

        # Create subscribers for different topics
        self.status_subscriber = self.create_subscription(
            String, 'robot_status', self.status_callback, 10)
        self.battery_subscriber = self.create_subscription(
            Float64, 'battery_level', self.battery_callback, 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'laser_scan', self.scan_callback, 10)

        # Store data for processing
        self.battery_history = []
        self.min_scan_distance = float('inf')

        self.get_logger().info('Data processor started')

    def status_callback(self, msg):
        self.get_logger().info(f'Status received: {msg.data}')

    def battery_callback(self, msg):
        battery_level = msg.data
        self.battery_history.append(battery_level)

        # Keep only last 10 readings
        if len(self.battery_history) > 10:
            self.battery_history.pop(0)

        avg_battery = statistics.mean(self.battery_history)
        self.get_logger().info(
            f'Battery: {battery_level:.2f}%, Average: {avg_battery:.2f}%')

    def scan_callback(self, msg):
        # Find minimum distance in scan
        valid_ranges = [r for r in msg.ranges if 0.1 < r < 10.0]
        if valid_ranges:
            min_distance = min(valid_ranges)
            if min_distance < self.min_scan_distance:
                self.min_scan_distance = min_distance
                self.get_logger().info(f'New minimum distance: {min_distance:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    processor = DataProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Processor interrupted')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Service Implementation

### Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts, Trigger
from my_robot_package.srv import NavigateTo  # Custom service

class RobotServiceServer(Node):
    def __init__(self):
        super().__init__('robot_service_server')

        # Create service servers
        self.add_srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)
        self.trigger_srv = self.create_service(Trigger, 'robot_reset', self.reset_callback)
        self.navigate_srv = self.create_service(NavigateTo, 'navigate_to', self.navigate_callback)

        self.get_logger().info('Service server started')

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Calculated {request.a} + {request.b} = {response.sum}')
        return response

    def reset_callback(self, request, response):
        # Simulate robot reset
        self.get_logger().info('Resetting robot...')
        response.success = True
        response.message = 'Robot reset successfully'
        return response

    def navigate_callback(self, request, response):
        # Simulate navigation to target
        self.get_logger().info(f'Navigating to ({request.x}, {request.y})')

        # Simulate navigation success/failure
        import random
        success = random.choice([True, False])

        if success:
            response.success = True
            response.message = f'Reached target ({request.x}, {request.y})'
        else:
            response.success = False
            response.message = f'Failed to reach target ({request.x}, {request.y}) - obstacle detected'

        return response

def main(args=None):
    rclpy.init(args=args)
    server = RobotServiceServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Service server interrupted')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts, Trigger
from my_robot_package.srv import NavigateTo

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')

        # Create service clients
        self.add_client = self.create_client(AddTwoInts, 'add_two_ints')
        self.reset_client = self.create_client(Trigger, 'robot_reset')
        self.navigate_client = self.create_client(NavigateTo, 'navigate_to')

        # Wait for services to be available
        while not self.add_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Add service not available, waiting...')

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')

        while not self.navigate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigate service not available, waiting...')

        # Call services after a delay
        self.timer = self.create_timer(2.0, self.call_services)
        self.call_count = 0

    def call_services(self):
        if self.call_count == 0:
            self.call_add_service()
        elif self.call_count == 1:
            self.call_reset_service()
        elif self.call_count == 2:
            self.call_navigate_service()
        else:
            self.timer.cancel()  # Stop after all calls

        self.call_count += 1

    def call_add_service(self):
        request = AddTwoInts.Request()
        request.a = 10
        request.b = 20

        future = self.add_client.call_async(request)
        future.add_done_callback(self.add_response_callback)

    def call_reset_service(self):
        request = Trigger.Request()

        future = self.reset_client.call_async(request)
        future.add_done_callback(self.reset_response_callback)

    def call_navigate_service(self):
        request = NavigateTo.Request()
        request.x = 5.0
        request.y = 3.0

        future = self.navigate_client.call_async(request)
        future.add_done_callback(self.navigate_response_callback)

    def add_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Add service result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def reset_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Reset service result: {response.success}, {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def navigate_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Navigate service result: {response.success}, {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClient()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Client interrupted')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Custom Message and Service Types

### Creating Custom Messages

Custom messages are defined in `.msg` files in the `msg/` directory:

**msg/RobotPose.msg**:
```
# Robot pose with timestamp
builtin_interfaces/Time timestamp
geometry_msgs/Pose pose
float64 confidence
string source
```

**msg/RobotCommand.msg**:
```
# Robot command message
string command_type
float64[] parameters
builtin_interfaces/Time execution_time
bool emergency_stop
```

### Creating Custom Services

Custom services are defined in `.srv` files in the `srv/` directory:

**srv/NavigateTo.srv**:
```
# Request
float64 x
float64 y
float64 theta
---
# Response
bool success
string message
float64 execution_time
```

**srv/GetRobotState.srv**:
```
# Request
string robot_id
---
# Response
bool success
string message
RobotPose current_pose
RobotCommand last_command
```

### Using Custom Types in Python

```python
# In your Python code
from my_robot_package.msg import RobotPose, RobotCommand
from my_robot_package.srv import NavigateTo, GetRobotState

# Publisher with custom message
class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.pose_publisher = self.create_publisher(RobotPose, 'robot_pose', 10)
        self.cmd_publisher = self.create_publisher(RobotCommand, 'robot_command', 10)

        self.timer = self.create_timer(0.1, self.publish_state)

    def publish_state(self):
        # Create and publish robot pose
        pose_msg = RobotPose()
        pose_msg.timestamp = self.get_clock().now().to_msg()
        pose_msg.confidence = 0.95
        pose_msg.source = 'localization'

        # Set pose values (simplified)
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = 2.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_publisher.publish(pose_msg)
```

## Launch Files for Python Packages

### Python Launch Files

Launch files can be written in Python:

```python
# launch/robot_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    robot_name = LaunchConfiguration('robot_name')

    # Create nodes
    sensor_publisher = Node(
        package='my_robot_package',
        executable='sensor_node',
        name='sensor_publisher',
        parameters=[
            {'robot_name': robot_name}
        ],
        output='screen'
    )

    data_processor = Node(
        package='my_robot_package',
        executable='processor_node',
        name='data_processor',
        output='screen'
    )

    service_server = Node(
        package='my_robot_package',
        executable='service_server',
        name='service_server',
        output='screen'
    )

    return LaunchDescription([
        robot_name_arg,
        sensor_publisher,
        data_processor,
        service_server
    ])
```

## Testing Python Nodes

### Unit Testing with pytest

```python
# test/test_nodes.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_package.nodes.sensor_node import SensorPublisher

class TestSensorPublisher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_node_creation(self):
        """Test that the node can be created successfully."""
        node = SensorPublisher()
        self.assertIsNotNone(node)
        self.assertEqual(node.get_name(), 'sensor_publisher')
        node.destroy_node()

    def test_publisher_exists(self):
        """Test that publishers are created."""
        node = SensorPublisher()
        self.assertIsNotNone(node.string_publisher)
        self.assertIsNotNone(node.float_publisher)
        node.destroy_node()

def test_with_executor():
    """Test with executor to ensure node functionality."""
    rclpy.init()
    try:
        node = SensorPublisher()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Run for a short time to ensure publishers work
        import time
        start_time = time.time()
        while time.time() - start_time < 1.0:  # Run for 1 second
            executor.spin_once(timeout_sec=0.1)

        node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
# test/test_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts

class TestNode(Node):
    def __init__(self):
        super().__init__('integration_test_node')
        self.subscription = self.create_subscription(
            String, 'robot_status', self.status_callback, 10)
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        self.status_received = False

    def status_callback(self, msg):
        self.status_received = True
        self.get_logger().info(f'Received status: {msg.data}')

def test_system_integration():
    """Test the integration of multiple nodes."""
    rclpy.init()
    try:
        test_node = TestNode()

        # Wait for service
        while not test_node.client.wait_for_service(timeout_sec=1.0):
            test_node.get_logger().info('Service not available...')

        # Call service
        future = test_node.client.call_async(AddTwoInts.Request(a=5, b=3))

        # Spin to process messages
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(test_node)

        count = 0
        while not test_node.status_received and count < 50:  # 5 seconds max
            executor.spin_once(timeout_sec=0.1)
            count += 1

        # Check service response
        rclpy.spin_until_future_complete(test_node, future, timeout_sec=1.0)
        result = future.result()

        print(f"Service result: {result.sum if result else 'None'}")
        print(f"Status received: {test_node.status_received}")

        test_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    test_system_integration()
```

## Code Examples with Explanations

### Complete Robot Control Node

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from my_robot_package.srv import NavigateTo
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.status_publisher = self.create_publisher(String, 'robot_status', qos_profile)

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile)
        self.emergency_subscription = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_callback, qos_profile)

        # Service server
        self.navigate_service = self.create_service(
            NavigateTo, 'navigate_to', self.navigate_callback)

        # Robot state
        self.current_scan = None
        self.emergency_stop = False
        self.target_x = 0.0
        self.target_y = 0.0
        self.has_target = False

        # Control parameters
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.safe_distance = 0.5

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Robot controller initialized')

    def scan_callback(self, msg):
        """Callback for laser scan data."""
        self.current_scan = msg

    def emergency_callback(self, msg):
        """Callback for emergency stop commands."""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.stop_robot()
            self.get_logger().warn('Emergency stop activated!')

    def navigate_callback(self, request, response):
        """Service callback for navigation requests."""
        self.target_x = request.x
        self.target_y = request.y
        self.has_target = True

        response.success = True
        response.message = f'Navigating to ({request.x}, {request.y})'
        response.execution_time = 0.0

        self.get_logger().info(f'Received navigation request to ({request.x}, {request.y})')
        return response

    def control_loop(self):
        """Main control loop."""
        if self.emergency_stop:
            self.stop_robot()
            return

        cmd_vel = Twist()

        if self.has_target and self.current_scan:
            # Simple navigation logic
            cmd_vel = self.calculate_navigation_command()
        elif self.current_scan:
            # Obstacle avoidance
            cmd_vel = self.avoid_obstacles()

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Publish status
        status_msg = String()
        if self.has_target:
            status_msg.data = f'Navigating to ({self.target_x:.2f}, {self.target_y:.2f})'
        else:
            status_msg.data = 'Idle'
        self.status_publisher.publish(status_msg)

    def calculate_navigation_command(self):
        """Calculate navigation command based on target."""
        cmd_vel = Twist()

        # This is a simplified navigation - in practice, use proper path planning
        if self.current_scan:
            # Check if path is clear
            front_ranges = self.current_scan.ranges[:len(self.current_scan.ranges)//5] + \
                          self.current_scan.ranges[-len(self.current_scan.ranges)//5:]

            min_distance = min([r for r in front_ranges if 0.1 < r < 10.0], default=float('inf'))

            if min_distance > self.safe_distance:
                cmd_vel.linear.x = self.linear_speed
                cmd_vel.angular.z = 0.0  # Simplified - would need proper navigation
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed  # Turn to avoid obstacle
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        return cmd_vel

    def avoid_obstacles(self):
        """Simple obstacle avoidance behavior."""
        cmd_vel = Twist()

        if not self.current_scan:
            return cmd_vel

        # Get front, left, and right ranges
        num_ranges = len(self.current_scan.ranges)
        front_idx = num_ranges // 2
        left_idx = num_ranges // 4
        right_idx = 3 * num_ranges // 4

        front_dist = self.current_scan.ranges[front_idx]
        left_dist = self.current_scan.ranges[left_idx]
        right_dist = self.current_scan.ranges[right_idx]

        # Simple obstacle avoidance
        if front_dist < self.safe_distance:
            # Obstacle in front - turn
            if left_dist > right_dist:
                cmd_vel.angular.z = self.angular_speed
            else:
                cmd_vel.angular.z = -self.angular_speed
        else:
            # No obstacles - move forward
            cmd_vel.linear.x = self.linear_speed

        return cmd_vel

    def stop_robot(self):
        """Stop the robot immediately."""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Controller interrupted')
    finally:
        controller.stop_robot()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Explanation**:
- Implements a complete robot controller with multiple ROS 2 communication patterns
- Uses QoS profiles for reliable communication
- Combines publisher, subscriber, and service functionality
- Implements safety features and emergency stop
- Demonstrates proper resource management and cleanup

## Hands-on Exercises

### Exercise 1: Custom Package Creation

**Objective**: Create a complete ROS 2 package with multiple nodes and custom message types.

**Requirements**:
1. Create a new ROS 2 package named `sensor_fusion_package`
2. Define custom message types for sensor data fusion
3. Implement nodes for sensor data processing
4. Create launch files to start the complete system
5. Write unit tests for your nodes

**Implementation Steps**:
1. Use `ros2 pkg create --build-type ament_python sensor_fusion_package`
2. Define custom message types in `msg/` directory
3. Implement sensor fusion algorithm in Python
4. Create publisher and subscriber nodes
5. Write comprehensive unit tests
6. Create launch files for system startup

**Expected Outcome**: A working sensor fusion package with proper structure and testing.

### Exercise 2: Multi-Robot Communication System

**Objective**: Build a system where multiple robots communicate and coordinate.

**Requirements**:
1. Create nodes that can identify themselves with unique IDs
2. Implement communication protocols for robot coordination
3. Add service interfaces for robot management
4. Include action-based task assignment
5. Implement proper error handling and recovery

**Implementation Steps**:
1. Design unique robot identification system
2. Implement topic-based communication between robots
3. Create service for robot discovery and management
4. Add action interface for task assignment
5. Test with multiple simulated robots
6. Implement fault tolerance mechanisms

**Expected Outcome**: A multi-robot communication system with proper coordination.

### Exercise 3: Advanced Python Package with External Dependencies

**Objective**: Create a ROS 2 package that integrates with external Python libraries.

**Requirements**:
1. Use external libraries like NumPy, OpenCV, or SciPy
2. Handle dependencies properly in setup.py
3. Implement complex data processing algorithms
4. Include visualization capabilities
5. Ensure proper memory management

**Implementation Steps**:
1. Identify and list external dependencies
2. Configure setup.py to install required packages
3. Implement algorithms using external libraries
4. Add visualization nodes using matplotlib or similar
5. Test memory usage and performance
6. Document dependency management

**Expected Outcome**: A sophisticated ROS 2 package with external library integration.

## Summary

Chapter 3 has provided comprehensive coverage of building ROS 2 packages with Python. You've learned about package structure, node implementation, communication patterns, custom message types, launch files, and testing strategies. The examples demonstrate practical applications of these concepts in robot control systems. The exercises will help you apply these concepts to create sophisticated ROS 2 packages.

## Further Reading

1. Rockenbach, M., et al. (2020). "Python in ROS 2: A practical guide to building robotic applications." *Journal of Open Robotics Software*, 1(1), 1-15.
2. Doodaghian, H., et al. (2021). "Performance evaluation of Python vs C++ in ROS 2 applications." *Robotics and Autonomous Systems*, 142, 103789.
3. ROS 2 Documentation. (2023). "Python Client Library (rclpy) Developer Guide." Retrieved from https://docs.ros.org/
4. Lütkebohle, I., et al. (2012). "The ROS node concept for lightweight process separation." *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems*, 4428-4435.

## Navigation

[Previous: Chapter 2 - Nodes, Topics, Services, and Actions](/docs/module1/chapter2) | [Next: Chapter 4 - Launch Files and Parameter Management](/docs/module1/chapter4)