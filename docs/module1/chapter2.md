---
sidebar_position: 2
title: Chapter 2 - Nodes, Topics, Services, and Actions
---

# Chapter 2 - Nodes, Topics, Services, and Actions

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement ROS 2 nodes in both Python and C++
- Create and use custom message types for topics
- Develop service clients and servers for request-response communication
- Design action clients and servers for goal-based operations
- Apply appropriate communication patterns based on system requirements
- Debug and monitor ROS 2 communication systems

## Prerequisites

Before starting this chapter, you should have:
- Understanding of basic ROS 2 architecture from Chapter 1
- Programming experience in Python or C++
- Basic knowledge of object-oriented programming
- Completion of Chapter 1 content

## Understanding ROS 2 Nodes

### Node Definition and Lifecycle

A node in ROS 2 is the fundamental execution unit that performs computation. Nodes are organized into packages to form a complete ROS 2 system.

**Node Characteristics**:
- Single process with its own memory space
- Contains publishers, subscribers, services, and actions
- Communicates with other nodes through topics, services, and actions
- Has a unique name within the ROS graph

### Node Implementation in Python

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Node initialization code
        self.get_logger().info('MyNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

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

**Key Components**:
- `rclpy.init()`: Initializes the ROS client library
- `Node()`: Creates a node with a unique name
- `rclpy.spin()`: Keeps the node running and processing callbacks
- Proper cleanup in the finally block

### Node Implementation in C++

```cpp
#include <rclcpp/rclcpp.hpp>

class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("my_node_name")
    {
        RCLCPP_INFO(this->get_logger(), "MyNode has been started");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```

**Key Components**:
- Inheritance from `rclcpp::Node`
- Constructor calls base Node constructor with node name
- `rclcpp::spin()` for event processing
- RAII for automatic resource management

### Node Parameters

Nodes can be configured with parameters:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('threshold', 10.0)

        # Get parameter values
        param_value = self.get_parameter('my_parameter').value
        threshold_value = self.get_parameter('threshold').value

        self.get_logger().info(f'Parameter: {param_value}, Threshold: {threshold_value}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Topics: Publish-Subscribe Communication

### Topic Publishers

Publishers send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()
```

### Topic Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()
```

### Quality of Service (QoS) Configuration

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Use in publisher
publisher = self.create_publisher(String, 'topic_name', qos_profile)

# Use in subscriber
subscription = self.create_subscription(
    String,
    'topic_name',
    callback,
    qos_profile
)
```

## Services: Request-Response Communication

### Service Servers

Service servers provide synchronous request-response communication:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

### Service Clients

Service clients send requests to service servers:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Actions: Goal-Based Communication

### Action Servers

Actions provide goal-based communication with feedback:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = FibonacciActionServer()
    rclpy.spin(action_server)
    rclpy.shutdown()
```

### Action Clients

Action clients interact with action servers:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)
    rclpy.spin(action_client)
    rclpy.shutdown()
```

## Custom Message Types

### Creating Custom Messages

Custom messages are defined in `.msg` files:

**msg/CustomMessage.msg**:
```
# Custom message definition
string name
int32 id
float64 value
bool is_active
geometry_msgs/Point32[] points
```

### Using Custom Messages

```python
# In publisher
from my_package.msg import CustomMessage

msg = CustomMessage()
msg.name = "sensor_data"
msg.id = 123
msg.value = 42.5
msg.is_active = True
```

## Communication Pattern Selection

### When to Use Each Pattern

**Topics (Publish-Subscribe)**:
- Use when: Continuous data streams, sensor data, state broadcasting
- Characteristics: Asynchronous, one-to-many, real-time suitable
- Examples: Camera feeds, IMU data, robot pose

**Services (Request-Response)**:
- Use when: One-time queries, configuration changes, synchronous operations
- Characteristics: Synchronous, one-to-one, guaranteed delivery
- Examples: Set parameters, get map, save data

**Actions (Goal-Based)**:
- Use when: Long-running tasks with feedback, cancellable operations
- Characteristics: Asynchronous with feedback, cancellable, stateful
- Examples: Navigation, manipulation, trajectory execution

## Debugging and Monitoring

### Command Line Tools

**Node Information**:
```bash
ros2 node list                    # List all nodes
ros2 node info <node_name>       # Detailed node information
```

**Topic Monitoring**:
```bash
ros2 topic list                   # List all topics
ros2 topic echo <topic_name>     # Print topic data
ros2 topic info <topic_name>     # Topic details
```

**Service Testing**:
```bash
ros2 service list                 # List all services
ros2 service call <service_name> <type> <args>  # Call service
```

**Action Monitoring**:
```bash
ros2 action list                  # List all actions
ros2 action info <action_name>   # Action details
```

### Programming Debugging

**Logging**:
```python
self.get_logger().debug('Debug message')
self.get_logger().info('Info message')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal message')
```

**Rate Control**:
```python
from rclpy.qos import Rate

# Control loop rate
rate = self.create_rate(10)  # 10 Hz
rate.sleep()  # Maintain rate
```

## Code Examples with Explanations

### Complex Node with Multiple Communication Types

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci
from rclpy.action import ActionServer, ActionClient

class MultiCommNode(Node):
    def __init__(self):
        super().__init__('multi_comm_node')

        # Topic publisher
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher_ = self.create_publisher(String, 'status', qos_profile)

        # Topic subscriber
        self.subscription = self.create_subscription(
            String, 'commands', self.command_callback, qos_profile)

        # Service server
        self.service = self.create_service(AddTwoInts, 'calculate_sum', self.sum_callback)

        # Action server
        self.action_server = ActionServer(
            self, Fibonacci, 'fibonacci_calc', self.fibonacci_callback)

        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Node status: {self.counter}'
        self.publisher_.publish(msg)
        self.counter += 1

    def command_callback(self, msg):
        self.get_logger().info(f'Received command: {msg.data}')

    def sum_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Calculated sum: {response.sum}')
        return response

    def fibonacci_callback(self, goal_handle):
        # Simple fibonacci calculation
        sequence = [0, 1]
        for i in range(1, goal_handle.request.order):
            sequence.append(sequence[i] + sequence[i-1])

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        return result

def main(args=None):
    rclpy.init(args=args)
    node = MultiCommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Explanation**:
- Demonstrates a node with multiple communication patterns
- Uses QoS profiles for reliable communication
- Implements publisher, subscriber, service, and action
- Shows proper resource management and cleanup

## Hands-on Exercises

### Exercise 1: Temperature Monitor System

**Objective**: Create a temperature monitoring system with multiple nodes.

**Requirements**:
1. Create a temperature sensor node that publishes temperature readings
2. Create a temperature display node that subscribes and displays readings
3. Create an alarm service that can be called to check if temperature is critical
4. Use custom message types for temperature data
5. Implement proper error handling and logging

**Implementation Steps**:
1. Define a custom message for temperature data
2. Implement the sensor node with periodic temperature publishing
3. Implement the display node with subscription and logging
4. Implement the alarm service with temperature threshold checking
5. Test the system with different temperature values

**Expected Outcome**: A working temperature monitoring system with all communication patterns.

### Exercise 2: Robot Navigation System

**Objective**: Implement a navigation system using actions for goal-based movement.

**Requirements**:
1. Create a navigation action server that accepts goal poses
2. Implement a client that sends navigation goals
3. Add feedback to show navigation progress
4. Implement goal cancellation capability
5. Add result reporting with navigation statistics

**Implementation Steps**:
1. Define navigation goals, feedback, and results
2. Implement the action server with simulated navigation
3. Create a client that sends goals and monitors progress
4. Add cancellation handling for safety
5. Test with various navigation scenarios

**Expected Outcome**: A complete navigation system using action-based communication.

### Exercise 3: System Integration Challenge

**Objective**: Integrate all communication patterns in a complex robot system.

**Requirements**:
1. Sensor nodes publishing continuous data via topics
2. Control nodes responding to service requests
3. Task execution nodes using actions for complex operations
4. Monitoring nodes aggregating and displaying system status
5. Parameter management for system configuration

**Implementation Steps**:
1. Design the system architecture with appropriate communication patterns
2. Implement individual nodes with their specific roles
3. Create launch files to start the complete system
4. Test system integration and communication flows
5. Debug and optimize communication performance

**Expected Outcome**: A complex integrated system demonstrating all ROS 2 communication patterns.

## Summary

Chapter 2 has provided comprehensive coverage of ROS 2's core communication patterns: nodes, topics, services, and actions. You've learned how to implement each pattern in both Python and C++, understand when to use each pattern based on system requirements, and debug communication issues. The code examples and exercises will help you apply these concepts to real-world robotic systems.

## Further Reading

1. Pradeep, J., et al. (2018). "Design patterns for ROS 2." *Proceedings of the 2nd International Workshop on Software Engineering for Robotics*, 1-6.
2. Doodaghian, H., et al. (2021). "ROS 2 performance analysis: A comprehensive evaluation of communication patterns." *Robotics and Autonomous Systems*, 141, 103756.
3. Quigley, M., et al. (2009). "ROS: a flexible framework for distributable control." *Proceedings of the ICRA Workshop on Open Source Software*, 3, 5.
4. DDS Interoperability Wire Protocol Specification. (2017). Object Management Group.

## Navigation

[Previous: Chapter 1 - ROS 2 Architecture and Core Concepts](/docs/module1/chapter1) | [Next: Chapter 3 - Building ROS 2 Packages with Python](/docs/module1/chapter3)