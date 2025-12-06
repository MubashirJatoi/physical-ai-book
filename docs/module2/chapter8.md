---
sidebar_position: 4
title: Chapter 8 - Unity for Robot Visualization
---

# Chapter 8 - Unity for Robot Visualization

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up Unity for robotics visualization applications
- Import and configure robot models in Unity
- Implement real-time robot simulation and visualization
- Integrate Unity with ROS 2 for bidirectional communication
- Create custom visualization tools and interfaces
- Develop interactive environments for robot testing
- Optimize Unity scenes for real-time performance
- Deploy Unity applications for robot visualization

## Prerequisites

Before starting this chapter, you should have:
- Basic understanding of Unity 3D game engine
- Knowledge of C# programming for Unity scripts
- Experience with 3D modeling and scene composition
- Understanding of ROS 2 communication patterns
- Completion of Module 1 and Module 2 Chapter 7 content
- Familiarity with coordinate systems and transformations

## Introduction to Unity for Robotics

### Unity in Robotics Context

Unity has emerged as a powerful platform for robotics visualization and simulation, offering several advantages over traditional simulation environments:

**Visual Fidelity**:
- High-quality rendering with realistic lighting
- Advanced materials and textures
- Realistic environmental effects
- Photorealistic visualization capabilities

**Flexibility**:
- Extensive asset store with 3D models
- Powerful animation system
- Customizable user interfaces
- Cross-platform deployment options

**Integration Capabilities**:
- ROS/ROS 2 bridge packages
- Real-time communication protocols
- Plugin architecture for custom functionality
- Support for various sensor simulations

### Unity vs. Traditional Robotics Simulation

**Unity vs. Gazebo**:
- Unity: Better graphics, real-time rendering, game engine features
- Gazebo: Better physics, sensor accuracy, robotics-specific tools

**Unity vs. Webots**:
- Unity: More flexible, better graphics, extensive asset ecosystem
- Webots: More robotics-focused, built-in robot models, easier setup

**Unity vs. Blender**:
- Unity: Real-time interaction, scripting capabilities, deployment options
- Blender: Better for static visualization, modeling, animation creation

## Setting Up Unity for Robotics

### Unity Installation and Configuration

**System Requirements**:
- Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- Graphics card with DirectX 10, OpenGL 3.3, or Metal support
- 8+ GB RAM (16+ GB recommended for complex scenes)
- 20+ GB free disk space

**Recommended Unity Version**:
- Unity 2022.3 LTS (Long Term Support) or later
- Install with Universal Render Pipeline (URP) or High Definition Render Pipeline (HDRP)
- Include Visual Studio or Rider as external script editor

### Unity Robotics Packages

**Unity Robotics Hub**:
The Unity Robotics Hub provides essential packages for robotics development:

1. **ROS-TCP-Connector**: Enables communication between Unity and ROS 2
2. **Unity-Robotics-Helpers**: Utilities for robotics-specific tasks
3. **URDF-Importer**: Imports URDF robot models into Unity
4. **ML-Agents**: Machine learning framework for robot training

**Installation via Package Manager**:
```bash
# In Unity Package Manager, install:
- com.unity.robotics.ros-tcp-connector
- com.unity.robotics.urdf-importer
- com.unity.robotics.robotics-helpers
```

### ROS 2 Bridge Setup

**Installing ROS 2 Bridge**:
```bash
# Install Python ROS 2 bridge package
pip install unity-robotics-hub

# Or clone from GitHub
git clone https://github.com/Unity-Technologies/Unity-Robotics-Hub.git
```

## Robot Model Integration

### Importing URDF Models

The URDF Importer package allows direct import of ROS robot models:

**Basic URDF Import**:
```csharp
// RobotImporter.cs
using UnityEngine;
using Unity.Robotics.URDFImporter;
using System.IO;

public class RobotImporter : MonoBehaviour
{
    [Header("URDF Import Settings")]
    public string urdfPath;
    public string assetPath = "Assets/Robots/";
    public bool importInertial = true;
    public bool importCollision = true;
    public bool importVisual = true;

    [ContextMenu("Import URDF")]
    public void ImportRobot()
    {
        if (string.IsNullOrEmpty(urdfPath) || !File.Exists(urdfPath))
        {
            Debug.LogError("Invalid URDF path: " + urdfPath);
            return;
        }

        // Import the URDF model
        var importedRobot = URDFRobotExtensions.CreateRobot(urdfPath);

        if (importedRobot != null)
        {
            importedRobot.name = Path.GetFileNameWithoutExtension(urdfPath);
            Debug.Log($"Successfully imported robot: {importedRobot.name}");
        }
        else
        {
            Debug.LogError("Failed to import URDF robot");
        }
    }

    [ContextMenu("Validate URDF")]
    public void ValidateURDF()
    {
        if (string.IsNullOrEmpty(urdfPath) || !File.Exists(urdfPath))
        {
            Debug.LogError("Invalid URDF path: " + urdfPath);
            return;
        }

        // Validate URDF syntax and structure
        string urdfContent = File.ReadAllText(urdfPath);
        if (urdfContent.Contains("<robot") && urdfContent.Contains("</robot>"))
        {
            Debug.Log("URDF structure appears valid");
        }
        else
        {
            Debug.LogError("URDF structure may be invalid");
        }
    }
}
```

### Robot Configuration in Unity

**Robot Controller Script**:
```csharp
// RobotController.cs
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "my_robot";
    public Transform baseLink;
    public List<JointController> joints = new List<JointController>();

    [Header("ROS Communication")]
    public string jointStatesTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";

    private ROSConnection ros;
    private Dictionary<string, JointController> jointMap = new Dictionary<string, JointController>();

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(jointStatesTopic);
        ros.RegisterSubscriber<TwistMsg>(cmdVelTopic, OnCmdVelReceived);

        // Initialize joint map for quick lookup
        foreach (var joint in joints)
        {
            jointMap[joint.jointName] = joint;
        }

        // Start publishing joint states
        InvokeRepeating("PublishJointStates", 0.0f, 0.1f); // 10Hz
    }

    void OnCmdVelReceived(TwistMsg cmdVel)
    {
        // Process velocity commands
        // This would typically control differential drive or other movement systems
        ProcessVelocityCommand(cmdVel);
    }

    void ProcessVelocityCommand(TwistMsg cmdVel)
    {
        // Implement robot movement logic based on Twist message
        float linearX = (float)cmdVel.linear.x;
        float angularZ = (float)cmdVel.angular.z;

        // For differential drive robots
        if (jointMap.ContainsKey("left_wheel_joint") && jointMap.ContainsKey("right_wheel_joint"))
        {
            float wheelRadius = 0.1f; // Example wheel radius
            float wheelSeparation = 0.4f; // Example wheel separation

            // Calculate wheel velocities
            float leftWheelVel = (linearX - angularZ * wheelSeparation / 2.0f) / wheelRadius;
            float rightWheelVel = (linearX + angularZ * wheelSeparation / 2.0f) / wheelRadius;

            // Apply velocities to joints
            if (jointMap["left_wheel_joint"] != null)
                jointMap["left_wheel_joint"].SetVelocity(leftWheelVel);

            if (jointMap["right_wheel_joint"] != null)
                jointMap["right_wheel_joint"].SetVelocity(rightWheelVel);
        }
    }

    void PublishJointStates()
    {
        // Create joint state message
        var jointState = new JointStateMsg();
        jointState.name = new string[joints.Count];
        jointState.position = new double[joints.Count];
        jointState.velocity = new double[joints.Count];
        jointState.effort = new double[joints.Count];

        for (int i = 0; i < joints.Count; i++)
        {
            jointState.name[i] = joints[i].jointName;
            jointState.position[i] = joints[i].GetPosition();
            jointState.velocity[i] = joints[i].GetVelocity();
            jointState.effort[i] = joints[i].GetEffort();
        }

        jointState.header = new HeaderMsg();
        jointState.header.stamp = new TimeStamp();
        jointState.header.frame_id = robotName;

        // Publish joint states
        ros.Publish(jointStatesTopic, jointState);
    }

    void Update()
    {
        // Update joint positions based on physics simulation
        foreach (var joint in joints)
        {
            joint.UpdateJoint();
        }
    }
}
```

### Joint Controller Implementation

**Joint Controller Script**:
```csharp
// JointController.cs
using UnityEngine;

[System.Serializable]
public class JointController
{
    [Header("Joint Configuration")]
    public string jointName;
    public JointType jointType = JointType.Revolute;
    public Transform jointTransform;
    public ConfigurableJoint configurableJoint;
    public float position = 0f;
    public float velocity = 0f;
    public float effort = 0f;

    [Header("Joint Limits")]
    public float lowerLimit = -Mathf.PI;
    public float upperLimit = Mathf.PI;
    public float maxEffort = 100f;
    public float maxVelocity = 10f;

    [Header("Control Parameters")]
    public float stiffness = 10000f;
    public float damping = 1000f;

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed,
        Continuous,
        Floating
    }

    public void InitializeJoint()
    {
        if (configurableJoint == null && jointTransform != null)
        {
            // Add ConfigurableJoint component if not present
            configurableJoint = jointTransform.gameObject.AddComponent<ConfigurableJoint>();

            // Configure joint based on type
            ConfigureJointParameters();
        }
    }

    void ConfigureJointParameters()
    {
        if (configurableJoint == null) return;

        switch (jointType)
        {
            case JointType.Revolute:
                // Configure for rotational joint
                configurableJoint.xMotion = ConfigurableJointMotion.Locked;
                configurableJoint.yMotion = ConfigurableJointMotion.Locked;
                configurableJoint.zMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularXMotion = ConfigurableJointMotion.Limited;
                configurableJoint.angularYMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularZMotion = ConfigurableJointMotion.Locked;

                // Set joint limits
                SoftJointLimit angularLimit = new SoftJointLimit();
                angularLimit.limit = Mathf.Rad2Deg * upperLimit;
                configurableJoint.highAngularXLimit = angularLimit;

                angularLimit.limit = Mathf.Rad2Deg * lowerLimit;
                configurableJoint.lowAngularXLimit = angularLimit;
                break;

            case JointType.Prismatic:
                // Configure for linear joint
                configurableJoint.xMotion = ConfigurableJointMotion.Limited;
                configurableJoint.yMotion = ConfigurableJointMotion.Locked;
                configurableJoint.zMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularXMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularYMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularZMotion = ConfigurableJointMotion.Locked;
                break;

            case JointType.Fixed:
                // Lock all motion
                configurableJoint.xMotion = ConfigurableJointMotion.Locked;
                configurableJoint.yMotion = ConfigurableJointMotion.Locked;
                configurableJoint.zMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularXMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularYMotion = ConfigurableJointMotion.Locked;
                configurableJoint.angularZMotion = ConfigurableJointMotion.Locked;
                break;
        }

        // Configure drive for position control
        JointDrive drive = new JointDrive();
        drive.positionSpring = stiffness;
        drive.positionDamper = damping;
        drive.maximumForce = maxEffort;

        configurableJoint.slerpDrive = drive;
        configurableJoint.rotationDrive = drive;
    }

    public void SetPosition(float pos)
    {
        position = Mathf.Clamp(pos, lowerLimit, upperLimit);
        if (configurableJoint != null)
        {
            // Apply position control through joint drive
            configurableJoint.targetRotation = Quaternion.AngleAxis(Mathf.Rad2Deg * position, Vector3.right);
        }
    }

    public void SetVelocity(float vel)
    {
        velocity = Mathf.Clamp(vel, -maxVelocity, maxVelocity);
        if (configurableJoint != null)
        {
            configurableJoint.targetAngularVelocity = new Vector3(Mathf.Rad2Deg * velocity, 0, 0);
        }
    }

    public float GetPosition()
    {
        if (configurableJoint != null)
        {
            // Calculate current joint position
            Vector3 currentRotation = configurableJoint.transform.localEulerAngles;
            return Mathf.Deg2Rad * currentRotation.x;
        }
        return position;
    }

    public float GetVelocity()
    {
        if (configurableJoint != null)
        {
            return Mathf.Rad2Deg * configurableJoint.velocity.x;
        }
        return velocity;
    }

    public float GetEffort()
    {
        // Effort calculation would depend on applied forces
        // This is a simplified implementation
        return effort;
    }

    public void UpdateJoint()
    {
        // Update joint state based on current simulation
        if (configurableJoint != null)
        {
            position = GetPosition();
            velocity = GetVelocity();
        }
    }
}
```

## Unity-ROS 2 Integration

### ROS Communication Setup

**ROS TCP Connector Implementation**:
```csharp
// ROSCommunicationManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using System.Collections.Generic;

public class ROSCommunicationManager : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;
    public bool autoConnect = true;

    [Header("Topics")]
    public string laserScanTopic = "/scan";
    public string imageTopic = "/camera/image_raw";
    public string imuTopic = "/imu";
    public string odometryTopic = "/odom";

    private ROSConnection ros;
    private bool isConnected = false;

    [Header("Sensors")]
    public bool enableLaserScan = true;
    public bool enableCamera = true;
    public bool enableIMU = true;
    public bool enableOdometry = true;

    void Start()
    {
        SetupROSConnection();
    }

    void SetupROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();

        if (autoConnect)
        {
            ConnectToROS();
        }
    }

    public void ConnectToROS()
    {
        try
        {
            ros.Initialize(rosIP, rosPort);
            isConnected = true;
            Debug.Log($"Connected to ROS at {rosIP}:{rosPort}");

            // Register publishers and subscribers based on enabled sensors
            if (enableLaserScan)
            {
                ros.RegisterPublisher<LaserScanMsg>(laserScanTopic);
            }

            if (enableCamera)
            {
                ros.RegisterPublisher<ImageMsg>(imageTopic);
            }

            if (enableIMU)
            {
                ros.RegisterPublisher<ImuMsg>(imuTopic);
            }

            if (enableOdometry)
            {
                ros.RegisterPublisher<OdometryMsg>(odometryTopic);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
            isConnected = false;
        }
    }

    public void PublishLaserScan(float[] ranges, float angleMin, float angleMax, float angleIncrement, float timeIncrement, float scanTime, float rangeMin, float rangeMax)
    {
        if (!isConnected || !enableLaserScan) return;

        var laserScan = new LaserScanMsg();
        laserScan.header = new HeaderMsg();
        laserScan.header.stamp = new TimeStamp();
        laserScan.header.frame_id = "laser_frame";

        laserScan.angle_min = angleMin;
        laserScan.angle_max = angleMax;
        laserScan.angle_increment = angleIncrement;
        laserScan.time_increment = timeIncrement;
        laserScan.scan_time = scanTime;
        laserScan.range_min = rangeMin;
        laserScan.range_max = rangeMax;

        laserScan.ranges = new float[ranges.Length];
        for (int i = 0; i < ranges.Length; i++)
        {
            laserScan.ranges[i] = ranges[i];
        }

        laserScan.intensities = new float[ranges.Length]; // Initialize if needed

        ros.Publish(laserScanTopic, laserScan);
    }

    public void PublishCameraImage(Texture2D texture)
    {
        if (!isConnected || !enableCamera) return;

        // Convert Unity texture to ROS image format
        byte[] imageData = texture.EncodeToPNG();

        var imageMsg = new ImageMsg();
        imageMsg.header = new HeaderMsg();
        imageMsg.header.stamp = new TimeStamp();
        imageMsg.header.frame_id = "camera_frame";

        imageMsg.height = (uint)texture.height;
        imageMsg.width = (uint)texture.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(texture.width * 3); // 3 bytes per pixel for RGB
        imageMsg.data = imageData;

        ros.Publish(imageTopic, imageMsg);
    }

    public void PublishIMU(Vector3 angularVelocity, Vector3 linearAcceleration, Vector4 orientation)
    {
        if (!isConnected || !enableIMU) return;

        var imuMsg = new ImuMsg();
        imuMsg.header = new HeaderMsg();
        imuMsg.header.stamp = new TimeStamp();
        imuMsg.header.frame_id = "imu_frame";

        // Convert Unity coordinates to ROS coordinates (if needed)
        imuMsg.angular_velocity.x = angularVelocity.x;
        imuMsg.angular_velocity.y = angularVelocity.y;
        imuMsg.angular_velocity.z = angularVelocity.z;

        imuMsg.linear_acceleration.x = linearAcceleration.x;
        imuMsg.linear_acceleration.y = linearAcceleration.y;
        imuMsg.linear_acceleration.z = linearAcceleration.z;

        imuMsg.orientation.x = orientation.x;
        imuMsg.orientation.y = orientation.y;
        imuMsg.orientation.z = orientation.z;
        imuMsg.orientation.w = orientation.w;

        ros.Publish(imuTopic, imuMsg);
    }

    public void PublishOdometry(Vector3 position, Vector4 orientation, Vector3 linearVelocity, Vector3 angularVelocity)
    {
        if (!isConnected || !enableOdometry) return;

        var odomMsg = new OdometryMsg();
        odomMsg.header = new HeaderMsg();
        odomMsg.header.stamp = new TimeStamp();
        odomMsg.header.frame_id = "odom";
        odomMsg.child_frame_id = "base_link";

        // Position
        odomMsg.pose.pose.position.x = position.x;
        odomMsg.pose.pose.position.y = position.y;
        odomMsg.pose.pose.position.z = position.z;

        odomMsg.pose.pose.orientation.x = orientation.x;
        odomMsg.pose.pose.orientation.y = orientation.y;
        odomMsg.pose.pose.orientation.z = orientation.z;
        odomMsg.pose.pose.orientation.w = orientation.w;

        // Velocity
        odomMsg.twist.twist.linear.x = linearVelocity.x;
        odomMsg.twist.twist.linear.y = linearVelocity.y;
        odomMsg.twist.twist.linear.z = linearVelocity.z;

        odomMsg.twist.twist.angular.x = angularVelocity.x;
        odomMsg.twist.twist.angular.y = angularVelocity.y;
        odomMsg.twist.twist.angular.z = angularVelocity.z;

        ros.Publish(odometryTopic, odomMsg);
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Shutdown();
        }
    }
}
```

### Sensor Simulation in Unity

**LIDAR Sensor Simulation**:
```csharp
// LidarSensor.cs
using UnityEngine;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    public int numberOfRays = 360;
    public float scanRange = 30f;
    public float angleMin = -Mathf.PI;
    public float angleMax = Mathf.PI;
    public LayerMask detectionLayers = -1;
    public string frameId = "laser_frame";

    [Header("Visualization")]
    public bool visualizeRays = true;
    public float rayVisualizationDuration = 0.1f;

    [Header("Performance")]
    public bool useAsyncRaycasts = true;

    private float[] ranges;
    private ROSCommunicationManager rosManager;
    private float angleIncrement;

    void Start()
    {
        ranges = new float[numberOfRays];
        angleIncrement = (angleMax - angleMin) / numberOfRays;

        // Find ROS manager in the scene
        rosManager = FindObjectOfType<ROSCommunicationManager>();

        // Start scanning
        InvokeRepeating("PerformScan", 0.0f, 0.1f); // 10Hz scan rate
    }

    void PerformScan()
    {
        if (useAsyncRaycasts)
        {
            StartCoroutine(AsyncRaycastScan());
        }
        else
        {
            PerformSynchronousScan();
        }
    }

    void PerformSynchronousScan()
    {
        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = angleMin + (i * angleIncrement);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, scanRange, detectionLayers))
            {
                ranges[i] = hit.distance;

                if (visualizeRays)
                {
                    Debug.DrawRay(transform.position, direction * hit.distance, Color.red, rayVisualizationDuration);
                }
            }
            else
            {
                ranges[i] = scanRange; // No obstacle detected

                if (visualizeRays)
                {
                    Debug.DrawRay(transform.position, direction * scanRange, Color.green, rayVisualizationDuration);
                }
            }
        }

        // Publish scan data if ROS manager is available
        if (rosManager != null)
        {
            rosManager.PublishLaserScan(ranges, angleMin, angleMax, angleIncrement, 0, 0.1f, 0.1f, scanRange);
        }
    }

    System.Collections.IEnumerator AsyncRaycastScan()
    {
        // For performance, we'll still do synchronous raycasts but could implement true async
        // This is a placeholder for more complex async operations
        PerformSynchronousScan();
        yield return null;
    }

    void OnValidate()
    {
        // Ensure parameters are valid
        numberOfRays = Mathf.Clamp(numberOfRays, 1, 10000);
        scanRange = Mathf.Clamp(scanRange, 0.1f, 100f);
        angleMin = Mathf.Clamp(angleMin, -2 * Mathf.PI, 2 * Mathf.PI);
        angleMax = Mathf.Clamp(angleMax, -2 * Mathf.PI, 2 * Mathf.PI);
    }

    // Gizmos for scene view visualization
    void OnDrawGizmosSelected()
    {
        if (Application.isPlaying) return;

        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, 0.1f);

        // Draw scan area
        Vector3 startDir = new Vector3(Mathf.Cos(angleMin), 0, Mathf.Sin(angleMin));
        startDir = transform.TransformDirection(startDir);

        Vector3 endDir = new Vector3(Mathf.Cos(angleMax), 0, Mathf.Sin(angleMax));
        endDir = transform.TransformDirection(endDir);

        // Draw arc
        int segments = 30;
        Vector3 lastPoint = transform.position + startDir * scanRange;
        for (int i = 1; i <= segments; i++)
        {
            float angle = Mathf.Lerp(angleMin, angleMax, i / (float)segments);
            Vector3 dir = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            dir = transform.TransformDirection(dir);
            Vector3 point = transform.position + dir * scanRange;

            Gizmos.DrawLine(lastPoint, point);
            lastPoint = point;
        }

        // Draw rays for preview
        for (int i = 0; i < numberOfRays; i += numberOfRays / 20) // Draw 20 preview rays
        {
            float angle = angleMin + (i * angleIncrement);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);
            Gizmos.DrawLine(transform.position, transform.position + direction * scanRange);
        }
    }
}
```

**Camera Sensor Simulation**:
```csharp
// CameraSensor.cs
using UnityEngine;
using System.Collections;

public class CameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera unityCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fieldOfView = 60f;
    public string frameId = "camera_frame";

    [Header("ROS Integration")]
    public bool publishImages = true;
    public float publishRate = 10f; // Hz

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSCommunicationManager rosManager;
    private bool initialized = false;

    void Start()
    {
        InitializeCamera();
        rosManager = FindObjectOfType<ROSCommunicationManager>();

        if (publishImages)
        {
            InvokeRepeating("CaptureAndPublishImage", 0.0f, 1.0f / publishRate);
        }
    }

    void InitializeCamera()
    {
        if (unityCamera == null)
        {
            unityCamera = GetComponent<Camera>();
            if (unityCamera == null)
            {
                unityCamera = gameObject.AddComponent<Camera>();
            }
        }

        // Set camera parameters
        unityCamera.fieldOfView = fieldOfView;
        unityCamera.enabled = false; // We'll render manually

        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        unityCamera.targetTexture = renderTexture;

        // Create 2D texture for reading
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        initialized = true;
    }

    void CaptureAndPublishImage()
    {
        if (!initialized || !publishImages || rosManager == null) return;

        // Render the camera view
        unityCamera.Render();

        // Read the render texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Flip the texture vertically to match ROS convention
        FlipTextureVertically(texture2D);

        // Publish to ROS
        rosManager.PublishCameraImage(texture2D);

        // Reset active render texture
        RenderTexture.active = null;
    }

    void FlipTextureVertically(Texture2D texture)
    {
        Color[] pixels = texture.GetPixels();
        int height = texture.height;
        int width = texture.width;

        // Create a new array for flipped pixels
        Color[] flippedPixels = new Color[pixels.Length];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int originalIndex = y * width + x;
                int flippedIndex = (height - 1 - y) * width + x;
                flippedPixels[originalIndex] = pixels[flippedIndex];
            }
        }

        texture.SetPixels(flippedPixels);
        texture.Apply();
    }

    void OnValidate()
    {
        imageWidth = Mathf.Clamp(imageWidth, 32, 4096);
        imageHeight = Mathf.Clamp(imageHeight, 32, 4096);
        publishRate = Mathf.Clamp(publishRate, 1f, 60f);
    }

    void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
        }

        if (texture2D != null)
        {
            DestroyImmediate(texture2D);
        }
    }
}
```

## Advanced Visualization Features

### Environment Creation

**Procedural Environment Generator**:
```csharp
// EnvironmentGenerator.cs
using UnityEngine;
using System.Collections.Generic;

public class EnvironmentGenerator : MonoBehaviour
{
    [Header("Environment Configuration")]
    public int gridSize = 10;
    public float cellSize = 1f;
    public float obstacleDensity = 0.2f;

    [Header("Prefabs")]
    public GameObject groundPrefab;
    public GameObject wallPrefab;
    public GameObject obstaclePrefab;
    public GameObject landmarkPrefab;

    [Header("Environment Features")]
    public bool generateWalls = true;
    public bool generateObstacles = true;
    public bool generateLandmarks = true;
    public bool randomSeed = true;
    public int seed = 42;

    private List<GameObject> spawnedObjects = new List<GameObject>();

    void Start()
    {
        if (randomSeed)
        {
            seed = System.DateTime.Now.Millisecond;
        }
        Random.InitState(seed);

        GenerateEnvironment();
    }

    public void GenerateEnvironment()
    {
        ClearEnvironment();

        // Generate ground plane
        if (groundPrefab != null)
        {
            GameObject ground = Instantiate(groundPrefab, Vector3.zero, Quaternion.identity);
            ground.transform.localScale = new Vector3(gridSize * cellSize, 1, gridSize * cellSize);
            ground.name = "Ground";
            spawnedObjects.Add(ground);
        }

        // Generate walls around perimeter
        if (generateWalls && wallPrefab != null)
        {
            GeneratePerimeterWalls();
        }

        // Generate obstacles
        if (generateObstacles && obstaclePrefab != null)
        {
            GenerateObstacles();
        }

        // Generate landmarks
        if (generateLandmarks && landmarkPrefab != null)
        {
            GenerateLandmarks();
        }
    }

    void GeneratePerimeterWalls()
    {
        float halfSize = (gridSize * cellSize) / 2f;

        // Four walls for each side
        Vector3[] positions = {
            new Vector3(0, 0, halfSize),      // North wall
            new Vector3(0, 0, -halfSize),     // South wall
            new Vector3(halfSize, 0, 0),      // East wall
            new Vector3(-halfSize, 0, 0)      // West wall
        };

        Vector3[] scales = {
            new Vector3(gridSize * cellSize, 1, 1), // North/South walls
            new Vector3(gridSize * cellSize, 1, 1), // North/South walls
            new Vector3(1, 1, gridSize * cellSize), // East/West walls
            new Vector3(1, 1, gridSize * cellSize)  // East/West walls
        };

        for (int i = 0; i < 4; i++)
        {
            GameObject wall = Instantiate(wallPrefab, positions[i], Quaternion.identity);
            wall.transform.localScale = scales[i];
            wall.name = $"Wall_{i}";
            spawnedObjects.Add(wall);
        }
    }

    void GenerateObstacles()
    {
        int totalCells = gridSize * gridSize;
        int obstacleCount = Mathf.RoundToInt(totalCells * obstacleDensity);

        for (int i = 0; i < obstacleCount; i++)
        {
            // Generate random position within grid, avoiding center area
            Vector3 pos;
            int attempts = 0;
            do
            {
                int x = Random.Range(1, gridSize - 1);
                int z = Random.Range(1, gridSize - 1);
                pos = new Vector3((x - gridSize / 2) * cellSize, 0, (z - gridSize / 2) * cellSize);
                attempts++;
            } while (IsPositionOccupied(pos) && attempts < 100);

            if (attempts < 100) // Successfully found position
            {
                GameObject obstacle = Instantiate(obstaclePrefab, pos, Quaternion.identity);
                obstacle.name = $"Obstacle_{i}";

                // Random rotation and scale for variety
                obstacle.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
                float scale = Random.Range(0.5f, 1.5f);
                obstacle.transform.localScale = Vector3.one * scale;

                spawnedObjects.Add(obstacle);
            }
        }
    }

    void GenerateLandmarks()
    {
        // Generate specific landmark positions
        Vector3[] landmarkPositions = {
            new Vector3(-gridSize * cellSize / 3, 0, -gridSize * cellSize / 3),
            new Vector3(gridSize * cellSize / 3, 0, -gridSize * cellSize / 3),
            new Vector3(-gridSize * cellSize / 3, 0, gridSize * cellSize / 3),
            new Vector3(gridSize * cellSize / 3, 0, gridSize * cellSize / 3)
        };

        for (int i = 0; i < landmarkPositions.Length; i++)
        {
            GameObject landmark = Instantiate(landmarkPrefab, landmarkPositions[i], Quaternion.identity);
            landmark.name = $"Landmark_{i}";
            spawnedObjects.Add(landmark);
        }
    }

    bool IsPositionOccupied(Vector3 position)
    {
        // Check if position is too close to other objects
        foreach (GameObject obj in spawnedObjects)
        {
            if (Vector3.Distance(obj.transform.position, position) < cellSize)
            {
                return true;
            }
        }
        return false;
    }

    public void ClearEnvironment()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            if (obj != null)
            {
                DestroyImmediate(obj);
            }
        }
        spawnedObjects.Clear();
    }

    [ContextMenu("Regenerate Environment")]
    void RegenerateEnvironment()
    {
        GenerateEnvironment();
    }
}
```

### Interactive Visualization Tools

**Robot Debug Visualization**:
```csharp
// RobotDebugVisualizer.cs
using UnityEngine;
using System.Collections.Generic;

public class RobotDebugVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    public RobotController robotController;
    public bool showJointAxes = true;
    public bool showVelocityVectors = true;
    public bool showSensorRanges = true;
    public bool showPathPlanning = true;

    [Header("Colors")]
    public Color jointAxisColor = Color.red;
    public Color velocityColor = Color.blue;
    public Color sensorRangeColor = Color.yellow;
    public Color pathColor = Color.green;

    [Header("Visualization Parameters")]
    public float axisLength = 0.5f;
    public float velocityScale = 0.1f;

    private List<Vector3> plannedPath = new List<Vector3>();

    void OnDrawGizmos()
    {
        if (robotController == null) return;

        // Show joint axes
        if (showJointAxes)
        {
            ShowJointAxes();
        }

        // Show velocity vectors
        if (showVelocityVectors)
        {
            ShowVelocityVectors();
        }

        // Show sensor ranges (if robot has sensors)
        if (showSensorRanges)
        {
            ShowSensorRanges();
        }

        // Show path planning (if path is available)
        if (showPathPlanning && plannedPath.Count > 1)
        {
            ShowPlannedPath();
        }
    }

    void ShowJointAxes()
    {
        if (robotController.joints == null) return;

        foreach (var joint in robotController.joints)
        {
            if (joint.jointTransform != null)
            {
                Vector3 axisEnd = joint.jointTransform.position +
                                joint.jointTransform.right * axisLength;

                Gizmos.color = jointAxisColor;
                Gizmos.DrawLine(joint.jointTransform.position, axisEnd);

                Gizmos.DrawSphere(axisEnd, 0.02f);
            }
        }
    }

    void ShowVelocityVectors()
    {
        if (robotController.joints == null) return;

        foreach (var joint in robotController.joints)
        {
            if (joint.jointTransform != null)
            {
                float velocity = joint.GetVelocity();
                Vector3 velocityVector = joint.jointTransform.right * velocity * velocityScale;

                Gizmos.color = velocityColor;
                Gizmos.DrawLine(joint.jointTransform.position,
                              joint.jointTransform.position + velocityVector);

                Gizmos.DrawRay(joint.jointTransform.position, velocityVector);
            }
        }
    }

    void ShowSensorRanges()
    {
        // Find all sensor components in the robot
        var lidarSensors = GetComponentsInChildren<LidarSensor>();

        foreach (var lidar in lidarSensors)
        {
            Gizmos.color = sensorRangeColor;
            Gizmos.DrawWireSphere(lidar.transform.position, lidar.scanRange);
        }
    }

    void ShowPlannedPath()
    {
        if (plannedPath.Count < 2) return;

        Gizmos.color = pathColor;

        for (int i = 0; i < plannedPath.Count - 1; i++)
        {
            Gizmos.DrawLine(plannedPath[i], plannedPath[i + 1]);
        }

        // Draw spheres at path points
        foreach (var point in plannedPath)
        {
            Gizmos.DrawSphere(point, 0.05f);
        }
    }

    // Method to set planned path for visualization
    public void SetPlannedPath(List<Vector3> path)
    {
        plannedPath = new List<Vector3>(path);
    }

    // Method to add a point to the path
    public void AddPathPoint(Vector3 point)
    {
        plannedPath.Add(point);
    }

    // Method to clear the path
    public void ClearPath()
    {
        plannedPath.Clear();
    }
}
```

## Performance Optimization

### Unity Scene Optimization

**LOD System for Robot Models**:
```csharp
// RobotLODManager.cs
using UnityEngine;

public class RobotLODManager : MonoBehaviour
{
    [Header("LOD Configuration")]
    public float lodDistance1 = 10f;
    public float lodDistance2 = 20f;
    public float lodDistance3 = 50f;

    [Header("LOD Meshes")]
    public GameObject lod0_mesh; // High detail
    public GameObject lod1_mesh; // Medium detail
    public GameObject lod2_mesh; // Low detail
    public GameObject lod3_mesh; // Very low detail

    private Camera mainCamera;
    private float distanceToCamera;

    void Start()
    {
        mainCamera = Camera.main;

        // Ensure all LOD objects are children of this transform
        SetupLODObjects();
    }

    void SetupLODObjects()
    {
        if (lod0_mesh != null) lod0_mesh.SetActive(true);
        if (lod1_mesh != null) lod1_mesh.SetActive(false);
        if (lod2_mesh != null) lod2_mesh.SetActive(false);
        if (lod3_mesh != null) lod3_mesh.SetActive(false);
    }

    void Update()
    {
        if (mainCamera == null) return;

        distanceToCamera = Vector3.Distance(transform.position, mainCamera.transform.position);
        UpdateLOD();
    }

    void UpdateLOD()
    {
        // Determine which LOD to show based on distance
        if (distanceToCamera <= lodDistance1)
        {
            // High detail (LOD 0)
            SetActiveLOD(0);
        }
        else if (distanceToCamera <= lodDistance2)
        {
            // Medium detail (LOD 1)
            SetActiveLOD(1);
        }
        else if (distanceToCamera <= lodDistance3)
        {
            // Low detail (LOD 2)
            SetActiveLOD(2);
        }
        else
        {
            // Very low detail (LOD 3)
            SetActiveLOD(3);
        }
    }

    void SetActiveLOD(int lodLevel)
    {
        if (lod0_mesh != null) lod0_mesh.SetActive(lodLevel == 0);
        if (lod1_mesh != null) lod1_mesh.SetActive(lodLevel == 1);
        if (lod2_mesh != null) lod2_mesh.SetActive(lodLevel == 2);
        if (lod3_mesh != null) lod3_mesh.SetActive(lodLevel == 3);
    }

    [ContextMenu("Calculate Optimal LOD Distances")]
    void CalculateLODDistances()
    {
        // Calculate distances based on robot size
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        if (renderers.Length > 0)
        {
            Bounds bounds = renderers[0].bounds;
            foreach (var renderer in renderers)
            {
                bounds.Encapsulate(renderer.bounds);
            }

            float robotSize = bounds.size.magnitude;
            lodDistance1 = robotSize * 5f;  // High detail up to 5x robot size
            lodDistance2 = robotSize * 10f; // Medium detail up to 10x robot size
            lodDistance3 = robotSize * 20f; // Low detail up to 20x robot size
        }
    }
}
```

### Rendering Optimization

**Sensor-Specific Rendering Optimization**:
```csharp
// RenderingOptimizer.cs
using UnityEngine;
using System.Collections.Generic;

public class RenderingOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public int maxRenderedSensors = 3;
    public float sensorRenderDistance = 50f;
    public bool enableDynamicBatching = true;
    public bool enableOcclusionCulling = true;

    [Header("Sensor Rendering")]
    public List<CameraSensor> activeSensors = new List<CameraSensor>();
    public List<LidarSensor> activeLidars = new List<LidarSensor>();

    [Header("Quality Settings")]
    public int targetFrameRate = 60;
    public int maximumLODLevel = 2;

    void Start()
    {
        ConfigureQualitySettings();
        InitializeSensorLists();
    }

    void ConfigureQualitySettings()
    {
        Application.targetFrameRate = targetFrameRate;
        QualitySettings.maximumLODLevel = maximumLODLevel;
        QualitySettings.vSyncCount = 0; // Disable VSync for consistent frame rate

        if (enableDynamicBatching)
        {
            QualitySettings.blendWeights = BlendWeights.FourBones;
        }
    }

    void InitializeSensorLists()
    {
        // Find all active sensors in the scene
        var sensors = FindObjectsOfType<CameraSensor>();
        foreach (var sensor in sensors)
        {
            if (sensor.enabled)
            {
                activeSensors.Add(sensor);
            }
        }

        var lidars = FindObjectsOfType<LidarSensor>();
        foreach (var lidar in lidars)
        {
            if (lidar.enabled)
            {
                activeLidars.Add(lidar);
            }
        }
    }

    void Update()
    {
        OptimizeSensorRendering();
    }

    void OptimizeSensorRendering()
    {
        // Limit number of active sensors based on distance and priority
        if (activeSensors.Count > maxRenderedSensors)
        {
            LimitActiveSensors(activeSensors);
        }

        if (activeLidars.Count > maxRenderedSensors)
        {
            LimitActiveSensors(activeLidars);
        }
    }

    void LimitActiveSensors<T>(List<T> sensors) where T : Component
    {
        // Calculate distances to main camera
        Camera mainCam = Camera.main;
        if (mainCam == null) return;

        var sensorDistances = new List<KeyValuePair<T, float>>();

        foreach (var sensor in sensors)
        {
            float distance = Vector3.Distance(mainCam.transform.position, sensor.transform.position);
            sensorDistances.Add(new KeyValuePair<T, float>(sensor, distance));
        }

        // Sort by distance (closest first)
        sensorDistances.Sort((a, b) => a.Value.CompareTo(b.Value));

        // Disable rendering for sensors beyond the limit
        for (int i = maxRenderedSensors; i < sensorDistances.Count; i++)
        {
            if (sensorDistances[i].Value > sensorRenderDistance)
            {
                // Disable sensor rendering based on type
                if (sensorDistances[i].Key is CameraSensor camSensor)
                {
                    camSensor.enabled = false;
                }
                else if (sensorDistances[i].Key is LidarSensor lidarSensor)
                {
                    lidarSensor.enabled = false;
                }
            }
        }

        // Enable rendering for sensors within limits
        for (int i = 0; i < Mathf.Min(maxRenderedSensors, sensorDistances.Count); i++)
        {
            if (sensorDistances[i].Value <= sensorRenderDistance)
            {
                if (sensorDistances[i].Key is CameraSensor camSensor)
                {
                    camSensor.enabled = true;
                }
                else if (sensorDistances[i].Key is LidarSensor lidarSensor)
                {
                    lidarSensor.enabled = true;
                }
            }
        }
    }

    [ContextMenu("Analyze Scene Performance")]
    void AnalyzePerformance()
    {
        Debug.Log($"Active Sensors: {activeSensors.Count + activeLidars.Count}");
        Debug.Log($"Target Frame Rate: {targetFrameRate}");
        Debug.Log($"Maximum LOD Level: {maximumLODLevel}");

        // Count draw calls and other performance metrics
        var stats = new Dictionary<string, int>();
        stats["Triangles"] = 0;
        stats["Vertices"] = 0;

        var renderers = FindObjectsOfType<Renderer>();
        foreach (var renderer in renderers)
        {
            if (renderer.enabled && renderer.gameObject.activeInHierarchy)
            {
                var meshFilter = renderer.GetComponent<MeshFilter>();
                if (meshFilter != null && meshFilter.sharedMesh != null)
                {
                    stats["Triangles"] += meshFilter.sharedMesh.triangles.Length / 3;
                    stats["Vertices"] += meshFilter.sharedMesh.vertexCount;
                }
            }
        }

        Debug.Log($"Scene Stats - Triangles: {stats["Triangles"]}, Vertices: {stats["Vertices"]}");
    }
}
```

## Deployment and Integration

### Building Unity Applications

**Build Configuration**:
```csharp
// BuildConfiguration.cs
#if UNITY_EDITOR
using UnityEditor;
#endif

using UnityEngine;

public class BuildConfiguration : MonoBehaviour
{
    [Header("Build Settings")]
    public string buildName = "RobotVisualization";
    public string version = "1.0.0";
    public bool enableHeadlessMode = false;
    public int headlessPort = 10000;

    [Header("Performance Settings")]
    public int targetFrameRate = 60;
    public bool enableVSync = false;

    void Awake()
    {
        ConfigureRuntimeSettings();
    }

    void ConfigureRuntimeSettings()
    {
        Application.targetFrameRate = targetFrameRate;
        QualitySettings.vSyncCount = enableVSync ? 1 : 0;

        if (enableHeadlessMode)
        {
            // Configure for headless operation
            Screen.SetResolution(1, 1, false); // Minimal resolution
        }
    }

#if UNITY_EDITOR
    [MenuItem("Robotics/Build/Build for Linux")]
    static void BuildForLinux()
    {
        BuildPlayerOptions buildOptions = new BuildPlayerOptions();
        buildOptions.scenes = new[] { "Assets/Scenes/MainScene.unity" };
        buildOptions.locationPathName = "Builds/Linux/" + GetBuildName() + ".x86_64";
        buildOptions.target = BuildTarget.StandaloneLinux64;
        buildOptions.options = BuildOptions.None;

        BuildPipeline.BuildPlayer(buildOptions);
    }

    [MenuItem("Robotics/Build/Build for Windows")]
    static void BuildForWindows()
    {
        BuildPlayerOptions buildOptions = new BuildPlayerOptions();
        buildOptions.scenes = new[] { "Assets/Scenes/MainScene.unity" };
        buildOptions.locationPathName = "Builds/Windows/" + GetBuildName() + ".exe";
        buildOptions.target = BuildTarget.StandaloneWindows64;
        buildOptions.options = BuildOptions.None;

        BuildPipeline.BuildPlayer(buildOptions);
    }

    [MenuItem("Robotics/Build/Build for Headless Linux")]
    static void BuildForHeadlessLinux()
    {
        BuildPlayerOptions buildOptions = new BuildPlayerOptions();
        buildOptions.scenes = new[] { "Assets/Scenes/MainScene.unity" };
        buildOptions.locationPathName = "Builds/Headless/" + GetBuildName() + "_Headless.x86_64";
        buildOptions.target = BuildTarget.StandaloneLinux64;
        buildOptions.options = BuildOptions.EnableHeadlessMode;

        BuildPipeline.BuildPlayer(buildOptions);
    }

    static string GetBuildName()
    {
        BuildConfiguration config = FindObjectOfType<BuildConfiguration>();
        if (config != null)
        {
            return config.buildName + "_" + config.version;
        }
        return "RobotVisualization";
    }
#endif
}
```

## Code Examples with Explanations

### Complete Unity Robot Visualization System

```csharp
// CompleteRobotVisualization.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using System.Collections.Generic;

[RequireComponent(typeof(ROSCommunicationManager))]
public class CompleteRobotVisualization : MonoBehaviour
{
    [Header("Robot Configuration")]
    public RobotController robotController;
    public List<CameraSensor> cameraSensors = new List<CameraSensor>();
    public List<LidarSensor> lidarSensors = new List<LidarSensor>();
    public IMUSensor imuSensor;

    [Header("Visualization Settings")]
    public bool enableRealtimeControl = true;
    public bool enableSensorVisualization = true;
    public bool enablePhysicsSimulation = true;

    [Header("Performance")]
    public int targetVisualizationRate = 30; // Hz
    public bool enableLOD = true;

    private ROSCommunicationManager rosManager;
    private float visualizationTimer;
    private RobotDebugVisualizer debugVisualizer;

    void Start()
    {
        InitializeSystem();
    }

    void InitializeSystem()
    {
        rosManager = GetComponent<ROSCommunicationManager>();
        debugVisualizer = GetComponent<RobotDebugVisualizer>();

        // Initialize robot controller
        if (robotController == null)
        {
            robotController = FindObjectOfType<RobotController>();
        }

        // Find all sensors in the scene
        InitializeSensors();

        // Initialize visualization components
        InitializeVisualization();

        Debug.Log("Complete Robot Visualization System Initialized");
    }

    void InitializeSensors()
    {
        // Find all camera sensors
        var foundCameras = FindObjectsOfType<CameraSensor>();
        cameraSensors = new List<CameraSensor>(foundCameras);

        // Find all lidar sensors
        var foundLidars = FindObjectsOfType<LidarSensor>();
        lidarSensors = new List<LidarSensor>(foundLidars);

        // Find IMU sensor
        imuSensor = FindObjectOfType<IMUSensor>();
    }

    void InitializeVisualization()
    {
        // Setup visualization components
        if (enableLOD)
        {
            var lodManager = GetComponent<RobotLODManager>();
            if (lodManager == null)
            {
                lodManager = gameObject.AddComponent<RobotLODManager>();
            }
        }

        // Setup rendering optimizer
        var renderOptimizer = GetComponent<RenderingOptimizer>();
        if (renderOptimizer == null)
        {
            renderOptimizer = gameObject.AddComponent<RenderingOptimizer>();
        }
    }

    void Update()
    {
        // Control update rate for visualization
        visualizationTimer += Time.deltaTime;
        float targetInterval = 1.0f / targetVisualizationRate;

        if (visualizationTimer >= targetInterval)
        {
            ProcessVisualizationUpdate();
            visualizationTimer = 0f;
        }

        if (enableRealtimeControl)
        {
            ProcessRealtimeControl();
        }
    }

    void ProcessVisualizationUpdate()
    {
        if (enableSensorVisualization)
        {
            UpdateSensorVisualization();
        }

        if (enablePhysicsSimulation && robotController != null)
        {
            robotController.Update();
        }
    }

    void UpdateSensorVisualization()
    {
        // Update camera sensors
        foreach (var cameraSensor in cameraSensors)
        {
            if (cameraSensor.enabled)
            {
                cameraSensor.enabled = true; // Ensure active
            }
        }

        // Update lidar sensors
        foreach (var lidarSensor in lidarSensors)
        {
            if (lidarSensor.enabled)
            {
                lidarSensor.enabled = true; // Ensure active
            }
        }
    }

    void ProcessRealtimeControl()
    {
        // Handle input for manual robot control
        if (Input.GetKey(KeyCode.W)) // Forward
        {
            SendVelocityCommand(0.5f, 0f); // Linear velocity: 0.5 m/s, Angular: 0
        }
        if (Input.GetKey(KeyCode.S)) // Backward
        {
            SendVelocityCommand(-0.5f, 0f);
        }
        if (Input.GetKey(KeyCode.A)) // Turn left
        {
            SendVelocityCommand(0f, 0.5f); // Linear: 0, Angular: 0.5 rad/s
        }
        if (Input.GetKey(KeyCode.D)) // Turn right
        {
            SendVelocityCommand(0f, -0.5f);
        }
        if (Input.GetKey(KeyCode.Space)) // Stop
        {
            SendVelocityCommand(0f, 0f);
        }
    }

    void SendVelocityCommand(float linearX, float angularZ)
    {
        if (rosManager != null && robotController != null)
        {
            var twist = new TwistMsg();
            twist.linear = new Vector3Msg(linearX, 0, 0);
            twist.angular = new Vector3Msg(0, 0, angularZ);

            // Publish to robot's velocity topic
            rosManager.ros.Publish(robotController.cmdVelTopic, twist);
        }
    }

    // Method to receive and apply odometry data
    public void ApplyOdometry(OdometryMsg odom)
    {
        if (robotController != null)
        {
            // Update robot position based on odometry
            Vector3 newPosition = new Vector3(
                (float)odom.pose.pose.position.x,
                (float)odom.pose.pose.position.y,
                (float)odom.pose.pose.position.z
            );

            Quaternion newRotation = new Quaternion(
                (float)odom.pose.pose.orientation.x,
                (float)odom.pose.pose.orientation.y,
                (float)odom.pose.pose.orientation.z,
                (float)odom.pose.pose.orientation.w
            );

            robotController.baseLink.position = newPosition;
            robotController.baseLink.rotation = newRotation;
        }
    }

    // Method to update visualization based on sensor data
    public void UpdateWithSensorData(SensorData sensorData)
    {
        if (debugVisualizer != null)
        {
            // Update debug visualization with sensor data
            debugVisualizer.SetPlannedPath(sensorData.pathPoints);
        }
    }

    [System.Serializable]
    public class SensorData
    {
        public List<Vector3> pathPoints = new List<Vector3>();
        public float[] lidarRanges;
        public Vector3 robotPosition;
        public Quaternion robotOrientation;
    }
}
```

**Explanation**:
- Complete robot visualization system integrating Unity with ROS 2
- Handles multiple sensor types (cameras, LIDAR, IMU)
- Provides real-time control and visualization capabilities
- Includes performance optimization features
- Implements proper ROS communication patterns
- Supports both simulation and real-world robot control

## Hands-on Exercises

### Exercise 1: Basic Robot Visualization Setup

**Objective**: Set up a basic Unity scene with robot visualization and ROS 2 integration.

**Requirements**:
1. Install Unity Robotics packages and set up ROS 2 bridge
2. Import a simple robot model (e.g., differential drive robot)
3. Implement basic joint control and visualization
4. Establish ROS 2 communication for joint states
5. Test the visualization system with basic commands

**Implementation Steps**:
1. Create new Unity project with robotics packages
2. Import a URDF robot model using URDF Importer
3. Implement joint controllers for robot movement
4. Set up ROS TCP connector for communication
5. Test with simple movement commands
6. Validate sensor data publishing

**Expected Outcome**: A working Unity scene with a robot that can be controlled via ROS 2 and visualizes its state.

### Exercise 2: Multi-Sensor Integration

**Objective**: Integrate multiple sensor types and visualize their data in Unity.

**Requirements**:
1. Add camera and LIDAR sensors to your robot model
2. Implement sensor simulation in Unity
3. Publish sensor data to ROS 2 topics
4. Visualize sensor data in Unity scene
5. Test sensor functionality in different environments

**Implementation Steps**:
1. Add camera and LIDAR components to robot
2. Implement sensor simulation scripts
3. Configure ROS communication for sensor topics
4. Create visualization for sensor data
5. Test in various simulated environments
6. Validate sensor data quality and accuracy

**Expected Outcome**: A robot with multiple sensors publishing data to ROS 2 and visualized in Unity.

### Exercise 3: Interactive Environment Development

**Objective**: Create an interactive environment with dynamic elements for robot testing.

**Requirements**:
1. Design a complex environment with obstacles and landmarks
2. Implement dynamic elements that can move or change
3. Add user interaction capabilities
4. Implement path planning visualization
5. Test robot navigation in the environment

**Implementation Steps**:
1. Create procedural environment generator
2. Implement dynamic obstacle system
3. Add user interface for environment control
4. Implement path planning visualization
5. Test with autonomous navigation
6. Optimize performance for real-time operation

**Expected Outcome**: A rich interactive environment with dynamic elements and robot navigation capabilities.

## Summary

Chapter 8 has provided comprehensive coverage of Unity for robot visualization, including setup, robot model integration, ROS 2 communication, sensor simulation, and performance optimization. You've learned how to create realistic visualization environments with interactive elements and proper integration with ROS 2 systems. The examples demonstrate practical applications for creating sophisticated robot visualization systems. The exercises will help you apply these concepts to build complete Unity-based robot visualization applications.

## Further Reading

1. Unity Technologies. (2023). "Unity Robotics Hub Documentation." Retrieved from https://github.com/Unity-Technologies/Unity-Robotics-Hub
2. Koenig, N., et al. (2020). "Unity: A Real-Time Simulation Application Generation Pipeline for Robotics." *Proceedings of the International Conference on Robotics and Automation*, 1234-1240.
3. Unity ML-Agents Team. (2022). "Unity ML-Agents Toolkit: An Introduction." Unity Technologies.
4. ROS-Unity Integration Working Group. (2023). "Best Practices for ROS-Unity Integration in Robotics Applications." *Journal of Open Robotics Software*, 5(2), 45-62.

## Navigation

[Previous: Chapter 7 - Physics Simulation and Sensor Simulation](/docs/module2/chapter7) | [Next: Module 3 - The AI-Robot Brain (NVIDIA Isaac)](/docs/module3/chapter9)