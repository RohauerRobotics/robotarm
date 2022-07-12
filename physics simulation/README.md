# 3 Joint Robot Arm Physics Simulation
By Dylan Rohauer
 <p align="center">
   <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/header_image.PNG" align="centre">
 </p>
This is a physics simulation of a three joint robot arm that I am in the process of making. The purpose of this simulation was originally just to estimate the torque requirement on the stepper motors but it has turned into a fully animated 3D simulation. This simulation is still in development and has become a crutial component of my robot design, providing verification for the image recognition and path planning. This is a version modified from the current working version to demonstrate serveral capabilities, but I may continue development of this a seperate project.

# Core Functions
This project can be broken up into two classes, each solving essesential parts of the simulation.

# class Path - inverse_kinematics()

The first part of making a good simulation is knowing where your arm needs to be. The goal for my arm is to eventually pick up an object located with image recognition and move it to a desired position. In order to go from a point in space to knowing the angles your joints need to be at you need inverse kinematics. Knowing the lengths of the arm segments and the position of the end effector you can calculate the angles nessary to reach this point. The following image shows the first step of this process which is to determine the interior angles of the triangle which represents the first two segments of the robot arm.
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/arm_triangle.png?raw=True)

Now with these angles we can find the joint angles of the arm relative to one another.
<p align="center">
   <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/arm_angles.png" align="centre" >
 </p>

Assuming the point is reachable we now have the first key nessisary to making our arm get to where it needs to go.

# class Path - acceleration_path()

With the initial angles(declared either by the operator or the last position) we now want to find the time it would take for each motor to finish it's motion at maximum acceleration following the trapezoidal motion profile. The trapezoidal motion profile means that the motor will accelerate at a fixed rate for a period of time and then decelerate until it stops at the correct position. 
<p align="center">
   <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/motion_profile.png" align="centre" >
 </p>

Using our set acceleration, we can calculate the time it will take for each joint to move from its initial position to it's end position. This is what the acceleration_path() function does, it also makes sure that the path it takes is the shortest one with some if statements. After each time to move has been calcutated, it selects the longest time and calculates what the acceleration would need to be for each joint for each to reach their final angular positions at the same time. 

# class Path - animation_path()

In order for our animation to work we need to have a select number of snapshots of our path and be able to plot them. To do this animation_path creates a list of the angular values of all of the joints sampled throughout the duration of their motion. 
<p align="center">
   <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/path_sampling.png" align="centre" >
 </p>

# class Plot - loop()

Now that we have our samples of angles for the duration of the arm's travel, we want to turn them from angles into coordinates. Loop is the function that brings everything together, it iterates through the list of samples and calls on the position function which gives it the coordinates and it in turn displays those coordinates on a matplotlib 3d display. 

# class Plot - position() 

In order to go from angles to coordinates we must understand the relationships between points. To do this we use coordinate planes, coordinate planes give us relative relationships through which we can define a point or line's location. By creating multiple coordinate planes we can define the joints and end effector of our robot arm as the orgin of these coordinate systems. Then with several homogenous transfer matricies we can specify the displacement and rotation between points. So from simply the angles and lengths of the arm segments we can consistantly create the points in space which we need to represent our robot arm.
 <p align="center">
   <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/rotation_matrix.png" align="centre" width="550" >
  <img src="https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/frame_displacement.png" align="centre" width="550" >
 </p>
