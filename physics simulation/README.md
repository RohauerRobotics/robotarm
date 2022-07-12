# 3 Joint Robot Arm Physics Simulation
By Dylan Rohauer
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/header_image.PNG?raw=True)

This is a physics simulation of a three joint robot arm that I am in the process of making. The purpose of this simulation was originally just to estimate the torque requirement on the stepper motors but it has turned into a fully animated 3D simulation. This simulation is still in development and has become a crutial component of my robot design, providing verification for the image recognition and path planning. This is a version modified from the current working version to demonstrate serveral capabilities, but I may continue development of this a seperate project.

# Core Functions
This project can be broken up into two classes, each solving essesential parts of the simulation.

class Path - inverse_kinematics()

The first part of making a good simulation is knowing where your arm needs to be. The goal for my arm is to eventually pick up an object located with image recognition and move it to a desired position. In order to go from a point in space to knowing the angles your joints need to be at you need inverse kinematics. Knowing the lengths of the arm segments and the position of the end effector you can calculate the angles nessary to reach this point. The following image shows the first step of this process which is to determine the interior angles of the triangle which represents the first two segments of the robot arm.
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/arm_triangle.png?raw=True)

Now with these angles we can find the joint angles of the arm relative to one another.
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/arm_angles.png?raw=True)

Assuming the point is reachable we now have the first key nessisary to making our arm get to where it needs to go.

class Path - acceleration_path()

With the initial angles(declared either by the operator or the last position) we now want to find the time it would take for each motor to finish it's motion at maximum acceleration following the trapezoidal motion profile. The trapezoidal motion profile means that the motor will accelerate at a fixed rate for a period of time and then decelerate until it stops at the correct position. 
