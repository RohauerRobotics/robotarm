# 3 Joint Robot Arm Physics Simulation
By Dylan Rohauer
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/header_image.PNG?raw=True)

This is a physics simulation of a three joint robot arm that I am in the process of making. The purpose of this simulation was originally just to estimate the torque requirement on the stepper motors but it has turned into a fully animated 3D simulation. This simulation is still in development and has become a crutial component of my robot design, providing verification for the image recognition and path planning. This is a version modified from the current working version to demonstrate serveral capabilities, but I may continue development of this a seperate project.

# Core Functions
This project can be broken up into two classes, each solving essesential parts of the simulation.

class Path
The class Path performs a few essential functions. The first of which is inverse kinematics, this is the process of taking a point input and computing the angles of the robot arm required for the end effector to meet this goal position.
![alt text](https://github.com/RohauerRobotics/robotarm/blob/working/physics%20simulation/images/arm_triangle.png?raw=True)

