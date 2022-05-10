# rotational matrices test
import numpy as np
angle_1 = 30
angle_2 = 90

a1 = 5 # length of link a1 in cm
a2 = 6 # length of link a2 in cm
a3 = 5.5 # length of link a3 in cm
a4 = 5.5 # length of link a4 in cm

def R_x(angle):
    rad = np.radians(angle)
    return [[1,0,0],[0,np.cos(rad),-np.sin(rad)],[0,np.sin(rad),np.cos(rad)]]

def R_y(angle):
    rad = np.radians(angle)
    return [[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0, np.cos(rad)]]

def R_z(angle):
    rad = np.radians(angle)
    return [[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]

R0_1 = R_z(angle_1)
R1_2 = R_z(angle_2)

R0_2 = np.dot(R0_1,R1_2)

# print(np.matrix(R0_1),"\n")

d0_1 = [[a2*np.cos(np.radians(angle_1))],[a2*np.sin(np.radians(angle_1))],[a1]]
d1_2 = [[a4*np.cos(np.radians(angle_2))],[a4*np.sin(np.radians(angle_2))],[a3]]

# print(np.matrix(d0_1),"\n")

# 1 designates left, right order
H0_1 = np.concatenate((R0_1,d0_1),1)
H0_1 = np.concatenate((H0_1,[[0,0,0,1]]),0)

# print(np.matrix(H0_1),"\n")
# print(np.matrix(H_1),"\n")

H1_2 = np.concatenate((R1_2,d1_2),1)
H1_2 = np.concatenate((H1_2,[[0,0,0,1]]),0)

# print(np.matrix(H1_2),"\n")

H0_2 = np.dot(H0_1,H1_2)

print(np.matrix(H0_2))
