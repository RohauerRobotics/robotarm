# rotational matrices test
import numpy as np
angle_1 = 30
angle_2 = 60
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

print(R0_2)
