import numpy as np

angle1 = 45
angle2 = 0
angle3 = 0
angle4 = 45

l1 = 0.150
l2 = 0.200
l3 = 0.050

def R_x(angle):
    rad = np.radians(angle)
    return [[1,0,0],[0,np.cos(rad),-np.sin(rad)],[0,np.sin(rad),np.cos(rad)]]

def R_y(angle):
    rad = np.radians(angle)
    return [[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0, np.cos(rad)]]

def R_z(angle):
    rad = np.radians(angle)
    return [[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]


# rotation matrices
r_ab = R_z(angle4)
# print("ab rotation:\n",np.matrix(r_ab))
r_bc = R_y(angle1)
print("bc rotation:\n",np.matrix(r_bc))
r_cd = R_y(angle2)
r_de = R_y(angle3)

# displacement matricies
d_ab = [[0],[0],[0]]
d_bc = [[np.sin(np.radians(angle1))*l1],[0],[np.cos(np.radians(angle1))*l1]]
d_cd = [[np.sin(np.radians(angle2))*l2],[0],[np.cos(np.radians(angle2))*l2]]
d_de = [[np.sin(np.radians(angle3))*l3],[0],[np.cos(np.radians(angle3))*l3]]
# matricies = [r_ab, r_bc, r_cd, r_de]
# homogenous transfer matricies
h_ab = np.concatenate((r_ab,d_ab),1)
h_ab = np.concatenate((h_ab,[[0,0,0,1]]),0)
#
h_bc = np.concatenate((r_bc,d_bc),1)
h_bc = np.concatenate((h_bc,[[0,0,0,1]]),0)
#
h_cd = np.concatenate((r_cd,d_cd),1)
h_cd = np.concatenate((h_cd,[[0,0,0,1]]),0)
# #
h_de = np.concatenate((r_de,d_de),1)
h_de = np.concatenate((h_de,[[0,0,0,1]]),0)

h_ac = np.dot(h_ab, h_bc)
print("\n x: ", round(h_ac[0][3],3), "y: ", round(h_ac[1][3],3), "z: ", round(h_ac[2][3],3))

h_ad = np.dot(h_ac, h_cd)
# print("\n x: ", round(h_ce[0][3],3), "y: ", round(h_ce[1][3],3), "z: ", round(h_ce[2][3],3))

h_ae = np.dot(h_ad, h_de)
# h_ae = np.dot(h_ad, h_de)
# r_ae = np.dot(r_ab, r_bc, r_cd, r_de)

# print(np.matrix(h_ae))
print("\n x: ", round(h_ae[0][3],3), "y: ", round(h_ae[1][3],3), "z: ", round(h_ae[2][3],3))
