import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import csv
import numpy as np
import time
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d

def dictionary_inputs():
    len_list = []
    mass_list = []
    with open("model_inputs.csv", mode = "r" ) as file:
        info = csv.reader(file, delimiter = ",")
        for row in info:
            if row[0] != 'arm_lengths_milimeters':
                len_list.append(int(row[0]))
            else:
                pass
            if row[1] != 'arm_mass_grams':
                mass_list.append(int(row[1]))
            else:
                pass
            #print(row[1])
    d = {'arm_lengths_milimeters': len_list,'arm_mass_grams': mass_list}
    return(d)

values = dictionary_inputs()
inital_angles = [80,80,269]

def line_data(vertical_angle, horizontal_angle, length, orgin):
    x_val = np.cos(np.radians(vertical_angle))*length + orgin[0]
    y_val = np.sin(np.radians(horizontal_angle))*length + orgin[1]
    z_val = np.sin(np.radians(vertical_angle))*length + orgin[2]
    return [x_val, y_val, z_val]

def find_angles(x,z):
    h = np.sqrt((x**2 + z**2))
    l2 = values['arm_lengths_milimeters'][1]
    l1 = values['arm_lengths_milimeters'][0]
    angle_1 = np.arccos(-(l2**2-l1**2-h**2)/(2*l1*h))
    angle_1_addition = np.arctan(z/x)
    angle_2 = (np.arccos(((np.sin(angle_1)*l1)/l2)) + (np.pi/2) - angle_1) + np.pi + angle_1+ angle_1_addition
    angle_3 = 270
    if np.radians(angle_2) >= 360:
        angle_2 -= 2*np.pi
    return [np.degrees(angle_1+angle_1_addition),np.degrees(angle_2),angle_3]

def path_to(inital, final):
    path = [[],[],[]]
    for w in range(0, 3):
        print(inital[w])
        if(inital[w] > final[w]):
            for k in range(0, abs(int(inital[w]-final[w]))+1):
                path[w].append(inital[w]-k)
                # print(w,"is moving down", abs(int(inital[w]-final[w]))+1,"units")
        elif(inital[w] < final[w]):
            for q in range(0, abs(int(inital[w]-final[w]))+1):
                path[w].append(inital[w]+q)
                # print(w,"is moving up", abs(int(inital[w]-final[w]))+1,"units")
        else:
            path[w] = [inital[w],initial[w]]
    return path

def path_smoother(path):
    list_len = [len(i) for i in path]
    # print(list_len)
    print(len(path))
    for w in range(0, len(path)):
        path[w].extend([path[w][-1]]*(max(list_len)-len(path[w])))
    return path

def end_effector_position(angles):
    z = 0
    x = 0
    for i in range(0,3):
        z += np.sin(np.radians(angles[i]))*values['arm_lengths_milimeters'][i]
        # print(z)
    for i in range(0,3):
        x += np.cos(np.radians(angles[i]))*values['arm_lengths_milimeters'][i]
        # print(x)
    return [x, 0, z]

test = find_angles(225,100)
path_1 = path_to(inital_angles, test)
print("Initial Angles", inital_angles, "Final Angles", test)
path = path_smoother(path_1)

plt.ion()

fig = plt.figure()
ax = p3.Axes3D(fig)
# Setting the axes properties
ax.set_xlim3d([-500, 500])
ax.set_xlabel('X')

ax.set_ylim3d([-500, 500])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 500])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# ln1 = [0,0,0]
# ln2 = [5,5,5]
set = [[[],[]],
[[],[]],
[[],[]]]
start = end_effector_position(inital_angles)
end = end_effector_position(test)
print(start)
pts = [[start[0],end[0]],[start[1],end[1]],[start[2],end[2]]]
line0, = ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
line1, = ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
line2, = ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
dots, = ax.plot3D(pts[0],pts[1],pts[2], ".")

while True:
    for x in range(0,len(path[0])):
        # line 1
        xyz0 = line_data(path[0][x],0,values['arm_lengths_milimeters'][0],[0,0,0])
        line0.set_data_3d([0,xyz0[0]],[0,xyz0[1]],[0,xyz0[2]])
        # line 2
        xyz1 = line_data(path[1][x],0,values['arm_lengths_milimeters'][1],xyz0)
        line1.set_data_3d([xyz0[0],xyz1[0]],[xyz0[1],xyz1[1]],[xyz0[2],xyz1[2]])
        # line 3
        xyz2 = line_data(path[2][x],0,values['arm_lengths_milimeters'][2],xyz1)
        line2.set_data_3d([xyz1[0],xyz2[0]],[xyz1[1],xyz2[1]],[xyz1[2],xyz2[2]])
        # path of end effector
        # for w in range(0,3):
        #     pts[w].append(xyz2[w])
        # dots.set_data_3d(pts[0],pts[1],pts[2])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.0001)
    pts = [[],[],[]]

# plt.plot(x_vals, y_vals, 'bo', linestyle = 'solid')
# plt.text(orgin_pt[0]-0.015, orgin_pt[1]+0.25, "Point1")
# plt.text(pt1[0]-0.050, pt1[1]-0.25, "Point2")
# plt.show()
