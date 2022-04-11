from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import csv
import numpy as np
import time

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

def line_data(vertical_angle, horizontal_angle, length, orgin):
    x_val = np.cos(np.radians(vertical_angle))*length + orgin[0]
    y_val = np.sin(np.radians(horizontal_angle))*length + orgin[1]
    z_val = np.sin(np.radians(vertical_angle))*length + orgin[2]
    return [x_val, y_val, z_val]

# makes sequence path for animation
sequence = []
# seq = np.array([0,0,0])
iter = 180
for x in range(0,iter):
    sequence.append(line_data(x,0,values['arm_lengths_milimeters'][0],[0,0,0]))
    # np.append(seq,line_data(x,0,values['arm_lengths_milimeters'][0],[0,0,0]))
# print(sequence)

def update_lines(num, dataLines, lines):
    # lines.set_xdata([0,dataLines[num][0]])
    # lines.set_ydata([0,dataLines[num][1]])
    # lines.set_data(dataLines[num][0:2])
    lines.set_data_3d([0,dataLines[num][0]],[0,dataLines[num][1]],[0,dataLines[num][2]])
    return lines

lines, = ax.plot([0,0,0],[50,50,50],'bo', linestyle='solid')

line_animation = animation.FuncAnimation(fig, update_lines, 180, fargs=(sequence, lines),
                        interval=10,blit=False)

plt.show()
