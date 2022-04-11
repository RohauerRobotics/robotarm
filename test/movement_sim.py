import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
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

orgin_pt = [0,0]

def get_pts(angle, length):
    x = np.cos(np.radians(angle))*length
    y = np.sin(np.radians(angle))*length
    return([x,y])

pt1 = get_pts(0,values['arm_lengths_milimeters'][0])

x_vals = [orgin_pt[0], pt1[0]]
y_vals = [orgin_pt[1],pt1[1]]

# figure(figsize=(8, 6), dpi=80)

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([-500,500])
plt.ylim([0,1000])
xline1 = []
yline1 = []
xline2 = []
yline2 = []
xline3 = []
yline3 = []
x_dots = []
y_dots = []
line1, = ax.plot(xline1, yline1, 'bo', linestyle='solid')
line2, = ax.plot(xline2, xline2, 'bo', linestyle='solid')
line3, = ax.plot(xline3, xline3, 'bo', linestyle='solid')
dots, = ax.plot(x_dots, y_dots, ".")

while 1:
    for x in range(0,180):
        # line 1
        line_1_updated = get_pts(x,values['arm_lengths_milimeters'][0])
        line1.set_xdata([0,line_1_updated[0]])
        line1.set_ydata([0,line_1_updated[1]])
        # line 2
        line_2_updated = get_pts(int(x/4),values['arm_lengths_milimeters'][1])
        line_2_updated = [a+b for a,b in zip(line_1_updated, line_2_updated)]
        line2.set_xdata([line_1_updated[0], line_2_updated[0]])
        line2.set_ydata([line_1_updated[1], line_2_updated[1]])
        # line 3
        line_3_updated = get_pts(270,values['arm_lengths_milimeters'][2])
        line_3_updated = [a+b for a,b in zip(line_2_updated, line_3_updated)]
        line3.set_xdata([line_2_updated[0], line_3_updated[0]])
        line3.set_ydata([line_2_updated[1], line_3_updated[1]])

        x_dots.append(line_1_updated[0])
        y_dots.append(line_1_updated[1])
        dots.set_xdata(x_dots)
        dots.set_ydata(y_dots)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.0001)
    x_dots = []
    y_dots = []

# plt.plot(x_vals, y_vals, 'bo', linestyle = 'solid')
# plt.text(orgin_pt[0]-0.015, orgin_pt[1]+0.25, "Point1")
# plt.text(pt1[0]-0.050, pt1[1]-0.25, "Point2")
# plt.show()
