# simulation test 2
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import csv
import numpy as np
import time
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d

class Path(object):
    def __init__(self, inital_angles,lengths_m):
        self.values = {"iA":inital_angles, "len":lengths_m,
        # define stepper settings
        "micro_steps":200, "w_max": np.pi, "accel":np.pi/8
        }
        # print("inital angles:\n", np.matrix(self.values['iA']))
        # print("lengths:\n", np.matrix(self.values['len']))
        final = self.inverse_kinematics([0.15,0.1,0.1])
        self.path, theta = self.animation_path(self.values['iA'],final)
        print(self.step_path)
        print(theta)
        self.stepper_path(theta)
        # print(path,'\n')
        # print(theta)

    def inverse_kinematics(self, end_e):
        x = end_e[0]
        y = end_e[1]
        z = end_e[2]
        # lengths
        l1 = self.values['len'][0]
        l2 = self.values['len'][1]
        # h length of triangle
        h = np.sqrt((x**2 + z**2))
        # print("hypotenuse", h)
        #
        if h > np.sqrt((l1**2 + l2**2)):
            print("Unable to reach")
            path = False
        elif h <= np.sqrt((l1**2 + l2**2)):
            print("Able to Reach")
            path = True
        else:
            pass

        if path:
            # adjacent angles
            if (l1**2+h**2-l2**2) >= (2*l1*h):
                phi1 = 0
            else:
                phi1 = np.arccos((l1**2+h**2-l2**2)/(2*l1*h))
                # print("phi1: ",np.degrees(phi1))
            # phi 2
            if phi1 == 0:
                phi2 = (np.pi)
            else:
                phi2 = np.arccos((l2**2+l1**2-h**2)/(2*l2*l1))
                # print("phi 2: ",np.degrees(phi2))
            # avoid divide by 0 error
            if x != 0:
                phi4 = np.arctan(z/x)
                # print("phi 4: ",np.degrees(phi4))
            elif x == 0:
                phi4 = (np.pi/2)
            else:
                pass
            # angles for postion finder
            # print("angle relative to x axis",np.degrees(phi1+phi4))
            angle1 = (np.pi/2)- (phi4 + phi1)
            angle2 = np.pi - phi2
            #
            if abs(angle1-angle2) > np.radians(15):
                angle3 = np.pi - angle1 - angle2
            else:
                angle3 = 0
            #
            if x == 0:
                angle4 = 0
            elif x != 0:
                angle4 = np.arctan(y/x)
            # print(np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4))
            return [np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4)]
        else:
            pass

    def animation_path(self, inital, final):
        path = [[],[],[],[]]
        theta = [0,0,0,0]
        self.step_path = [[],[],[],[]]
        for w in range(0, 4):
            couple = [final[w],inital[w]]
            # print(couple)
            # adjust values if outside range
            if(inital[w])>360:
                inital[w] = inital[w]-360
            elif(inital[w])<0:
                inital[w] = inital[w] + 360
            elif(final[w])>360:
                final[w] = final[w]-360
            elif(final[w])<0:
                final[w] = final[w] + 360
            else:
                pass
            # determine shortest number of steps between angles
            if (abs(final[w]-inital[w]) > 180):
                negative_max = abs(max(couple) - 360)
                steps = negative_max + min(couple)
                theta[w] = round(np.radians(steps),3)
                steps = int(steps)
                if inital[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]+x)
                    self.step_path[w].append(True)
                elif final[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]-x)
                    self.step_path[w].append(False)
                else:
                    print("error")

            elif (abs(final[w]-inital[w]) <= 180):
                steps = max(couple) - min(couple)
                theta[w] = round(np.radians(steps),3)
                steps = int(steps)
                if inital[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]-x)
                    self.step_path[w].append(False)
                elif final[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]+x)
                    self.step_path[w].append(True)
                else:
                    print("error")
            else:
                pass

            if (final[w]==inital[w]):
                path[w].append(0)
                theta[w] = 0
            else:
                pass

        list_len = [len(i) for i in path]
        for w in range(0, len(path)):
            path[w].extend([path[w][-1]]*(max(list_len)-len(path[w])))
        return path, theta

    def velocity(self, angle):
        return np.sqrt(2*self.values["w_max"]*angle)

    def time_at(self, angle):
        return np.sqrt((2*angle)/self.values["accel"])

    def stepper_path(self, theta):
        # for x in range(0, )
        # radians per step
        self.stepper_times = []
        rps = (np.pi)/self.values["micro_steps"]
        for w in range(0, len(self.step_path)):
            #
            # print(int(theta[w]*(100/np.pi)))
            self.stepper_times.append(self.time_at(theta[w]))
            for x in range(1, int(theta[w]*(100/np.pi))+1):
                angle = x*rps
                if (self.velocity(angle) > self.values["w_max"]):
                    delay = 0.01
                    self.step_path[w].append(delay)
                elif (self.velocity(angle) <= self.values["w_max"]):
                    delay = (self.time_at(angle)/x)
                    self.step_path[w].append(delay)
                else:
                    pass
        # print(self.step_path[2])\
        # print(self.stepper_times)

class Plot(object):
    def __init__(self, path, lengths_m,inital_angles):
        self.values = {"iA":inital_angles, "len":lengths_m,
        # define stepper settings
        "micro_steps":200, "w_max": np.pi, "accel":np.pi/8,
        "mass": [0.15,0.2,0.4]
        }
        self.l1 = lengths_m[0]
        self.l2 = lengths_m[1]
        self.l3 = lengths_m[2]
        self.mass = [0.4,0.3,0.2]
        plt.ion()
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        # Setting the axes properties
        self.ax.set_xlim3d([-500, 500])
        self.ax.set_xlabel('X')
        #
        self.ax.set_ylim3d([-500, 500])
        self.ax.set_ylabel('Y')
        #
        self.ax.set_zlim3d([0.0, 500])
        self.ax.set_zlabel('Z')
        #
        self.ax.set_title('3D Test')
        set = [[[],[]],[[],[]],[[],[]],[[],[]]]
        self.line0, = self.ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
        self.line1, = self.ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
        self.line2, = self.ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
        self.line3, = self.ax.plot(set[3][0], set[3][1], 'bo', linestyle='solid', color='red')
        # self.torque_path()
        for x in range(0,10):
            self.loop(path)

    def R_x(self, angle):
        rad = np.radians(angle)
        return [[1,0,0],[0,np.cos(rad),-np.sin(rad)],[0,np.sin(rad),np.cos(rad)]]

    def R_y(self, angle):
        rad = np.radians(angle)
        return [[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0, np.cos(rad)]]

    def R_z(self, angle):
        rad = np.radians(angle)
        return [[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]

    def position(self, angles):
        angle1 = angles[0]
        angle2 = angles[1]
        angle3 = angles[2]
        angle4 = angles[3]
        # rotation matrices
        r_ab = self.R_z(angle4)
        # print("ab rotation:\n",np.matrix(r_ab))
        r_bc = self.R_y(angle1)
        # print("bc rotation:\n",np.matrix(r_bc))
        r_cd = self.R_y(angle2)
        r_de = self.R_y(angle3)

        # displacement matricies
        d_ab = [[0],[0],[0]]
        d_bc = [[np.sin(np.radians(angle1))*self.l1],[0],[np.cos(np.radians(angle1))*self.l1]]
        d_cd = [[np.sin(np.radians(angle2))*self.l2],[0],[np.cos(np.radians(angle2))*self.l2]]
        d_de = [[np.sin(np.radians(angle3))*self.l3],[0],[np.cos(np.radians(angle3))*self.l3]]
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
        self.h_ac = h_ac
        h_ad = np.dot(h_ac, h_cd)
        self.h_ad = h_ad
        h_ae = np.dot(h_ad, h_de)
        self.h_ae = h_ae
        self.torque_path()
        return [[[0,h_ac[0][3]],[0,h_ac[1][3]],[0,h_ac[2][3]]],
        [[h_ac[0][3],h_ad[0][3]],[h_ac[1][3],h_ad[1][3]],[h_ac[2][3],h_ad[2][3]]]
        ,[[h_ad[0][3],h_ae[0][3]],[h_ad[1][3],h_ae[1][3]],[h_ad[2][3],h_ae[2][3]]]]

    def torque_path(self):
        self.orgin_t = [0,0,0]
        t1 = np.cross([self.h_ac[0][3],self.h_ac[1][3],self.h_ac[2][3]],[0,0,9.81*self.values["mass"][0]])
        # print("torque \n",np.matrix(t1))
        t2 = np.cross([self.h_ad[0][3],self.h_ad[1][3],self.h_ad[2][3]],[0,0,9.81*self.values["mass"][1]])
        # print("torque \n",np.matrix(t2))
        t3 = np.cross([self.h_ae[0][3],self.h_ae[1][3],self.h_ae[2][3]],[0,0,9.81*self.values["mass"][2]])
        # print("torque \n",np.matrix(t3))
        for x in range(0,3):
            self.orgin_t[x] = round(t1[x] + t2[x] + t3[x],3)
        # print("torque \n",np.matrix(self.orgin_t))

        print("Torque Magnitude:",round(np.sqrt((self.orgin_t[0]**2)+(self.orgin_t[1]**2)+(self.orgin_t[2]**2)),3),"Nm")


    def loop(self, path):
        for x in range(0, len(path[0])):
            # line 1
            lines = self.position([path[0][x], path[1][x], path[2][x], path[3][x]])
            # xyz0 = line_data(path[0][x],path[3][x],values['arm_lengths_milimeters'][0],[0,0,0])
            self.line0.set_data_3d(np.dot(lines[0][0],1000),np.dot(lines[0][1],1000),np.dot(lines[0][2],1000))
            # print(lines[0][0]*1000,lines[0][1]*1000,lines[0][2]*1000,"\n")
            # line 2
            # xyz1 = line_data(path[1][x],path[3][x],values['arm_lengths_milimeters'][1],xyz0)
            self.line1.set_data_3d(np.dot(lines[1][0],1000),np.dot(lines[1][1],1000),np.dot(lines[1][2],1000))
            # print(np.dot(lines[1][0],1000),np.dot(lines[1][1],1000),np.dot(lines[1][2],1000),"\n")
            # print(lines[1][0]*1000,lines[1][1]*1000,lines[1][2]*1000,"\n")
            # line 3
            # xyz2 = line_data(path[2][x],path[3][x],values['arm_lengths_milimeters'][2],xyz1)
            self.line2.set_data_3d(np.dot(lines[2][0],1000),np.dot(lines[2][1],1000),np.dot(lines[2][2],1000))
            # print(np.dot(lines[2][0],1000),np.dot(lines[2][1],1000),np.dot(lines[2][2],1000),"\n")
            self.line3.set_data_3d(np.dot([0,self.orgin_t[0]],1000),np.dot([0,self.orgin_t[1]],1000),np.dot([0,self.orgin_t[2]],1000))
            # path of end effector
            # for w in range(0,3):
            #     pts[w].append(xyz2[w])
            # dots.set_data_3d(pts[0],pts[1],pts[2])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.0001)
            # pts = [[],[],[]]

bob = Path([0,0,0,0.0],[0.20, 0.150, 0.050])
print(bob.path)
plot = Plot(bob.path,[0.20, 0.150, 0.050],[0,0,0,0.0])
