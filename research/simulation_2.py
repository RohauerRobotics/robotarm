  # simulation test 2
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import csv
import numpy as np
import time
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import math

class Path(object):
    def __init__(self, inital_angles,lengths_m):
        self.values = {"iA":inital_angles, "len":lengths_m,
        # define stepper settings
        "micro_steps":200, "w_max": np.pi, "accel":np.pi/8
        }
        # print("inital angles:\n", np.matrix(self.values['iA']))
        # print("lengths:\n", np.matrix(self.values['len']))
        final = self.inverse_kinematics([0.15,-0.1,0.1])
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
            print(np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4))
            return [np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4)]
        else:
            pass

    def animation_path(self, inital, final):
        path = [[],[],[],[]]
        theta = [0,0,0,0]
        self.step_path = [[],[],[],[]]
        for w in range(0, 4):
            couple = [final[w],inital[w]]
            abs_couple = [abs(final[w]),abs(inital[w])]
            if final[w] == 0:
                abs_couple[0] = 0
            elif inital[w] == 0:
                abs_couple[1] = 0
            else:
                pass
            print(abs_couple)
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
            if (abs(couple[0]-couple[1]) > 180):
                negative_max = abs(max(couple) - 360)
                steps = negative_max + min(abs_couple)
                theta[w] = round(np.radians(steps),3)
                steps = int(steps)
                if inital[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]+x)
                    self.step_path[w].append(True)
                elif final[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]-x)
                    # print("correct path", w)
                    self.step_path[w].append(False)
                else:
                    print("error 1", w)

            elif (abs(couple[0]-couple[1]) <= 180):
                steps = max(abs_couple) - min(abs_couple)
                theta[w] = round(np.radians(steps),3)
                steps = int(steps)
                # inital
                if inital[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]-x)
                    print("correct path", w)
                    self.step_path[w].append(False)
                # final
                elif final[w] == max(couple):
                    for x in range(0, steps):
                        path[w].append(inital[w]+x)
                    self.step_path[w].append(True)
                else:
                    print("error 2")
            else:
                print("Passed List")
                pass
            if (final[w]==inital[w]):
                path[w].append(0)
                print("appended 0")
                theta[w] = 0
            else:
                pass

        list_len = [len(i) for i in path]
        print("Path Length: ",path)
        self.dir = [self.step_path[i][0] for i in range(0,4)]
        self.theta = theta
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
    def __init__(self, path, lengths_m,inital_angles,dir,theta,step_times):
        self.values = {"iA":inital_angles, "len":lengths_m,
        # define stepper settings
        "micro_steps":200, "w_max": np.pi, "accel":np.pi/8,
        "mass": [0.15,0.2,0.4]
        }
        self.l1 = lengths_m[0]
        self.l2 = lengths_m[1]
        self.l3 = lengths_m[2]
        self.mass = [0.4,0.3,0.2]
        self.dir = dir
        self.theta = theta
        self.step_times = step_times
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
        set = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        self.line0, = self.ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
        self.line1, = self.ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
        self.line2, = self.ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
        self.line3, = self.ax.plot(set[3][0], set[3][1], 'bo', linestyle='solid', color='red')
        self.line4, = self.ax.plot(set[4][0], set[4][1], 'bo', linestyle='solid', color='red')
        self.line5, = self.ax.plot(set[5][0], set[5][1], 'bo', linestyle='solid', color='red')
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

    def magnitude(self,vector):
        return math.sqrt(sum(pow(element, 2) for element in vector))

    def position(self, angles):
        angle1 = angles[0]
        self.angle1 = np.radians(angle1)
        angle2 = angles[1]
        angle3 = angles[2]
        angle4 = angles[3]
        # rotation matrices
        r_ab = self.R_z(angle4)
        self.r_ab = r_ab
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

        r_ac = np.dot(r_ab,r_bc)
        # # print("r_ac dot product: "np.matrix(r_ac)))
        # self.h_cd = np.concatenate((r_ac,d_cd),1)
        # self.h_cd = np.concatenate((self.h_cd,[[0,0,0,1]]),0)
        # # r_ae = np.dot(r_ac,r_de)
        # self.h_de = np.concatenate((r_de,d_de),1)
        # self.h_de = np.concatenate((self.h_de,[[0,0,0,1]]),0)
        # self.h_ce = np.dot(self.h_cd,self.h_de)
        # #
        h_de = np.concatenate((r_de,d_de),1)
        h_de = np.concatenate((h_de,[[0,0,0,1]]),0)
        #
        # self.h_ce = np.dot(h_cd,h_de)

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

    def motor1_torque(self):
        self.orgin_t = [0,0,0]
        current_time = (self.step_times[0]/self.path_len)*self.x
        #
        ac = [self.h_ac[0][3],self.h_ac[1][3],self.h_ac[2][3]]
        tg1 = np.cross(ac,[0,0,9.81*self.values["mass"][0]])
        #
        tm1 = np.cross(ac,[(-np.cos(self.angle1)*self.magnitude(ac)*(np.pi/8)*self.values["mass"][0]),
        0, (np.sin(self.angle1)*self.magnitude(ac)*(np.pi/8)*self.values["mass"][0])])
        #
        ad = [self.h_ad[0][3],self.h_ad[1][3],self.h_ad[2][3]]
        tg2 = np.cross(ad,[0,0,9.81*self.values["mass"][1]])

        tm2 = np.cross(ad,[(-np.cos(self.angle1)*self.magnitude(ad)*(np.pi/8)*self.values["mass"][1]),
        0, (np.sin(self.angle1)*self.magnitude(ad)*(np.pi/8)*self.values["mass"][1])])
        #
        ae = [self.h_ae[0][3],self.h_ae[1][3],self.h_ae[2][3]]
        tg3 = np.cross(ae,[0,0,9.81*self.values["mass"][2]])
        #
        tm3 = np.cross(ae,[(-np.cos(self.angle1)*self.magnitude(ae)*(np.pi/8)*self.values["mass"][2]),
        0, (np.sin(self.angle1)*self.magnitude(ae)*(np.pi/8)*self.values["mass"][2])])
        #
        for x in range(0,3):
            self.orgin_t[x] = round(tg1[x] + tg2[x] + tg3[x]+tm1[x] + tm2[x] + tm3[x],3)

    def motor2_torque(self):
        self.motor2_t = [0,0,0]
        current_time = (self.step_times[0]/self.path_len)*self.x
        #
        cd = [self.h_ad[0][3]-self.h_ac[0][3],self.h_ad[1][3]-self.h_ac[1][3],self.h_ad[2][3]-self.h_ac[2][3]]
        tg2 = np.cross(cd,[0,0,9.81*self.values["mass"][1]])
        #
        tm2 = np.cross(cd,[(-np.cos(self.angle1)*self.magnitude(cd)*(np.pi/8)*self.values["mass"][1]),
        0, (np.sin(self.angle1)*self.magnitude(cd)*(np.pi/8)*self.values["mass"][1])])
        #
        ce = [self.h_ae[0][3]-self.h_ac[0][3],self.h_ae[1][3]-self.h_ac[1][3],self.h_ae[2][3]-self.h_ac[2][3]]
        tg3 = np.cross(ce,[0,0,9.81*self.values["mass"][2]])

        tm3 = np.cross(ce,[(-np.cos(self.angle1)*self.magnitude(ce)*(np.pi/8)*self.values["mass"][2]),
        0, (np.sin(self.angle1)*self.magnitude(ce)*(np.pi/8)*self.values["mass"][2])])

        for x in range(0,3):
            self.motor2_t[x] = round(tg2[x] + tg3[x] + tm2[x] + tm3[x],3)

    def motor3_torque(self):
        self.motor3_t = [0,0,0]
        current_time = (self.step_times[0]/self.path_len)*self.x
        #
        de = [self.h_ae[0][3]-self.h_ad[0][3],self.h_ae[1][3]-self.h_ad[1][3],self.h_ae[2][3]-self.h_ad[2][3]]
        tg3 = np.cross(de,[0,0,9.81*self.values["mass"][2]])
        #
        tm3 = np.cross(de,[(-np.cos(self.angle1)*self.magnitude(de)*(np.pi/8)*self.values["mass"][2]),
        0, (np.sin(self.angle1)*self.magnitude(de)*(np.pi/8)*self.values["mass"][2])])
        #
        for x in range(0,3):
            self.motor3_t[x] = round(tg3[x] + tm3[x],3)

    def torque_path(self):
        self.motor1_torque()
        self.motor2_torque()
        self.motor3_torque()
        #
        # print("Motor 1 Torque Magnitude:",round(np.sqrt((self.orgin_t[0]**2)+(self.orgin_t[1]**2)+(self.orgin_t[2]**2)),3),"Nm")
        # print("Motor 2 Torque Magnitude:",round(np.sqrt((self.motor2_t[0]**2)+(self.motor2_t[1]**2)+(self.motor2_t[2]**2)),3),"Nm")
        # print("Motor 3 Torque Magnitude:",round(np.sqrt((self.motor3_t[0]**2)+(self.motor3_t[1]**2)+(self.motor3_t[2]**2)),3),"Nm")


    def loop(self, path):
        self.path_len = len(path[0])
        for self.x in range(0, len(path[0])):
            # line 1
            lines = self.position([path[0][self.x], path[1][self.x], path[2][self.x], path[3][self.x]])
            # scale for model
            scale = 1000
            self.line0.set_data_3d(np.dot(lines[0][0],scale),np.dot(lines[0][1],scale),np.dot(lines[0][2],scale))
            #
            self.line1.set_data_3d(np.dot(lines[1][0],scale),np.dot(lines[1][1],scale),np.dot(lines[1][2],scale))
            #
            self.line2.set_data_3d(np.dot(lines[2][0],scale),np.dot(lines[2][1],scale),np.dot(lines[2][2],scale))
            #
            self.line3.set_data_3d(np.dot([0,self.orgin_t[0]],scale),np.dot([0,self.orgin_t[1]],scale),np.dot([0,self.orgin_t[2]],scale))
            #
            self.line4.set_data_3d(np.dot([lines[1][0][0],self.motor2_t[0]+lines[1][0][0]],scale),np.dot([lines[1][1][0],self.motor2_t[1]+lines[1][1][0]],scale),
            np.dot([lines[1][2][0],self.motor2_t[2]+lines[1][2][0]],scale))
            #
            self.line5.set_data_3d(np.dot([lines[2][0][0],self.motor3_t[0]+lines[2][0][0]],scale),np.dot([lines[2][1][0],self.motor3_t[1]+lines[2][1][0]],scale),
            np.dot([lines[2][2][0],self.motor3_t[2]+lines[2][2][0]],scale))
            # print("x: ",self.magnitude(np.dot([lines[1][0][0],self.motor2_t[0]],1000)))
            # print("y: ",self.magnitude(np.dot([lines[1][1][0],self.motor2_t[1]],1000)))
            # print("z: ",self.magnitude(np.dot([lines[1][2][0],self.motor2_t[2]],1000)))
            # path of end effector
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.0001)

bob = Path([0,0,0,0.0],[0.20, 0.150, 0.050])

print("Times collected:", bob.stepper_times)
plot = Plot(bob.path,[0.20, 0.150, 0.050],[0,0,0,0.0],bob.dir,bob.theta,bob.stepper_times)
