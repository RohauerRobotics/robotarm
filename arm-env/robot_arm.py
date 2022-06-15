# simulation test 2
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import csv
import numpy as np
import time
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import math
from imageprocess import Image_Processing
#
from multiprocessing import Lock, Process, Queue, current_process
import queue # imported for using queue.Empty exception
import cv2

class Path(object):
  def __init__(self, inital_angles,lengths_m):
      self.values = {"iA":inital_angles, "len":lengths_m,
      # define stepper settings
      "micro_steps":200, "w_max": np.pi, "accel":np.pi/8
      }

  def find_path_to(self, goal):
      final, bool = self.inverse_kinematics(goal)
      if bool:
          print("Angles: ", final)
          self.path, theta = self.animation_path(self.values['iA'],final)
          self.stepper_path(theta)
      else:
          self.path = None
          self.step_path = None

      return self.path, self.step_path, bool

  def inverse_kinematics(self, end_e):
      x = end_e[0]
      y = end_e[1]
      z = end_e[2]
      # lengths
      l1 = self.values['len'][0]
      l2 = self.values['len'][1]
      # h length of triangle
      h = np.sqrt((x**2 + z**2))
      print("Distance to object:",h)
      # print("hypotenuse", h)
      #
      if h > np.sqrt((l1**2 + l2**2)):
          print("Unable to reach")
          bool = False
      elif h <= np.sqrt((l1**2 + l2**2)):
          print("Able to Reach")
          bool = True
      else:
          pass

      if bool:
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
              phi4 = np.arctan(abs(z)/abs(x))
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
              if ((x < 0) ^ (y<0)):
                  angle4 = np.arctan(y/x) + np.pi
              else:
                  angle4 = np.arctan(y/x)
          # print("Angles:",np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4))
          path= [np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4)]
      else:
          pass
      return path, bool

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
          # print(abs_couple)
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
                  # print("correct path", w)
                  self.step_path[w].append(False)
              # final
              elif final[w] == max(couple):
                  # print("Number of Steps:", steps)
                  # print("Path :", w)
                  # print("inital", inital[w])
                  # print("path", path[w])
                  for x in range(0, steps):
                      path[w].append(inital[w]+x)
                  self.step_path[w].append(True)
              else:
                  print("error 2")
          else:
              # print("Passed List")
              pass
          if (final[w]==inital[w]):
              path[w].append(0)
              # print("appended 0")
              theta[w] = 0
          else:
              pass
      list_len = [len(i) for i in path]
      # print("Path Length: ",path)
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
  def __init__(self,lengths_m):
      self.values = {"len":lengths_m,
      # define stepper settings
      "micro_steps":200, "w_max": np.pi, "accel":np.pi/8,
      "mass": [0.15,0.2,0.4]
      }
      self.l1 = lengths_m[0]
      self.l2 = lengths_m[1]
      self.l3 = lengths_m[2]
      self.mass = [0.4,0.3,0.2]
      self.R_id =[[1,1,1],[1,1,1],[1,1,1]]

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

  def position(self, angles, grip_width):
      angle1 = angles[0]
      self.angle1 = np.radians(angle1)
      angle2 = angles[1]
      angle3 = angles[2]
      angle4 = angles[3]
      angle5 = angles[4]
      # rotation matrices
      r_ab = self.R_z(angle4)
      self.r_ab = r_ab
      r_bc = self.R_y(angle1)
      r_cd = self.R_y(angle2)
      r_de = self.R_y(angle3)
      r_ef = self.R_z(angle5)

      # displacement matricies
      d_ab = [[0],[0],[0]]
      d_bc = [[np.sin(np.radians(angle1))*self.l1],[0],[np.cos(np.radians(angle1))*self.l1]]
      d_cd = [[np.sin(np.radians(angle2))*self.l2],[0],[np.cos(np.radians(angle2))*self.l2]]
      d_de = [[np.sin(np.radians(angle3))*self.l3],[0],[np.cos(np.radians(angle3))*self.l3]]
      # displacement matricies for claw
      d_ef = [[0],[0],[-self.l3/2]]
      d_f1 = [[grip_width/2],[0],[0]]
      d_f2 = [[-grip_width/2],[0],[0]]

      # homogenous transfer matricies
      h_ab = np.concatenate((r_ab,d_ab),1)
      h_ab = np.concatenate((h_ab,[[0,0,0,1]]),0)
      #
      h_bc = np.concatenate((r_bc,d_bc),1)
      h_bc = np.concatenate((h_bc,[[0,0,0,1]]),0)
      #
      h_cd = np.concatenate((r_cd,d_cd),1)
      h_cd = np.concatenate((h_cd,[[0,0,0,1]]),0)
      #
      h_de = np.concatenate((r_de,d_de),1)
      h_de = np.concatenate((h_de,[[0,0,0,1]]),0)
      #
      h_ef = np.concatenate((r_ef,d_ef),1)
      h_ef = np.concatenate((h_ef,[[0,0,0,1]]),0)
      #
      h_f1 = np.concatenate((self.R_id,d_f1),1)
      h_f1 = np.concatenate((h_f1,[[0,0,0,1]]),0)
      #
      h_f2 = np.concatenate((self.R_id,d_f2),1)
      h_f2 = np.concatenate((h_f2,[[0,0,0,1]]),0)
      #
      h_ac = np.dot(h_ab, h_bc)
      h_ad = np.dot(h_ac, h_cd)
      h_ae = np.dot(h_ad, h_de)
      # end effector test
      h_af = np.dot(h_ae,h_ef)
      h_a1 = np.dot(h_af,h_f1)
      h_a2 = np.dot(h_af,h_f2)
      # print("h_af: ", h_af)
      # print('/n')
      # print("h_a1: ", h_a1)

      return [[[0,h_ac[0][3]],[0,h_ac[1][3]],[0,h_ac[2][3]]],
      [[h_ac[0][3],h_ad[0][3]],[h_ac[1][3],h_ad[1][3]],[h_ac[2][3],h_ad[2][3]]]
      ,[[h_ad[0][3],h_ae[0][3]],[h_ad[1][3],h_ae[1][3]],[h_ad[2][3],h_ae[2][3]]],
      [[h_a1[0][3],h_a2[0][3]],[h_a1[1][3],h_a2[1][3]],[h_a1[2][3],h_a2[2][3]]]]

class Features(object):
    def __init__(self):
        # print("Feature Class Activated")
        self.class_lib_sides = {'remote':4,'suitcase':4,
        "laptop":4,"keyboard":4,"cell phone":4,"book":4}

    def pass_filter_variables(self, image, object_info, state_info):
        self.img = image
        # declare variables used to sort filters
        self.object_name = object_info['name']
        # used for area estimation
        self.overhead_area = object_info['over_A']
        # overhead resolution in form [height(y), width(x)]
        self.overhead_res = object_info['over_res']
        # define number of sides
        self.object_sides = self.retrive_sides(self.object_name)
        # declare variables used to postion arm relative to object
        self.end_effector_xyz = state_info['xyz']
        self.base_rotation = state_info['angle4']
        self.end_effector_rotation = state_info['angle5']
        self.ultrasonic_height = state_info['ultra_h']
        # slope of h/x & h/y
        self.mx = state_info['m_hx']
        self.my = state_info['m_hy']

    def retrive_sides(self, name):
        num_sides = None
        for item in self.class_lib_sides:
            if item == name:
                num_sides = self.class_lib_sides[item]
            else:
                pass
        return num_sides

    def estimate_area(self):
        # define ratios of hand to overhead resolution
        # so if switching to 720p from 1280 the area will be
        # changed in resolution as well
        hand_over_x = self.img.shape[1]/self.overhead_res[1]
        hand_over_y = self.img.shape[0]/self.overhead_res[0]
        # based on form of eqution xf = h*mx*x0
        # so xf*yf = A0
        resized_area = self.overhead_area*hand_over_x*hand_over_y
        area_est = ((self.ultrasonic_height**2)*self.mx*self.my*resized_area)
        # try estimating perimeter instead of area
        self.perm_est = 2*np.pi*np.sqrt((area_est/np.pi))
        print("Original Area: ",self.overhead_area)
        print("Object in new image perimeter estimation: ", self.perm_est)
        return self.perm_est

    def increase_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def item_outline(self, passed_img):
        threshold = 200
        perm_adj = 50
        color = (255,255,255)
        gray = cv2.cvtColor(passed_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray,(3,3))
        cn_out = cv2.Canny(gray,threshold,threshold*2)
        contours, _ = cv2.findContours(cn_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rotated_rectangles = []
        boxes = []
        areas = []
        for x in range(0,len(contours)):
            # print("Contour Area:", cv2.contourArea(contours[x]))
            if ((cv2.arcLength(contours[x],False) > self.perm_est-perm_adj)):
                areas.append(cv2.contourArea(contours[x]))
                poly = cv2.approxPolyDP(contours[x],3,True)
                rect = cv2.minAreaRect(poly)
                rotated_rectangles.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxes.append(box)
            else:
                pass
        # create empty screen to draw shape on
        drawing = np.zeros((cn_out.shape[0], cn_out.shape[1], 3), dtype=np.uint8)
        # draw contours
        for i in range(0, len(boxes)):
            cv2.drawContours(drawing,[boxes[i]],0,color,2)
        if len(boxes) == 0:
            found =  False
        else:
            found = True

        return found, drawing, rotated_rectangles

    def look_for_object(self, hand_poll, hand_push,object_info, state_info):
        self.estimate_area()
        # apply kernal filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        passed_img = cv2.filter2D(self.img, -1, kernel)
        cv2.imwrite('passed_img.jpg',passed_img)
        found, drawing, rectangles = self.item_outline(passed_img)
        if not found:
            # polls hand camera for image
            hand_poll.put("Get")
            # waits
            while hand_push.empty():
                pass
            # displays image
            image = hand_push.get()
            cv2.imshow("Original Picture: ", image)
            cv2.waitKey(0)
            # load new information into class
            self.pass_filter_variables(image, object_info, state_info)
            self.estimate_area()
            # apply kernal filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            passed_img = cv2.filter2D(image, -1, kernel)
            # apply brightness increase
            passed_img = self.increase_brightness(passed_img,35)
            found, drawing, rectangles = self.item_outline(passed_img)
            if not found:
                print("Cannot Find Image")
            else:
                pass
            cv2.imshow("drawing", drawing)
            cv2.waitKey(0)
        else:
            cv2.imshow("drawing", drawing)
            cv2.waitKey(0)

class Path_Exe(object):
    def __init__(self):
        self.nan = Image_Processing()
        self.overhead_vid_path = 0
        self.hand_vid_path = 1
        arm_lengths = [0.35, 0.20, 0.050]
        self.path = Path([0,0,0,0.0],arm_lengths)
        # for multiprocessing testing only
        # travel path
        test = [True, ['remote', [132, 172, 242, 244]], ['person', [333, 44, 641, 436]]]
        self.custom_or_search(test)
        # dat = self.nan.obj_search('object')
        # if dat[0]:
        #     travel = self.nan.initialization_move_to(dat)
        #     print("Travel Path Created", travel)
        #     # print(np.matrix(travel))
        #     self.custom_or_search(travel)
        #
        # else:
        #     pass

    def custom_or_search(self, travel):
        if travel[0]:
            self.begin_multi(travel)

        elif travel[0] != True:
            print("custom directions")

        else:
            pass

    def calc_world_pos(self, obj):
        # estimated values found from bounding box while camera was on top of board
        kx = 1.244
        ky = 1.288
        center = [((obj[0]+obj[2])/2)-320,(-(obj[1]+obj[3])/2)+240,0]
        # print('/n Object Cented at:(in pxls)',center)
        w_center = [(center[0]*kx)/1000,(center[1]*ky)/1000,0]
        # print('/n Object Cented at:(in meters)',w_center)
        return w_center

    def two_three_dim(self, xy_box):
        xy_zero = self.calc_world_pos(xy_box)
        flat_coord = np.sqrt((xy_zero[0]*xy_zero[0]) + (xy_zero[1]*xy_zero[1]))
        l1 = self.path.values['len'][0]
        l2 = self.path.values['len'][1]
        max_h = (l1 + l2) - 0.15
        for x in range(0,100):
            adj = (x * 0.01)
            w = np.cos(np.radians(60))*(max_h-adj)
            if ((flat_coord-0.02) <= w) or ((flat_coord+0.02) >= w):
                xy_zero[2] = np.sin(np.radians(60))*(max_h-adj)
                break
            else:
                pass
        return xy_zero

    def cameras(self,hand_poll, hand_push, overhead_poll, overhead_push):
        # waits for poll request from image processor
        # upon request delivers current frame
        try:
            vid0 = cv2.VideoCapture(int(self.overhead_vid_path))
            vid1 = cv2.VideoCapture(int(self.hand_vid_path))
            vid1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            vid1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            vid0 = cv2.VideoCapture(self.overhead_vid_path)
            vid1 = cv2.VideoCapture(self.hand_vid_path)

        while True:
            # ret, frame = vid0.read()s
            if not overhead_poll.empty():
                ret0, frame0 = vid0.read()
                overhead_poll.get()
                overhead_push.put(frame0)
            else:
                pass

            if not hand_poll.empty():
                ret1, frame1 = vid1.read()
                hand_poll.get()
                hand_push.put(frame1)
            else:
                pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid0.release()
        vid1.relase()
        cv2.destroyAllWindows()

    def initialize_plot(self):
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
        set = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        self.line0, = self.ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
        self.line1, = self.ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
        self.line2, = self.ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
        self.line3, = self.ax.plot(set[3][0], set[3][1], 'bo', linestyle='solid')
        # self.point1, = self.ax.scatter()
        self.point1, = self.ax.plot(set[4][0], set[4][1], 'bo', linestyle='solid')

    def live_sim(self, sim_pull, sim_push):
        # plot = Instance_Plot()
        self.initialize_plot()
        while True:
            if not sim_push.empty():
                package = sim_push.get()

            elif sim_push.empty():
                for x in range(0, len(package)):
                    lines = package[x]
                    scale = 1000
                    self.line0.set_data_3d(np.dot(lines[0][0],scale),np.dot(lines[0][1],scale),np.dot(lines[0][2],scale))
                    #
                    self.line1.set_data_3d(np.dot(lines[1][0],scale),np.dot(lines[1][1],scale),np.dot(lines[1][2],scale))
                    #
                    self.line2.set_data_3d(np.dot(lines[2][0],scale),np.dot(lines[2][1],scale),np.dot(lines[2][2],scale))
                    #
                    self.line3.set_data_3d(np.dot(lines[3][0],scale),np.dot(lines[3][1],scale),np.dot(lines[3][2],scale))
                    #
                    self.point1.set_data_3d([np.dot(lines[4][0],scale),np.dot(lines[4][0],scale)],[np.dot(lines[4][1],scale),np.dot(lines[4][1],scale)],
                    [np.dot(lines[4][2],scale),np.dot(lines[4][2],scale)])
                    # path of end effector
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    # time.sleep(0.0001)
            else:
                pass

    def path_exe(self, travel, overhead_poll, overhead_push, hand_poll, hand_push, sim_pull, sim_push):
        # use inverse kinematics to determine end effector postion
        xyz = self.two_three_dim(travel[1][1])
        # define animation and stepper motor path
        # bool says whether object is within reach
        path, step_path, bool = self.path.find_path_to(xyz)
        # shift point down for goal
        pt = [xyz[0],xyz[1]]
        pt.append(xyz[2]-(self.path.values['len'][2]/2))
        # stand-in for claw angle which will be found later
        angle5_standin = 90
        gripper_width = 0.1
        # needed for image orientation
        end_angles = [path[0][-1],path[1][-1],path[2][-1],path[3][-1], angle5_standin]
        print("end_angles: ", end_angles)
        # define object_info library
        object_info = {'name':travel[1][0],
        'over_A':((travel[1][1][0]-travel[1][1][2])*(travel[1][1][1]-travel[1][1][3])),
        'over_res':[480,640]}
        # define state info
        # stand in for ultra sonic sensor
        ultra_h_standin = xyz[2]-self.path.values['len'][2]

        state_info = {'xyz':pt,'angle4':end_angles[3], 'angle5':end_angles[4],
        'ultra_h':ultra_h_standin,'m_hx':4,'m_hy':4}
        if bool:
            plotter = Plot(self.path.values['len'])
            suite = []
            for x in range(0,len(path[0])):
                post = plotter.position([path[0][x], path[1][x], path[2][x], path[3][x], angle5_standin],gripper_width)
                post.append(pt)
                suite.append(post)
            sim_push.put(suite)

            # polls hand camera for image
            hand_poll.put("Get")
            # waits
            while hand_push.empty():
                pass
            # displays image
            image = hand_push.get()
            print("Image Size:", image.shape)
            cv2.imshow("Hand Cam", image)
            cv2.waitKey(0)
            feat = Features()
            feat.pass_filter_variables(image, object_info, state_info)
            feat.look_for_object(hand_poll, hand_push, object_info, state_info)
        else:
            pass

    def begin_multi(self, travel):
        # queues for passsing data between functions
        hand_poll = Queue()
        hand_push = Queue()
        overhead_poll = Queue()
        overhead_push = Queue()
        sim_push = Queue()
        sim_pull = Queue()
        # process intiialization for running multiple functions
        # at the same time
        path_exe_p = Process(target=self.path_exe, args=(travel, overhead_poll, overhead_push,
        hand_poll, hand_push, sim_pull, sim_push))
        path_exe_p.start()
        #
        cameras_p = Process(target=self.cameras, args=(hand_poll, hand_push,
        overhead_poll, overhead_push))
        cameras_p.start()
        #
        live_sim_p = Process(target=self.live_sim, args=(sim_pull, sim_push))
        live_sim_p.start()
        processes = [path_exe_p,cameras_p,live_sim_p]
        # join processes together
        for p in processes:
            p.join()

if __name__=="__main__":
    nan = Path_Exe()
