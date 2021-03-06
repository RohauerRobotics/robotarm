# simulation test 2
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import csv
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
	  "micro_steps":400, "w_max": np.pi, "accel":np.pi/8
	  }
	  # self.accel = self.values['accel']
	  # self.decel = self.values['accel']
	  self.wa_max = self.values['w_max']
	  self.wc_max = self.values['w_max']

  def find_path_to(self, goal, grip_anglef, grasp_time):
	  if goal != None:
		  final, bool = self.inverse_kinematics(goal)
		  if grasp_time:
			  final.append(grip_anglef-final[3])
		  elif not grasp_time:
			  final.append(grip_anglef)
	  elif goal == None:
		  bool = False
	  else:
		  pass
	  if bool:
		  accel = self.acceleration_path(self.values['iA'],final)
		  self.path, theta = self.animation_path(self.values['iA'],final,accel)
		  self.step_path = self.stepper_path(self.step_path)
	  else:
		  self.path = None
		  self.step_path = None

	  return self.path, self.step_path, bool

  def trapezoid_function(self, time, theta, accel, max_omega=False):
	  theta = np.radians(theta)
	  decel = accel
	  if not max_omega:
		  time_a = np.sqrt((2*theta)/(accel+(accel**2/decel)))
		  if time <= time_a:
			  r = 0.5*accel*time**2
		  elif time > time_a:
			  adj_time = time - time_a
			  r = (0.5*accel*time_a**2) + (accel*time_a*adj_time) - (0.5*decel*adj_time**2)
		  else:
			  pass
	  elif max_omega:
		  time_a = self.wa_max/accel
		  # print("time a: ", time_a)
		  time_b = (theta - (self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel)))/self.wa_max
		  # print("time b", time_b)
		  if time < time_a:
			  r = 0.5*accel*time**2
		  elif (time_a <= time < (time_b + time_a)):
			  r = (0.5*accel*time_a**2) + accel*time_a*(time-time_a)
		  elif (time >= (time_b + time_a)):
			  r = (0.5*accel*time_a**2) + (accel*time_a*time_b) - (0.5*decel*(time-time_a-time_b)**2)
		  else:
			  pass
	  else:
		  pass
	  return r

  def inverted_trap(self, theta, total, accel):
	  decel = accel
	  theta = np.radians(theta)
	  total = np.radians(total)
	  time_half = np.sqrt((2*total)/(accel+(accel**2/decel)))
	  # print("time_half", time_half)
	  t_try = np.sqrt((2*theta)/accel)
	  if t_try < time_half:
		  t_ret = t_try
	  elif t_try >= time_half:
		  a = -0.5*accel
		  b = accel*time_half
		  # print('b: ',b)
		  c = theta - (total/2)
		  # print("c: ", c)
		  if ((b**2)+(4*a*c)) < 0:
			  root = np.sqrt((b**2)-(4*a*c))
		  elif ((b**2)+(4*a*c)) >= 0:
			  root = np.sqrt((b**2)+(4*a*c))
		  # print('root: ', root)
		  t_ret = -((b-root)/(2*a)) + time_half
		  # print("Time Return: ", t_ret)
	  else:
		  print("Error")
	  return t_ret

  def check_if_max(self, angle, accel):
	  angle = np.radians(angle)
	  decel = accel
	  max_theta = (self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel))
	  max = None
	  if angle < max_theta:
		  max = False
	  elif angle >= max_theta:
		  max = True
	  return max

  def find_total_time(self, angle, max, accel):
	  angle = np.radians(angle)
	  decel = accel
	  if not max:
		  time_a = np.sqrt((2*angle)/(accel+(accel**2/decel)))
		  # print("time_a", time_a)
		  time_c = time_a*(accel/decel)
		  total_time = time_a + time_c
	  elif max:
		  time_a = self.wa_max/accel
		  # print("time_a")
		  time_b = (angle - ((self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel))))/self.wa_max
		  time_c = self.wc_max/decel
		  total_time = time_a + time_b + time_c
	  else:
		  pass

	  return total_time

  def search_crit(self, elem):
	  return elem[0]

  def acceleration_path(self, inital, final):
	  path = [[0,0],[0,0],[0,0],[0,0],[0,0]]
	  a = self.values['accel']
	  for w in range(0, 5):
		  if final[w] != inital[w]:
			  couple = [final[w],inital[w]]
			  abs_couple = [abs(final[w]),abs(inital[w])]
			  #
			  if final[w] == 0:
				  abs_couple[0] = 0
			  elif inital[w] == 0:
				  abs_couple[1] = 0
			  else:
				  pass
				  #
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
				  angle = negative_max + min(abs_couple)
				  steps = int(angle)
				  maximum = self.check_if_max(angle,a)
				  if steps == 0:
					  path[w][0] = 1
					  path[w][1] = 0
				  else:
					  total_time = self.find_total_time(angle, maximum,a)
					  path[w][0] = total_time
					  path[w][1] = angle

			  elif (abs(couple[0]-couple[1]) <= 180):
				  angle = max(abs_couple) - min(abs_couple)
				  steps = int(angle)
				  # inital
				  maximum = self.check_if_max(angle,a)
				  # if w == 0:
				  if steps == 0:
					  path[w][0] = 1
					  path[w][1] = 0
				  else:
					  total_time = self.find_total_time(angle, maximum,a)
					  path[w][0] = total_time
					  path[w][1] = angle
			  else:
				  pass

		  elif (final[w]==inital[w]):
			  path[w].append(0)
		  else:
			  pass
	  # print("Time and angle path:", path)
	  copy = list(path)
	  path.sort(key=self.search_crit, reverse=True)
	  base_time = path[0][0]/2
	  print('Total Time',path[0][0])
	  accel = [0,0,0,0,0]
	  for x in range(0, 5):
		  a = np.radians(copy[x][1])/(base_time**2)
		  accel[x] = a
	  print("accel", accel)
	  return accel

  def inverse_kinematics(self, end_e):
	  x = end_e[0]
	  y = end_e[1]
	  z = end_e[2]
	  # lengths
	  l1 = self.values['len'][0]
	  l2 = self.values['len'][1]
	  # h length of triangle
	  w = np.sqrt((x**2 + y**2))
	  h = np.sqrt((w**2 + z**2))
	  # print("Distance to object:",h)
	  # print("hypotenuse", h)
	  #
	  if h > (l1+l2):
		  print("Unable to reach")
		  bool = False
	  elif h <= (l1+l2):
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
		  if w != 0:
			  phi4 = np.arctan(abs(z)/abs(w))
			  # print("phi 4: ",np.degrees(phi4))
		  elif w == 0:
			  phi4 = (np.pi/2)
		  else:
			  pass
		  # angles for postion finder
		  # print("Phi 1, 2, 4: ",np.degrees(phi1),",",np.degrees(phi2),",",np.degrees(phi4))
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
			  elif ((x<0)&(y<0)):
				  angle4 = np.arctan(y/x) + np.pi
			  else:
				  angle4 = np.arctan(y/x)
		  # print("Angles:",np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4))
		  # print("Final Angle 4:", np.degrees(angle4))
		  path = [np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4)]
	  else:
		  pass
	  return path, bool

  def animation_path(self, inital, final, accel):
	  path = [[],[],[],[],[]]
	  theta = [0,0,0,0,0]
	  self.step_path = [[],[],[],[],[]]
	  self.step_key = []
	  p = 30
	  R = self.values['micro_steps']/(2*np.pi)
	  for w in range(0, 5):
		  a = accel[w]
		  if final[w] != inital[w]:
			  couple = [final[w],inital[w]]
			  abs_couple = [abs(final[w]),abs(inital[w])]
			  # print("Initial Angles: ", couple)
			  # print("Abs Couple", abs_couple)
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
				  angle = negative_max + min(abs_couple)
				  theta[w] = round(np.radians(angle),3)
				  steps = int(np.radians(angle)*R)
				  maximum = self.check_if_max(angle,a)
				  # print("angle",w,": ", angle)
				  # print("maximum: ", maximum)
				  # print("# of Steps: ", steps)
				  if steps < 3:
					  for x in range(1, p+1):
						  path[w].append(final[w])
				  else:
					  total_time = self.find_total_time(angle, maximum,a)
					  if inital[w] == max(couple):
						  # print("Step A")
						  for x in range(1, p+1):
							  t = (x*(total_time/p))
							  # print("iterated time:", t)
							  # print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
							  path[w].append(inital[w]+np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
						  self.step_key.append(True)
						  for z in range(0,steps+1):
							  count = z*(angle/steps)
							  self.step_path[w].append(self.inverted_trap(count,angle,a))
					  elif final[w] == max(couple):
						  # print("Step B")
						  for x in range(1, p+1):
							  t = (x*(total_time/p))
							  # print("iterated time:", t)
							  # print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
							  path[w].append(inital[w]-np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
						  # print("correct path", w)
						  self.step_key.append(False)
						  for z in range(0,steps+1):
							  count = z*(angle/steps)
							  self.step_path[w].append(self.inverted_trap(count,angle,a))
					  else:
						  print("error 1", w)

			  elif (abs(couple[0]-couple[1]) <= 180):
				  angle = max(abs_couple) - min(abs_couple)
				  theta[w] = round(np.radians(angle),3)
				  steps = int(np.radians(angle)*R)
				  # print("# of Steps: ", steps)
				  # inital
				  maximum = self.check_if_max(angle,a)
				  # if w == 0:
				  # print("angle: ", angle)
				  # print("maximum: ", maximum)
				  if steps < 3:
					  for x in range(1, p+1):
						  path[w].append(final[w])
				  else:
					  total_time = self.find_total_time(angle, maximum,a)
					  # if w == 0:
					  # print("Angle",w,"Total Time: ", total_time)
					  if inital[w] == max(couple):
						  # print("Step C")
						  for x in range(1, p+1):
							  t = (x*(total_time/p))
							  a = accel[w]
							  # print("iterated time:", t)
							  # print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
							  path[w].append(inital[w]-np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
						  # print("correct path", w)
						  self.step_key.append(False)
						  for z in range(0,steps+1):
							  count = z*(angle/steps)
							  self.step_path[w].append(self.inverted_trap(count,angle,a))
							  # print("Appended: ",self.inverted_trap(count,angle,a))
					  # final
					  elif final[w] == max(couple):
						  # print("Step D")
						  for x in range(1, p+1):
							  t = (x*(total_time/p))
							  a = accel[w]
							  # if w == 0:
							  # print("iterated time:", t)
							  # print("trapezod function val:",np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
							  path[w].append(inital[w]+np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))
						  self.step_key.append(True)
						  for z in range(0,steps+1):
							  count = z*(angle/steps)
							  self.step_path[w].append(self.inverted_trap(count,angle,a))
							  # print("Appended: ",self.inverted_trap(count,angle,a))
					  else:
						  print("error 2")
			  else:
				  pass
		  elif (final[w]==inital[w]):
			  path[w].append(final[w])
			  # print("appended 0")
			  theta[w] = 0
		  else:
			  pass
	  list_len = [len(i) for i in path]
	  # print("Path Length: ",path)
	  # self.dir = [self.step_path[i][0] for i in range(0,4)]
	  self.theta = theta
	  for w in range(0, len(path)):
		  path[w].extend([path[w][-1]]*(max(list_len)-len(path[w])))
	  return path, theta

  def velocity(self, angle):
	  return np.sqrt(2*self.values["w_max"]*angle)

  def time_at(self, angle):
	  return np.sqrt((2*angle)/self.values["accel"])

  def stepper_path(self, times):
      # print("Times: ", times)
      # function for finding the value of the delays between the steps taken

      # delay is equal to the pause after step HIGH and step LOW
      time_ranges = {"A":[],"B":[],"C":[],"D":[],"E":[]}
      delay = 0.010

      index = [["A",0],["B",1],["C",2],["D",3],["E",4]]

      for w in range(0,len(index)):
          i = index[w][1]
          key = index[w][0]
          for x in range(0,len(times[i])):
              upper_bound = times[i][x] + delay
              lower_bound = times[i][x] - delay
              time_ranges[key].append([lower_bound, upper_bound])

      # find time difference between lower_bound and upper bound

      delay_times = {"A":[],"B":[],"C":[],"D":[],"E":[]}
      for w in range(0, len(index)):
          key = index[w][0]
          if time_ranges[key]:
              first_delay = time_ranges[key][1][0]
              delay_times[key].append(first_delay)

      # print("delay times", delay_times)

      for w in range(0, len(index)):
          i = index[w][1]
          key = index[w][0]
          if time_ranges[key]:
              # print("time ranges key len: ", len(time_ranges[key]))
              for x in range(1,len(time_ranges[key])-1):
                  delay = time_ranges[key][x][0] - time_ranges[key][x-1][1]
                  if delay >= 0:
                      delay_times[key].append(delay)
                  elif delay < 0:
                      delay_times[key].append(0)

      return delay_times

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

	def R_z(self, angle):
		rad = np.radians(angle)
		return [[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]

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
		self.xyz = state_info['xyz']
		self.angle4 = state_info['angle4']
		self.angle5 = state_info['angle5']
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
		# add fixed perimeter estimation for testing
		self.perm_est = 1162.5762034949778
		# print("Original Area: ",self.overhead_area)
		# print("Object in new image perimeter estimation: ", self.perm_est)
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

	def search_val(self,elem):
		return elem[1]

	def find_move_angle(self, rectangles, boxes):
		kx = 1.414
		ky = 1.53
		center = [rectangles[0][0][0], rectangles[0][0][1]]
		points = [[boxes[0][0][0],boxes[0][0][1]],
		[boxes[0][1][0],boxes[0][1][1]],
		[boxes[0][2][0],boxes[0][2][1]],
		[boxes[0][3][0],boxes[0][3][1]]]
		# define shift values for box
		shiftx = 640 - center[0]
		shifty = 360 - center[1]
		# shift all points so center of rectangle is in center
		# of screen
		for x in range(0,4):
			points[x][0] = points[x][0] + shiftx
			points[x][1] = points[x][1] + shifty
		# verify that center is at center
		# print("center shifted x: ", (center[0] + shiftx) - 640)
		# print("center shifted y: ", -(center[1] + shifty) + 360)
		# convert points to cartesian plane
		for x in range(0,4):
			points[x][0] = points[x][0] - 640
			points[x][1] = -points[x][1] + 360
		# print("Cartesian Points: ", points)
		# use first pair of cartesian coordinates to
		# define indexes for 3 different magnitudes
		mag_lst = []
		for i in range(1,4):
			dx = points[i][0] - points[0][0]
			dy = points[i][1] - points[0][1]
			mag_lst.append([i, np.sqrt(dy**2+dx**2)])
		# order the magnitude list from largest magnitude to
		# smallest value
		mag_lst.sort(reverse=True,key = self.search_val)
		# find magnitude of gripper grip_width
		wdx = (points[mag_lst[2][0]][0]-points[0][0])*kx
		wdy = (points[mag_lst[2][0]][1]-points[0][1])*ky
		grip_width = np.sqrt(wdx**2+wdy**2)/1000
		print("Grip Width: ", grip_width)
		# print("Ordered Magnitude List: ", mag_lst)
		# midpoint of long side 1
		mid_x1 = points[0][0] + ((points[mag_lst[1][0]][0]-points[0][0])/2)
		mid_y1 = points[0][1] + ((points[mag_lst[1][0]][1]-points[0][1])/2)
		# print("Midpoint Corner(x,y),(x1,y1): ",[points[0][0],points[0][1]],[points[mag_lst[1][0]][0],points[mag_lst[1][0]][1]])
		# print("Midpoint (x,y)", [mid_x1,mid_y1])
		# midpoint of long side 2
		mid_x2 = points[mag_lst[2][0]][0] + ((points[mag_lst[0][0]][0]-points[mag_lst[2][0]][0])/2)
		mid_y2 = points[mag_lst[2][0]][1] + ((points[mag_lst[0][0]][1]-points[mag_lst[2][0]][1])/2)
		# makes unit vectos of the midpoints
		unit_scale1 = np.sqrt(mid_x1**2 + mid_y1**2)
		unit_scale2 = np.sqrt(mid_x2**2 + mid_y2**2)
		m1 = [mid_x1/unit_scale1,mid_y1/unit_scale1]
		m2 = [mid_x2/unit_scale2,mid_y2/unit_scale2]
		mdps = [m1,m2]
		# print("These are the mid points: ", mdps)
		# find 2 angles based on the unit vector form of
		# the x and y values of each midpoint
		if mdps[0][1] > 0:
			theta1 = np.arccos(mdps[0][0])
			theta2 = np.arcsin(mdps[1][1])
			print("A")
		elif mdps[1][1] > 0:
			theta1 = np.arccos(mdps[1][0])
			theta2 = np.arcsin(mdps[0][1])
			print("B")
		else:
			print("something else happened")
		# select smaller angle to return
		val = 0
		if abs(theta1) < abs(theta2):
			val = np.degrees(theta1)
		elif abs(theta1) > abs(theta2):
			val = np.degrees(theta2)
		else:
			print("theta1 must equal theta 2")
			val = np.degrees(theta2)

		print("Angle5 move value: ", val)
		return val, grip_width

	def scale_points(self, rectangles, boxes):
		pts = []
		kx = 1.414
		ky = 1.53
		# print("self.xyz", self.xyz)
		h = self.xyz[2]-self.ultrasonic_height
		# print("Rectangles[0]",rectangles[0])
		# print("Rectangles[0][0]",rectangles[0][0])
		# print("Rectangles[0][0][0]",rectangles[0][0][0])
		center = [((rectangles[0][0][0]-640)*kx)/1000, ((-rectangles[0][0][1]+360)*ky)/1000, h]
		# print("/nRectangle Center: ", center)
		pts.append(center)
		# estimated values found from bounding box while camera was on top of board
		# print("Boxes[0]",boxes[0])
		# print("Boxes[0][0]",boxes[0][0])
		# print("Boxes[0][0][0]",boxes[0][0][0])
		pts.append([[((boxes[0][0][0]-640)*kx)/1000,((boxes[0][1][0]-640)*kx)/1000,((boxes[0][2][0]-640)*kx)/1000,((boxes[0][3][0]-640)*kx)/1000],
		[((-boxes[0][0][1]+360)*ky)/1000,((-boxes[0][1][1]+360)*ky)/1000,((-boxes[0][2][1]+360)*ky)/1000,((-boxes[0][3][1]+360)*ky)/1000],
		[h,h,h,h]])
		# find the angle of rotation
		width = rectangles[0][1][0]
		height = rectangles[0][1][1]
		angle = rectangles[0][2]
		# finds amount of rotation to long side of angle
		move, grip_width = self.find_move_angle(rectangles, boxes)
		return pts, move, grip_width

	def xyz_reframe(self, pts):
		# define homogenous transfer matricies
		h = self.xyz[2]-self.ultrasonic_height
		r_ab = self.R_z(self.angle4)
		r_bf = self.R_z(self.angle5)
		d_af = [[self.xyz[0]],[self.xyz[1]], [h]]
		r_af = np.dot(r_ab,r_bf)
		h_af = np.concatenate((r_af,d_af),1)
		h_af = np.concatenate((h_af,[[0,0,0,1]]),0)
		center = [h_af[0][3],h_af[1][3],h_af[2][3]]
		# define no rotation matrix
		# r_fx = [[1,1,1],[1,1,1],[1,1,1]]
		# d_fc = [[pts[0][0]],[pts[0][1]],[0]]
		# h_fc = np.concatenate((r_fx,d_fc),1)
		# h_fc = np.concatenate((h_fc,[[0,0,0,1]]),0)
		# h_ac = np.dot(h_af,h_fc)
		# xyz = [h_ac[0][3],h_ac[1][3],h_ac[2][3]]
		# try adding displacment vectors to see difference from concatonation
		xyz = [pts[0][0]+center[0],pts[0][1]+center[1],center[2]]
		# print("Center Location:", xyz)
		rect = [[pts[1][0][0]+center[0],pts[1][0][1]+center[0],pts[1][0][2]+center[0],pts[1][0][3]+center[0]],
		[pts[1][1][0]+center[1],pts[1][1][1]+center[1],pts[1][1][2]+center[1],pts[1][1][3]+center[1]],
		[center[2],center[2],center[2],center[2]]]
		# print("Rectangles: ", np.matrix(rect))
		return xyz, rect

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

		return found, drawing, rotated_rectangles, boxes

	def look_for_object(self, hand_poll, hand_push,object_info, state_info):
		self.estimate_area()
		# apply kernal filter
		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		passed_img = cv2.filter2D(self.img, -1, kernel)
		cv2.imwrite('passed_img.jpg',passed_img)
		found, drawing, rectangles, boxes = self.item_outline(passed_img)
		# define "empty" varibles to avoid issues
		xyz = None
		grip_width = 0.1
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
			self.img = image
			# load new information into class
			self.pass_filter_variables(image, object_info, state_info)
			self.estimate_area()
			# apply kernal filter
			kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
			passed_img = cv2.filter2D(image, -1, kernel)
			# apply brightness increase
			passed_img = self.increase_brightness(passed_img,35)
			found, drawing, rectangles, boxes = self.item_outline(passed_img)
			if not found:
				print("Cannot Find Image")
			elif found:
				cv2.imshow("drawing", drawing)
				cv2.waitKey(0)
				pts, move, grip_width = self.scale_points(rectangles, boxes)
				xyz, rect = self.xyz_reframe(pts)
		elif found:
			cv2.imshow("drawing", drawing)
			cv2.waitKey(0)
			pts, move, grip_width = self.scale_points(rectangles, boxes)
			xyz, rect = self.xyz_reframe(pts)
		else:
			print("Impossible")
		return xyz, rect, move, grip_width

class Path_Exe(object):
	def __init__(self):
		self.nan = Image_Processing()
		self.overhead_vid_path = 0
		self.hand_vid_path = 1
		arm_lengths = [0.4, 0.4, 0.050]
		self.path = Path([0,0,0,0,90],arm_lengths)
		# for multiprocessing testing only
		# travel path
		test = [True, ['remote', [894, 1, 980, 195]], ['person', [-4, 445, 352, 719]]]
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
		# kx = 1.244
		# ky = 1.288
		kx = 1.4464
		ky = 1.5403
		# based on
		center = [abs((obj[2]-obj[0])/2)+obj[0]-640,-(abs((obj[3]-obj[1])/2)+obj[1])+360,0]
		# print('\n Object Cented at:(in pxls)',center)
		w_center = [(center[0]*kx)/1000,(center[1]*ky)/1000,0]
		# print('/n Object Cented at:(in meters)',w_center)
		return w_center

	def two_three_dim(self, xy_box):
		xy_zero = self.calc_world_pos(xy_box)
		flat_coord = np.sqrt((xy_zero[0]*xy_zero[0]) + (xy_zero[1]*xy_zero[1]))
		l1 = self.path.values['len'][0]
		l2 = self.path.values['len'][1]
		l3 = self.path.values['len'][2]
		tot_length = (l1 + l2)
		max_w = np.sqrt((tot_length**2-l3**2))
		if flat_coord > max_w:
			print("Cannot Reach")
			end_effector = None
		elif flat_coord < max_w:
			for x in range(0,90):
				w = np.cos(np.radians(90-x))*tot_length
				h = np.sin(np.radians(90-x))*tot_length
				if (tot_length) >= (np.sqrt(flat_coord**2 + h**2)):
					xy_zero[2] = h
					end_effector = [xy_zero[0],xy_zero[1],xy_zero[2]]
					break
				else:
					end_effector = None
		else:
			end_effector = None
		print("xyz is: ", end_effector)
		return end_effector

	def cameras(self,hand_poll, hand_push, overhead_poll, overhead_push):
		# waits for poll request from image processor
		# upon request delivers current frame
		try:
			vid0 = cv2.VideoCapture(int(self.overhead_vid_path))
			vid1 = cv2.VideoCapture(int(self.hand_vid_path))
			#
			vid0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
			vid0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
			#
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
		self.ax.set_xlim3d([-800, 800])
		self.ax.set_xlabel('X')
		#
		self.ax.set_ylim3d([-800, 800])
		self.ax.set_ylabel('Y')
		#
		self.ax.set_zlim3d([0.0, 800])
		self.ax.set_zlabel('Z')
		#
		self.ax.set_title('3D Test')
		set = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
		self.line0, = self.ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
		self.line1, = self.ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
		self.line2, = self.ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
		self.line3, = self.ax.plot(set[3][0], set[3][1], 'bo', linestyle='solid')
		# self.point1, = self.ax.scatter()
		# self.point1, = self.ax.plot(set[4][0], set[4][1], 'bo', linestyle='solid')

	def live_sim(self, sim_static, sim_live):
		# plot = Instance_Plot()
		self.initialize_plot()
		live_pkg = None
		static_pkg = None
		while True:
			# gets angle values from queue
			if not sim_live.empty():
				live_pkg = sim_live.get()
			else:
				pass
			# get points from static library
			if not sim_static.empty():
				static_pkg = sim_static.get()
			else:
				pass

			if (sim_live.empty() & sim_static.empty() & (live_pkg != None) & (static_pkg != None)):
				# print("Static Pkg", static_pkg)
				scale = 1000
				goals = self.ax.scatter(np.dot(static_pkg['goals'][0],scale),np.dot(static_pkg['goals'][1],scale),np.dot(static_pkg['goals'][2],scale), color = 'r')
				outline = self.ax.scatter(np.dot(static_pkg['outline'][0],scale),np.dot(static_pkg['outline'][1],scale),np.dot(static_pkg['outline'][2],scale), color = 'g')
				for x in range(0, len(live_pkg)):
					lines = live_pkg[x]
					scale = 1000
					self.line0.set_data_3d(np.dot(lines[0][0],scale),np.dot(lines[0][1],scale),np.dot(lines[0][2],scale))
					#
					self.line1.set_data_3d(np.dot(lines[1][0],scale),np.dot(lines[1][1],scale),np.dot(lines[1][2],scale))
					#
					self.line2.set_data_3d(np.dot(lines[2][0],scale),np.dot(lines[2][1],scale),np.dot(lines[2][2],scale))
					#
					self.line3.set_data_3d(np.dot(lines[3][0],scale),np.dot(lines[3][1],scale),np.dot(lines[3][2],scale))
					#
					# self.point1.set_data_3d([np.dot(lines[4][0],scale),np.dot(lines[4][0],scale)],[np.dot(lines[4][1],scale),np.dot(lines[4][1],scale)],
					# [np.dot(lines[4][2],scale),np.dot(lines[4][2],scale)])
					# path of end effector
					self.fig.canvas.draw()
					self.fig.canvas.flush_events()
					# time.sleep(0.0001)
				goals.remove()
				outline.remove()
			else:
				pass

			if 0xFF == ord('q'):
				break

	def make_points(self, travel, xyz, n):
		pts = []
		# xfyfzf = self.two_three_dim(travel[2][1])
		pts.append([[xyz[0]],
					[xyz[1]],
					[xyz[2]]])
		# estimated values found from bounding box while camera was on top of board
		kx = 1.4464
		ky = 1.5403
		pts.append([[((travel[n][1][0]-640)*kx)/1000,((travel[n][1][0]-640)*kx)/1000,((travel[n][1][2]-640)*kx)/1000,((travel[n][1][2]-640)*kx)/1000],
		[((-travel[n][1][1]+360)*ky)/1000,((-travel[n][1][3]+360)*ky)/1000,((-travel[n][1][1]+360)*ky)/1000,((-travel[n][1][3]+360)*ky)/1000],
		[0,0,0,0]])
		return pts

	def arrange_static(self,pts, rect, new_center, ultra_h_standin, xyz, i, num_updates):
		distance = ultra_h_standin
		print("divisor: ",num_updates-i)
		z_val = (distance - ((distance)/(num_updates-i)))
		# make sure it doesn't go through ground
		if z_val < self.path.values['len'][2]:
			z_val = self.path.values['len'][2]
		else:
			pass
		pts[0][0].append(new_center[0])
		pts[0][1].append(new_center[1])
		pts[0][2].append(z_val)
		pts[1][0] = rect[0]
		pts[1][1] = rect[1]
		pts[1][2] = rect[2]
		return pts, [new_center[0],new_center[1],z_val]

	def iterative_approach(self,num_updates,ultra_h_standin,path,xyz,state_info,pts,suite,applied_width,sim_live,sim_static,hand_poll,hand_push):
		# plotter = Plot(self.path.values['len'])
		# suite = []
		feat = Features()
		end_angles = [path[0][-1], path[1][-1], path[2][-1], path[3][-1], path[4][-1]]
		ultra_h_standin = (xyz[2]-0.01)
		last_move = 0
		for i in range(0, num_updates):
			# polls hand camera for image
			hand_poll.put("Get")
			# waits
			while hand_push.empty():
				pass
			# displays image
			image = hand_push.get()
			cv2.imshow("Hand Cam", image)
			cv2.waitKey(0)
			angle4 = end_angles[3]
			# use feature class to find center of object, angle, ect
			feat.pass_filter_variables(image, self.object_info, state_info)
			new_center, rect, move, grip_width = feat.look_for_object(hand_poll, hand_push, self.object_info, state_info)
			#
			actual_move = move - last_move
			last_move = move
			if (new_center != None):
				pts, verify = self.arrange_static(pts, rect, new_center, ultra_h_standin, xyz, i, num_updates)
				static_pkg = {'goals':pts[0],'outline':pts[1]}
				self.path.values['iA'] = end_angles
				path1, step_path1, bool1 = self.path.find_path_to(verify,actual_move+angle4+90,False)
				# will grasp item if count is final
				if i == (num_updates-1):
					applied_width = grip_width
				else:
					pass

				if bool1:
					for x in range(0, len(path1[0])):
						post = self.plotter.position([path1[0][x], path1[1][x], path1[2][x], path1[3][x], path1[4][x]], applied_width)
						# post.append(pt)
						suite.append(post)
					# post data to the simulation
					sim_live.put(suite)
					sim_static.put(static_pkg)
					# determine xyz of the end effector
					end_position = self.plotter.position([path1[0][-1], path1[1][-1], path1[2][-1], path1[3][-1], path1[4][-1]], grip_width)
					xyz = [end_position[2][0][1], end_position[2][1][1], end_position[2][2][0]]
					# determine last angles of the arm
					end_angles = [path1[0][-1], path1[1][-1], path1[2][-1], path1[3][-1], path1[4][-1]]
					ultra_h_standin = (xyz[2]-0.01)
					state_info = {'xyz':[xyz[0],xyz[1],xyz[2]],
					'angle4':end_angles[3], 'angle5':end_angles[4],'ultra_h':ultra_h_standin,'m_hx':4,'m_hy':4}
				else:
					print("F")
		#  ------------------------------
		print("end angles:",end_angles)
		print("applied width:", applied_width)
		return end_angles, applied_width

	def path_to_above(self,travel,end_angles, suite, sim_static, sim_live, gripper_width = 0.1, angle5 = 90, First=True):
		# use inverse kinematics to determine end effector postion
		if First:
			xyz = self.two_three_dim(travel[1][1])
			n = 1
		elif not First:
			xyz = self.two_three_dim(travel[2][1])
			self.path.values['iA'] = end_angles
			n = 2
		print("Final Position: ", xyz)
		# define animation and stepper motor path
		# bool says whether object is within reach
		path, step_path, bool = self.path.find_path_to(xyz,angle5,False)
		print("Stepper Path:", np.matrix(step_path))
		# needed for image orientation
		end_angles = [path[0][-1],path[1][-1],path[2][-1],path[3][-1], path[4][-1]]
		# print("end_angles: ", end_angles)
		# define object_info library

		self.object_info = {'name':travel[n][0],
		'over_A':((travel[n][1][0]-travel[n][1][2])*(travel[n][1][1]-travel[n][1][3])),
		'over_res':[720,1280]}
		# define state info
		# stand in for ultra sonic sensor
		ultra_h_standin = xyz[2] - 0.01
		# arranges static data for plot
		pts = self.make_points(travel,xyz,n)
		state_info = {'xyz':[xyz[0],xyz[1],xyz[2]],
		'angle4':end_angles[3], 'angle5':end_angles[4],'ultra_h':ultra_h_standin,'m_hx':4,'m_hy':4}
		# pts = self.make_points(travel,xyz)
		static_pkg = {'goals':pts[0],'outline':pts[1]}
		if bool:
			for x in range(0,len(path[0])):
				post = self.plotter.position([path[0][x], path[1][x], path[2][x], path[3][x], path[4][x]],gripper_width)
				suite.append(post)
			sim_live.put(suite)
			sim_static.put(static_pkg)
		else:
			print("Couldn't Plot Path")

		return bool, ultra_h_standin, path, xyz, state_info, pts, suite, gripper_width

	def path_straight_to(self, end_angles, end_pt, grip_width, sim_static, sim_live):
		pts = [[],[]]
		end_position = self.plotter.position(end_angles, grip_width)
		start_pt = [end_position[2][0][1], end_position[2][1][1], end_position[2][2][0]]
		pts[0] = [start_pt, end_pt]
		static_pkg = {'goals':pts[0],'outline':pts[1]}
		self.path.values['iA'] = end_angles
		path, step_path, bool = self.path.find_path_to(end_pt, end_angles[4], False)
		if bool:
			suite = []
			for x in range(0,len(path[0])):
				post = self.plotter.position([path[0][x], path[1][x], path[2][x], path[3][x], path[4][x]],grip_width)
				suite.append(post)
			sim_live.put(suite)
			sim_static.put(static_pkg)
		else:
			print("Couldn't Plot Path")
		final_angles = [path[0][-1], path[1][-1], path[2][-1], path[3][-1], path[4][-1]]
		return bool, final_angles, suite

	def path_exe(self, travel, overhead_poll, overhead_push, hand_poll, hand_push, sim_static, sim_live):
		end_angles = [0,0,0,0,0]
		self.plotter = Plot(self.path.values['len'])
		suite = []
		bool, ultra_h_standin, path, xyz, state_info, pts, suite, gripper_width = self.path_to_above(travel,end_angles,suite,sim_static, sim_live,
		gripper_width = 0.1, angle5 = 90, First=True)
		bool = True
		if bool:
			num_updates = 3
			end_angles,  hold_width = self.iterative_approach(num_updates, ultra_h_standin, path, xyz, state_info,
			pts, suite, gripper_width, sim_live, sim_static, hand_poll, hand_push)
			fool, tranistory_inflation, suite = self.path_straight_to(end_angles, xyz, hold_width, sim_static, sim_live)
			if fool:
				bool1, ultra_h_standin1, path1, xyz1, state_info1, pts1, suite1, gripper_width1 = self.path_to_above(travel, tranistory_inflation,suite,sim_static,
				sim_live, gripper_width = gripper_width, angle5 = end_angles[4], First=False)
		else:
			pass

	def begin_multi(self, travel):
		# queues for passsing data between functions
		hand_poll = Queue()
		hand_push = Queue()
		overhead_poll = Queue()
		overhead_push = Queue()
		sim_live = Queue()
		sim_static = Queue()
		# process intiialization for running multiple functions
		# at the same time
		path_exe_p = Process(target=self.path_exe, args=(travel, overhead_poll, overhead_push,
		hand_poll, hand_push, sim_static, sim_live))
		path_exe_p.start()
		#
		cameras_p = Process(target=self.cameras, args=(hand_poll, hand_push,
		overhead_poll, overhead_push))
		cameras_p.start()
		#
		live_sim_p = Process(target=self.live_sim, args=(sim_static, sim_live))
		live_sim_p.start()
		processes = [path_exe_p,cameras_p,live_sim_p]
		# join processes together
		for p in processes:
			p.join()

if __name__=="__main__":
	nan = Path_Exe()
