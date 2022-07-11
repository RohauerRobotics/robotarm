# ----------------------------------

# Robotic Arm Physics Simulation
# By Dylan Rohauer (7/10/2022)
# Thanks to the following sources for helping me figure out stuff
# to solve all of the problems needed to build this:
# Angela Sodemann (YouTuber) - homogenous transfer matricies
# Matt O'Dell (Calc III Professor) - matricies practice
# Kira Burt (Physics I & II Professor) - angular motion problems
# Mark - inverse kinematics math
# Damon Sisk (YouTuber) - motion profiles
# Stewart Technologies (YouTuber) - motion profiles and joint interpolation

# ----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import math

# ----------------------------------
# The class Path focuses on the creation of motion planning
# including inverse kinematics, joint motion interpolation,
# and the picking of joint angles for a realistic animation path
# ----------------------------------

class Path(object):
	def __init__(self, inital_angles,lengths_m):
		# define the initial angles of the arm
		# define the # of microsteps needed to complete
		# 1 rotation of the stepper motor
		# define max angular velocity
		# define max acceleration
		self.values = {"iA":inital_angles, "len":lengths_m, "micro_steps":200, "w_max": np.pi, "accel":np.pi/8}
		self.wa_max = self.values['w_max']
		self.wc_max = self.values['w_max']

		# use the inverse kinematics function to
		# find the final angles of each joint
		final, bool = self.inverse_kinematics([0.15,0.01,0.125])

		# if the arm is able to reach continue
		if bool:
			# find the acceleration constants of each joint such
			# that all joints will reach the same place at the same
			# time
			self.accel, self.t_time = self.acceleration_path(self.values['iA'], final)
			self.path, self.divisions = self.animation_path(self.values['iA'], final, self.accel)

	# takes a goal point and finds the angles of the
	# robot arm required to meet that point with the
	# end effector
	def inverse_kinematics(self, end_e):
		# assign the x,y,z coordinates of the end
		# effector goal position shorter variable names
		x = end_e[0]
		y = end_e[1]
		z = end_e[2]

		# assign the first two lengths of the robot arm
		# with shorter varible names
		l1 = self.values['len'][0]
		l2 = self.values['len'][1]

		# w reprents the radial length of the arm projected
		# onto the xy-plane
		w = np.sqrt((x**2 + y**2))

		# h represnts the the hypotenuse of the triangle
		# created between the height and the length of
		# the arm projection
		h = np.sqrt((w**2 + z**2))

		# if the distance to the point from the
		# base of the arm (h) is greater than the total
		# length of the arm then the goal is out of reach
		if h > (l1+l2):
			print("Unable to reach")
			bool = False
		elif h <= (l1+l2):
			print("Able to Reach")
			bool = True
		else:
			pass
		# bool represents the boolean of whether or
		# not the arm is able to reach its goal
		if bool:
			# Φ1, Φ2, and Φ4 reprent internal
			# angles to the triangle created by the two arm
			# segments
			# --- image of these angles can be found in
			# --- documentation as arm_triangle.png

			# Φ1 can be found by using the law of cosines
			if (l1**2+h**2-l2**2) >= (2*l1*h):
				phi1 = 0
			else:
				phi1 = np.arccos((l1**2+h**2-l2**2)/(2*l1*h))

			# Φ2 can also be found by using the law of cosines
			if phi1 == 0:
				phi2 = (np.pi)
			else:
				phi2 = np.arccos((l2**2+l1**2-h**2)/(2*l2*l1))

			# avoid divide by 0 error
			# Φ4 can be found by inverse tangent of the arm's
			# radial length and the height above the ground
			if w != 0:
				phi4 = np.arctan(abs(z)/abs(w))
			elif w == 0:
				phi4 = (np.pi/2)
			else:
				pass

			# angles 1, 2, 3, & 4 represent the angles relative
			# to the z axis, l1, l2, and the x axis respecively
			# --- image of these angles can be found in documentation
			# --- as arm_angles.png

			angle1 = (np.pi/2)- (phi4 + phi1)
			angle2 = np.pi - phi2
			angle3 = np.pi - angle1 - angle2

			# if x is equal to zero there will be a divide by zero error
			if x == 0:
				angle4 = 0
			elif x != 0:
				if ((x < 0) ^ (y<0)):
					angle4 = np.arctan(y/x) + np.pi
				elif ((x<0)&(y<0)):
					angle4 = np.arctan(y/x) + np.pi
				else:
					angle4 = np.arctan(y/x)

			# path consists of a list of all of the angles at any given postion in degrees
			path = [np.degrees(angle1),np.degrees(angle2),np.degrees(angle3),np.degrees(angle4)]
		else:
			pass

		return path, bool

	# finds the acceleration constants of each joint
	# nessary to make each stepper stop moving
	# at the same time
	def acceleration_path(self, inital, final):
		# "path" consists of a list of the total time
		# taken to move each joint, the degree which
		# the joint will move through, and whether or
		# not the joint will reach maximum angular velocity
		# --- [total_time, angle_moved,] ---
		path = [[0,0,None],[0,0,None],[0,0,None],[0,0,None]]
		a = self.values['accel']

		# for loop will create a list of all the times
		# and angles with the set max acceleration/
		# deceleration
		for w in range(0, 4):
			# adjust values if outside of range 0-360
			if (inital[w]) > 360:
				inital[w] = inital[w] - 360
			elif (inital[w]) < 0:
				inital[w] = inital[w] + 360
			elif (final[w]) > 360:
				final[w] = final[w] - 360
			elif (final[w]) < 0:
				final[w] = final[w] + 360
			else:
				pass

			if final[w] != inital[w]:
				couple = [final[w],inital[w]]
				abs_couple = [abs(final[w]),abs(inital[w])]

				# determine shortest number of steps between angles
				if (abs(couple[0]-couple[1]) > 180):
					# if the absolute difference between the final and
					# inital angles is greater than 180 degrees
					# the end effector should travel in the opposite
					# direction to take less time
					negative_max = abs(max(couple) - 360)
					angle = negative_max + min(abs_couple)
					steps = int(angle)

					# check if max looks at whether or not the
					# joint will reach maximum angular velocity
					# given the distance to travel and the maximum
					# angular acceleration
					maximum = self.check_if_max(angle,a)

					# if the number of steps is zero
					# time will return 1 and angle will return
					# 0 to avoid a divide by zero error
					if steps == 0:
						path[w][0] = 1
						path[w][1] = 0
						path[w][2] = False

					# total_time finds the time to travel the
					# angular displacement at the max acceleration
					else:
						total_time, divs = self.find_total_time(angle, maximum, a)
						path[w][0] = total_time
						path[w][1] = angle
						path[w][2] = maximum

				elif (abs(couple[0]-couple[1]) <= 180):
					# if the absolute difference between the final
					# and inital angles is less than 180 the travel
					# path will be as short as possible
					angle = max(abs_couple) - min(abs_couple)
					steps = int(angle)

					# checks if max angular velocity is reached
					maximum = self.check_if_max(angle,a)

					# returns 1 and 0 to avoid 0/0 error
					if steps == 0:
						path[w][0] = 1
						path[w][1] = 0
						path[w][2] = False
					# finds total time based on shortest angular displacment
					else:
						total_time, divs = self.find_total_time(angle, maximum, a)
						path[w][0] = total_time
						path[w][1] = angle
						path[w][2] = maximum
				else:
					pass

			# returns 1 and 0 to avoid 0/0 error
			elif (final[w]==inital[w]):
				path[w][0] = 1
				path[w][1] = 0
				path[w][2] = False

			else:
				pass

		# copy is made so stepper motor order can be preserved
		copy = list(path)
		print("Acceleation Path:", copy)
		# sorts based on time, search crit specifies that
		# the 0 index is the time varible
		path.sort(key=self.search_crit, reverse=True)

		# base_time is representative of half of the total
		# max travel time because half of the time is
		# equal to the time needed to accelerate and
		# decelerate
		base_time = path[0][0]/2
		t_time = path[0][0]
		print('Total Time', t_time)
		accel = [0,0,0,0]
		for x in range(0, 4):
			if not copy[w][2]:
				a = np.radians(copy[x][1])/(base_time**2)
				accel[x] = a
			elif copy[w][2]:
				a = self.values['accel']
			else:
				pass
		print("accel", accel)

		return accel, t_time

	# determines if a stepper motor will reach
	# maximum speed when going through an angular displacement
	def check_if_max(self, angle, accel):
		angle = np.radians(angle)
		decel = accel
		# max_theta represents the angular displacement of a
		# stepper motor accelerating and decelerating at the maximum
		# acceleration value.
		#
		# So if the angle needed to travel is greater than the max
		# angular displacement, then the maximum angular velocity will be
		# reached and max will return true.
		max_theta = (self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel))
		max = None
		if angle < max_theta:
			max = False
		elif angle >= max_theta:
			max = True

		return max

	# finds the total time needed to move though an angle
	def find_total_time(self, angle, max, accel):
		angle = np.radians(angle)
		decel = accel
		divs = [0,0,0]
		if not max:
			# found by adding the angular displacement of the
			# acceleration to the displacement of the deceleration
			# and solving for time_a using the relationship that
			# the absolute value of accel*time_a = decel*time_c
			time_a = np.sqrt((2*angle)/(accel+(accel**2/decel)))

			# found by using the relationship that
			# the absolute value of accel*time_a = decel*time_c
			time_c = time_a*(accel/decel)

			# define divs
			divs[0] = time_a
			divs[2] = time_c

			# sum components
			total_time = time_a + time_c

		elif max:
			# found by using the relationship that wt = a
			time_a = self.wa_max/accel

			# found by dividing the displacement at max angular
			# velocity by the max velocity
			time_b = (angle - ((self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel))))/self.wa_max

			# same as time_a
			time_c = self.wc_max/decel

			# define divs
			divs[0] = time_a
			divs[1] = time_b
			divs[2] = time_c

			# sum components
			total_time = time_a + time_b + time_c

		else:
			pass

		return total_time, divs

	# selects an index of an element
	def search_crit(self, elem):
		# specifies that the 0 index is the
		# time variable
		return elem[0]

	# determines the path a stepper motor will travel through to reach a
	# certain goal point
	def animation_path(self, inital, final, accel):
		# path will consist of a list of lists where
		# each index of the path represents the list of
		# angular values each joint will be at during the motion
		path = [[],[],[],[]]

		# p is the number of divisions each path will
		# broken up into
		p = 200
		self.p = p
		# R is the number of steps per stepper motor rotation
		R = self.values['micro_steps']/(2*np.pi)

		# division of time definition
		divisions = [0,0,0,0]

		for w in range(0, 4):
			a = accel[w]
			if final[w] != inital[w]:
				couple = [final[w], inital[w]]
				abs_couple = [abs(final[w]), abs(inital[w])]

				# adjust values if outside of range 0-360
				if (inital[w]) > 360:
					inital[w] = inital[w] - 360
				elif (inital[w]) < 0:
					inital[w] = inital[w] + 360
				elif (final[w]) > 360:
					final[w] = final[w] - 360
				elif (final[w]) < 0:
					final[w] = final[w] + 360
				else:
					pass

				# determine shortest number of steps between angles
				#
				# if the absolute difference between the final and
				# inital angles is greater than 180 degrees
				# the end effector should travel in the opposite
				# direction to take less time
				if (abs(couple[0] - couple[1]) > 180):
					# finds the distance to the goal in the opposing
					# direction which will be less than 180
					negative_max = abs(max(couple) - 360)

					# the displacement to the postion specified
					# is the sum of the absolute max + min
					angle = negative_max + min(abs_couple)

					# steps is found by multipling the constant R
					# by the angle displaced
					steps = int(np.radians(angle)*R)

					# checks if maximum angular velocity is reached
					maximum = self.check_if_max(angle, a)

					# if there are 0 steps then the animation
					# path will act as if there are 0
					if steps == 0:
						for x in range(1, p+1):
							path[w].append(final[w])
					else:
						# finds total time
						total_time, divs = self.find_total_time(angle, maximum, a)

						# define division of times
						divisions[w] = divs
						# if the intial angle is the max angle then the end
						# the shortest distance to the goal point is past the
						# inital point
						if inital[w] == max(couple):
							# print("Step A")

							# iterate through the range of points defined by p
							for x in range(1, p+1):
								# t represents different increments of time
								t = (x*(total_time/p))

								# print("iterated time:", t)
								# print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# the trapezoid function takes a the time,
								# anglular displacement and acceleration and
								# returns the value of the angle at the specified time
								path[w].append(inital[w]+np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# if the angles are increasing the joint is accelerating towards the ground and is negative
								self.accel[w] = -self.accel[w]

						# if the final angle is max then the shortest path is
						# away from it
						elif final[w] == max(couple):
							# print("Step B")
							for x in range(1, p+1):
								# t represents different increments of time
								t = (x*(total_time/p))

								# print("iterated time:", t)
								# print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# the trapezoid function takes a the time,
								# anglular displacement and acceleration and
								# returns the value of the angle at the specified time
								path[w].append(inital[w]-np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# define that the acceleration is in the negative direction

								# print("correct path", w)
						else:
							print("error 1", w)

				# if the absoulte difference from the start and end
				# points is less than 180
				elif (abs(couple[0]-couple[1]) <= 180):
					# the angle needed to travel is the difference between
					# the absolute maximum and minimum
					angle = max(abs_couple) - min(abs_couple)

					# the number of steps is defined
					steps = int(np.radians(angle)*R)

					# print("# of Steps: ", steps)

					# checks if the maximum angular velocity is reached
					maximum = self.check_if_max(angle,a)

					# print("angle: ", angle)
					# print("maximum: ", maximum)

					if steps == 0:
						for x in range(1, p+1):
							path[w].append(final[w])
					else:
						# total time is found
						total_time, divs = self.find_total_time(angle, maximum, a)

						# define division of times
						divisions[w] = divs

						# print("Angle",w,"Total Time: ", total_time)

						# if the initial value is max then traveling towards
						# it is the fastest path
						if inital[w] == max(couple):

							# print("Step C")

							# iterate through time to find path
							for x in range(1, p+1):
								# time incrimentally found
								t = (x*(total_time/p))
								# print("iterated time:", t)

								# print("trapezod function val:", np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# trapezoid function is used to find angle
								path[w].append(inital[w]-np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# print("correct path", w)

						# if the final angle is greater than the inital
						# then the shortest path is away from the initial
						elif final[w] == max(couple):
							# print("Step D")

							for x in range(1, p+1):
								# time incrimentally found
								t = (x*(total_time/p))

								# print("iterated time:", t)
								# print("trapezod function val:",np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# trapezoid function is used to find angle
								path[w].append(inital[w]+np.degrees(self.trapezoid_function(t,angle,a,max_omega=maximum)))

								# if the angles are increasing the joint is accelerating towards the ground and is negative
								self.accel[w] = -self.accel[w]


			# if the inital and final angles are the same
			# then no distance is travelled
			elif (final[w]==inital[w]):
				path[w].append(final[w])

				# print("appended 0")

			else:
				pass

		return path, divisions

	# is used to determine the angular position of a stepper
	# motor on any given path
	def trapezoid_function(self, time, theta, accel, max_omega=False):
		# define total anglular displacement as theta
		theta = np.radians(theta)
		decel = accel

		# if the max angular velocity isn't reached
		if not max_omega:
			# time a is time inital acceleration takes
			time_a = np.sqrt((2*theta)/(accel+(accel**2/decel)))

			# before time a is reached the angular diplacement is
			# due directly to acceleration
			if time <= time_a:
				r = 0.5*accel*time**2

			# after time to finish acceleration
			# displacement is now due to inital displacement
			# plus angular velocity plus deceleration
			elif time > time_a:
				adj_time = time - time_a
				r = (0.5*accel*time_a**2) + (accel*time_a*adj_time) - (0.5*decel*adj_time**2)
			else:
				pass
		# if the maximum angular velocity is reached
		elif max_omega:
			# time a is the time to reach max
			# angular acceleration
			time_a = self.wa_max/accel

			# time b is the time to get the extra angular
			# displacement not gotten from acceleration and
			# deceleration
			time_b = (theta - (self.wa_max**2/(2*accel)) + (self.wc_max**2/(2*decel)))/self.wa_max

			# before time a is reached the angular diplacement is
			# due directly to acceleration
			if time < time_a:
				r = 0.5*accel*time**2

			# after intial acceleration time but before
			# deceleration displacement is due to initial
			# displacement plus max angular velocity displacement
			elif (time_a <= time < (time_b + time_a)):
				r = (0.5*accel*time_a**2) + accel*time_a*(time-time_a)

			# after time b, displacement is due to current displacemnt
			# plus displacement due to angular velocity
			elif (time >= (time_b + time_a)):
				r = (0.5*accel*time_a**2) + (accel*time_a*time_b) - (0.5*decel*(time-time_a-time_b)**2)

			else:
				pass
		else:
			pass

		return r

# ----------------------------------
# The class Plot focuses on taking the angles generated in Path,
# converting them to points and plotting them in a 3d simulation in
# matplotlib.
# ----------------------------------

class Plot(object):
	def __init__(self, path, lengths_m, inital_angles, t_time, p, divisions, accel):
		# define stepper settings
		self.values = {"iA":inital_angles, "len":lengths_m,
		"micro_steps":200, "w_max": np.pi, "accel":np.pi/8,
		"mass": [0.15,0.2,0.4]
		}
		# define class variable versions of arm segments
		self.l1 = lengths_m[0]
		self.l2 = lengths_m[1]
		self.l3 = lengths_m[2]
		#
		self.p = p
		self.t_time = t_time
		self.accel = accel
		self.divisions = divisions

		# declare plt is interactive
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

		# set serves as empy lists for defining lines
		# to be plotted
		set = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
		self.line0, = self.ax.plot(set[0][0], set[0][1], 'bo', linestyle='solid')
		self.line1, = self.ax.plot(set[1][0], set[1][1], 'bo', linestyle='solid')
		self.line2, = self.ax.plot(set[2][0], set[2][1], 'bo', linestyle='solid')
		self.line3, = self.ax.plot(set[3][0], set[3][1], 'bo', linestyle='solid', color='red')
		self.line4, = self.ax.plot(set[4][0], set[4][1], 'bo', linestyle='solid', color='red')
		self.line5, = self.ax.plot(set[5][0], set[5][1], 'bo', linestyle='solid', color='red')

		# loops animation 25 times
		for x in range(0,26):
			self.loop(path)

	# define x axis rotation matrix
	def R_x(self, angle):
		rad = np.radians(angle)
		return [[1,0,0],[0,np.cos(rad),-np.sin(rad)],[0,np.sin(rad),np.cos(rad)]]

	# define y axis rotation matrix
	def R_y(self, angle):
		rad = np.radians(angle)
		return [[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0, np.cos(rad)]]

	# define z axis rotation matrix
	def R_z(self, angle):
		rad = np.radians(angle)
		return [[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]]

	# define function for finding the magnitude of a vector
	def magnitude(self,vector):
		return math.sqrt(sum(pow(element, 2) for element in vector))

	# define postion function for finding x,y,z coordinates of
	# arm segments and torques for arm
	def position(self, angles, t):
		# define class variables for the
		# input angles
		self.angle1 = np.radians(angles[0])
		self.angle2 = np.radians(angles[1])
		self.angle3 = np.radians(angles[2])
		self.angle4 = np.radians(angles[3])

		# rotation matrices
		r_ab = self.R_z(np.degrees(self.angle4))
		r_bc = self.R_y(np.degrees(self.angle1))
		r_cd = self.R_y(np.degrees(self.angle2))
		r_de = self.R_y(np.degrees(self.angle3))

		# displacement matricies
		d_ab = [[0],[0],[0]]
		d_bc = [ [np.sin(self.angle1)*self.l1], [0], [np.cos(self.angle1)*self.l1] ]
		d_cd = [ [np.sin(self.angle2)*self.l2], [0], [np.cos(self.angle2)*self.l2] ]
		d_de = [ [np.sin(self.angle3)*self.l3], [0], [np.cos(self.angle3)*self.l3] ]

		# homogenous transfer matricies
		h_ab = np.concatenate((r_ab,d_ab), 1)
		h_ab = np.concatenate((h_ab,[[0,0,0,1]]), 0)
		#
		h_bc = np.concatenate((r_bc,d_bc), 1)
		h_bc = np.concatenate((h_bc,[[0,0,0,1]]), 0)
		#
		h_cd = np.concatenate((r_cd,d_cd), 1)
		h_cd = np.concatenate((h_cd,[[0,0,0,1]]), 0)
		#
		h_de = np.concatenate((r_de,d_de), 1)
		h_de = np.concatenate((h_de,[[0,0,0,1]]), 0)

		# define homogenous transformation
		# matricies for finding torque
		self.h_ac = np.dot(h_ab, h_bc)
		self.h_ad = np.dot(self.h_ac, h_cd)
		self.h_ae = np.dot(self.h_ad, h_de)

		# call torque path to find torque values for each joint
		self.torque_path(t)

		# returns list of x,y,z for each arm joint and end effectorself.
		return [ [ [0,self.h_ac[0][3]], [0,self.h_ac[1][3]], [0,self.h_ac[2][3]] ],
		[ [self.h_ac[0][3],self.h_ad[0][3]], [self.h_ac[1][3],self.h_ad[1][3]], [self.h_ac[2][3],self.h_ad[2][3]] ],
		[ [self.h_ad[0][3],self.h_ae[0][3]], [self.h_ad[1][3],self.h_ae[1][3]], [self.h_ad[2][3],self.h_ae[2][3]] ] ]

	# estimates the torque on stepper 1
	def stepper1_torque(self, t):
		# define the list for torque values in
		# the x,y,z direction around the orgin
		self.orgin_t = [0,0,0]

		# divisions mark the acceleration of the stepper
		# motor throughout its travel path
		if t <= self.divisions[0][0]:
			a = self.accel[0]
		elif (self.divisions[0][0] + self.divisions[0][1]) > t > self.divisions[0][0]:
			a = 0
		elif t >= (self.divisions[0][0] + self.divisions[0][1]):
			a = - self.accel[0]
		else:
			pass
		# print("Acceleration for Stepper 1:", a)
		if t <= self.divisions[3][0]:
			a4 = self.accel[3]
		elif (self.divisions[3][0] + self.divisions[3][1]) > t > self.divisions[3][0]:
			a4 = 0
		elif t >= (self.divisions[3][0] + self.divisions[3][1]):
			a4 = - self.accel[3]
		else:
			pass

		# ac represents the position vector of stepper 2
		# relative to stepper 1
		ac = [self.h_ac[0][3], self.h_ac[1][3], self.h_ac[2][3]]

		# tg2 represents the torque due to gravity on
		# stepper 1 (representing stepper motor 1) from stepper 2
		tg2 = np.cross(ac, [0, 0, 9.81*self.values["mass"][0]])

		# tm2 represents the torque on stepper 1 due to the
		# rotational acceleration of the mass at point ac
		tm2 = np.cross(ac, [(np.cos(self.angle4)*a4*self.values["mass"][0]),
		(np.sin(self.angle4)*a4*self.values["mass"][0]), (a*self.values["mass"][0])])

		# ad represents the position vector from the orgin to
		# stepper motor 3
		ad = [self.h_ad[0][3], self.h_ad[1][3], self.h_ad[2][3]]

		# tg3 represents the torque on stepper 1 from the weight of stepper 3
		tg3 = np.cross(ad,[0, 0, 9.81*self.values["mass"][1]])

		# tm3 represents the torque on s tepper 1 from the rotational acceleration of stepper motor 3
		tm3 = np.cross(ad,[(np.cos(self.angle1)*self.magnitude(ad)*a4*self.values["mass"][1]),
		(np.sin(self.angle4)*a4*self.values["mass"][1]), (a*self.values["mass"][1])])

		# ae represents the position vector of the end effector relative to stepper 1
		ae = [self.h_ae[0][3],self.h_ae[1][3],self.h_ae[2][3]]

		# tge represents the torque due to gravity on stepper 1
		# from the mass of the end effector
		tge = np.cross(ae,[0, 0, 9.81*self.values["mass"][2]])

		# tme represents the torque on stepper 1 due to the rotational
		# acceleration of the end effector
		tme = np.cross(ae,[(np.sin(self.angle1)*self.magnitude(ad)*a4*self.values["mass"][2]),
		(np.cos(self.angle4)*a4*self.values["mass"][2]), (a*self.values["mass"][2])])

		# vector sum of all of the torques on stepper motor 1
		for x in range(0,3):
			self.orgin_t[x] = round(tg2[x] + tg3[x] + tge[x] + tm2[x] + tm3[x] + tme[x], 3)

	# estimates the torque on stepper 2
	def stepper2_torque(self, t):
		# define class variable for the torque on
		# stepper 2
		self.motor2_t = [0,0,0]

		# define the acceleration of the joints connected to
		# stepper motor 2
		if t <= self.divisions[1][0]:
			a = self.accel[1]
		elif (self.divisions[1][0] + self.divisions[1][1]) > t > self.divisions[1][0]:
			a = 0
		elif t >= (self.divisions[1][0] + self.divisions[1][1]):
			a = - self.accel[1]
		else:
			pass

		# define the acceleration due to the rotation of stepper 4
		if t <= self.divisions[3][0]:
			a4 = self.accel[3]
		elif (self.divisions[3][0] + self.divisions[3][1]) > t > self.divisions[3][0]:
			a4 = 0
		elif t >= (self.divisions[3][0] + self.divisions[3][1]):
			a4 = - self.accel[3]
		else:
			pass

		# cd represents the position vector of stepper 3 relative to stepper 1
		cd = [self.h_ad[0][3]-self.h_ac[0][3],self.h_ad[1][3]-self.h_ac[1][3],self.h_ad[2][3]-self.h_ac[2][3]]

		# torque on stepper 2 due to the weight of stepper 3
		tg3 = np.cross(cd, [0, 0, 9.81*self.values["mass"][1]])

		# torque on stepper 2 due to the rotational acceleration of stepper 3
		tm3 = np.cross(cd,[(np.cos(self.angle4)*a4*self.values["mass"][1]),
		(np.sin(self.angle4)*a4*self.values["mass"][1]), (a*self.values["mass"][1])])

		# ce represents the position vector of the end effector relative to
		# stepper 2
		ce = [self.h_ae[0][3]-self.h_ac[0][3],self.h_ae[1][3]-self.h_ac[1][3],self.h_ae[2][3]-self.h_ac[2][3]]

		# tge represents the torque due to gravity on stepper 2
		# due to the weight of the end effector
		tge = np.cross(ce,[0,0,9.81*self.values["mass"][2]])

		# tme represents the torque on stepper 2 due to the
		# rotational acceleration of the end effector
		tme = np.cross(ce,[(np.cos(self.angle4)*a4*self.values["mass"][1]),
		(np.sin(self.angle4)*a4*self.values["mass"][1]), (a*self.values["mass"][1])])

		# vector sum of the torques
		for x in range(0,3):
			self.motor2_t[x] = round(tg3[x] + tge[x] + tm3[x] + tme[x],3)

	# estimates the torque on stepper 3
	def stepper3_torque(self, t):
		# define class variable of stepper 3 torque
		self.motor3_t = [0,0,0]

		# define the acceleration of the end effector
		# connected to stepper 3
		if t <= self.divisions[2][0]:
			a = self.accel[2]
		elif (self.divisions[2][0] + self.divisions[2][1]) > t > self.divisions[2][0]:
			a = 0
		elif t >= (self.divisions[2][0] + self.divisions[2][1]):
			a = - self.accel[2]
		else:
			pass

		# define the acceleration of things connected to
		# stepper 4
		if t <= self.divisions[3][0]:
			a4 = self.accel[3]
		elif (self.divisions[3][0] + self.divisions[3][1]) > t > self.divisions[3][0]:
			a = 0
		elif t >= (self.divisions[3][0] + self.divisions[3][1]):
			a4 = - self.accel[3]
		else:
			pass

		# de represents the position vector of the end effector
		# relative to stepper 3
		de = [self.h_ae[0][3]-self.h_ad[0][3],self.h_ae[1][3]-self.h_ad[1][3],self.h_ae[2][3]-self.h_ad[2][3]]

		# tge represents the torque on stepper 3 due to the
		# weight of the end effector
		tge = np.cross(de,[0,0,9.81*self.values["mass"][2]])

		# tme represents the torque on stepper 3 due to the
		# acceleration of stepper 4
		tme = np.cross(de,[(np.cos(self.angle4)*a4*self.values["mass"][2]),
		(np.sin(self.angle4)*a4*self.values["mass"][2]), (a*self.values["mass"][2])])

		# vector sum of torques
		for x in range(0,3):
			self.motor3_t[x] = round(tge[x] + tme[x],3)

	# calls the torque functions
	def torque_path(self, t):
		self.stepper1_torque(t)
		self.stepper2_torque(t)
		self.stepper3_torque(t)

		# print("Stepper 1 Torque Magnitude:",round(np.sqrt((self.orgin_t[0]**2)+(self.orgin_t[1]**2)+(self.orgin_t[2]**2)),3),"Nm")
		# print("Stepper 2 Torque Magnitude:",round(np.sqrt((self.motor2_t[0]**2)+(self.motor2_t[1]**2)+(self.motor2_t[2]**2)),3),"Nm")
		# print("Stepper 3 Torque Magnitude:",round(np.sqrt((self.motor3_t[0]**2)+(self.motor3_t[1]**2)+(self.motor3_t[2]**2)),3),"Nm")

	# runs the simulation
	def loop(self, path):
		self.path_len = len(path[0])
		# iterates through the list of angles
		for self.x in range(0, self.p):
			#
			t = (self.t_time/self.p)*(self.x+1)
			# print("Time:", t)

			# defines lines as a list of points representing the joints and end effector of the arm
			lines = self.position([path[0][self.x], path[1][self.x], path[2][self.x], path[3][self.x]], t)

			# scales up the size for model
			scale = 1000
			self.line0.set_data_3d(np.dot(lines[0][0],scale), np.dot(lines[0][1],scale), np.dot(lines[0][2],scale))
			#
			self.line1.set_data_3d(np.dot(lines[1][0],scale), np.dot(lines[1][1],scale), np.dot(lines[1][2],scale))
			#
			self.line2.set_data_3d(np.dot(lines[2][0],scale), np.dot(lines[2][1],scale), np.dot(lines[2][2],scale))
			#
			self.line3.set_data_3d(np.dot([0, self.orgin_t[0]], scale), np.dot([0, self.orgin_t[1]], scale), np.dot([0, self.orgin_t[2]], scale))
			#
			self.line4.set_data_3d(np.dot([lines[1][0][0], self.motor2_t[0] + lines[1][0][0]],scale), np.dot([lines[1][1][0], self.motor2_t[1] + lines[1][1][0]], scale),
			np.dot([lines[1][2][0],self.motor2_t[2] + lines[1][2][0]], scale))
			#
			self.line5.set_data_3d(np.dot([lines[2][0][0],self.motor3_t[0] + lines[2][0][0]],scale),np.dot([lines[2][1][0], self.motor3_t[1] + lines[2][1][0]],scale),
			np.dot([lines[2][2][0], self.motor3_t[2] + lines[2][2][0]], scale))

			# print("x: ",self.magnitude(np.dot([lines[1][0][0],self.motor2_t[0]],1000)))
			# print("y: ",self.magnitude(np.dot([lines[1][1][0],self.motor2_t[1]],1000)))
			# print("z: ",self.magnitude(np.dot([lines[1][2][0],self.motor2_t[2]],1000)))

			# path of end effector
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
			time.sleep(0.0001)

path = Path([0,0,0,0.0], [0.20, 0.150, 0.050])
plot = Plot(path.path, [0.20, 0.150, 0.050], [0,0,0,0.0], path.t_time, path.p, path.divisions, path.accel)
