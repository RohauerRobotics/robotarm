# Moment Arm Physics Simulation
# Only Runs in 2D to keep things simple
import csv
import numpy as np

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


class Calulations(object):
    def __init__(self, dictionary_inputs):
        print("Calculations class running")
        self.data = dictionary_inputs
        self.data['angle0'] = 0
        self.data['angle1'] = 0
        self.data['angle2'] = 0
        self.holding_torque_calculation()

    def lever_arm_length(self,joint_count):
        # pass
        leverarm_sum = 0
        for x in range(0,joint_count):
            leverarm_sum += self.data['arm_lengths_milimeters'][x]*np.cos(np.radians(self.data['angle'+str(x)]))
            #print("arm length added:", self.data['arm_lengths_milimeters'][x])
        return leverarm_sum

    def holding_torque_calculation(self):
        whole_lever_arm = self.lever_arm_length(3)
        torque_m3 = 100*(whole_lever_arm/1000)*(self.data['arm_mass_grams'][2]/1000)*9.81
        #print(torque_m3,"Newton Centimetes")
        two_arm_pieces = self.lever_arm_length(2)
        torque_m2 = 100*(two_arm_pieces/1000)*(self.data['arm_mass_grams'][1]/1000)*9.81
        #print(torque_m2,"Newton Centimetes")
        one_arm = self.lever_arm_length(1)
        torque_m1 = 100*(one_arm/1000)*(self.data['arm_mass_grams'][0]/1000)*9.81
        #print(torque_m1,"Newton Centimetes")
        holding_toque_sum = torque_m1 + torque_m2 + torque_m3
        print("Sum of holding torque", holding_toque_sum, "Newton Centimeters")

    def inertial_torque_calculation(self):
        whole_lever_arm = self.lever_arm_length(3)
        torque_m3 = 100*(whole_lever_arm/1000)*(self.data['arm_mass_grams'][2]/1000)*9.81
        #print(torque_m3,"Newton Centimetes")
        two_arm_pieces = self.lever_arm_length(2)
        torque_m2 = 100*(two_arm_pieces/1000)*(self.data['arm_mass_grams'][1]/1000)*9.81
        #print(torque_m2,"Newton Centimetes")
        one_arm = self.lever_arm_length(1)
        torque_m1 = 100*(one_arm/1000)*(self.data['arm_mass_grams'][0]/1000)*9.81
        #print(torque_m1,"Newton Centimetes")
        holding_toque_sum = torque_m1 + torque_m2 + torque_m3
        print("Sum of holding torque", holding_toque_sum, "Newton Centimeters")

test = Calulations(values)
