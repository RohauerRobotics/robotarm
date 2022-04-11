# angle map test
import csv
import numpy as np
import time

inital_angles = [45,0,100]

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

def find_angles(x,z):
    h = np.sqrt((x**2 + z**2))
    l2 = values['arm_lengths_milimeters'][1]
    l1 = values['arm_lengths_milimeters'][0]
    angle_1 = np.arccos(-(l2**2-l1**2-h**2)/(2*l1*h))
    angle_1_addition = np.arctan(z/x)
    angle_2 = np.arccos(((np.sin(angle_1)*l1)/l2)) + (np.pi/2) - angle_1
    angle_3 = 270
    return [np.degrees(angle_1),np.degrees(angle_2),angle_3]

def path_to(inital, final):
    path = [[],[],[]]
    for w in range(0, 3):
        print(inital[w])
        if(inital[w] > final[w]):
            for k in range(0, abs(int(inital[w]-final[w]))+1):
                path[w].append(inital[w]-1*k)
        elif(inital[w] < final[w]):
            for q in range(0, abs(int(inital[w]-final[w]))+1):
                path[w].append(inital[w]+1*q)
        else:
            path[w] = inital[w]
    return path

def path_smoother(path):
    list_len = [len(i) for i in path]
    # print(list_len)
    print(len(path))
    for w in range(0, len(path)):
        path[w].extend([path[w][-1]]*(max(list_len)-len(path[w])))
    return path


test = find_angles(125,100)
path = path_to(inital_angles, test)
patched_path = path_smoother(path)
print(patched_path)
# print(path)
# print(test)
