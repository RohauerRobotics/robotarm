import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import json
# packages for multiprocessing
from multiprocessing import Lock, Process, Queue, current_process
import queue
# packages for multiplexing and interfacing with hardware
import board
import busio
import adafruit_tca9548a
import digitalio as dg

class Client(object):
    def __init__(self):
        # command for finding camera locations
        # v4l2-ctl --list-devices

        # PC = True
        # if PC == True:
        #     cam1 = 0
        #     cam2 = 1
        # else:
        #     cam1 = "/dev/video3"
        #     cam2 = "/dev/video1"
        #
        # self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client.connect(('192.168.1.9', 8485))
        # connection = self.client.makefile('wb')
        #
        # # define handcam
        # self.handcam = cv2.VideoCapture(cam1, cv2.CAP_V4L2)
        # self.handcam.set(3, 1280)
        # self.handcam.set(4, 720)
        # cv2.waitKey(2)
        #
        # # define overheadcam
        # self.overheadcam = cv2.VideoCapture(cam2, cv2.CAP_V4L2)
        # self.overheadcam.set(3, 1280)
        # self.overheadcam.set(4, 720)
        # cv2.waitKey(2)

        # for x in range(0,10):
        #     self.wait_for_request()
        #     # print("Taking a break")
        #     time.sleep(0.25)
        pass

    def operate_connection(self, movement_instructions, state, state_request, initialization):
        # command for finding camera locations
        # v4l2-ctl --list-devices

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(('192.168.1.10', 8485))
        connection = self.client.makefile('wb')

        PC = False
        if PC == True:
            cam1 = 0
            cam2 = 1
            # define handcam
            self.handcam = cv2.VideoCapture(cam1)
            self.handcam.set(3, 1280)
            self.handcam.set(4, 720)
            cv2.waitKey(2)

            # define overheadcam
            self.overheadcam = cv2.VideoCapture(cam2)
            self.overheadcam.set(3, 1280)
            self.overheadcam.set(4, 720)
            cv2.waitKey(2)

        else:
            cam1 = "/dev/video1"
            cam2 = "/dev/video2"

            # define handcam
            self.handcam = cv2.VideoCapture(cam1, cv2.CAP_V4L2)
            self.handcam.set(3, 1280)
            self.handcam.set(4, 720)
            cv2.waitKey(2)

            # define overheadcam
            self.overheadcam = cv2.VideoCapture(cam2, cv2.CAP_V4L2)
            self.overheadcam.set(3, 1280)
            self.overheadcam.set(4, 720)
            cv2.waitKey(2)

        for x in range(0,250):
            self.wait_for_request(movement_instructions, state, state_request, initialization)
            # print("Taking a break")
            time.sleep(0.25)

    def wait_for_request(self, movement_instructions, state, state_request, initialization):
        request = b""
        request = self.client.recv(4096)
        # request.decode('utf-8')
        request = str(request, 'utf-8')
        # print(request)
        copy = list(request)
        # print(copy)
        pick = ""
        for i in range(0, len(copy)):
            pick += copy[i]

        print(pick)

        if pick == "send_state":
            self.send_state(state, state_request)

        elif pick == "movement_instructions":
            self.movement(movement_instructions)

        elif pick == "zero":
            print("correct pick")
            self.zero_initialization(initialization)

        elif pick == "send_handcam" or "send_overheadcam":
            self.send_cam(request)

    def send_state(self, state, state_request):
        state_request.put("Get State")
        while state.empty():
            pass
        dict = state.get()
        json_dict = pickle.dumps(dict, 0)
        # test = bytes("good",'utf-8')
        self.client.sendall(json_dict)
        print("sent package")

        acknowledge = self.client.recv(4096)
        acknowledge.decode('utf-8')
        print(acknowledge)

    def send_cam(self, request):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        if request == "send_overheadcam":
            if self.overheadcam.grab():
                ret, frame = self.overheadcam.retrieve(0)
            else:
                print("Error no camera data!")
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(frame, 0)
            size = len(data)
            self.client.sendall(struct.pack(">L", size) + data)
            print("sent image")

            acknowledge = self.client.recv(4096)
            acknowledge.decode('utf-8')
            print(acknowledge)

        elif request == "send_handcam":
            if self.handcam.grab():
                ret, frame = self.handcam.retrieve(0)
            else:
                print("Error no camera data!")
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(frame, 0)
            size = len(data)
            self.client.sendall(struct.pack(">L", size) + data)
            print("sent image")

            acknowledge = self.client.recv(4096)
            acknowledge.decode('utf-8')
            print(acknowledge)
        else:
            pass

    def movement(self, movement_instructions):
        ready = bytes("ready for instructions",'utf-8')
        self.client.sendall(ready)

        instructions = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        while len(instructions) < payload_size:
            print("Recv: {}".format(len(instructions)))
            instructions += self.client.recv(4096)

        print("Done Recv: {}".format(len(instructions)))
        packed_msg_size = instructions[:payload_size]
        instructions = instructions[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(instructions) < msg_size:
            instructions += self.client.recv(4096)

        instruction_data = instructions[:msg_size]
        instructions = pickle.loads(instruction_data, fix_imports=True, encoding="bytes")
        # print(instructions)

        recieved = bytes("instructions sent", 'utf-8')
        self.client.sendall(recieved)

        movement_instructions.put(instructions)

    def zero_initialization(self, initialization):
        ready = bytes("begining initialization",'utf-8')
        self.client.sendall(ready)

        initialization.put("yes")


class States(object):
    def __init__(self):
        # define class variable for socket communication
        connection =  Client()

        # define hardware interface pins
        self.initialize_pins()
        self.initialize_multiplexer()

        # self.
        # define queues for passing information between threads
        movement_instructions = Queue()
        state = Queue()
        state_request = Queue()
        initialization = Queue()

        # define processes

        # collects state conditions
        collect_state_p = Process(target = self.collect_state, args=(state, state_request))
        collect_state_p.start()

        # recieves communication from gaming pc
        connection_p = Process(target = connection.operate_connection, args=(
                                movement_instructions, state,state_request, initialization))
        connection_p.start()

        # executes instructions from pc
        execute_movement_p = Process(target=self.execute_movement, args=(
                                movement_instructions, state, state_request, initialization))
        execute_movement_p.start()


        proceses = [connection_p, execute_movement_p, collect_state_p]

        for p in proceses:
            p.join()

    def initialize_pins(self):
        # stepper 1
        self.DIR1 = dg.DigitalInOut(board.D4)
        self.DIR1.direction = dg.Direction.OUTPUT
        self.STEP1 = dg.DigitalInOut(board.D17)
        self.STEP1.direction = dg.Direction.OUTPUT

        # stepper 2
        self.DIR2 = dg.DigitalInOut(board.D27)
        self.DIR2.direction = dg.Direction.OUTPUT
        self.STEP2 = dg.DigitalInOut(board.D22)
        self.STEP2.direction = dg.Direction.OUTPUT

        # stepper 3
        self.DIR3 = dg.DigitalInOut(board.D10)
        self.DIR3.direction = dg.Direction.OUTPUT
        self.STEP3 = dg.DigitalInOut(board.D9)
        self.STEP3.direction = dg.Direction.OUTPUT

        # stepper 4
        self.DIR4 = dg.DigitalInOut(board.D11)
        self.DIR4.direction = dg.Direction.OUTPUT
        self.STEP4 = dg.DigitalInOut(board.D0)
        self.STEP4.direction = dg.Direction.OUTPUT

        # stepper 5
        self.DIR5 = dg.DigitalInOut(board.D5)
        self.DIR5.direction = dg.Direction.OUTPUT
        self.STEP5 = dg.DigitalInOut(board.D6)
        self.STEP5.direction = dg.Direction.OUTPUT

        # stepper 6
        self.DIR6 = dg.DigitalInOut(board.D13)
        self.DIR6.direction = dg.Direction.OUTPUT
        self.STEP6 = dg.DigitalInOut(board.D19)
        self.STEP6.direction = dg.Direction.OUTPUT

        # ultrasonic sensor
        self.TRIG = dg.DigitalInOut(board.D25)
        self.TRIG.direction = dg.Direction.OUTPUT
        self.ECHO = dg.DigitalInOut(board.D8)
        self.ECHO.direction = dg.Direction.INPUT

        # set Trig value to false
        self.TRIG.value = False
        # time.sleep(1.5)

        # Hall effect sensors

        # hall effect sensor 1
        self.HALL1 = dg.DigitalInOut(board.D26)
        self.HALL1.direction = dg.Direction.INPUT

        # hall effect sensor 2
        self.HALL2 = dg.DigitalInOut(board.D14)
        self.HALL2.direction = dg.Direction.INPUT

        # hall effect sensor 3
        self.HALL3 = dg.DigitalInOut(board.D15)
        self.HALL3.direction = dg.Direction.INPUT

        # hall effect sensor 4
        self.HALL4 = dg.DigitalInOut(board.D18)
        self.HALL4.direction = dg.Direction.INPUT

        # hall effect sensor 5
        self.HALL5 = dg.DigitalInOut(board.D23)
        self.HALL5.direction = dg.Direction.INPUT

        # hall effect sensor 6
        self.HALL6 = dg.DigitalInOut(board.D24)
        self.HALL6.direction = dg.Direction.INPUT

    def initialize_multiplexer(self):
        # define i2c connection to multiplexer
        self.i2c = (busio.I2C(board.SCL, board.SDA, 400))
        time.sleep(0.1)
        self.tca = adafruit_tca9548a.TCA9548A(self.i2c)

        # make list of connected encoders
        self.encoder_verify = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None}
        for key in self.encoder_verify.keys():
            q = key
            if self.tca[q].try_lock():
                address = self.tca[q].scan()
                print("Found",hex(address[0]))
                print("Raw address:", address[0])
                self.tca[q].unlock()
                if hex(address[0]) == hex(54):
                    self.encoder_verify[key] = True
                else:
                    self.encoder_verify[key] = False
            else:
                self.encoder_verify[key] = False

        print(f"Encoders lib: {self.encoder_verify}")

    def read_angles(self):
        lib = {'a1':None, 'a2':None, 'a3':None, 'a4':None, 'a5':None, 'a6':None}

        for key in self.encoder_verify.keys():
            q = key
            # print(q)
            if self.encoder_verify[key]:
                print(f"Indexing: {q}")
                angle_count = 0
                angle_list = []
                for x in range(0, 2):
                    if self.check_magnent_noprint(q):
                        angle_list.append(self.read_angle_raw(q))
                        angle_count += 1
                    else:
                        print("No Magnent Detected!")
                idx = "a" + str(q+1)
                if sum(angle_list) != 0:
                    avg = sum(angle_list)/angle_count
                    lib[idx] = avg
                else:
                    lib[idx] = None
            else:
                pass
        # print(lib)
        return lib

    def read_angle_raw(self, address):
        angleHigh_buff = bytearray(2)
        mstat_address = bytes([0x0D])
        if self.tca[address].try_lock():
            self.tca[address].writeto(0x36, mstat_address, stop=False)
            time.sleep(0.05)
            self.tca[address].readfrom_into(0x36, angleHigh_buff)
            # angleHigh_buff = angleHigh_buff << 8;
            # print("High angle",angleHigh_buff[1])
            angleLow_buff = bytearray(2)
            mstat_address = bytes([0x0C])
            self.tca[address].writeto(0x36, mstat_address, stop=False)
            time.sleep(0.05)
            self.tca[address].readfrom_into(0x36, angleLow_buff)
            high = angleHigh_buff[1:2]
            low = angleLow_buff[1:2]
            # print(high)
            rawangle = struct.pack('cc',bytes(high),bytes(low))
            # rawangle.append(int(bytes(angleLow_buff[1:2])))
            # print(low)
            int_angle = int.from_bytes(rawangle,"big")
            float_angle = int_angle*0.087890625
            print(float_angle)
            self.tca[address].unlock()
            return(float_angle)
        else:
            print(f"Couldn't Lock {address}!")
            return(0)

    def ultrasonic_distance(self):
        # set Trigger to HIGH
        # GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
        self.TRIG.value = True

        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        # GPIO.output(GPIO_TRIGGER, GPIO.LOW)
        self.TRIG.value = False

        StartTime = time.time()
        StopTime = time.time()
        # print("Starting Time")
        # save StartTime
        while self.ECHO.value == 0:
            StartTime = time.time()
            # print("infinite loop?")

        # save time of arrival
        # print(f"echo output: {self.ECHO.value}")
        while self.ECHO.value == 1:
            StopTime = time.time()
            if (StopTime - StartTime) > 0.25:
                break
            # print("Stop time: ",StopTime)
            # print("loop 2?")
        # print("Stoping Time")
        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2

        # sanity check the distance sensor
        if distance > 1000:
            distance = 0

        else:
            self.ultra_count += 1

        print(f"Ultrasonic Distance: {distance}")
        return distance

    def check_magnent_noprint(self, address):
        # collects raw angle from
        data_buffer_in = bytearray(1)
        # print("Checking Magnent Status")
        while(bytearray.decode(data_buffer_in) != 'g'):
            data_buffer_in = bytearray(1)
            mstat_address = bytes([0x0B])
            if self.tca[address].try_lock():
                self.tca[address].writeto(0x36, mstat_address, stop=False)
                time.sleep(0.005)
                self.tca[address].readfrom_into(0x36, data_buffer_in)
                self.tca[address].unlock()
            else:
                pass
            # print("Magnent Status", bytearray.decode(data_buffer_in))
        # print("Magnent Detected")
        return True

    def find_encoder_direction(self):
        self.angle_dirs = [None, None, None, None, None, None]
        self.first_dirs = [None, None, None, None, None, None]

        for key in self.encoder_verify.keys():
            if self.encoder_verify[key] == True:
                exit_dir = None
                dir_sample = []
                num_samples = 6
                delay = 0.05

                if not self.halls[key].value:
                    print("Hall Sensor Detected while finding direction!")
                    for x in range(1, 250):
                        # print(f"Hall Value: {self.halls[key].value}")
                        # print(f"Num of Steps: {x}")

                        self.dirs[key].value = True
                        self.step(self.steps[key], x, delay)

                        if self.halls[key].value:
                            exit_dir = True
                            break

                        else:
                            self.dirs[key].value = False
                            self.step(self.steps[key], x, delay)

                        self.dirs[key].value = False
                        self.step(self.steps[key], x, delay)

                        if self.halls[key].value:
                            exit_dir = False
                            break

                        else:
                            self.dirs[key].value = True
                            self.step(self.steps[key], x, delay)

                if exit_dir == None:
                    for q in range(0, num_samples):
                        # print(f"hall sensor: {halls[key].value}")
                        if self.halls[key].value:
                            self.dirs[key].value = True
                            self.steps[key].value = True
                            time.sleep(delay)
                            self.steps[key].value = False
                            time.sleep(delay)

                            if self.check_magnent_noprint(key):
                                dir_sample.append(self.read_angle_raw(key))
                            else:
                                print("No Magnent Detected")

                        elif not self.halls[key].value:
                            if self.check_magnent_noprint(key):
                                self.stop_1[key] = self.read_angle_raw(key)
                                self.first_dirs[key] = True
                            else:
                                print("No Magnent Detected")
                            break



                elif exit_dir != None:
                    self.step(self.steps[key], 45, delay)
                    for q in range(0, num_samples):
                        # print(f"hall sensor: {halls[key].value}")
                        if self.halls[key].value:
                            self.dirs[key].value = True
                            self.steps[key].value = True
                            time.sleep(delay)
                            self.steps[key].value = False
                            time.sleep(delay)

                            if self.check_magnent_noprint(key):
                                dir_sample.append(self.read_angle_raw(key))
                            else:
                                print("No Magnent Detected")

                        elif not self.halls[key].value:
                            if self.check_magnent_noprint(key):
                                self.stop_1[key] = self.read_angle_raw(key)
                                self.first_dirs[key] = True
                            else:
                                print("No Magnent Detected")
                            break

                for i in range(0, num_samples-1):
                    a = dir_sample[i]
                    b = dir_sample[i + 1]
                    if abs(b-a) < 2.5:
                        if b > a:
                            self.angle_dirs[key] = "positive"
                        elif b < a:
                            self.angle_dirs[key] = "negative"
                        else:
                            pass
                    else:
                        pass
            else:
                pass

    def step(self, stepper, num_steps, delay):
        for x in range(0, num_steps):
            stepper.value = True
            time.sleep(delay)
            stepper.value = False
            time.sleep(delay)

    def zero_steppers(self):
        print("zeroing steppers")
        # define lists for processing all of the joint zeroing
        self.dirs = [self.DIR1, self.DIR2, self.DIR3, self.DIR4, self.DIR5, self.DIR6]
        self.steps = [self.STEP1, self.STEP2, self.STEP3, self.STEP4, self.STEP5, self.STEP6]
        self.halls = [self.HALL1, self.HALL2, self.HALL3, self.HALL4, self.HALL5, self.HALL6]

        # define stop_1, stop_2, angle direction, and first direction
        self.stop_1 = [None, None, None, None, None, None]
        self.stop_2 = [None, None, None, None, None, None]
        self.angle_dirs = [None, None, None, None, None, None]
        self.first_dirs = [None, None, None, None, None, None]

        # set 2nd encoder verification to False to speed up test
        self.encoder_verify[2] = False

        # determine angle_dirs
        self.find_encoder_direction()
        # print(f"angle dirs: {self.angle_dirs}")

        # move steppers to the first stopping points
        delay = 0.05
        for key in self.encoder_verify.keys():
            if self.encoder_verify[key] == True:
                for q in range(0, 2000):
                    print("count: ", q)

                    if self.halls[key].value:
                        self.dirs[key].value = True
                        self.step(self.steps[key], 1, delay)

                    elif not self.halls[key].value:
                        if self.check_magnent_noprint(key):
                            self.stop_1[key] = self.read_angle_raw(key)
                            self.first_dirs[key] = True
                        else:
                            print("No Magnent 1")
                        break

        # print(f"List of Stop 1s: {self.stop_1}")

        for key in self.encoder_verify.keys():
            if self.encoder_verify[key] == True:

                for w in range(0, 45):
                    self.dirs[key].value = False
                    self.step(self.steps[key], 1, delay)

                for q in range(0, 2000):
                    print("count: ", q)

                    if self.halls[key].value:
                        self.dirs[key].value = False
                        self.step(self.steps[key], 1, delay)

                    elif not self.halls[key].value:
                        if self.check_magnent_noprint(key):
                            self.stop_2[key] = self.read_angle_raw(key)
                        else:
                            print("No Magnent 1")
                        break

        print(f"List of Stop 1s: {self.stop_1}")
        print(f"List of Stops: {self.stop_2}")
        print(f"angle dirs: {self.angle_dirs}")

    def follow_path(self, path):
        print("Following path:", path)

    def execute_movement(self, movement_instructions, state, state_request, initialization):
        print("executing movement")
        while True:
            if not movement_instructions.empty():
                x = movement_instructions.get()
                print("Movement instructions:", x)

            if not initialization.empty():
                w = initialization.get()
                print("initialization:", w)
                self.zero_steppers()

            else:
                pass

    def collect_state(self, state, state_request):
        while True:
            if not state_request.empty():
                request = state_request.get()
                print(request)
                lib = self.read_angles()

                # get an average sample from ultrasonic sensor
                self.ultra_count = 0
                num_samples = 3
                ultra_sam = []
                for x in range(0, num_samples):
                    ultra_sam.append(self.ultrasonic_distance())

                if self.ultra_count > 0:
                    ultra_avg = sum(filter(None, ultra_sam))/self.ultra_count
                else:
                    ultra_avg = "None"
                lib['ultra'] = ultra_avg

                # print state out and post to queue
                print(lib)
                state.put(lib)
                print("Posted State")
            else:
                pass



if __name__=="__main__":
	test = States()
