# multiplexer working version
# code for testing multiplexer with magnetic encoders
import board
import busio
import time
import struct
import adafruit_tca9548a

# define i2c object - Pin 28 for SCL and Pin 27 for SDA - 400 hz frequency
i2c = (busio.I2C(board.SCL, board.SDA, 400))
time.sleep(0.1)
tca = adafruit_tca9548a.TCA9548A(i2c)
# address = tca[0].scan()

# mag_id = i2c.scan()
# for channel in range(8):
#     if tca[channel].try_lock():
#         print("Channel {}:".format(channel), end="")
#         addresses = tca[channel].scan()
#         print([hex(address) for address in addresses if address != 0x70])
#         tca[channel].unlock()
q = 0

if tca[q].try_lock():
    address = tca[q].scan()
    print("Found",hex(address[0]))
    tca[q].unlock()

# if tca[2].try_lock():
#     address = tca[2].scan()
#     print("Found",hex(address[0]))
#     tca[2].unlock()

def select_bus(bus_num):
    bus_add = bytes([bus_num])
    i2c.writeto(0x70, bus_add, stop=False)
    time.sleep(0.1)
    print("Bus",bus_num,"selected")

def check_magnent(address):
    # collects raw angle from
    data_buffer_in = bytearray(1)
    print("Checking Magnent Status")
    while(bytearray.decode(data_buffer_in) != 'g'):
        data_buffer_in = bytearray(1)
        mstat_address = bytes([0x0B])
        if tca[address].try_lock():
            tca[address].writeto(0x36, mstat_address, stop=False)
            time.sleep(0.005)
            tca[address].readfrom_into(0x36, data_buffer_in)
            # print(f"Non decoded version: {data_buffer_in}")
            print("Magnent Status", bytearray.decode(data_buffer_in,'utf-8'))
            tca[address].unlock()
        else:
            pass
    print("Magnent Detected")
    return True

def read_angle_raw(address):
    angleHigh_buff = bytearray(2)
    mstat_address = bytes([0x0D])
    if tca[address].try_lock():
        tca[address].writeto(0x36, mstat_address, stop=False)
        time.sleep(0.05)
        tca[address].readfrom_into(0x36, angleHigh_buff)
        # angleHigh_buff = angleHigh_buff << 8;
        # print("High angle",angleHigh_buff[1])
        angleLow_buff = bytearray(2)
        mstat_address = bytes([0x0C])
        tca[address].writeto(0x36, mstat_address, stop=False)
        time.sleep(0.05)
        tca[address].readfrom_into(0x36, angleLow_buff)
        high = angleHigh_buff[1:2]
        low = angleLow_buff[1:2]
        # print(high)
        rawangle = struct.pack('cc',bytes(high),bytes(low))
        # rawangle.append(int(bytes(angleLow_buff[1:2])))
        # print(low)
        int_angle = int.from_bytes(rawangle,"big")
        print(int_angle*0.087890625)
        tca[address].unlock()

def check_magnent_noprint(address):
    # collects raw angle from
    data_buffer_in = bytearray(1)
    # print("Checking Magnent Status")
    while(bytearray.decode(data_buffer_in) != 'g'):
        data_buffer_in = bytearray(1)
        mstat_address = bytes([0x0B])
        if tca[address].try_lock():
            tca[address].writeto(0x36, mstat_address, stop=False)
            time.sleep(0.005)
            tca[address].readfrom_into(0x36, data_buffer_in)
            tca[address].unlock()
        else:
            pass
        # print("Magnent Status", bytearray.decode(data_buffer_in))
    # print("Magnent Detected")
    return True

# select_bus(0)
# mag_id = i2c.scan()
# print("Id Scan", mag_id)
check_magnent(q)

while(check_magnent_noprint(q) == True):
    read_angle_raw(q)

i2c.deinit()
