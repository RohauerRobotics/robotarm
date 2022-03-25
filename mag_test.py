# https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi

# I2C control of magnetic encoder

import board
import busio
import time
import struct

i2c = (busio.I2C(board.SCL_1, board.SDA_1, 400))
time.sleep(1)
# set up I2C Bus connection == wire.begin

def check_magnent():
    # collects raw angle from
    data_buffer_in = bytearray(1)
    print("Checking Magnent Status")
    while(bytearray.decode(data_buffer_in) != 'g'):
        data_buffer_in = bytearray(1)
        mstat_address = bytes([0x0B])
        i2c.writeto(0x36, mstat_address, stop=False)
        time.sleep(0.005)
        i2c.readfrom_into(0x36, data_buffer_in)
        print("Magnent Status", bytearray.decode(data_buffer_in))
    print("Magnent Detected")
    return True

def check_magnent_noprint():
    # collects raw angle from
    data_buffer_in = bytearray(1)
    # print("Checking Magnent Status")
    while(bytearray.decode(data_buffer_in) != 'g'):
        data_buffer_in = bytearray(1)
        mstat_address = bytes([0x0B])
        i2c.writeto(0x36, mstat_address, stop=False)
        time.sleep(0.005)
        i2c.readfrom_into(0x36, data_buffer_in)
        # print("Magnent Status", bytearray.decode(data_buffer_in))
    # print("Magnent Detected")
    return True

def read_angle_raw():
    angleHigh_buff = bytearray(2)
    mstat_address = bytes([0x0D])
    i2c.writeto(0x36, mstat_address, stop=False)
    time.sleep(0.05)
    i2c.readfrom_into(0x36, angleHigh_buff)
    # angleHigh_buff = angleHigh_buff << 8;
    # print("High angle",angleHigh_buff[1])
    angleLow_buff = bytearray(2)
    mstat_address = bytes([0x0C])
    i2c.writeto(0x36, mstat_address, stop=False)
    time.sleep(0.05)
    i2c.readfrom_into(0x36, angleLow_buff)
    high = angleHigh_buff[1:2]
    low = angleLow_buff[1:2]
    # print(high)
    rawangle = struct.pack('cc',bytes(high),bytes(low))
    # rawangle.append(int(bytes(angleLow_buff[1:2])))
    # print(low)
    int_angle = int.from_bytes(rawangle,"big")
    print(int_angle*0.087890625)

# returns a scan of the i2c pins and prints the board ID
mag_id = i2c.scan()
print("Id Scan", hex(mag_id[0]))
check_magnent()
while(check_magnent_noprint() == True):
    read_angle_raw()
# i2c.writeto_then_readfrom(i2c_address, out_buffer: circuitpython_typing.ReadableBuffer,
# in_buffer: circuitpython_typing.WriteableBuffer, *, out_start: int = 0, out_end: int = sys.maxsize, in_start: int = 0, in_end: int = sys.maxsize)

# run to de initialize the board
i2c.deinit()
