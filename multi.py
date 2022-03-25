# code for testing multiplexer with magnetic encoders
import board
import busio
import time
import struct

# define i2c object - Pin 28 for SCL and Pin 27 for SDA - 400 hz frequency
i2c = (busio.I2C(board.SCL_1, board.SDA_1, 400))
time.sleep(1)
mag_id = i2c.scan()
print("Id Scan", hex(mag_id[0]))

def

i2c.deinit()
