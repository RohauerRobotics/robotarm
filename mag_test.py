# I2C control of magnetic encoder

import board
import busio
import time

i2c = (busio.I2C(board.SCL_1, board.SDA_1, 400))

# set up I2C Bus connection == wire.begin
i2c_address = bytes(0x36, 'utf-8')
i2c_address = bytes(0x36, 'utf-8')
i2c = (busio.I2C(board.SCL_1, board.SDA_1, 400))

# returns a scan of the i2c pins and prints the board ID
print(i2c.scan())

i2c.writeto_then_readfrom(i2c_address, out_buffer: circuitpython_typing.ReadableBuffer,
in_buffer: circuitpython_typing.WriteableBuffer, *, out_start: int = 0, out_end: int = sys.maxsize, in_start: int = 0, in_end: int = sys.maxsize)

# run to de initialize the board
i2c.deinit()

def check_magnent():
    mstat_address = bytes(0x36, 'utf-8')
