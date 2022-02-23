# download link for library dependencies 
# https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi

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
# I2C control of magnetic encoder
import board
import busio
import time

i2c = (busio.I2C(board.SCL_1, board.SDA_1, 400))
data_buffer_in = int(0, 'utf-8')
# set up I2C Bus connection == wire.begin
i2c_address = bytes(0x36, 'utf-8')

# returns a scan of the i2c pins and prints the board ID
print(i2c.scan())
check_magnent()
# i2c.writeto_then_readfrom(i2c_address, out_buffer: circuitpython_typing.ReadableBuffer,
# in_buffer: circuitpython_typing.WriteableBuffer, *, out_start: int = 0, out_end: int = sys.maxsize, in_start: int = 0, in_end: int = sys.maxsize)

# run to de initialize the board
i2c.deinit()

def check_magnent():
    # collects raw angle from
    print("Checking Magnent Status")
    while (data_buffer_in != 32):
        data_buffer_in = int(0, 'utf-8')
        mstat_address = bytes(0x0B, 'utf-8')
        data_buffer_in = bytes(None, 'utf-8')
        i2c.writeto_then_readfrom(i2c_address, mstat_address: circuitpython_typing.ReadableBuffer,
        data_buffer_in: circuitpython_typing.WriteableBuffer*, out_start: int = 0, out_end: int = 6)
        print("Magnent Status", data_buffer_in)
    print("Magnent Detected")

def check_magnent():
    mstat_address = bytes(0x36, 'utf-8')
