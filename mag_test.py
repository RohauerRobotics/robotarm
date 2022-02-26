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
        i2c.writeto_then_readfrom(i2c_address, mstat_address: circuitpython_typing.ReadableBuffer,
        data_buffer_in: circuitpython_typing.WriteableBuffer*, out_start: int = 0, out_end: int = 6)
        print("Magnent Status", data_buffer_in)
    print("Magnent Detected")

def read_angle():
    first_four_bits = int(None, 'utf-8')
    raw1_address = bytes(0x0D, 'utf-8')
    i2c.writeto_then_readfrom(i2c_address, raw1_address: circuitpython_typing.ReadableBuffer,
    first_four_bits: circuitpython_typing.WriteableBuffer*, out_start: int = 0, out_end: int = 6, in_start: int = 0, in_stop: int=3)
    last_four_bits = int(None, 'utf-8')
    raw2_address = bytes(0x0C, 'utf-8')
    i2c.writeto_then_readfrom(i2c_address, raw2_address: circuitpython_typing.ReadableBuffer,
    first_four_bits: circuitpython_typing.WriteableBuffer*, out_start: int = 0, out_end: int = 6, in_start: int = 4, in_stop: int=7)
