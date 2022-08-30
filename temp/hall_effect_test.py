# hall effect sensor test
import time
import board
import digitalio as dg

hall_1 = dg.DigitalInOut(board.D26)
hall_1.direction = dg.Direction.INPUT

def check_sensor(delay_in_seconds):
    start = time.time()
    end = time.time()
    while (end-start) < delay_in_seconds:
        end = time.time()
        print(hall_1.value)
        time.sleep(0.1)

check_sensor(20)
