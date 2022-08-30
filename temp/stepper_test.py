# control stepper motor jetson nano
import time
import board
import digitalio as dg
import math

DIR = dg.DigitalInOut(board.D4)
DIR.direction = dg.Direction.OUTPUT

STEP = dg.DigitalInOut(board.D17)
STEP.direction = dg.Direction.OUTPUT

def one_small_step(dir_bool):
    print("That's one small step for a man, one giant leap for mankind.")
    DIR.value = dir_bool
    w = 0.00225
    for i in range(0,2000):
        STEP.value = True
        print(wait(w,i))
        time.sleep(wait(w,i))
        STEP.value = False
        time.sleep(wait(w,i))

def wait(w,i):
    q = (w-(0.0000000025*i**2))
    if (q > 0.00075):
        send = q
    elif(0.00075> q):
        send = 0.00075
    else:
        pass
    return(send)

one_small_step(True)
