#Libraries
# import RPi.GPIO as GPIO
import time
import digitalio as dg
import board
#GPIO Mode (BOARD / BCM)
# GPIO.setmode(GPIO.BCM)



TRIG = dg.DigitalInOut(board.D11)
TRIG.direction = dg.Direction.OUTPUT

ECHO = dg.DigitalInOut(board.D8)
ECHO.direction = dg.Direction.INPUT

TRIG.value = False
time.sleep(1.5)
# #set GPIO direction (IN / OUT)
# GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
# GPIO.setup(GPIO_ECHO, GPIO.IN)
# GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
# time.sleep(2)

def distance():
    # set Trigger to HIGH
    # GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
    TRIG.value = True

    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    # GPIO.output(GPIO_TRIGGER, GPIO.LOW)
    TRIG.value = False

    StartTime = time.time()
    StopTime = time.time()
    print("Starting Time")
    # save StartTime
    while ECHO.value == 0:
        StartTime = time.time()
        print("infinite loop?")

    # save time of arrival
    print(f"echo output: {ECHO.value}")
    while ECHO.value == 1:
        StopTime = time.time()
        # print("Stop time: ",StopTime)
        # print("loop 2?")
    print("Stoping Time")
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2

    return distance

if __name__ == '__main__':
    try:
        while True:
            dist = distance()
            print ("Measured Distance = %.1f cm" % dist)
            time.sleep(.1)

        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        # GPIO.cleanup()
