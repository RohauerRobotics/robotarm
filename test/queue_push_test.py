# push + log queue data
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception


def control(control_push, control_log):
    count = 0
    while True:
        print("running control")
        time.sleep(0.1)
        if control_log.empty():
            if count < 10:
                move_to = [1,2,3,4,5]
                control_push.put(move_to)
                time.sleep(0.1)
                print("Pushed Angles")
                count += 1
            else:
                pass
        elif not control_log.empty():
            print(control_log.get())
        else:
            pass

def verify(control_push, control_log):
    while True:
        print("running verify")
        time.sleep(0.1)
        if not control_push.empty():
            print("Verified Coordinates:", control_push.get())
        elif control_push.empty():
            time.sleep(0.1)
            control_log.put("Verified!")
        else:
            pass


def main():
    control_push = Queue()
    control_log = Queue()
    verificaton_push = Queue()
    verification_log = Queue()
    control_process = Process(target=control, args=(control_push, control_log))
    control_process.start()
    verify_process = Process(target=verify, args=(control_push, control_log))
    verify_process.start()
    processes = [control_process, verify_process]
    # activate processes
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
