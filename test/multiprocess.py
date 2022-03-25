# multi-process test

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception

def mag_verification():
    print("checked angles")
    time.sleep(2)

def kinematics_calculation():
    print("Calculating Kinematics")
    time.sleep(3)

def image_processing():
    print("Procesing Images")
    time.sleep(4)

def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will
                raise queue.Empty exception if the queue is empty.
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion
                message to task_that_are_done queue
            '''
            if task == 1:
                mag_verification()
                message = "mag_verification"
            elif task == 2:
                kinematics_calculation()
                message = "kinematics_calculation"
            elif task == 3:
                image_processing()
                message = "image_processing"
            else:
                message = "nothing"

            # print(task)
            tasks_that_are_done.put(message + ' is done by ' + current_process().name)
            time.sleep(.5)
    return True



def main():
    number_of_task = 3
    number_of_processes = 3
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for i in range(1, number_of_task+1):
        tasks_to_accomplish.put(i)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    print("Other Program here")
    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()
