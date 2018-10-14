import threading
import time

exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("Thread Start: %s" % self.name)
        threadLock.acquire()
        print_time(self.name, self.counter, 5)
        threadLock.release()
        print("Thread Exit: %s" % self.name)

threadLock = threading.Lock()
threads = []

def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1


t1 = myThread(1, "Thread（1）", 1)
t2 = myThread(2, "Thread（2）", 2)

threads.append(t1)
threads.append(t2)

t1.start()
t2.start()
t1.join()
t2.join()
print("Exit main Thread")
