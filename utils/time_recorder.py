import time
import numpy as np
"""
This is a class for keeping time and possible recording it to a file
"""


class time_keeper(object):
    def __init__(self, time_keeping_file="time_keeper.txt", max_running_time=9999):
        self.start = time.time()
        self.max_running_time = max_running_time * 60 * 60
        self.time_keeping_file = time_keeping_file
        self.end = -1
        self.duration = -1

    def record(self, write_number):
        """
        Record the time to the time_keeping_file, the time marked is the interval between current time and the start time
        :param write_number:
        :return:
        """
        with open(self.time_keeping_file, "a") as f:
            self.end = time.time()
            self.duration = self.end - self.start
            f.write('{},{}\n'.format(write_number, self.duration))
            if (self.duration > self.max_running_time):
                raise ValueError('Your program has run over the maximum time limit set by Ben in time_keeper function')
