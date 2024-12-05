import time


class Timer:
    def __init__(self, name=''):
        self.start_time = 0
        self.end_time = 0
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        print(f"Elapsed time {self.name}: {self.end_time - self.start_time:.2f} seconds")