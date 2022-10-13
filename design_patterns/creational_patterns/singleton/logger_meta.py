import datetime
from design_patterns.creational_patterns.singleton.singleton_meta import Singleton


class Logger(metaclass=Singleton):  # specifying Singleton as metaclass instead of inheriting from object
    log_file = None

    def open_log(self, path):
        if self.log_file is None:
            self.log_file = open(path, mode='w')

    def write_log(self, log_record):
        now = str(datetime.datetime.now())
        record = f'{now}: {log_record}\n'
        self.log_file.write(record)

    def close_log(self):
        self.log_file.close()
        self.log_file = None
