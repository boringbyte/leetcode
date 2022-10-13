import datetime


class Logger(object):
    log_file = None

    @staticmethod
    def instance():
        if '_instance' not in Logger.__dict__:
            Logger._instance = Logger()
        return Logger._instance

    def open_log(self, path):
        self.log_file = open(path, mode='w')

    def write_log(self, log_record):
        now = str(datetime.datetime.now())
        record = f'{now}: {log_record}\n'
        self.log_file.write(record)

    def close_log(self):
        self.log_file.close()


if __name__ == '__main__':
    """
    1. Singleton violates single responsibility principle
    2. Non-standard class access
    3. Harder to test
    4. Carry global state
    5. Hard to sub class
    6. Singletons considered harmful
    """
    logger = Logger.instance()
    logger.open_log('my.log')
    logger.write_log('Logging with classic Singleton pattern')
    logger.close_log()

    with open('my.log', 'r') as fp:
        for line in fp:
            print(line, end='')
