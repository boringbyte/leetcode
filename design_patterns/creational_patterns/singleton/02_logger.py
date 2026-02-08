import threading
from datetime import datetime


class Logger:

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.log_file = "app.log"
            self.logs = []
            self._initialized = True

    def _write_to_file(self, message):
        with open(self.log_file, "a") as fp:
            fp.write(message + "\n")

    def info(self, message):
        log_entry = f"[INFO] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)

    def error(self, message):
        log_entry = f"[ERROR] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)

    def warn(self, message):
        log_entry = f"[WARN] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)

    def get_all_logs(self):
        return self.logs


if __name__ == '__main__':
    # module1.py
    logger = Logger()
    logger.info("Application started")

    # module2.py
    logger = Logger()
    logger.error("An error occurred")

    # module3.py
    logger = Logger()  # Same instance
    logger.warn("Low memory warning")

    print(f"Total logs: {len(logger.get_all_logs())}")
