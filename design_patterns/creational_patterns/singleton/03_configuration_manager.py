import json
import threading


class ConfigManager:

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
            self.config = {}
            self._initialized = True

    def load_from_file(self, file_path):
        """Load configuration from JSON file"""
        try:
            with open(file_path, "r") as fp:
                self.config = json.load(fp)
                print(f"Configuration loaded from {file_path}")
        except FileNotFoundError:
            print(f"Config file {file_path} not found. Using defaults")
            self.config = self._get_default_config()

    @staticmethod
    def _get_default_config():
        return {
            "app_name": "MyApp",
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "port": 5462,
                "name": "mydb"
            },
            "api": {
                "timeout": 30,
                "retries": 3
            }
        }

    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_all(self):
        """Get entire configuration"""
        return self.config


if __name__ == '__main__':
    # app.py
    config = ConfigManager()
    config.load_from_file("config.json")

    # database.py
    config = ConfigManager()
    db_host = config.get("database.host")
    db_port = config.get("database.port")
    print(f"Connecting to {db_host}:{db_port}")

    # api_client.py
    config = ConfigManager()
    timeout = config.get("api.timeout")
    retries = config.get("api.retries")
    print(f"API timeout: {timeout}s, retries: {retries}")

    # admin.py
    config = ConfigManager()  # Same instance
    config.set("api.timeout", 60)  # Update globally
    print(config.get("api.timeout"))  # 60
