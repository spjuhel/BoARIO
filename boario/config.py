from configparser import ConfigParser
import os
from pathlib import Path


class Config:
    def __init__(self, user_conf_path=None):
        # Define paths for default and user configuration
        self.default_conf_path = (
            Path(__file__).parent / "boario-tools.conf"
        )  # Package folder
        self.user_conf_path = Path(user_conf_path or Path.home() / ".boario-tools.conf")

        # Initialize the ConfigParser and load configurations
        self.settings = self._load_config()

    def _load_config(self):
        config = ConfigParser(os.environ)

        config.read(self.default_conf_path)

        if self.user_conf_path.exists():
            config.read(self.user_conf_path)
        return config

    def get(self, section, option, fallback=None):
        return self.settings.get(section, option, fallback=fallback)


# Then pass this config to custom_utils when using its functionality
config = Config()
