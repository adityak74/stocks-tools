"""Config module."""

from dotenv import dotenv_values


class Config:
    """Config singleton class."""

    def __init__(self):
        self.config = None

    def get_config(self):
        """Get config from env."""
        if self.config is None:
            self.config = dotenv_values()
        return self.config
