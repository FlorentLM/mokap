import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
from dotenv import load_dotenv
from mokap.utils.fileio import read_config
from pathlib import Path

logger = logging.getLogger(__name__)


class AbstractTrigger(ABC):
    """
    Abstract Base Class for a hardware trigger
    Defines a common interface for different hardware trigger implementations
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config if config else read_config(Path(__file__).parents[3] / 'config.yaml').get('trigger', {})
        self._connected: bool = False
        load_dotenv()

    @abstractmethod
    def _connect(self) -> None:
        """ Establishes the connection to the hardware device """
        pass

    @property
    def connected(self) -> bool:
        """ Returns the connection status """
        return self._connected

    @abstractmethod
    def start(self, frequency: float, duty_cycle_percent: int = 50) -> None:
        """
        Starts the trigger signal

        Args:
            frequency (float): The frequency of the signal in Hz
            duty_cycle_percent (int): The duty cycle (0-100), 50% is standard
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """ Stops the trigger signal """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """ Closes the connection to the hardware device """
        pass

    def __enter__(self):
        """ Context manager entry point """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Context manager exit point. Ensures stop() and disconnect() are called """
        if self.connected:
            logger.debug("Exiting context: stopping and disconnecting trigger.")
            self.stop()
            self.disconnect()