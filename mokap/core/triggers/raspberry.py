# In hardware.py

import os
import time
from dotenv import load_dotenv
from typing import NoReturn, Optional

import warnings
from cryptography.utils import CryptographyDeprecationWarning

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
    import paramiko


class RaspberryTrigger:
    """
    Manages a hardware trigger signal from a Raspberry Pi using pigpio

    This class establishes a persistent SSH connection to send commands that start and
    stop a Pulse Width Modulation signal on a specified GPIO pin

    It is designed to be used as a context manager to ensure that the SSH
    connection is always closed properly

    Requires the following environment variables to be set in a .env file or
    in the system environment:
        - TRIGGER_HOST: The IP address or hostname of the Raspberry Pi
        - TRIGGER_USER: The username for the SSH connection
        - TRIGGER_PASS: The password for the SSH connection
        - GPIO_PIN: The BCM pin number to use for the PWM signal (e.g., 18)

        # TODO: GPIO pin has no business being in the env file, it's not a secret
    """

    def __init__(self, silent: bool = False):
        self._silent = silent
        self.client: Optional[paramiko.SSHClient] = None
        self._connected: bool = False

        # Load configuration from .env file
        load_dotenv()
        self.host = os.getenv('TRIGGER_HOST')
        self.user = os.getenv('TRIGGER_USER')
        self.password = os.getenv('TRIGGER_PASS')
        self.gpio_pin = os.getenv('GPIO_PIN')

        self._connect()

    def _connect(self):
        """ Establishes the SSH connection to the Raspberry Pi """
        required_vars = {
            "TRIGGER_HOST": self.host,
            "TRIGGER_USER": self.user,
            "TRIGGER_PASS": self.password,
            "GPIO_PIN": self.gpio_pin
        }
        missing_vars = [name for name, val in required_vars.items() if val is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

        try:
            if not self._silent:
                print(f"[INFO] Connecting to trigger at {self.host}...")

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                hostname=self.host,
                username=self.user,
                password=self.password,
                timeout=5,
                look_for_keys=False  # Important for password-based auth
            )
            self._connected = True
            if not self._silent:
                print("[INFO] Trigger connected successfully.")

        except Exception as e:
            print(f"[ERROR] Trigger connection failed: {e}")
            self.client = None
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self, frequency: float, duty_cycle_percent: int = 50):
        """
        Starts the PWM signal on the configured GPIO pin

        Args:
            frequency (float): The frequency of the signal in Hz
            duty_cycle_percent (int): The duty cycle (0-100) 50% is standard
        """
        if not self.connected:
            print("[ERROR] Cannot start trigger: not connected.")
            return

        # pigpiod's 'pigs hp' command uses a duty cycle value from 0 to 1,000,000 (for parts per million)
        duty_cycle_value = int(duty_cycle_percent * 10000)
        command = f'pigs hp {self.gpio_pin} {int(frequency)} {duty_cycle_value}'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                print(f"[ERROR] Trigger start command failed: {err}")
            elif not self._silent:
                print(f"[INFO] Trigger started at {frequency} Hz with {duty_cycle_percent}% duty cycle.")

        except Exception as e:
            print(f"[ERROR] Failed to send 'start' command: {e}")
            self.disconnect()  # Assume connection is dead

    def stop(self):
        """ Stops the PWM signal and sets the pin to a low state """
        if not self.connected:
            # No need to print an error if already disconnected
            return
        # 'pigs hp {pin} 0 0' turns off the hardware PWM
        # 'pigs w {pin} 0' ensures the pin is left in a low state
        command = f'pigs hp {self.gpio_pin} 0 0 && pigs w {self.gpio_pin} 0'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                print(f"[ERROR] Trigger stop command failed: {err}")
            elif not self._silent:
                print("[INFO] Trigger stopped.")

        except Exception as e:
            print(f"[ERROR] Failed to send 'stop' command: {e}")
        finally:
            # we still want to disconnect cleanly
            self.disconnect()

    def disconnect(self):
        """ Closes the SSH connection if it is open """
        if self.client:
            self.client.close()
            self.client = None
            self._connected = False
            if not self._silent:
                print("[INFO] Trigger disconnected.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == '__main__':
    # This just a debug mini script
    # .env file with the required variables is needed

    print("--- Testing RaspberryTrigger ---")

    try:
        with RaspberryTrigger(silent=False) as trigger:
            if trigger.connected:
                print("\nStarting trigger for 5 seconds...")
                trigger.start(frequency=100)
                time.sleep(5)
                print("Stopping trigger...")

        print("\nTest complete. Trigger should be stopped and disconnected.")

    except EnvironmentError as e:
        print(f"\nConfiguration Error: Please check your .env file. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")