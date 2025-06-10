import os
import time
from typing import Optional, Dict
from mokap.core.triggers.interface import AbstractTrigger
import paramiko


class RaspberryTrigger(AbstractTrigger):
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
    """

    def __init__(self, config: Optional[Dict] = None, silent: bool = False):
        super().__init__(config=config, silent=silent)
        self.client: Optional[paramiko.SSHClient] = None
        self._connected: bool = False

        # Load configuration from .env file
        self.host = os.getenv('TRIGGER_HOST')
        self.user = os.getenv('TRIGGER_USER')
        self.password = os.getenv('TRIGGER_PASS')

        if self._config.get('kind', '') == 'raspberry':
            self.gpio_pin = self._config.get('gpio_pin', 18)
        else:
            raise EnvironmentError(f"Missing required config (did you define the Raspberry Pi trigger in the config file?")

        self._connect()

    def _connect(self):
        """ Establishes the SSH connection to the Raspberry Pi """
        required_vars = {
            "TRIGGER_HOST": self.host,
            "TRIGGER_USER": self.user,
            "TRIGGER_PASS": self.password,
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


if __name__ == '__main__':
    # This just a debug mini script
    # .env file with the required variables is needed

    secs = 5
    freq = 10

    print("--- Testing RaspberryTrigger ---")

    try:
        with RaspberryTrigger(silent=False) as trigger:
            if trigger.connected:
                print(f"Starting trigger for {secs} seconds...")
                trigger.start(frequency=freq)
                time.sleep(secs)
                print("Stopping trigger...")

        print("\nTest complete. Trigger should be stopped and disconnected.")

    except EnvironmentError as e:
        print(f"\nConfiguration Error: Please check your .env file. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")