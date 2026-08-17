import logging
import os
import time
from typing import Optional, Dict
from mokap.core.triggers.interface import AbstractTrigger
import paramiko

logger = logging.getLogger(__name__)


class RaspberryTrigger(AbstractTrigger):
    """
    Manages a hardware trigger signal from a Raspberry Pi.
    - Pre-Pi 5: Uses `pigpio` (https://abyz.me.uk/rpi/pigpio/) for precise DMA hardware PWM.
    - Pi 5 and later: Uses native sysfs hardware PWM.

    Pi 5 prerequisites:
        Enable hardware PWM by adding the overlay to `/boot/firmware/config.txt`:
           - If using GPIO 18 & 19: Add `dtoverlay=pwm-2chan`
           - If using GPIO 12 & 13: Add `dtoverlay=pwm`
        The SSH user needs passwordless sudo (NOPASSWD) for the sysfs PWM commands to work
        non-interactively - `start`/`stop` will otherwise fail with a permission error.

    Requires the following environment variables to be set in a .env file or system env:
        - TRIGGER_HOST: The IP address or hostname of the Raspberry Pi
        - TRIGGER_USER: The username for the SSH connection
        - TRIGGER_PASS: The password for the SSH connection
    """

    PWM_CHANNEL_MAP = {
        12: 0, 13: 1, 14: 2, 18: 2, 15: 3, 19: 3
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)
        self.client: Optional[paramiko.SSHClient] = None
        self._connected: bool = False
        self._supports_sysfs: bool = False
        self.pwm_channel: Optional[int] = None

        # Load configuration from .env file
        self.host = os.getenv('TRIGGER_HOST')
        self.user = os.getenv('TRIGGER_USER')
        self.password = os.getenv('TRIGGER_PASS')

        if self._config.get('type', '') == 'raspberry':
            self.gpio_pin = self._config.get('gpio_pin', 18)
        else:
            raise EnvironmentError('Missing required config (did you define the Raspberry Pi trigger in the config file?)')

        logger.debug(f'Raspberry trigger at {self.user}@{self.host}, using GPIO pin {self.gpio_pin}.')

        self._connect()

    def _connect(self):
        """Establishes the SSH connection to the Raspberry Pi (and detects the model)."""
        required_vars = {
            "TRIGGER_HOST": self.host,
            "TRIGGER_USER": self.user,
            "TRIGGER_PASS": self.password,
        }
        missing_vars = [name for name, val in required_vars.items() if val is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

        try:
            logger.debug(f"Connecting to Raspberry Trigger at {self.host}...")

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                hostname=self.host,
                username=self.user,
                password=self.password,
                timeout=5,
                look_for_keys=False  # Important for password-based auth
            )

            # Detect Pi Model
            stdin, stdout, stderr = self.client.exec_command('cat /proc/device-tree/model')
            model_info = stdout.read().decode().strip()
            self._supports_sysfs = 'Raspberry Pi 5' in model_info

            if self._supports_sysfs:
                logger.info(f'Detected {model_info}. Configuring for Sysfs Hardware PWM.')
                if self.gpio_pin not in self.PWM_CHANNEL_MAP:
                    raise ValueError(f'GPIO {self.gpio_pin} does not support hardware PWM on Pi 5. '
                                     f'Supported pins: {list(self.PWM_CHANNEL_MAP.keys())}')
                self.pwm_channel = self.PWM_CHANNEL_MAP[self.gpio_pin]
            else:
                logger.info(f"Detected {model_info}. Configuring for legacy 'pigpio'.")

            self._connected = True

            logger.info("Trigger connected successfully.")

        except Exception as e:
            logger.error(f"Trigger connection failed: {e}")
            self.client = None
            self._connected = False

    def start(self, frequency: float, duty_cycle_percent: int = 50):
        """
        Starts the PWM signal on the configured GPIO pin.

        Args:
            frequency (float): The frequency of the signal in Hz
            duty_cycle_percent (int): The duty cycle (0-100) 50% is standard
        """
        if not self.connected:
            logger.error('Cannot start trigger: not connected.')
            return

        if self._supports_sysfs:
            self._start_sysfs(frequency, duty_cycle_percent)
        else:
            self._start_pigpio(frequency, duty_cycle_percent)

    def stop(self):
        """Stops the PWM signal and sets the pin to a low state."""

        if not self.connected:
            return

        if self._supports_sysfs:
            self._stop_sysfs()
        else:
            self._stop_pigpio()

        self.disconnect()

    def _start_pigpio(self, frequency: float, duty_cycle_percent: int):
        """Hardware PWM implementation using pigpio (for Pi 4 and below)."""

        duty_cycle_value = int(duty_cycle_percent * 10000)
        command = f'pigs hp {self.gpio_pin} {int(frequency)} {duty_cycle_value}'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                logger.error(f'pigpio trigger start command failed: {err}')
            else:
                logger.info(f'Trigger started via pigpio at {frequency} Hz with {duty_cycle_percent}% duty cycle.')
        except Exception as e:
            logger.error(f"Failed to send pigpio 'start' command: {e}")
            self.disconnect()

    def _stop_pigpio(self):
        """Hardware PWM implementation using pigpio (for Pi 4 and below)."""

        command = f'pigs hp {self.gpio_pin} 0 0 && pigs w {self.gpio_pin} 0'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                logger.error(f'pigpio trigger stop command failed: {err}')
            else:
                logger.info('Trigger stopped via pigpio.')
        except Exception as e:
            logger.error(f"Failed to send pigpio 'stop' command: {e}")

    def _start_sysfs(self, frequency: float, duty_cycle_percent: int):
        """Sysfs hardware PWM implementation (for Pi 5 and later)."""

        period_ns = int(1e9 / frequency)
        duty_ns = int(period_ns * (duty_cycle_percent / 100.0))

        command = f"""
        sudo sh -c '
        for chip in /sys/class/pwm/pwmchip*; do
            [ ! -d $chip/pwm{self.pwm_channel} ] && echo {self.pwm_channel} > $chip/export 2>/dev/null
            if [ -d $chip/pwm{self.pwm_channel} ]; then
                echo 0 > $chip/pwm{self.pwm_channel}/enable 2>/dev/null
                echo 0 > $chip/pwm{self.pwm_channel}/duty_cycle 2>/dev/null
                echo {period_ns} > $chip/pwm{self.pwm_channel}/period 2>/dev/null
                echo {duty_ns} > $chip/pwm{self.pwm_channel}/duty_cycle 2>/dev/null
                echo 1 > $chip/pwm{self.pwm_channel}/enable 2>/dev/null
                echo "SUCCESS"
                break
            fi
        done
        '
        """
        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()

            if 'SUCCESS' not in out:
                logger.error(f'Sysfs trigger start failed. Did you add dtoverlay to config.txt? Err: {err}')
            else:
                logger.info(f'Trigger started via Sysfs at {frequency} Hz with {duty_cycle_percent}% duty cycle.')

        except Exception as e:
            logger.error(f"Failed to send Sysfs 'start' command: {e}")
            self.disconnect()

    def _stop_sysfs(self):
        """Sysfs hardware PWM implementation (for Pi 5 and later)."""

        command = f"""
        sudo sh -c '
        for chip in /sys/class/pwm/pwmchip*; do
            if [ -d $chip/pwm{self.pwm_channel} ]; then
                echo 0 > $chip/pwm{self.pwm_channel}/duty_cycle 2>/dev/null
                echo 0 > $chip/pwm{self.pwm_channel}/enable 2>/dev/null
                echo "SUCCESS"
                break
            fi
        done
        '
        """
        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()

            if 'SUCCESS' not in out:
                logger.error(f'Sysfs trigger stop command failed: {err}')
            else:
                logger.info('Trigger stopped via Sysfs.')
        except Exception as e:
            logger.error(f"Failed to send Sysfs 'stop' command: {e}")

    def disconnect(self):
        """Closes the SSH connection if it is open."""
        if self.client:
            self.client.close()
            self.client = None
            self._connected = False

            logger.info("Trigger disconnected.")


if __name__ == '__main__':
    # This just a debug mini script.
    # `.env` file with the required variables is needed.
    # `config.yaml` file in the project root is needed too.

    secs = 5
    freq = 10

    print("[ Testing RaspberryTrigger ]")

    try:
        with RaspberryTrigger() as trigger:
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