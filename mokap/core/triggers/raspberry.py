import logging
import os
import time
import subprocess
import textwrap
from typing import Optional, Dict
from mokap.core.triggers.interface import AbstractTrigger
import paramiko

logger = logging.getLogger(__name__)


class RaspberryTrigger(AbstractTrigger):
    """
    Manages a hardware trigger signal from a Raspberry Pi.
    
    Automatically detects Pi version and uses:
    - gpiozero for Pi 5
    - pigpio for Pi 4
    
    Uses SSH to execute trigger commands remotely.

    Requires the following environment variables to be set in a .env file or
    in the system environment:
        - TRIGGER_HOST: The IP address or hostname of the Raspberry Pi
        - TRIGGER_USER: The username for the SSH connection
        - TRIGGER_PASS: The password for the SSH connection
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)
        self.client: Optional[paramiko.SSHClient] = None
        self._connected: bool = False
        self.pi_version: Optional[int] = None
        self._pwm_process_id: Optional[int] = None  # Track background PWM process

        # Load configuration from .env file
        self.host = os.getenv('TRIGGER_HOST')
        self.user = os.getenv('TRIGGER_USER')
        self.password = os.getenv('TRIGGER_PASS')

        if self._config.get('type', '') == 'raspberry':
            self.gpio_pin = self._config.get('pin', 18)
        else:
            raise EnvironmentError(f"Missing required config (did you define the Raspberry Pi trigger in the config file?")

        logger.debug(f'Raspberry trigger at {self.user}@{self.host}, using GPIO pin {self.gpio_pin}.')

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
            self._connected = True

            logger.info("Trigger connected successfully.")
            
            # Detect Pi version after successful connection
            self._detect_pi_version_remote()

        except Exception as e:
            logger.error(f"Trigger connection failed: {e}")
            self.client = None
            self._connected = False

    def _detect_pi_version_remote(self):
        """Detect Pi version via SSH"""
        try:
            stdin, stdout, stderr = self.client.exec_command("cat /proc/device-tree/model")
            model = stdout.read().decode().strip().lower()
            
            if 'raspberry pi 5' in model or 'pi 5' in model:
                self.pi_version = 5
                logger.info("Detected Raspberry Pi 5 - using gpiozero")
                self._check_pi5_gpio_permissions()
            elif 'raspberry pi 4' in model or 'pi 4' in model:
                self.pi_version = 4
                logger.info("Detected Raspberry Pi 4 - using pigpio")
            else:
                logger.warning(f"Unknown Pi model: {model}. Trying gpiozero first (Pi 5), fallback to pigpio (Pi 4)")
                self.pi_version = None
        except Exception as e:
            logger.warning(f"Failed to detect Pi version: {e}. Will auto-detect at runtime.")
            self.pi_version = None

    def _check_pi5_gpio_permissions(self):
        """Check if user has GPIO access on Pi 5"""
        try:
            # Check if user is in gpio group
            stdin, stdout, stderr = self.client.exec_command("id | grep -q '(gpio)' && echo 'YES' || echo 'NO'")
            has_gpio = stdout.read().decode().strip() == 'YES'
            
            if has_gpio:
                logger.info("GPIO access confirmed (user in gpio group)")
            else:
                logger.warning("User is NOT in gpio group. PWM may fail. On Pi 5, run: sudo usermod -a -G gpio $USER")
            
            # Also check if gpiozero/libgpiod is installed
            stdin2, stdout2, stderr2 = self.client.exec_command("python3 -c 'import gpiozero; print(\"OK\")' 2>&1")
            import_result = stdout2.read().decode().strip()
            if import_result != 'OK':
                logger.warning(f"gpiozero may not be installed: {import_result}")
            else:
                logger.info("gpiozero is installed")
                
        except Exception as e:
            logger.debug(f"Could not check GPIO permissions: {e}")

    def start(self, frequency: float, duty_cycle_percent: int = 6):
        """
        Starts the PWM signal on the configured GPIO pin

        Args:
            frequency (float): The frequency of the signal in Hz
            duty_cycle_percent (int): The duty cycle (0-100) 50% is standard
        """
        if not self.connected:
            logger.error("Cannot start trigger: not connected.")
            return

        frequency = int(round(frequency))

        if self.pi_version == 5:
            self._restart_gpiozero(frequency, duty_cycle_percent)
        elif self.pi_version == 4:
            self._start_pigpio(frequency, duty_cycle_percent)
        else:
            # Unknown version: try gpiozero first (Pi 5), fallback to pigpio (Pi 4)
            self._restart_gpiozero(frequency, duty_cycle_percent)

    def _start_pigpio(self, frequency: float, duty_cycle_percent: int):
        """Start PWM using pigpio (Pi 4)"""
        # pigpiod's 'pigs hp' command uses a duty cycle value from 0 to 1,000,000 (for parts per million)
        duty_cycle_value = int(duty_cycle_percent * 10000)
        command = f'pigs hp {self.gpio_pin} {int(frequency)} {duty_cycle_value}'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                logger.error(f"Trigger start command failed: {err}")

            logger.info(f"Trigger started at {frequency} Hz with {duty_cycle_percent}% duty cycle (pigpio).")

        except Exception as e:
            logger.error(f"Failed to send 'start' command: {e}")

    def _start_gpiozero(self, frequency: float, duty_cycle_percent: int):
        """Start PWM using gpiozero (Pi 5) - keeps PWM running in background"""
        duty_cycle = duty_cycle_percent / 100.0  # gpiozero uses 0-1
        
        # Create a persistent Python script that keeps PWM alive
        script = textwrap.dedent(f'''
            import time
            import sys
            from gpiozero import PWMOutputDevice

            sys.stdout.reconfigure(line_buffering=True)
            pwm = PWMOutputDevice({self.gpio_pin}, frequency={frequency})
            pwm.value = {duty_cycle}
            sys.stdout.write("PWM_STARTED\\n")
            sys.stdout.flush()
            try:
                while True:
                    time.sleep(1)  # Keep process alive indefinitely
            except KeyboardInterrupt:
                pwm.off()
        ''').strip()

        # Run with nohup to keep process alive even if SSH connection closes
        command = f"nohup python3 << 'EOF' > /tmp/mokap_pwm_{self.gpio_pin}.log 2>&1 &\n{script}\nEOF"

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            time.sleep(0.5)  # Give process time to start and write to log
            
            # Check if process started by verifying log file has output
            check_cmd = f"(grep -q 'PWM_STARTED' /tmp/mokap_pwm_{self.gpio_pin}.log 2>/dev/null && echo 'CONFIRMED') || echo 'PENDING'"
            stdin2, stdout2, stderr2 = self.client.exec_command(check_cmd)
            status = stdout2.read().decode().strip()
            
            # Also check if process is actually running
            ps_cmd = f"pgrep -f 'PWMOutputDevice.*{self.gpio_pin}' > /dev/null && echo 'RUNNING' || echo 'NOT_RUNNING'"
            stdin3, stdout3, stderr3 = self.client.exec_command(ps_cmd)
            ps_status = stdout3.read().decode().strip()
            
            if status == 'CONFIRMED' and ps_status == 'RUNNING':
                logger.info(f"Trigger started at {frequency} Hz with {duty_cycle_percent}% duty cycle (gpiozero). PWM process confirmed running.")
            else:
                logger.warning(f"Trigger started at {frequency} Hz with {duty_cycle_percent}% duty cycle (gpiozero). Log status: {status}, Process status: {ps_status}")
                if ps_status == 'NOT_RUNNING':
                    logger.error("PWM process failed to start or exited immediately. Check /tmp/mokap_pwm_18.log on Pi for errors.")

        except Exception as e:
            logger.error(f"Failed to start gpiozero PWM: {e}")

    def _stop_gpiozero_process(self):
        """Stops the gpiozero PWM worker without closing the SSH connection."""
        command = f"pkill -f 'PWMOutputDevice.*{self.gpio_pin}' || true"

        try:
            self.client.exec_command(command)
        except Exception as e:
            logger.debug(f"Failed to stop gpiozero PWM worker cleanly: {e}")

    def _restart_gpiozero(self, frequency: float, duty_cycle_percent: int):
        """Restart gpiozero PWM so runtime frequency updates take effect immediately."""
        self._stop_gpiozero_process()
        time.sleep(0.05)
        self._start_gpiozero(frequency, duty_cycle_percent)



    def stop(self):
        """ Stops the PWM signal and sets the pin to a low state """
        if not self.connected:
            # No need to print an error if already disconnected
            return

        if self.pi_version == 5:
            self._stop_gpiozero()
        elif self.pi_version == 4:
            self._stop_pigpio()
        else:
            # Unknown version: try gpiozero first, fallback to pigpio
            self._stop_gpiozero()

    def _stop_pigpio(self):
        """Stop PWM using pigpio (Pi 4)"""
        # 'pigs hp {pin} 0 0' turns off the hardware PWM
        # 'pigs w {pin} 0' ensures the pin is left in a low state
        command = f'pigs hp {self.gpio_pin} 0 0 && pigs w {self.gpio_pin} 0'

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            err = stderr.read().decode().strip()
            if err:
                logger.error(f"Trigger stop command failed: {err}")

            logger.info("Trigger stopped (pigpio).")

        except Exception as e:
            logger.error(f"Failed to send 'stop' command: {e}")

        finally:
            # we still want to disconnect cleanly
            self.disconnect()

    def _stop_gpiozero(self):
        """Stop PWM using gpiozero (Pi 5) - kills background process"""
        try:
            self._stop_gpiozero_process()
            time.sleep(0.2)
            logger.info(f"Trigger stopped (gpiozero). GPIO {self.gpio_pin} PWM process terminated.")

        except Exception as e:
            logger.error(f"Failed to stop gpiozero PWM: {e}")

        finally:
            self.disconnect()

    def disconnect(self):
        """ Closes the SSH connection if it is open """
        if self.client:
            self.client.close()
            self.client = None
            self._connected = False

            logger.info("Trigger disconnected.")


# if __name__ == '__main__':
#     # This just a debug mini script
#     # .env file with the required variables is needed
#     # you also need a config.yaml file in the project root
#     # (or to pass the config dictionary directly)

#     secs = 5
#     freq = 10

#     print("--- Testing RaspberryTrigger ---")

#     try:
#         with RaspberryTrigger() as trigger:
#             if trigger.connected:
#                 print(f"Starting trigger for {secs} seconds...")
#                 trigger.start(frequency=freq)
#                 time.sleep(secs)
#                 print("Stopping trigger...")

#         print("\nTest complete. Trigger should be stopped and disconnected.")

#     except EnvironmentError as e:
#         print(f"\nConfiguration Error: Please check your .env file. Details: {e}")
#     except Exception as e:
#         print(f"\nAn unexpected error occurred during the test: {e}")