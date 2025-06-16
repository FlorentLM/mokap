import logging
import time
from typing import Optional, Dict
import serial
from mokap.core.triggers.interface import AbstractTrigger

logger = logging.getLogger(__name__)


class ArduinoTrigger(AbstractTrigger):
    """
    Manages a hardware trigger signal from an Arduino-compatible board
    over a serial (USB) connection

    It sends simple commands ("START freq duty", "STOP") to the device
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config=config)

        self.ser: serial.Serial | None = None

        if self._config.get('kind', '') == 'arduino':
            self.port = self._config.get('port')
            self.baud_rate = self._config.get('baudrate', 115200)
            self.gpio_pin = self._config.get('gpio_pin', 11)
        else:
            raise EnvironmentError(f"Missing required config (did you define the Arduino trigger in the config file?")

        logger.debug(f'Arduino trigger using GPIO pin {self.gpio_pin} on port {self.port} with baud rate {self.baud_rate}.')

        self._connect()

    def _connect(self) -> None:
        """ Establishes the serial connection to the Arduino """

        try:
            logger.debug(f'Connecting to Arduino trigger at {self.port}...')

            self.ser = serial.Serial(self.port, self.baud_rate, timeout=2)
            time.sleep(2)  # wait for arduino to reset and initialize

            # simple handshake to confirm connection
            self.ser.write(b"PING\n")
            response = self.ser.readline().decode().strip()
            if response != 'PONG':
                raise serial.SerialException(f'Arduino handshake failed. Expected "PONG", got "{response}"')

            # Identify which firmware is on there
            self.ser.write(b"ID?\n")
            id_response = self.ser.readline().decode().strip()
            if 'PWM_TRIGGER' in id_response:
                self.firmware_type = id_response

                logger.debug(f'Detected firmware: "{self.firmware_type}"')

            else:
                self.firmware_type = 'UNKNOWN'
                logger.warning(f'Could not identify firmware. Got "{id_response}". Functionality may be limited.')

            self._connected = True
            logger.info('Trigger connected successfully.')

        except serial.SerialException as e:
            logger.error(f'Arduino trigger connection failed: {e}')

            self.ser = None
            self._connected = False

    def _send_command(self, command: str) -> bool:
        """ Sends a command and waits for an "OK" response """
        if not self.connected:
            return False

        try:
            self.ser.write(f'{command}\n'.encode())
            response = self.ser.readline().decode().strip()
            if response == 'OK':
                return True
            else:
                logger.error(f'Arduino command "{command}" failed. Response: "{response}"')
                return False

        except serial.SerialException as e:
            logger.error(f'Lost connection to Arduino: {e}')
            self.disconnect()
            return False

    def start(self, frequency: float, duty_cycle_percent: int = 50) -> None:
        if not self.connected:
            logger.error('Cannot start trigger: not connected.')
            return

        if frequency < 31 and 'TONE' in self.firmware_type:
            logger.error(f'The installed "{self.firmware_type}" firmware does not support frequencies below 31 Hz.\n'
                         f'        You requested {frequency} Hz. You should use flash the `trigger_millis_v1.ino` file.')
            return

        command = f"START {self.gpio_pin} {frequency} {duty_cycle_percent}"

        if self._send_command(command):
            logger.info(f"Trigger started at {frequency} Hz with 50% duty cycle.")

    def stop(self) -> None:
        if not self.connected:
            return

        command = f'STOP {self.gpio_pin}'
        if self._send_command(command):
            logger.info('Trigger stopped.')

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            self._connected = False

            logger.info('Trigger disconnected.')

if __name__ == '__main__':
    # This just a debug mini script
    # To run this, you need a config.yaml file in the project root
    # (or to pass the config dictionary directly)

    secs = 5
    freq = 31

    print("--- Testing ArduinoTrigger ---")

    try:
        with ArduinoTrigger() as trigger:
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