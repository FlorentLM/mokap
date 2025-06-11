import time
import threading
from typing import Optional, Dict
import serial
from mokap.core.triggers.interface import AbstractTrigger


class FTDITrigger(AbstractTrigger):
    """
    Manages a hardware trigger signal from a USB-to-TTL Serial adapter

    This implementation generates a PWM signal in software by toggling a serial port
    control line (RTS or DTR) in a separate thread

    This is subject to the OS's accuracy and may not be suitable for
    very high-frequency or high-precision timing applications
    """

    def __init__(self, config: Optional[Dict] = None, silent: bool = False):
        super().__init__(config=config, silent=silent)

        self.ser: Optional[serial.Serial] = None
        self._trigger_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

        if self._config.get('kind', '') == 'ftdi':
            self.port: str = self._config.get('port')
            self.pin: str = self._config.get('pin', 'RTS').upper()  # RTS pin by default (but can still use DTR)
            self.baud_rate: int = self._config.get('baudrate', 9600)
        else:
            raise EnvironmentError("Missing or incorrect config for the FTDI trigger.")

        if self.pin not in ['RTS', 'DTR']:
            raise ValueError("Invalid pin specified for FTDI trigger. Must be 'RTS' or 'DTR'.")

        self._connect()

    def _connect(self) -> None:
        """ Establishes the serial connection to the FTDI device """

        if not self._silent:
            print(f'[INFO] Connecting to FTDI trigger at {self.port}...')
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            # Connection success is just the port opening
            self._connected = True

            # ensure the pin is in a low state initially
            self._set_pin_state(False)

            if not self._silent:
                print(f'[INFO] FTDI trigger connected successfully on pin {self.pin}.')

        except serial.SerialException as e:
            print(f'[ERROR] FTDI trigger connection failed: {e}')
            self.ser = None
            self._connected = False

    def _set_pin_state(self, state: bool):
        """ Sets the state of the selected control pin (RTS or DTR) """

        if not self.connected:
            return
        try:
            if self.pin == 'RTS':
                self.ser.rts = state
            elif self.pin == 'DTR':
                self.ser.dtr = state

        except serial.SerialException as e:
            print(f'[ERROR] Lost connection to FTDI device while setting pin state: {e}')
            self.disconnect()

    def _trigger_loop(self, on_time: float, off_time: float):
        """ The loop that generates the PWM signal, run in a thread """

        while not self._stop_event.is_set():
            self._set_pin_state(True)
            time.sleep(on_time)

            # check if stop was requested during the 'on' phase
            if self._stop_event.is_set():
                break

            self._set_pin_state(False)
            time.sleep(off_time)

        # ensure the pin is left in a low state
        self._set_pin_state(False)

    def start(self, frequency: float, duty_cycle_percent: int = 50) -> None:
        """ Starts the trigger signal by launching a software PWM thread """
        if not self.connected:
            print('[ERROR] Cannot start trigger: not connected.')
            return

        if self._trigger_thread and self._trigger_thread.is_alive():
            if not self._silent:
                print('[WARNING] Trigger is already running. Stopping it before starting new frequency.')
            self.stop()

        if frequency <= 0:
            print('[ERROR] Frequency must be positive.')
            return

        if not (0 <= duty_cycle_percent <= 100):
            print('[ERROR] Duty cycle must be between 0 and 100.')
            return

        period = 1.0 / frequency
        on_time = period * (duty_cycle_percent / 100.0)
        off_time = period - on_time

        self._stop_event = threading.Event()
        self._trigger_thread = threading.Thread(
            target=self._trigger_loop,
            args=(on_time, off_time),
            daemon=True
        )
        self._trigger_thread.start()

        if not self._silent:
            print(
                f"[INFO] Trigger started at {frequency:.2f} Hz with {duty_cycle_percent}% duty cycle on pin {self.pin}.")

    def stop(self) -> None:
        """ Stops the trigger signal thread """

        if self._trigger_thread and self._trigger_thread.is_alive():
            self._stop_event.set()
            self._trigger_thread.join()  # wait for thread to finish
            if not self._silent:
                print('[INFO] Trigger stopped.')

        self._trigger_thread = None

        # ensure pin is low even if it was never started
        if self.connected:
            self._set_pin_state(False)

    def disconnect(self) -> None:
        """ Stops the trigger and closes the connection to the hardware device """
        if not self.connected:
            return

        self.stop()  # ensure the thread is stopped *before* closing the port

        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

        self._connected = False
        if not self._silent:
            print('[INFO] FTDI trigger disconnected.')


if __name__ == '__main__':
    # debug mini-script
    # To run this, you need a config.yaml file in the project root
    # (or to pass the config dictionary directly)

    secs = 5
    freq = 10

    print("--- Testing FtdiTrigger ---")

    try:
        with FTDITrigger(silent=False) as trigger:
            if trigger.connected:
                print(f"Starting trigger for {secs} seconds at {freq} Hz...")
                trigger.start(frequency=freq, duty_cycle_percent=50)
                time.sleep(secs)
                print("Stopping trigger...")

        print("\nTest complete. Trigger should be stopped and disconnected.")

    except EnvironmentError as e:
        print(f"\nConfiguration Error: Please check your config file. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")