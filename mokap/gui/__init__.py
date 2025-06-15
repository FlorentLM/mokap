import sys
from PySide6.QtWidgets import QMessageBox, QTextEdit

# The GUI Logger is the very first thing we declare and instantiate so it can capture everything!

class GUILogger:
    def __init__(self):
        self.text_area = None
        self._temp_output = ''

        # Store the original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Redirect the system streams to here
        sys.stdout = self
        sys.stderr = self

    def register_text_area(self, text_area: QTextEdit):
        self.text_area = text_area
        # if there was any output captured before the GUI was ready, write it now
        if self._temp_output:
            self.text_area.insertPlainText(self._temp_output)
            self._temp_output = ''

    def write(self, text: str):
        # write to the original console (always)
        self.original_stdout.write(text)

        # and if the GUI is ready, write to it
        if self.text_area:
            self.text_area.insertPlainText(text)
            self.text_area.ensureCursorVisible()
        else:
            # If the GUI is not ready yet, buffer the output.
            self._temp_output += text

    def flush(self):
        self.original_stdout.flush()
        # QTextEdit doesn't need an explicit flush

    def restore(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


GUI_LOGGER = GUILogger()

from mokap.gui.widgets.window_maincontrols import QApplication, MainControls
