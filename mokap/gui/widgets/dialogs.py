from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QFrame, QGridLayout, QPushButton


class SnapPopup(QFrame):
    """ Mini popup with 9 buttons to snap the window """

    def __init__(self, parent=None, move_callback=None):
        super().__init__(parent)
        self.move_callback = move_callback
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)

        layout = QGridLayout()
        layout.setSpacing(1)
        self.setLayout(layout)

        positions = [
            ("nw", "Top-left"), ("n", "Top-center"), ("ne", "Top-right"),
            ("w", "Left"),      ("c", "Center"),      ("e", "Right"),
            ("sw", "Bottom-left"), ("s", "Bottom-center"), ("se", "Bottom-right"),
        ]

        for idx, (pos_code, tooltip) in enumerate(positions):
            btn = QPushButton()
            btn.setToolTip(tooltip)
            btn.setProperty("position", pos_code)
            btn.setMaximumWidth(25)
            btn.setMaximumHeight(25)
            btn.clicked.connect(self.on_button_clicked)
            # btn.setIcon(QIcon(f"icons/{pos_code}.png"))   # TODO
            layout.addWidget(btn, idx // 3, idx % 3)

    def on_button_clicked(self):
        button = self.sender()
        pos_code = button.property("position")
        if self.move_callback:
            self.move_callback(pos_code)
        self.hide()

    def show_popup(self, pos: QPoint):
        self.move(pos)
        self.show()