from typing import Union
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QFrame, QGridLayout, QPushButton, QDialog, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, \
    QDialogButtonBox
from mokap.utils.datatypes import ChessBoard, CharucoBoard


class SnapPopup(QFrame):

    """
        Mini popup with 9 buttons to snap the window
    """

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


class BoardParamsDialog(QDialog):

    def __init__(self, board_params: Union[ChessBoard, CharucoBoard], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Board Settings")
        self.setModal(True)
        self.is_charuco = isinstance(board_params, CharucoBoard)
        self.board_params = board_params
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()

        grid_layout.addWidget(QLabel("Rows:"), 0, 0)
        self.row_spin = QSpinBox()
        self.row_spin.setMinimum(2)
        self.row_spin.setMaximum(30)
        self.row_spin.setValue(self.board_params.rows)
        grid_layout.addWidget(self.row_spin, 0, 1)

        grid_layout.addWidget(QLabel("Columns:"), 1, 0)
        self.col_spin = QSpinBox()
        self.col_spin.setMinimum(2)
        self.col_spin.setMaximum(30)
        self.col_spin.setValue(self.board_params.cols)
        grid_layout.addWidget(self.col_spin, 1, 1)

        grid_layout.addWidget(QLabel("Square length (cm):"), 2, 0)
        self.sq_spin = QDoubleSpinBox()
        self.sq_spin.setMinimum(0.01)
        self.sq_spin.setMaximum(1000.0)
        self.sq_spin.setDecimals(2)
        self.sq_spin.setValue(self.board_params.square_length)
        grid_layout.addWidget(self.sq_spin, 2, 1)

        if self.is_charuco:
            grid_layout.addWidget(QLabel("Marker margin:"), 3, 0)
            self.marker_spin = QSpinBox()
            self.marker_spin.setMinimum(0)
            self.marker_spin.setMaximum(10)
            self.marker_spin.setValue(self.board_params.marker_length)
            grid_layout.addWidget(self.marker_spin, 3, 1)

        main_layout.addLayout(grid_layout)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)
        self.setLayout(main_layout)

    def get_values(self):
        # Create and return a new board object based on the dialog values
        if self.is_charuco:
            return CharucoBoard(
                rows=self.row_spin.value(),
                cols=self.col_spin.value(),
                square_length=self.sq_spin.value(),
                margin=self.marker_spin.value()
            )
        else:
            return ChessBoard(
                rows=self.row_spin.value(),
                cols=self.col_spin.value(),
                square_length=self.sq_spin.value()
            )
