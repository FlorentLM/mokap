from typing import List, Optional
from PySide6.QtCore import QTimer, Slot
from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.gui.workers import GUI_UPDATE_TIMER, DEBUG_SIGNALS_FLOW
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.utils.datatypes import CalibrationData, DetectionPayload, ExtrinsicsPayload, IntrinsicsPayload


class MultiviewWorker(CalibrationProcessingWorker):

    def __init__(self, cameras_names: List[str], origin_camera_name: str):
        super().__init__(name='multiview')

        self._cameras_names = cameras_names
        self._origin_camera_name = origin_camera_name
        self.multiview_tool: Optional[MultiviewCalibrationTool] = None  # will be initialized later

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(GUI_UPDATE_TIMER)
        self.update_timer.timeout.connect(self.emit_online_extrinsics)

    @Slot(MultiviewCalibrationTool)
    def initialize_tool(self, tool: MultiviewCalibrationTool):
        """
        Receives the fully configured tool from the Coordinator when calibration stage > 0
        """
        self.multiview_tool = tool
        self.update_timer.start()
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Multiview tool initialized and ready.")

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):
        """
        This is ONLY for receiving detection data during online refinement
        """
        if DEBUG_SIGNALS_FLOW:
            print(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')

        # If tool isn't ready or payload isn't a detection, do nothing
        if self.multiview_tool is None or not isinstance(data.payload, DetectionPayload):
            return

        # core of the online process: register the detection
        cam_idx = self._cameras_names.index(data.camera_name)
        self.multiview_tool.register(cam_idx, data.payload)

    def emit_online_extrinsics(self):
        """
        Called by the QTimer. Checks if the tool has estimated extrinsics and emits them
        """
        if self.multiview_tool and self.multiview_tool._estimated:
            rvecs, tvecs = self.multiview_tool.initial_extrinsics
            for i, cam_name in enumerate(self._cameras_names):
                payload = ExtrinsicsPayload(rvecs[i], tvecs[i])
                # This goes to the Coordinator, which will forward it to the main thread UI
                self.send_payload.emit(CalibrationData(cam_name, payload))

    @Slot()
    def trigger_refinement(self):
        """
        Slot to be connected to a GUI button. Triggers the final BA
        """
        if self._paused or self.multiview_tool is None:
            return

        print(f"[{self.name.title()}] Attempting to run final Bundle Adjustment.")

        self.blocking.emit(True)
        success = self.multiview_tool.refine_all()
        self.blocking.emit(False)

        if success:
            print(f"[{self.name.title()}] Bundle Adjustment successful. Emitting refined results.")

            # emit the final results
            K_opts, D_opts = self.multiview_tool.refined_intrinsics
            r_opts, t_opts = self.multiview_tool.refined_extrinsics

            for i, cam_name in enumerate(self._cameras_names):
                # Send refined intrinsics
                intr_payload = IntrinsicsPayload(K_opts[i], D_opts[i])
                self.send_payload.emit(CalibrationData(cam_name, intr_payload))

                # Send refined extrinsics
                extr_payload = ExtrinsicsPayload(r_opts[i], t_opts[i])
                self.send_payload.emit(CalibrationData(cam_name, extr_payload))
        else:
            print(f"[{self.name.title()}] Bundle Adjustment failed.")

    @Slot()
    def reset(self):
        """ Destroys the tool and stops the timer, resets the worker's state """
        super().reset()

        if self.multiview_tool is not None:
            if DEBUG_SIGNALS_FLOW:
                print(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")

            self.update_timer.stop()
            self.multiview_tool = None

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)

        # If we move back to stage 0, we must destroy the tool and stop the timer
        if stage == 0 and self.multiview_tool is not None:
            print(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")
            self.update_timer.stop()
            self.multiview_tool = None
