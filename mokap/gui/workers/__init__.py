from dataclasses import dataclass
from lucida.calibration import Detection

@dataclass
class IndexedDetection(Detection):
    """
    Extends Lucida's Detection to include frame index for the GUI workers.
    """
    frame_idx: int = 0

from .coordinator import CalibrationCoordinator
from .detector_worker import DetectorWorker
from .monocular_worker import MonocularWorker
from .multiview_worker import MultiviewWorker