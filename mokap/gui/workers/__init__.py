from dataclasses import dataclass
import numpy as np
@dataclass
class DetectionResult:  # TODO: Not sure this is needed, Lucida already has a Detection dataclass
    """
    Lightweight container for detection results.
    """
    frame_idx: int
    image_points: np.ndarray  # (N, 2) with NaN for undetected points
    valid: bool  # True if enough points were detected

    @property
    def detected_mask(self) -> np.ndarray:
        """Boolean mask of which points were detected."""
        return ~np.isnan(self.image_points).any(axis=1)

    @property
    def detected_points(self) -> np.ndarray:
        """Only the points that were actually detected."""
        return self.image_points[self.detected_mask]

    @property
    def detected_ids(self) -> np.ndarray:
        """Indices of detected points."""
        return np.where(self.detected_mask)[0]


from .coordinator import CalibrationCoordinator
from .detector_worker import DetectorWorker
from .monocular_worker import MonocularWorker
from .multiview_worker import MultiviewWorker