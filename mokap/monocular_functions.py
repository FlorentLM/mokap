import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2


def undistortion(points2d, cam_mat, dist_coeffs):
    """
        Simple wrapper around OpenCV's undistortPoints
    """

    points2d = np.asarray(points2d)

    # OpenCV expects a shape (N, 1, 2)
    if not (points2d.ndim == 3 and points2d.shape[1:] == (1, 2)):
        points2d = points2d.reshape(-1, 1, 2)

    points_undist = cv2.undistortPoints(points2d, cameraMatrix=cam_mat, distCoeffs=dist_coeffs, P=cam_mat)

    return points_undist.squeeze()