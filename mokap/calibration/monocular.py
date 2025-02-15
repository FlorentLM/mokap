import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2

# This can be used to estimate thepretical camera matrices
# Values from https://www.digicamdb.com/sensor-sizes/
SENSOR_SIZES = {'1/4"': [3.20, 2.40],
                '1/3.6"': [4, 3],
                '1/3.4"': [4.23, 3.17],
                '1/3.2"': [4.5, 3.37],
                '1/3"': [4.8, 3.6],
                '1/2.9"': [4.96, 3.72],
                '1/2.7"': [5.33, 4],
                '1/2.5"': [5.75, 4.32],
                '1/2.4"': [5.90, 4.43],
                '1/2.35"': [6.03, 4.52],
                '1/2.33"': [6.08, 4.56],
                '1/2.3"': [6.16, 4.62],
                '1/2"': [6.4, 4.8],
                '1/1.9"': [6.74, 5.05],
                '1/1.8"': [7.11, 5.33],
                '1/1.76"': [7.27, 5.46],
                '1/1.75"': [7.31, 5.49],
                '1/1.72"': [7.44, 5.58],
                '1/1.7"': [7.53, 5.64],
                '1/1.65"': [7.76, 5.81],
                '1/1.63"': [7.85, 5.89],
                '1/1.6"': [8, 6],
                '8.64 x 6 mm': [8.64, 6],
                '2/3"': [8.8, 6.6],
                '10.82 x 7.52 mm': [10.82, 7.52],
                '1"': [13.2, 8.8],
                '14 x 9.3 mm': [14, 9.3],
                'Four Thirds': [17.3, 13],
                '18.1 x 13.5 mm': [18.1, 13.5],
                '1.5"': [18.7, 14],
                '20.7 x 13.8 mm': [20.7, 13.8],
                '21.5 x 14.4 mm': [21.5, 14.4],
                '22.2 x 14.8 mm': [22.2, 14.8],
                '22.3 x 14.9 mm': [22.3, 14.9],
                '22.4 x 15 mm': [22.4, 15],
                '22.5 x 15 mm': [22.5, 15],
                '22.7 x 15.1 mm': [22.7, 15.1],
                '22.8 x 15.5 mm': [22.8, 15.5],
                '23.1 x 15.4 mm': [23.1, 15.4],
                '23 x 15.5 mm': [23, 15.5],
                '23.2 x 15.4 mm': [23.2, 15.4],
                '23.3 x 15.5 mm': [23.3, 15.5],
                '23.4 x 15.6 mm': [23.4, 15.6],
                '23.5 x 15.6 mm': [23.5, 15.6],
                '23.7 x 15.5 mm': [23.7, 15.5],
                '23.6 x 15.6 mm': [23.6, 15.6],
                '23.5 x 15.7 mm': [23.5, 15.7],
                '23.7 x 15.6 mm': [23.7, 15.6],
                '23.6 x 15.7 mm': [23.6, 15.7],
                '23.7 x 15.7 mm': [23.7, 15.7],
                '23.6 x 15.8 mm': [23.6, 15.8],
                '24 x 16 mm': [24, 16],
                '27 x 18 mm': [27, 18],
                '27.65 x 18.43 mm': [27.65, 18.43],
                '27.9 x 18.6 mm': [27.9, 18.6],
                '28.7 x 18.7 mm': [28.7, 18.7],
                '28.7 x 19.1 mm': [28.7, 19.1],
                '35.6 x 23.8 mm': [35.6, 23.8],
                '35.7 x 23.8 mm': [35.7, 23.8],
                '35.8 x 23.8 mm': [35.8, 23.8],
                '35.8 x 23.9 mm': [35.8, 23.9],
                '35.9 x 23.9 mm': [35.9, 23.9],
                '36 x 23.9 mm': [36, 23.9],
                '35.9 x 24 mm': [35.9, 24],
                '36 x 24 mm': [36, 24],
                '45 x 30 mm': [45, 30],
                '44 x 33 mm': [44, 33]
                }


def estimate_camera_matrix(f_mm, sensor_wh_mm, image_wh_px):
    """
        Estimate the camera matrix K (a 3x3 matrix of the camera intrinsics parameters) using real-world values
            [ fx, 0, cx ]
        K = [ 0, fy, cy ]
            [ 0,  0,  1 ]

        fx and fy (in pixels) are the focal lengths along the x and y axes
            since they are estimated from the real-world focal length, they are identical

        cx and cy are the coordinates of the principal point (in pixels)
            This corresponds to the point where the optical axis intersects the image plane
            and is usually in the centre of the frame
    """

    sensor_w_mm, sensor_h_mm = sensor_wh_mm
    image_w_px, image_h_px = image_wh_px
    pixel_size_x = sensor_w_mm / image_w_px
    pixel_size_y = sensor_h_mm / image_h_px

    f_x = f_mm / pixel_size_x
    f_y = f_mm / pixel_size_y

    cx = image_w_px / 2.0
    cy = image_h_px / 2.0

    K = np.array([
        [f_x,   0, cx],
        [  0, f_y, cy],
        [  0,   0,  1]
    ], dtype=np.float32)

    return K


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