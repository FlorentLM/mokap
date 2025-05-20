import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2
import jax.numpy as jnp
from jax import jit

# This can be used to estimate theoretical camera matrices
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


def is_sharp(image, threshold=80.0):
    # TODO - Improve this
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gauss = cv2.GaussianBlur(gray, (13, 13), 0)
    laplacian = cv2.Laplacian(gauss, cv2.CV_64F)
    variance = laplacian.var() * 10
    return variance >= threshold


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
    image_w_px, image_h_px = np.asarray(image_wh_px)[:2]
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


def reprojection(points3d, camera_matrix, dist_coeffs, rvec, tvec):
    """ Reproject 3D points into 2D space using a custom 3D reprojection algorithm making use of JIT calculations
    :param points3d: 3D points to reproject
    :param camera_matrix: camera intrinsics
    :param dist_coeffs: distortion coefficients
    :param rvec: rotation vector
    :param tvec: translation vector
    """

    if dist_coeffs is not None and len(dist_coeffs) < 9:
        dist_coeffs_minimal = np.zeros(8, dtype=np.float32)
        dist_coeffs_minimal[:len(dist_coeffs)] = dist_coeffs
        dist_coeffs = dist_coeffs_minimal

    reproj = project_3d_points(points3d, rvec, tvec, camera_matrix, dist_coeffs)

    return reproj

@jit
def project_3d_points(points3d, rvec, tvec, camera_matrix, dist_coeffs):

    points3d = jnp.asarray(points3d)
    rvec = jnp.asarray(rvec)
    tvec = jnp.asarray(tvec)
    camera_matrix = jnp.asarray(camera_matrix)
    dist_coeffs = jnp.asarray(dist_coeffs)

    if rvec.shape == (3, 1) or rvec.shape == (3,):
        rmat, _ = cv2.Rodrigues(rvec)
    else:
        rmat = rvec

    if tvec.shape == (3,):
        tvec = tvec.reshape(3, 1)

    points_cam = (rmat @ points3d.T) + tvec

    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs

    # Normalize by Z coords
    normalised_2dpts = points_cam[:2] / points_cam[2]
    x, y = normalised_2dpts
    r2 = jnp.pow(normalised_2dpts, 2).sum(axis=0)
    r4 = jnp.pow(r2, 2)
    r6 = jnp.pow(r2, 3)

    # Apply radial and tangential distortion
    radial_distortion = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_distorted = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_distorted = y * radial_distortion + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Project to 2D image plane using the camera matrix
    points_2d = camera_matrix[:2, :2] @ jnp.vstack((x_distorted, y_distorted)) + camera_matrix[:2, 2].reshape(2, 1)

    return points_2d.T


def undistortion(points2d, camera_matrix, dist_coeffs):
    """
        Simple wrapper around OpenCV's undistortPoints
    """

    points2d = np.asarray(points2d)

    # OpenCV expects a shape (N, 1, 2)
    if not (points2d.ndim == 3 and points2d.shape[1:] == (1, 2)):
        points2d = points2d.reshape(-1, 1, 2)

    if len(dist_coeffs) < 4:
        dist_coeffs_minimal = np.zeros(4, dtype=np.float32)
        dist_coeffs_minimal[:len(dist_coeffs)] = dist_coeffs
        dist_coeffs = dist_coeffs_minimal
    points_undist = cv2.undistortPoints(points2d, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, P=camera_matrix)

    return points_undist.squeeze()