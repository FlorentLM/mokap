import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2
from functools import reduce
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from mokap.utils import geometry
from mokap.calibration import monocular

# Defines functions that relate to processing stuff from N cameras

def filter_outliers(values, strong=False):

    # Use z score to find outliers
    if (values == 0).all():
        return values
    else:
        if not (values.std(axis=0) == 0).any():
            z = (values - values.mean(axis=0)) / values.std(axis=0)
        else:
            z = (values - values.mean(axis=0))
        outliers_z = (np.abs(z) > 2.5).any(axis=1)

        # Use IQR to find outliers
        if strong:
            q_1 = np.quantile(values, 0.125, axis=0)
            q_3 = np.quantile(values, 0.875, axis=0)
        else:
            q_1 = np.quantile(values, 0.25, axis=0)
            q_3 = np.quantile(values, 0.75, axis=0)

        iqr = q_3 - q_1
        lower = q_1 - 1.5 * iqr
        upper = q_3 + 1.5 * iqr

        outliers_iqr = np.any(values < lower, axis=1) | np.any(values > upper, axis=1)

        outliers_both = outliers_iqr | outliers_z

        filtered = values[~outliers_both]

        return filtered


def bestguess_rtvecs(n_m_rvecs, n_m_tvecs):
    """
        Finds a first best guess, for N cameras, of their real extrinsics parameters, from M samples per camera
    """

    N = len(n_m_rvecs)

    n_optimised_rvecs = np.zeros((N, 3))
    n_optimised_tvecs = np.zeros((N, 3))

    # Huber loss instead of simply doing mahalanobis distance makes the influence of 'flipped' detections less strong
    # (flipped detections being the ones resulting from an ambiguity in the monocular pose estimation)
    def robust_loss(residual, delta=0.0):
        # delta is the threshold between quadratic and linear behavior
        abs_r = np.abs(residual)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return 0.5 * quadratic ** 2 + delta * linear

    def objective_func(estimated, observed, cov):
        # regularize covariance matrix to avoid singularities
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)
        residuals = observed - estimated
        # Mahalanobis residuals
        mahal = np.sqrt(np.einsum('ij,jk,ik->i', residuals, cov_inv, residuals))

        loss = robust_loss(mahal)
        return np.sum(loss)

    for i in range(N):
        if n_m_rvecs[i].size == 0:  # no samples -> skip
            continue

        all_rvecs = filter_outliers(n_m_rvecs[i])
        all_tvecs = filter_outliers(n_m_tvecs[i])

        median_rvecs = np.median(all_rvecs, axis=0)
        median_tvecs = np.median(all_tvecs, axis=0)

        # covariance matrices
        # if not enough samples, use identity matrix
        if all_rvecs.shape[0] > 1:
            cov_r = np.cov(all_rvecs, rowvar=False)
        else:
            cov_r = np.eye(3)
        if all_tvecs.shape[0] > 1:
            cov_t = np.cov(all_tvecs, rowvar=False)
        else:
            cov_t = np.eye(3)

        try:
            # We can use L-BFGS-B here even though it's quite sensitive - because the robust_loss should take care of
            # problematic values
            res_r = minimize(objective_func, x0=median_rvecs, args=(all_rvecs, cov_r), method='L-BFGS-B')
            opt_r = res_r.x
        # except np.linalg.LinAlgError:
        except Exception as e:
            print("Optimization error for rvec:", e)
            opt_r = median_rvecs
        try:
            res_t = minimize(objective_func, x0=median_tvecs, args=(all_tvecs, cov_t), method='L-BFGS-B')
            opt_t = res_t.x
        # except np.linalg.LinAlgError:
        except Exception as e:
            print("Optimization error for tvec:", e)
            opt_t = median_tvecs

        n_optimised_rvecs[i, :] = opt_r
        n_optimised_tvecs[i, :] = opt_t

    return n_optimised_rvecs, n_optimised_tvecs


def common_points(n_p_points, n_p_points_ids):
    """
        Extracts the points and their IDs that are common to N cameras
    """
    nb_cameras = len(n_p_points)

    common_points_ids = reduce(np.intersect1d, n_p_points_ids)

    common_points = np.zeros((nb_cameras, len(common_points_ids), n_p_points[0].shape[1]))

    for n in range(nb_cameras):
        _, _, c = np.intersect1d(common_points_ids, n_p_points_ids[n], return_indices=True)
        common_points[n, :] = n_p_points[n][c]

    return common_points, common_points_ids

# (n_p_points2d, n_cam_mats, n_dist_coeffs) = (common_points2d, n_cam_mats, n_dist_coeffs)
def undistortion(n_p_points2d, n_cam_mats, n_dist_coeffs):
    """
        Undistort 2D points for each of N cameras (each camera can have a different number of points)
    """
    nb_cameras = len(n_cam_mats)

    # we use a list here because it's not necessarily the same number of points in all cameras
    points2d_undist = [monocular.undistortion(n_p_points2d[n], n_cam_mats[n], n_dist_coeffs[n]) for n in range(nb_cameras)]

    return points2d_undist

# (n_p_points2d, n_p_points_ids, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs) = (this_m_points2d, this_m_points2d_ids, rvecs, tvecs, camera_matrices, distortion_coeffs)
def triangulation(n_p_points2d, n_p_points_ids, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs):
    """
        Triangulate observations from N cameras, with each camera having its own number of points P,
        and obtain the 3D coordinates for points that are common to N cameras.
    """
    nb_cameras = len(n_rvecs_world)

    P_mats = [geometry.projection_matrix(n_cam_mats[n], geometry.invert_extrinsics_matrix(
        geometry.extrinsics_matrix(n_rvecs_world[n], n_tvecs_world[n]))) for n in range(nb_cameras)]

    common_points2d, common_points_ids = common_points(n_p_points2d, n_p_points_ids)

    if len(common_points_ids) < 3:
        return None, None

    points2d_undist = undistortion(common_points2d, n_cam_mats, n_dist_coeffs)

    points3d_svd = geometry.triangulate_points_svd(points2d_undist, P_mats)

    return points3d_svd, common_points_ids


def reprojection(points3d_world, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs):
    """
        Compute the reprojection of 3D points (world coordinates) for each of N cameras
    """
    nb_cameras = len(n_rvecs_world)

    points2d_reproj = np.zeros((nb_cameras, points3d_world.shape[0], 2))
    for n in range(nb_cameras):

        # Need to be n-camera-centric, so we invert them from world-centric
        cam_rvec, cam_tvec = geometry.invert_extrinsics_2(n_rvecs_world[n], n_tvecs_world[n])

        points2d_reproj[n, :, :] = monocular.reprojection(points3d_world, n_cam_mats[n], n_dist_coeffs[n], cam_rvec, cam_tvec)

    return points2d_reproj


def compute_3d_errors(points2d, points2d_ids,
                      points3d_world, points3d_ids, points3d_theor,
                      n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs,
                      fill_value=np.nan):
    """
        Compute the multi-view reprojection error (i.e. the error in 2D of the reprojected 3D triangulation) for each of
        N cameras, AND the 3D consistency error (i.e. the error in the distances between points in 3D)
    """
    if type(fill_value) is str:
        _fill_value = np.nan
    else:
        _fill_value = fill_value

    nb_cameras = len(n_rvecs_world)

    max_nb_points = points3d_theor.shape[0]
    max_nb_dists = int((max_nb_points * (max_nb_points - 1)) / 2.0)

    errors_2d_reproj = np.ones((nb_cameras, max_nb_points, 2)) * _fill_value
    errors_3d_consistency = np.ones(max_nb_dists) * _fill_value

    tri_idx = np.tril_indices(max_nb_points, k=-1)

    ideal_distances = cdist(points3d_theor, points3d_theor)
    measured_distances = np.ones_like(ideal_distances) * _fill_value
    visibility_mask = np.zeros_like(ideal_distances, dtype=bool)

    reprojected_points = reprojection(points3d_world, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs)

    # We can only compare the reprojection to the points that are detected in 2D,
    # but it can be a different set of points for each camera
    for n in range(nb_cameras):
        common_indices, i_3d, i_2d = np.intersect1d(points3d_ids, points2d_ids[n], return_indices=True)
        errors_2d_reproj[n, common_indices, :] = reprojected_points[n, i_3d, :] - points2d[n][i_2d, :]

    sq_idx = np.ix_(points3d_ids, points3d_ids)

    measured_distances[sq_idx] = cdist(points3d_world, points3d_world)
    visibility_mask[sq_idx] = True

    visible_ideal_distances = np.copy(ideal_distances)
    visible_ideal_distances[~visibility_mask] = _fill_value

    tri_measured_distances = measured_distances[tri_idx]
    tri_ideal_distances = visible_ideal_distances[tri_idx]

    errors_3d_consistency[:] = tri_measured_distances - tri_ideal_distances

    if type(fill_value) is str:
        match fill_value:
            case 'mean':
                fill_e2d = np.nanmean(errors_2d_reproj, axis=1, keepdims=True)
                fill_e3d = np.nanmean(errors_3d_consistency)
            case 'median':
                fill_e2d = np.nanmedian(errors_2d_reproj, axis=1, keepdims=True)
                fill_e3d = np.nanmedian(errors_3d_consistency)
            case 'max':
                fill_e2d = np.nanmax(errors_2d_reproj, axis=1, keepdims=True)
                fill_e3d = np.nanmax(errors_3d_consistency)
            case _:
                raise AttributeError(f"fill value '{fill_value}' is unknown (must be either a scalar, or 'mean', 'median' or 'max')")
        nans_e2d = np.isnan(errors_2d_reproj)
        nans_e3d = np.isnan(errors_3d_consistency)
        errors_2d_reproj[nans_e2d] = np.broadcast_to(fill_e2d, errors_2d_reproj.shape)[nans_e2d]
        errors_3d_consistency[nans_e3d] = fill_e3d

    return errors_2d_reproj, errors_3d_consistency


def interpolate3d(points3d_svd, points3d_ids, points3d_theoretical):
    """
        Use triangulated 3D points and theoretical board layout to interpolate missing points
    """
    detected_theoretical = points3d_theoretical[points3d_ids]

    # Build a design matrix by appending a column of ones to account for translation
    # Each row is [x, y, z, 1] for a detected theoretical point
    N_detected = detected_theoretical.shape[0]
    A = np.hstack([detected_theoretical, np.ones((N_detected, 1), dtype=detected_theoretical.dtype)])

    # Solve for the transformation matrix T (a 4x3 matrix) that best maps
    # [theoretical_point, 1] * T  to measured point (i.e. tje SVD result)
    T, residuals, rank, s = np.linalg.lstsq(A, points3d_svd, rcond=None)

    # then apply T to all the theoretical board points
    N_total = points3d_theoretical.shape[0]
    A_full = np.hstack([points3d_theoretical, np.ones((N_total, 1), dtype=points3d_theoretical.dtype)])
    points3d_full = A_full.dot(T)

    points3d_ids_full = np.arange(N_total)
    return points3d_full, points3d_ids_full