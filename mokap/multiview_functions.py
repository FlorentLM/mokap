import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2
from functools import reduce
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from mokap import proj_geom, monocular_functions

# Defines functions that relate to processing stuff from N cameras

def filter_outliers(values):

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
    N = n_m_rvecs.shape[0]

    n_optimised_rvecs = np.zeros((N, 3))
    n_optimised_tvecs = np.zeros((N, 3))

    def objective_func(estimated_values, observed_values, covariance_matrix):
        cov_inv = np.linalg.inv(covariance_matrix)
        residuals = observed_values - estimated_values
        return np.sum(np.einsum('ij,jk,ik->i', residuals, cov_inv, residuals))

    for n, (m_rvecs, m_tvecs) in enumerate(zip(n_m_rvecs, n_m_tvecs)):

        all_rvecs = filter_outliers(m_rvecs)
        all_tvecs = filter_outliers(m_tvecs)

        mean_rvecs = np.mean(all_rvecs, axis=0)
        mean_tvecs = np.mean(all_tvecs, axis=0)

        cov_matrix_rvecs = np.cov(mean_rvecs, rowvar=False)
        cov_matrix_tvecs = np.cov(mean_tvecs, rowvar=False)

        try:
            result = minimize(objective_func, x0=mean_tvecs, args=(all_tvecs, cov_matrix_tvecs))
            optimized_tvecs = result.x
        except np.linalg.LinAlgError:
            optimized_tvecs = mean_tvecs

        try:
            result = minimize(objective_func, x0=mean_rvecs, args=(all_rvecs, cov_matrix_rvecs))
            optimized_rvecs = result.x
        except np.linalg.LinAlgError:
            optimized_rvecs = mean_rvecs

        n_optimised_rvecs[n, :] = optimized_rvecs
        n_optimised_tvecs[n, :] = optimized_tvecs

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


def undistortion(n_p_points2d, n_cam_mats, n_dist_coeffs):
    """
        Undistort 2D points for each of N cameras (each camera can have a different number of points)
    """
    nb_cameras = len(n_cam_mats)

    # we use a list here because it's not necessarily the same number of points in all cameras
    points2d_undist = [monocular_functions.undistortion(n_p_points2d[n], n_cam_mats[n], n_dist_coeffs) for n in range(nb_cameras)]

    return points2d_undist


def triangulation(n_p_points2d, n_p_points_ids, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs):
    """
        Triangulate observations from N cameras, with each camera having its own number of points P,
        and obtain the 3D coordinates for points that are common to N cameras.
    """
    nb_cameras = len(n_rvecs_world)

    P_mats = [proj_geom.projection_matrix(n_cam_mats[n], proj_geom.invert_extrinsics_matrix(
        proj_geom.extrinsics_matrix(n_rvecs_world[n], n_tvecs_world[n]))) for n in range(nb_cameras)]

    common_points2d, common_points_ids = common_points(n_p_points2d, n_p_points_ids)

    if len(common_points_ids) < 3:
        return None, None

    points2d_undist = undistortion(common_points2d, n_cam_mats, n_dist_coeffs)

    points3d_svd = proj_geom.triangulate_points_svd(points2d_undist, P_mats)

    return points3d_svd, common_points_ids


def reprojection(points3d_world, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs):
    """
        Compute the reprojection of 3D points (world coordinates) for each of N cameras
    """
    nb_cameras = len(n_rvecs_world)

    points2d_reproj = np.zeros((nb_cameras, points3d_world.shape[0], 2))
    for n in range(nb_cameras):

        # Need to be n-camera-centric, so we invert them from world-centric
        cam_rvec, cam_tvec = proj_geom.invert_extrinsics_2(n_rvecs_world[n], n_tvecs_world[n])

        reproj, jacobian = cv2.projectPoints(points3d_world,
                                             rvec=cam_rvec, tvec=cam_tvec,
                                             cameraMatrix=n_cam_mats[n],
                                             distCoeffs=n_dist_coeffs[n])

        points2d_reproj[n, :, :] = reproj.squeeze()

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
        errors_2d_reproj[n, common_indices, :] = reprojected_points[n, :, :] - points2d[n][i_2d, :]

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

