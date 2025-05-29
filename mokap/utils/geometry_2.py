import numpy as np
from functools import reduce
from scipy.optimize import minimize
from typing import List, Tuple
import jax.numpy as jnp
from mokap.utils import geometry_jax


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
    # TODO - This could be fully JAX-ified using the https://github.com/google/jaxopt addons

    N = len(n_m_rvecs)

    n_optimised_rvecs = np.zeros((N, 3))
    n_optimised_tvecs = np.zeros((N, 3))

    # Huber loss instead of simply doing mahalanobis distance makes the influence of 'flipped' detections less strong
    # (flipped detections being the ones resulting from an ambiguity in the monocular pose estimation)
    def huberloss(residual, delta=0.0):
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

        loss = huberloss(mahal)
        return np.sum(loss)

    for i in range(N):
        if n_m_rvecs[i].size == 0:  # no samples -> skip
            continue

        all_rvecs = filter_outliers(n_m_rvecs[i])
        all_tvecs = filter_outliers(n_m_tvecs[i])

        median_rvecs = np.median(all_rvecs, axis=0)
        median_tvecs = np.median(all_tvecs, axis=0)

        # covariance matrices
        if all_rvecs.shape[0] > 1:
            cov_r = np.cov(all_rvecs, rowvar=False)
        else:
            cov_r = np.eye(3)
        if all_tvecs.shape[0] > 1:
            cov_t = np.cov(all_tvecs, rowvar=False)
        else:
            cov_t = np.eye(3)   # if not enough samples, we use the identity matrix

        # we can use L-BFGS-B here even though it's quite sensitive because the huber loss should take care of problematic values
        # Minimize orientation
        try:
            res_r = minimize(objective_func, x0=median_rvecs, args=(all_rvecs, cov_r), method='L-BFGS-B')
            opt_r = res_r.x
        except Exception as e:
            print("Optimization error for rvec:", e)
            opt_r = median_rvecs

        # Minimize position
        try:
            res_t = minimize(objective_func, x0=median_tvecs, args=(all_tvecs, cov_t), method='L-BFGS-B')
            opt_t = res_t.x
        except Exception as e:
            print("Optimization error for tvec:", e)
            opt_t = median_tvecs

        n_optimised_rvecs[i, :] = opt_r
        n_optimised_tvecs[i, :] = opt_t

    return n_optimised_rvecs, n_optimised_tvecs


def common_points(
        points2d: List[np.ndarray],
        points_ids: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the common points and points IDs in a list of variable number of detected points

    Args:
        points2d: list containing N arrays of points coordinates, each of shape (?, 2)
        points_ids: list containing N arrays of points IDs, each of shape (?, )

    Returns:
        common_pts: 2D coordinates of the M points that are common to all cameras
        common_ids: IDs of the M points that are common to all cameras
    """

    common_ids = reduce(np.intersect1d, points_ids)

    N = len(points2d)
    M = len(common_ids)
    D = points2d[0].shape[1]
    common_pts = np.zeros((N, M, D), dtype=points2d[0].dtype)

    for i in range(N):
        _, _, idx_in_cam = np.intersect1d(common_ids,
                                          points_ids[i],
                                          return_indices=True)
        common_pts[i] = points2d[i][idx_in_cam]

    return common_pts, common_ids


def triangulation(
        points2d:           jnp.ndarray,
        visibility_mask:    jnp.ndarray,
        rvecs_world:        np.ndarray,
        tvecs_world:        np.ndarray,
        camera_matrices:    np.ndarray,
        dist_coeffs:        np.ndarray,
) -> Tuple[jnp.ndarray, np.ndarray]:
    """
    Triangulate points 2D seen by C cameras

    Args:
        points2d: points 2D detected by the C cameras (C, N, 2)
        visibility_mask: visibility mask for points 2D (C, N)
        rvecs_world: C rotation vectors (C, 3)
        tvecs_world: C translation vectors (C, 3)
        camera_matrices: C camera matrices (C, 3, 3)
        dist_coeffs: C distortion coefficients (C, ≤8)

    Returns:
        points3d: N 3D coordinates (N, 3)

    """

    rvecs_world = jnp.asarray(rvecs_world)
    tvecs_world = jnp.asarray(tvecs_world)
    camera_matrices = jnp.asarray(camera_matrices)
    dist_coeffs = jnp.asarray(dist_coeffs)

    # this is converted back to a float array because the triangulate_svd accepts actual weights
    # we can multiply the visibility weights by a confidence score
    visibility_mask = visibility_mask.astype(jnp.float32)

    # Recover camera-centric extrinsics matrices and compute the projection matrices
    E_all = geometry_jax.extrinsics_matrix(rvecs_world, tvecs_world)    # (C, 4, 4)
    E_inv_all = geometry_jax.invert_extrinsics_matrix(E_all)            # (C, 4, 4)
    P_all = geometry_jax.projection_matrix(camera_matrices, E_inv_all)  # (C, 3, 4)

    pts2d_ud = geometry_jax.undistort_multiple(points2d, camera_matrices, dist_coeffs)
    pts3d = geometry_jax.triangulate_svd(pts2d_ud, P_all, weights=visibility_mask)  # (N, 3)

    return pts3d