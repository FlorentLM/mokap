import numpy as np
from scipy.optimize import least_squares
from alive_progress import alive_bar
from mokap.calibration import multiview
from mokap.utils import CallbackOutputStream


def flatten_extrinsics(m_rvecs, m_tvecs):
    """
        Takes rvecs and tvecs from m observations and returns a flattened vector of (m*6) parameters
    """
    return np.hstack([np.array(m_rvecs), np.array(m_tvecs)]).ravel()


def unflatten_extrinsics(params):
    """
        Takes a flattened vector of (m*6) parameters and returns rvecs and tvecs for the m observations
    """
    m_rvecs, m_tvecs = np.split(params.reshape(-1, 6), [3], axis=1)
    return m_rvecs, m_tvecs


def flatten_intrinsics(n_camera_matrices, n_distortion_coeffs, simple_focal=True, simple_dist=False, complex_dist=False):
    """
        Takes camera_matrices and distortion_coeffs from n cameras and returns a flattened vector.

        When simple_focal is True (default), only one value is used for fx and fy
        When simple_dist is True, only K1 and K2 are used

    """
    cm = np.array(n_camera_matrices).reshape(-1, 9)[:, [0, 2, 4, 5]]
    dc = np.array(n_distortion_coeffs)

    if simple_focal:
        cm = cm[:, [0, 1, 3]]

    if simple_dist and complex_dist:
        raise AssertionError('Distortion cannot be both simple and complex.')
    elif simple_dist and not complex_dist:
        dc = dc[:, :2]
    elif complex_dist and not simple_dist:
        if dc.shape[1] < 8:
            compl_dc = np.zeros((dc.shape[0], 8))
            compl_dc[:, :dc.shape[1]] = dc
            dc = compl_dc
    else:
        dc = dc[:, :5]
    return np.hstack([cm, dc]).ravel()


def unflatten_intrinsics(params, simple_focal=True, simple_dist=False, complex_dist=False):
    """
        Takes a flattened vector of (n*9) parameters and returns camera_matrices and distortion_coeffs for the n observations
    """

    n_cm = 3 if simple_focal else 4

    if simple_dist and complex_dist:
        raise AssertionError('Distortion cannot be both simple and complex.')
    elif simple_dist and not complex_dist:
        n_dc = 2
    elif complex_dist and not simple_dist:
        n_dc = 8
    else:
        n_dc = 5

    cm_params, dc_params = np.split(params.reshape(-1, n_cm + n_dc), [n_cm], axis=1)

    cm_sparse = np.zeros((cm_params.shape[0], 9))
    cm_sparse[:, [2, 5]] = cm_params[:, [1, -1]]
    cm_sparse[:, [0, 4]] = cm_params[:, :1+n_cm//2:2]
    cm_sparse[:, -1] = 1
    n_camera_matrices = cm_sparse.reshape(-1, 3, 3)

    n_dist_coeffs = np.zeros((dc_params.shape[0], n_dc))
    n_dist_coeffs[:, :n_dc] = dc_params

    return n_camera_matrices, n_dist_coeffs


def flatten_params(n_camera_matrices, n_distortion_coeffs, rvecs, tvecs, simple_focal=True, simple_dist=False, complex_dist=False):
    """
        Takes camera_matrices, distortion_coeffs, and rvecs and tvecs from n cameras
        OR camera_matrices and distortion_coeffs from n cameras, and rvecs and tvecs for m observations
        and returns a flattened vector of (m*n*9) parameters
    """
    intrinsics = flatten_intrinsics(n_camera_matrices, n_distortion_coeffs, simple_focal=simple_focal, simple_dist=simple_dist, complex_dist=complex_dist)
    extrinsics = flatten_extrinsics(rvecs, tvecs)

    return np.concatenate([intrinsics, extrinsics])


def unflatten_params(params, nb_cams, simple_focal=True, simple_dist=False, complex_dist=False):
    """
        Takes a flattened vector of (m*n*9) parameters, and the number n of cameras;
         returns camera_matrices and distortion_coeffs for the n observations, and rvecs and tvecs for m observations
    """

    n_cm = 3 if simple_focal else 4
    if simple_dist and complex_dist:
        raise AssertionError('Distortion cannot be both simple and complex.')
    elif simple_dist and not complex_dist:
        n_dc = 2
    elif complex_dist and not simple_dist:
        n_dc = 8
    else:
        n_dc = 5
    split = (n_cm + n_dc) * nb_cams

    n_camera_matrices, n_distortion_coeffs = unflatten_intrinsics(params[:split], simple_focal=simple_focal, simple_dist=simple_dist, complex_dist=complex_dist)
    m_rvecs, m_tvecs = unflatten_extrinsics(params[split:])

    return n_camera_matrices, n_distortion_coeffs, m_rvecs, m_tvecs


def cost_func(params, points_2d, points_2d_ids, points_3d_th, weight_2d_reproj=1.0, weight_3d_consistency=1.0, simple_focal=True, simple_dist=False, complex_dist=False):
    N = len(points_2d)
    M = len(points_2d[0])

    max_nb_points = points_3d_th.shape[0]
    max_nb_dists = int((max_nb_points * (max_nb_points - 1)) / 2.0)

    camera_matrices, distortion_coeffs, rvecs, tvecs = unflatten_params(params,
                                                                        nb_cams=N,
                                                                        simple_focal=simple_focal,
                                                                        simple_dist=simple_dist,
                                                                        complex_dist=complex_dist)

    all_errors_reproj = np.zeros((M, N, max_nb_points, 2))
    all_errors_consistency = np.zeros((M, max_nb_dists))

    # fs = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) / 2.0
    # camera_matrices[:, 0, 0] = fs
    # camera_matrices[:, 1, 1] = fs

    for m in range(M):
        this_m_points2d = [cam[m] for cam in points_2d]
        this_m_points2d_ids = [cam[m] for cam in points_2d_ids]

        points3d_svd, points3d_ids = multiview.triangulation(this_m_points2d, this_m_points2d_ids,
                                                             rvecs, tvecs,
                                                             camera_matrices, distortion_coeffs)

        if points3d_svd is not None:
            # points3d_svd, points3d_ids = multiview.interpolate_missing_points3d(points3d_svd, points3d_ids, points_3d_th)

            errors_reproj, errors_consistency = multiview.compute_3d_errors(this_m_points2d,
                                                                                 this_m_points2d_ids,
                                                                                 points3d_svd,
                                                                                 points3d_ids,
                                                                                 points_3d_th,
                                                                                 rvecs,
                                                                                 tvecs,
                                                                                 camera_matrices,
                                                                                 distortion_coeffs,
                                                                                 fill_value='median')
            all_errors_reproj[m, :, :, :] = errors_reproj
            all_errors_consistency[m, :] = errors_consistency
        else:
            all_errors_reproj[m, :, :, :] = 0.0
            all_errors_consistency[m, :] = 0.0

    return np.concatenate([all_errors_reproj.ravel() * weight_2d_reproj, all_errors_consistency.ravel() * weight_3d_consistency])


def run_bundle_adjustment(camera_matrices, distortion_coeffs, rvecs, tvecs, points_2d, points_ids, points_3d, reproj_weight=1.0, consistency_weight=2.0, simple_focal=True, simple_distortion=False, complex_dist=False):

    nb_cams = len(camera_matrices)

    # Flatten all the optimisable variables into a 1-D array
    x0 = flatten_params(camera_matrices, distortion_coeffs, rvecs, tvecs, simple_focal=simple_focal, simple_dist=simple_distortion, complex_dist=complex_dist)

    # Note: Points 2D, points 3D and points IDs are fixed - We do not optimise those!

    with alive_bar(title='Bundle adjustment...', force_tty=True) as bar:
        with CallbackOutputStream(bar, keep_stdout=False):
            result = least_squares(cost_func, x0,       # x0 contains all the optimisable variables
                                   verbose=2,
                                   x_scale='jac',
                                   ftol=1e-8,
                                   method='trf',
                                   loss='cauchy',
                                   f_scale=0.75,
                                   args=(points_2d,     # Passed as a fixed parameters
                                         points_ids,    # Passed as a fixed parameters
                                         points_3d,     # Passed as a fixed parameters
                                         reproj_weight, consistency_weight,     # Weights for the two types of errors
                                         simple_focal, simple_distortion))

    camera_matrices_opt, distortion_coeffs_opt, rvecs_opt, tvecs_opt = unflatten_params(result.x, nb_cams=nb_cams, simple_focal=simple_focal, simple_dist=simple_distortion)

    return camera_matrices_opt, distortion_coeffs_opt, rvecs_opt, tvecs_opt