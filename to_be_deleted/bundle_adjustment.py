import numpy as np
from scipy.optimize import least_squares
from alive_progress import alive_bar
from to_be_deleted import multiview
# from mokap.calibration import multiview_jax as multiview
from mokap.utils import CallbackOutputStream


def flatten_params(Ks, dist_coeffs, rvecs, tvecs,
                   simple_focal=True,
                   simple_distortion=False,
                   complex_distortion=False,
                   shared=False):
    """
    Ks               : (n_cam, 3, 3) array of camera matrices
    dist_coeffs      : (n_cam, <=8) array of distortion coefficients per cam
    rvecs            : (n_view, 3) array of rotation vectors
    tvecs            : (n_view, 3) array of translation vectors
    simple_distortion  : use first 4 coefficients [k1, k2, p1, p2]
    complex_distortion : use up to 8 coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
    shared             : share intrnsics across all cameras
    """
    n_cam = Ks.shape[0]

    # determine distortion count
    if simple_distortion and complex_distortion:
        raise AssertionError('Distortion cannot be both simple and complex.')
    n_d = 4 if simple_distortion else (8 if complex_distortion else 5)

    if simple_focal:
        f = (Ks[:, 0, 0] + Ks[:, 1, 1]) * 0.5
        intr_elems = [f, Ks[:, 0, 2], Ks[:, 1, 2]]
    else:
        intr_elems = [Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]]

    # prepare distortion (pad to n_d if needed)
    dc = np.asarray(dist_coeffs)
    if dc.shape[1] < n_d:
        pad = np.zeros((n_cam, n_d - dc.shape[1]), dtype=dc.dtype)
        dc = np.hstack([dc, pad])
    dc = dc[:, :n_d]

    # stack intrinsics per cam
    intr_stack = np.column_stack(intr_elems + [dc])

    # shared: average then repeat
    if shared and n_cam > 1:
        blk = intr_stack.mean(axis=0, keepdims=True)
        intr_stack = np.repeat(blk, n_cam, axis=0)

    intr_flat = intr_stack.ravel()
    extr_flat = np.hstack([rvecs.ravel(), tvecs.ravel()])
    return np.hstack([intr_flat, extr_flat])


def unflatten_params(x, n_cam,
                     simple_focal=True,
                     simple_distortion=False,
                     complex_distortion=False,
                     shared=False):

    if simple_distortion and complex_distortion:
        raise AssertionError('Distortion cannot be both simple and complex.')

    n_k = 3 if simple_focal else 4
    n_d = 4 if simple_distortion else (8 if complex_distortion else 5)
    per_cam = n_k + n_d

    # intrinsics slice length
    total_intr = per_cam * n_cam
    intr_block = x[:total_intr].reshape(n_cam, per_cam)
    extr = x[total_intr:]

    # choose intrinsics per camera
    if shared and n_cam > 1:
        intr = np.repeat(intr_block[0:1], n_cam, axis=0)
    else:
        intr = intr_block

    # unpack intrinsics
    if simple_focal:
        fx = intr[:, 0]
        fy = fx
        cx = intr[:, 1]
        cy = intr[:, 2]
        dc = intr[:, 3:3+n_d]
    else:
        fx = intr[:, 0]
        fy = intr[:, 1]
        cx = intr[:, 2]
        cy = intr[:, 3]
        dc = intr[:, 4:4+n_d]

    # rebuild Ks array
    Ks = np.zeros((n_cam, 3, 3),dtype=fx.dtype)
    Ks[:, 0, 0] = fx; Ks[:, 1, 1] = fy
    Ks[:, 0, 2] = cx; Ks[:, 1, 2] = cy
    Ks[:, 2, 2] = 1.0

    # rebuild distortion
    dist_full = dc

    # derive nb of viewsd from extrinsics length
    n_view = extr.size // 6
    if n_view * 6 != extr.size:
        raise ValueError(f"Extrinsics length {extr.size} not divisible by 6.\n"
                         f"Expected 6*n_view.")

    # rebuild extrinsics
    rsz = 3 * n_view
    rvecs = extr[:rsz].reshape(n_view, 3)
    tvecs = extr[rsz:].reshape(n_view, 3)

    return Ks, dist_full, rvecs, tvecs


def intrinsics_bounds(simple_focal: bool, simple_distortion: bool, complex_distortion: bool):
    # focal and principal point
    # if simple_focal is True, we only optimise for [f, cx, cy] else [fx, fy, cx, cy]
    if simple_focal:
        intr_lo = [0.0, 0.0, 0.0]
        intr_hi = [np.inf, np.inf, np.inf]
    else:
        intr_lo = [0.0, 0.0, 0.0, 0.0]
        intr_hi = [np.inf, np.inf, np.inf, np.inf]

    # distortion
    if simple_distortion and not complex_distortion:
        # k1, k2, p1, p2  (4)
        intr_lo += [-0.5, -0.5, -0.1, -0.1]
        intr_hi += [ 0.5,  0.5,  0.1,  0.1]
    elif complex_distortion and not simple_distortion:
        # k1, k2, p1, p2, k3, k4, k5, k6  (8)
        intr_lo += [-0.5] * 2 + [-0.1, -0.1] + [-0.5] * 4
        intr_hi += [ 0.5] * 2 + [ 0.1,  0.1] + [ 0.5] * 4
    else:
        # default 5-coef: k1, k2, p1, p2, k3
        intr_lo += [-0.5, -0.5, -0.1, -0.1, -0.5]
        intr_hi += [ 0.5,  0.5,  0.1,  0.1,  0.5]

    return np.array(intr_lo), np.array(intr_hi)


# params, points_3d_th = x0, calib.dt.points3d
def cost_func(params, points_2d, points_ids, points_3d_th, weight_2d_reproj=1.0, weight_3d_consistency=1.0, simple_focal=True, simple_distortion=False, complex_distortion=False, interpolate=False, shared=False):
    N = len(points_2d)
    M = len(points_2d[0])

    max_nb_points = points_3d_th.shape[0]
    max_nb_dists = int((max_nb_points * (max_nb_points - 1)) / 2.0)

    camera_matrices, distortion_coeffs, rvecs, tvecs = unflatten_params(params,
                                                                        N,
                                                                        simple_focal=simple_focal,
                                                                        simple_distortion=simple_distortion,
                                                                        complex_distortion=complex_distortion,
                                                                        shared=shared)

    all_errors_reproj = np.zeros((M, N, max_nb_points, 2))
    all_errors_consistency = np.zeros((M, max_nb_dists))

    for m in range(M):
        this_m_points2d = [cam[m] for cam in points_2d]
        this_m_points2d_ids = [cam[m] for cam in points_ids]

        points3d_svd, points3d_ids = multiview.triangulation(this_m_points2d, this_m_points2d_ids,
                                                             rvecs, tvecs,
                                                             camera_matrices, distortion_coeffs)

        if points3d_svd is not None:

            if interpolate:
                points3d_svd, points3d_ids = multiview.interpolate3d(points3d_svd, points3d_ids, points_3d_th)

            # (points2d, points2d_ids, points3d_world, points3d_ids, points3d_theor, n_rvecs_world, n_tvecs_world, n_cam_mats, n_dist_coeffs) = (this_m_points2d, this_m_points2d_ids, points3d_svd, points3d_ids, points_3d_th, rvecs, tvecs, camera_matrices, distortion_coeffs)
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


def run_bundle_adjustment(camera_matrices, distortion_coeffs, rvecs, tvecs, points_2d, points_ids, points_3d, reproj_weight=1.0, consistency_weight=2.0, simple_focal=True, simple_distortion=False, complex_distortion=False, interpolate=False, shared=False):

    nb_cams = len(points_2d)

    # Flatten all the optimisable variables into a 1-D array
    x0 = flatten_params(camera_matrices, distortion_coeffs, rvecs, tvecs, simple_focal=simple_focal, simple_distortion=simple_distortion, complex_distortion=complex_distortion, shared=shared)

    # Set bounds on intrinsics parameters
    lo_intr, hi_intr = intrinsics_bounds(simple_focal=simple_focal, simple_distortion=simple_distortion, complex_distortion=complex_distortion)
    lb_intr = np.tile(lo_intr, nb_cams)
    ub_intr = np.tile(hi_intr, nb_cams)
    lb = np.concatenate([lb_intr, -np.inf * np.ones(x0.size - lb_intr.size)])
    ub = np.concatenate([ub_intr, np.inf * np.ones(x0.size - ub_intr.size)])

    # Note: Points 2D, points 3D and points IDs are fixed - We do not optimise those!

    with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
        with CallbackOutputStream(bar, keep_stdout=False):
            result = least_squares(cost_func, x0,       # x0 contains all the optimisable variables
                                   verbose=2,
                                   bounds=(lb, ub),
                                   x_scale='jac',
                                   ftol=1e-8,
                                   method='trf',
                                   loss='cauchy',
                                   f_scale=0.75,
                                   args=(points_2d,     # Passed as a fixed parameters
                                         points_ids,    # Passed as a fixed parameters
                                         points_3d,     # Passed as a fixed parameters
                                         reproj_weight, consistency_weight,     # Weights for the two types of errors
                                         simple_focal, simple_distortion, complex_distortion,
                                         interpolate, shared))

    camera_matrices_opt, distortion_coeffs_opt, rvecs_opt, tvecs_opt = unflatten_params(result.x, nb_cams, simple_focal=simple_focal, simple_distortion=simple_distortion, complex_distortion=complex_distortion, shared=shared)

    return camera_matrices_opt, distortion_coeffs_opt, rvecs_opt, tvecs_opt