import numpy as np
from scipy.optimize import least_squares
from alive_progress import alive_bar
from typing import Tuple
from mokap.utils import CallbackOutputStream
from mokap.utils import geometry_jax
import jax
import jax.numpy as jnp


def flatten_params(
        camera_matrices:    np.ndarray,
        dist_coeffs:        np.ndarray,
        cam_rvecs:          np.ndarray,
        cam_tvecs:          np.ndarray,
        board_rvecs:        np.ndarray,
        board_tvecs:        np.ndarray,
        simple_focal:       bool,
        simple_distortion:  bool,
        complex_distortion: bool,
        shared:             bool
) -> np.ndarray:
    """
    Flatten all optimisable parameters into a vector

    Args:
        camera_matrices: array of C camera matrices (C, 3, 3)
        dist_coeffs: array of distortion coefficients per cam (C, <=8)
        cam_rvecs: array of rotation vectors (M, 3)
        cam_tvecs: array of translation vectors (M, 3)
        simple_focal: fix fx = fy
        simple_distortion: use first 4 coefficients [k1, k2, p1, p2]
        complex_distortion: use up to 8 coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
        shared: share intrnsics across all cameras

    Returns:
        X: all the optimisable parameters flattened into a loooong vector
    """
    C = camera_matrices.shape[0]

    # determine distortion count
    if simple_distortion and complex_distortion:
        raise AssertionError('Distortion cannot be both simple and complex.')
    D = 4 if simple_distortion else (8 if complex_distortion else 5)

    if simple_focal:
        f = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) * 0.5
        intr_elems = [f, camera_matrices[:, 0, 2], camera_matrices[:, 1, 2]]
    else:
        intr_elems = [camera_matrices[:, 0, 0],
                      camera_matrices[:, 1, 1],
                      camera_matrices[:, 0, 2],
                      camera_matrices[:, 1, 2]]

    # prepare distortion (pad to D if needed)
    dc = np.asarray(dist_coeffs)
    if dc.shape[1] < D:
        pad = np.zeros((C, D - dc.shape[1]), dtype=dc.dtype)
        dc = np.hstack([dc, pad])
    dc = dc[:, :D]

    # stack intrinsics per cam
    intr_stack = np.column_stack(intr_elems + [dc])

    # shared: average then repeat
    if shared and C > 1:
        blk = intr_stack.mean(axis=0, keepdims=True)
        intr_stack = np.repeat(blk, C, axis=0)

    intr_flat = intr_stack.ravel()
    extr_flat = np.hstack([cam_rvecs.ravel(), cam_tvecs.ravel()])

    board_flat = np.hstack([board_rvecs.ravel(), board_tvecs.ravel()])
    X = np.hstack([intr_flat, extr_flat, board_flat])

    return X


def unflatten_params(
        X:                  np.ndarray,
        nb_cams:            int,
        nb_frames:          int,
        simple_focal:       bool,
        simple_distortion:  bool,
        complex_distortion: bool,
        shared:             bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unflatten the parameters vector back into usable arrays

    Args:
        X: the looong parameters vector
        nb_cams: the number C of cameras
        simple_focal: fix fx = fy
        simple_distortion: use first 4 coefficients [k1, k2, p1, p2]
        complex_distortion: use up to 8 coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
        shared: share intrnsics across all cameras

    Returns:
        camera_matrices: array of C camera matrices (C, 3, 3)
        dist_coeffs: array of distortion coefficients per cam (C, <=8)
        rvecs: array of rotation vectors (M, 3)
        tvecs: array of translation vectors (M, 3)
    """

    if simple_distortion and complex_distortion:
        raise AssertionError('Distortion cannot be both simple and complex.')

    C, P = nb_cams, nb_frames

    # how many intrinsics per camera
    n_k = 3 if simple_focal else 4
    n_d = 4 if simple_distortion else (8 if complex_distortion else 5)
    per_cam = n_k + n_d

    total_intr = per_cam * C
    intr_block = X[:total_intr].reshape(C, per_cam)

    # unpack intrinsics
    if shared and C > 1:
        intr = np.repeat(intr_block[0:1], C, axis=0)
    else:
        intr = intr_block

    # unpack intrinsics and rebuidl dist coeffs
    if simple_focal:
        fx = intr[:, 0]
        fy = fx
        cx = intr[:, 1]
        cy = intr[:, 2]
        dist_coeffs = intr[:, 3:3+n_d]
    else:
        fx = intr[:, 0]
        fy = intr[:, 1]
        cx = intr[:, 2]
        cy = intr[:, 3]
        dist_coeffs = intr[:, 4:4+n_d]

    # rebuild camera matrices
    camera_matrices = np.zeros((C, 3, 3), dtype=fx.dtype)
    camera_matrices[:, 0, 0] = fx
    camera_matrices[:, 1, 1] = fy
    camera_matrices[:, 0, 2] = cx
    camera_matrices[:, 1, 2] = cy
    camera_matrices[:, 2, 2] = 1.0

    # C camera extrinsics (6 parameters each)
    start_cam_extr = total_intr
    end_cam_extr = start_cam_extr + 6 * C
    cam_extr = X[start_cam_extr: end_cam_extr]  # length 6 * C
    cam_rvecs = cam_extr[:3 * C].reshape(C, 3)
    cam_tvecs = cam_extr[3 * C:6 * C].reshape(C, 3)

    # now the board poses: 6*P values remaining
    board_block = X[end_cam_extr:]  # this is X[total_intr + 6*C :]
    board_rvecs = board_block[:3 * P].reshape(P, 3)
    board_tvecs = board_block[3 * P:6 * P].reshape(P, 3)

    return camera_matrices, dist_coeffs, cam_rvecs, cam_tvecs, board_rvecs, board_tvecs


def intrinsics_bounds(
        simple_focal: bool,
        simple_distortion: bool,
        complex_distortion: bool
):
    k_lo = -0.25
    k_hi = 0.25
    p_lo = -0.1
    p_hi = 0.1
    f_lo = 0.0
    f_hi = np.inf
    c_lo = 0.0
    c_hi = np.inf  # TODO: use image centre +- a small margin

    # focal and principal point
    # if simple_focal is True, we only optimise for [f, cx, cy] else [fx, fy, cx, cy]
    if simple_focal:
        intr_lo = [f_lo, c_lo, c_lo]
        intr_hi = [f_hi, c_hi, c_hi]
    else:
        intr_lo = [f_lo, f_lo, c_lo, c_lo]
        intr_hi = [f_hi, f_hi, c_hi, c_hi]

    # distortion
    if simple_distortion and not complex_distortion:
        # k1, k2, p1, p2  (4)
        intr_lo += [k_lo, k_lo, p_lo, p_lo]
        intr_hi += [k_hi, k_hi, p_hi, p_hi]
    elif complex_distortion and not simple_distortion:
        # k1, k2, p1, p2, k3, k4, k5, k6  (8)
        intr_lo += [k_lo] * 2 + [p_lo, p_lo] + [k_lo] * 4
        intr_hi += [k_hi] * 2 + [p_hi, p_hi] + [k_hi] * 4
    else:
        # default 5-coef: k1, k2, p1, p2, k3
        intr_lo += [k_lo, k_lo, p_lo, p_lo, k_lo]
        intr_hi += [k_hi, k_hi, p_hi, p_hi, k_hi]

    return np.array(intr_lo), np.array(intr_hi)

# params = x0
def cost_func(
    params:             np.ndarray,
    points2d:           np.ndarray,     # (C, P, N, 2)
    visibility_mask:    np.ndarray,     # (C, P, N)
    points3d_th:        jnp.ndarray,    # (N, 3)
    simple_focal:       bool = False,
    simple_distortion:  bool = False,
    complex_distortion: bool = False,
    shared:             bool = False,
):

    C, P, N = visibility_mask.shape

    cam_mats, dist_coefs, cam_rvecs, cam_tvecs, board_rvecs, board_tvecs = unflatten_params(
        params,
        C, P,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared)

    # compute board -> world matrices
    E_board_w = geometry_jax.extrinsics_matrix(board_rvecs, board_tvecs)  # (P, 4, 4)

    # compute camera -> world matrices
    E_cam_w = geometry_jax.extrinsics_matrix(cam_rvecs, cam_tvecs)  # (C, 4, 4)
    E_world_cam = geometry_jax.invert_extrinsics_matrix(E_cam_w)    # (C, 4, 4)

    # reshape for broadcast
    E_world_cam = E_world_cam[None, :, :, :]    # (1, C, 4, 4)
    E_board_w = E_board_w[:, None, :, :]        # (P, 1, 4, 4)

    E_board_cam = jnp.matmul(E_world_cam, E_board_w)

    r_bc, t_bc = geometry_jax.extmat_to_rtvecs(E_board_cam) # each is (P, C, 3)

    # we vmap over the P dimension (project_multiple expects (C, 3) inputs)
    project_frame = lambda rv, tv: geometry_jax.project_multiple(points3d_th, rv, tv, cam_mats, dist_coefs)
    reproj = jax.vmap(project_frame, in_axes=(0, 0))(r_bc, t_bc)  # (P, C, N, 2)

    # bring points2d & mask into (P, C, N, ...) layout
    obs_pc = jnp.transpose(points2d, (1, 0, 2, 3))       # (P, C, N, 2)
    mask_pc = jnp.transpose(visibility_mask, (1, 0, 2))  # (P, C, N)
    m = mask_pc[..., None]  # (P, C, N, 1)

    resid_pc = jnp.where(m, reproj - obs_pc, 0.0)  # (P, C, N, 2)
    return np.array(resid_pc.ravel())

def run_bundle_adjustment(
        camera_matrices:    np.ndarray,  # (C, 3, 3)
        distortion_coeffs:  np.ndarray,  # (C, ≤8)
        cam_rvecs:          np.ndarray,  # (C, 3)
        cam_tvecs:          np.ndarray,  # (C, 3)
        board_rvecs:        np.ndarray,  # (C, 3)
        board_tvecs:        np.ndarray,  # (C, 3)
        points2d:           np.ndarray,  # (C, P, N, 2)
        visibility_mask:    np.ndarray,  # (C, P, N)
        points3d_th,
        image_size = None,     # (2,)    # TODO: Needs to support multiple sizes
        priors_weight=0.0,
        simple_focal=False,
        simple_distortion=False,
        complex_distortion=False,
        shared=False):

    # Recover the dimensions
    C, P, N = visibility_mask.shape

    # Flatten all the optimisable variables into a 1-D array
    x0 = flatten_params(
        camera_matrices,
        distortion_coeffs,
        cam_rvecs,
        cam_tvecs,
        board_rvecs, board_tvecs,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared)

    # Set bounds on intrinsics parameters
    lo_intr, hi_intr = intrinsics_bounds(simple_focal=simple_focal,
                                         simple_distortion=simple_distortion,
                                         complex_distortion=complex_distortion)
    lb_intr = np.tile(lo_intr, C)
    ub_intr = np.tile(hi_intr, C)
    lb = np.concatenate([lb_intr, -np.inf * np.ones(x0.size - lb_intr.size)])
    ub = np.concatenate([ub_intr, np.inf * np.ones(x0.size - ub_intr.size)])

    points3d_th = jnp.asarray(points3d_th)    # (N, 3)

    with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
        with CallbackOutputStream(bar):
            result = least_squares(cost_func,
                                   x0,          # x0 contains all the optimisable variables
                                   verbose=2,
                                   # bounds=(lb, ub),
                                   x_scale='jac',
                                   ftol=1e-8,
                                   method='trf',
                                   loss='cauchy',
                                   f_scale=1.0,
                                   args=(
                                       points2d,        # Passed as a fixed parameters
                                       visibility_mask, # Passed as a fixed parameters
                                       points3d_th,     # Passed as a fixed parameters

                                       simple_focal,
                                       simple_distortion,
                                       complex_distortion,
                                       shared))

    camera_matrices_opt, distortion_coeffs_opt, cam_rvecs_opt, cam_tvecs_opt, board_rvecs_opt, board_tvecs_opt = unflatten_params(
        result.x,
        C, P,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared)

    return camera_matrices_opt, distortion_coeffs_opt, cam_rvecs_opt, cam_tvecs_opt, board_rvecs_opt, board_tvecs_opt