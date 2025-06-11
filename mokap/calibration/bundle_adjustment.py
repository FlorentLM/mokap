import numpy as np
from numpy._typing import ArrayLike
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from mokap.utils import CallbackOutputStream
from alive_progress import alive_bar
from mokap.utils.geometry.projective import project_multiple
from mokap.utils.geometry.transforms import extrinsics_matrix, invert_extrinsics_matrix, extmat_to_rtvecs


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

    # shared: average the intrinsics
    if shared and C > 1:
        blk = intr_stack.mean(axis=0, keepdims=True)  # (1, intr_size)
        intr_flat = blk.ravel()
    else:
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

    if shared and C > 1:
        total_intr = per_cam  # only one set in X
        intr_block = X[:per_cam].reshape(1, per_cam)  # 1 set of intrinsics
        intr_block = np.repeat(intr_block, C, axis=0)  # tile for use
    else:
        total_intr = per_cam * C
        intr_block = X[:total_intr].reshape(C, per_cam)

    # unpack intrinsics and rebuidl dist coeffs
    if simple_focal:
        fx = intr_block[:, 0]
        fy = fx
        cx = intr_block[:, 1]
        cy = intr_block[:, 2]
        dist_coeffs = intr_block[:, 3:3+n_d]
    else:
        fx = intr_block[:, 0]
        fy = intr_block[:, 1]
        cx = intr_block[:, 2]
        cy = intr_block[:, 3]
        dist_coeffs = intr_block[:, 4:4+n_d]

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
    images_sizes_wh:    ArrayLike,
    simple_focal:       bool,
    simple_distortion:  bool,
    complex_distortion: bool,
    shared:             bool
) -> Tuple[np.ndarray, np.ndarray]:

    C = images_sizes_wh.shape[0]
    w, h = images_sizes_wh[0, :2]   # TODO: Actually use the multiple sizes!!

    # Focal length bounds
    f_lo = 0.0
    f_hi = np.inf

    # Principal point bounds (+- 20% margin)
    cx_lo = w / 2.0 - w * 0.35
    cx_hi = w / 2.0 + w * 0.35
    cy_lo = h / 2.0 - h * 0.35
    cy_hi = h / 2.0 + h * 0.35

    # k and p distortion coeffs bounds
    k_lo = -0.5
    k_hi = 0.5
    p_lo = -0.2
    p_hi = 0.2

    K_bounds = np.array([
        [f_lo, f_hi],    # fx
        [f_lo, f_hi],    # fy
        [cx_lo, cx_hi],  # cx
        [cy_lo, cy_hi],  # cy
    ])

    D_bounds = np.array([
        [k_lo, k_hi],  # k1
        [k_lo, k_hi],  # k2
        [p_lo, p_hi],  # p1
        [p_lo, p_hi],  # p2
        [k_lo, k_hi],  # k3
        [k_lo, k_hi],  # k4
        [k_lo, k_hi],  # k5
        [k_lo, k_hi],  # k6
    ])

    # Choose slices
    if simple_distortion and complex_distortion:
        raise ValueError("Cannot use both simple and complex distortion.")

    if simple_distortion:
        D_slice = D_bounds[:4]
    elif complex_distortion:
        D_slice = D_bounds[:8]
    else:
        D_slice = D_bounds[:5]

    if simple_focal:
        K_slice = K_bounds[[0, 2, 3]]  # f, cx, cy
    else:
        K_slice = K_bounds  # fx, fy, cx, cy

    bounds = np.vstack([K_slice, D_slice]).T

    # Expand across cameras
    if not shared:
        bounds = np.tile(bounds, (1, C))

    return bounds[0], bounds[1]


# params = x0
def cost_func(
    params:             np.ndarray,
    points2d:           jnp.ndarray,     # (P, C, N, 2)
    visibility_mask:    jnp.ndarray,     # (P, C, N)
    points3d_th:        jnp.ndarray,     # (N, 3)
    points_weights:     jnp.ndarray,     # (P, C, N)
    prior_weight:       float = 0.0,     # (P, C, N)
    rvecs_cam_init:     Optional[jnp.ndarray] = None, # (C, 3) or None
    tvecs_cam_init:     Optional[jnp.ndarray] = None, # (C, 3) or None
    simple_focal:       bool = False,
    simple_distortion:  bool = False,
    complex_distortion: bool = False,
    shared:             bool = False,
):

    P, C, N = visibility_mask.shape

    # Extract optimisable params from the looong vector and move them back to the GPU
    cam_mats, dist_coefs, cam_rvecs, cam_tvecs, board_rvecs, board_tvecs = unflatten_params(params, C, P,
        simple_focal=simple_focal, simple_distortion=simple_distortion,
        complex_distortion=complex_distortion, shared=shared)

    cam_mats = jnp.asarray(cam_mats)
    dist_coefs = jnp.asarray(dist_coefs)
    board_rvecs = jnp.asarray(board_rvecs)
    board_tvecs = jnp.asarray(board_tvecs)
    cam_rvecs = jnp.asarray(cam_rvecs)
    cam_tvecs = jnp.asarray(cam_tvecs)

    # compute board -> world matrices
    E_board_w = extrinsics_matrix(board_rvecs, board_tvecs)  # (P, 4, 4)

    # compute camera -> world matrices
    E_cam_w = extrinsics_matrix(cam_rvecs, cam_tvecs)  # (C, 4, 4)
    E_world_cam = invert_extrinsics_matrix(E_cam_w)    # (C, 4, 4)

    # reshape for broadcast
    E_world_cam = E_world_cam[None, :, :, :]    # (1, C, 4, 4)
    E_board_w = E_board_w[:, None, :, :]        # (P, 1, 4, 4)

    E_board_cam = jnp.matmul(E_world_cam, E_board_w)

    r_bc, t_bc = extmat_to_rtvecs(E_board_cam) # each is (P, C, 3)

    # we vmap over the P dimension (project_multiple expects (C, 3) inputs)
    project_frame = lambda rv, tv: project_multiple(points3d_th, rv, tv, cam_mats, dist_coefs)
    reproj = jax.vmap(project_frame, in_axes=(0, 0))(r_bc, t_bc)  # (P, C, N, 2)

    # Weighted residuals
    resid = reproj - points2d
    weighted_resid = jnp.where(visibility_mask[..., None], resid * points_weights[..., None], 0.0)

    # Calculate mean error for logging
    visible_error_magnitudes = jnp.where(visibility_mask, jnp.linalg.norm(resid, axis=-1), jnp.nan)
    mean_reprojection_error_px = jnp.nanmean(visible_error_magnitudes)

    print(f"Mean Reprojection Error: {mean_reprojection_error_px:.2f}px")

    # Prior regularization (optional)
    if rvecs_cam_init is not None and tvecs_cam_init is not None:
        rvec_resid = cam_rvecs - rvecs_cam_init
        tvec_resid = cam_tvecs - tvecs_cam_init
        prior_resid = jnp.concatenate([rvec_resid.ravel(), tvec_resid.ravel()]) * prior_weight
        final = jnp.concatenate([weighted_resid.ravel(), prior_resid])
    else:
        final = weighted_resid.ravel()

    return np.array(final)


def residual_weights(
    pts2d:              np.ndarray,      # (C, P, N, 2)
    visibility_mask:    np.ndarray,      # (C, P, N)
    camera_matrices:    np.ndarray,      # (C, 3, 3)
    reproj_error:       Optional[np.ndarray] = None  # (C, P, N) or None
) -> np.ndarray:
    """
    Compute per-observation weights for BA residuals based on
        - Visibility
        - Distance from image center
        - Number of views per point
        - Reprojection error (optional)
    """

    C, P, N, _ = pts2d.shape

    # Distance to center weighting
    cx = camera_matrices[:, 0, 2][:, None, None]    # (C, 1, 1)
    cy = camera_matrices[:, 1, 2][:, None, None]
    center = np.stack([cx, cy], axis=-1)      # (C, 1, 1, 2)

    dists = np.linalg.norm(pts2d - center, axis=-1)          # (C, P, N)
    max_dist = np.sqrt(cx[:, 0, 0] ** 2 + cy[:, 0, 0] ** 2)  # (C,)
    max_dist = max_dist[:, None, None] + 1e-8

    dist_weight = 1.0 / (1.0 + (dists / max_dist) ** 2)      # (C, P, N)

    # Weighting with the number of cameras seeing each point
    nb_views = visibility_mask.sum(axis=0)  # (P, N)
    nb_views_weight = np.tile(nb_views[None, :, :], (C, 1, 1))  # (C, P, N)
    nb_views_weight = nb_views_weight / (1.0 + nb_views_weight)

    # Optional reprojection weight
    if reproj_error is not None:
        reproj_weight = 1.0 / (1.0 + reproj_error)
        reproj_weight = np.clip(reproj_weight, 0.1, 1.0)
    else:
        reproj_weight = np.ones_like(visibility_mask, dtype=np.float32)

    # Combine
    weights = (
        visibility_mask.astype(np.float32) *
        dist_weight *
        nb_views_weight *
        reproj_weight
    )

    # Normalize
    weights /= (np.max(weights) + 1e-8)

    return weights


def make_jacobian_sparsity(
        C: int,
        P: int,
        N: int,
        intr_size: int,
        add_priors: bool,
        shared: bool
) -> np.ndarray:
    """
    Build a sparse mask S of shape (num_residuals, num_params) where each non‐zero
    entry = True indicates that the corresponding residual depends on that parameter.
    We flatten residuals in the order (p, c, n, coord)
    """

    rt = 6
    total_intr = intr_size
    total_ext = rt * C
    total_board = rt * P
    num_params = total_intr + total_ext + total_board

    # Each visible (p, c, n) contributes 2 residuals (x and y).
    num_residuals = 2 * P * C * N
    if add_priors:
        num_residuals += 6 * C

    S = lil_matrix((num_residuals, num_params), dtype=bool)

    # If intrinsics are shared: intr_stride = intr_size (one block for all C cams).
    # Otherwise: intr_stride = (intr_size // C).
    intr_stride = intr_size if shared else (intr_size // C)

    for p in range(P):
        for c in range(C):
            for n in range(N):
                base = (p * C + c) * N + n
                row0 = 2 * base
                row1 = row0 + 1

                # Intrinsic block
                if shared:
                    col_intr = 0
                else:
                    col_intr = c * intr_stride
                S[row0, col_intr : col_intr + intr_stride] = 1
                S[row1, col_intr : col_intr + intr_stride] = 1

                # Extrinsic (this camera c)
                col_ext = total_intr + c * rt
                S[row0, col_ext : col_ext + rt] = 1
                S[row1, col_ext : col_ext + rt] = 1

                # Board‐pose for frame p
                col_board = total_intr + total_ext + p * rt
                S[row0, col_board : col_board + rt] = 1
                S[row1, col_board : col_board + rt] = 1

    if add_priors:
        base_row = 2 * P * C * N
        for c in range(C):
            row_start = base_row + c * rt
            col_start = total_intr + c * rt
            S[row_start : row_start + rt, col_start : col_start + rt] = 1

    return S.tocsr()


def run_bundle_adjustment(
        camera_matrices:    np.ndarray,  # (C, 3, 3)
        distortion_coeffs:  np.ndarray,  # (C, ≤8)
        cam_rvecs:          np.ndarray,  # (C, 3)
        cam_tvecs:          np.ndarray,  # (C, 3)
        board_rvecs:        np.ndarray,  # (P, 3)
        board_tvecs:        np.ndarray,  # (P, 3)
        points2d:           np.ndarray,  # (C, P, N, 2)
        visibility_mask:    np.ndarray,  # (C, P, N)
        points3d_th:        np.ndarray,  # (N, 3)
        images_sizes_wh:    ArrayLike,   # (C, 2 or 3)
        priors_weight:      float = 0.0,
        simple_focal:       bool = False,
        simple_distortion:  bool = False,
        complex_distortion: bool = False,
        shared:             bool = False,
        fix_intrinsics:     bool = False,
        fix_extrinsics:     bool = False,
        fix_distortion:     bool = False
):

    # Recover the dimensions
    C, P, N = visibility_mask.shape

    images_sizes_wh = np.atleast_2d(images_sizes_wh)

    # Flatten all the optimisable variables into a 1-D array
    x0 = flatten_params(
        camera_matrices,
        distortion_coeffs,
        cam_rvecs,
        cam_tvecs,
        board_rvecs,
        board_tvecs,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared)

    # Set bounds on intrinsics parameters
    lb_intr, ub_intr = intrinsics_bounds(
        images_sizes_wh,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared
    )

    # Complete with (infinite) bounds for other parameters
    num_params = x0.size
    num_extrinsics = C * 6
    num_board_poses = P * 6
    num_intrinsics = num_params - num_extrinsics - num_board_poses
    lower_bounds = np.concatenate([lb_intr.ravel(), np.full(num_extrinsics + num_board_poses, -np.inf)])
    upper_bounds = np.concatenate([ub_intr.ravel(), np.full(num_extrinsics + num_board_poses, np.inf)])

    # fix parameters by setting their bounds to their initial values
    epsilon = 1e-8

    if fix_intrinsics:
        print("[BA] Fixing intrinsic parameters.")
        intr_slice = slice(0, num_intrinsics)
        fixed_values = x0[intr_slice]
        lower_bounds[intr_slice] = fixed_values - epsilon
        upper_bounds[intr_slice] = fixed_values + epsilon

    if fix_extrinsics:
        print("[BA] Fixing camera extrinsic parameters.")
        extr_slice = slice(num_intrinsics, num_intrinsics + num_extrinsics)
        fixed_values = x0[extr_slice]
        lower_bounds[extr_slice] = fixed_values - epsilon
        upper_bounds[extr_slice] = fixed_values + epsilon

    if fix_distortion:
        print("[BA] Fixing distortion parameters to ZERO.")
        # We need to find where the distortion params are in the flattened vector X
        n_k = 3 if simple_focal else 4
        n_d = 4 if simple_distortion else (8 if complex_distortion else 5)

        if shared:
            # Only one block of intrinsics
            start_idx = n_k
            end_idx = n_k + n_d
            lower_bounds[start_idx:end_idx] = 0.0 - epsilon
            upper_bounds[start_idx:end_idx] = 0.0 + epsilon
        else:
            # One block for each camera
            per_cam = n_k + n_d
            for i in range(C):
                cam_offset = i * per_cam
                start_idx = cam_offset + n_k
                end_idx = cam_offset + n_k + n_d
                lower_bounds[start_idx:end_idx] = 0.0 - epsilon
                upper_bounds[start_idx:end_idx] = 0.0 + epsilon

    # Compute residual weighting
    points_weights = residual_weights(points2d, visibility_mask, camera_matrices)  # TODO: Add reproj weight

    rvecs_cam_init = jnp.asarray(cam_rvecs.copy()) if priors_weight > 0.0 else None
    tvecs_cam_init = jnp.asarray(cam_tvecs.copy()) if priors_weight > 0.0 else None

    # Put points2d and related arrays into (P, C, N, ...) layout and push to GPU
    points2d = jnp.transpose(points2d, (1, 0, 2, 3))                # (P, C, N, 2)
    visibility_mask = jnp.transpose(visibility_mask, (1, 0, 2))     # (P, C, N)
    points_weights = jnp.transpose(points_weights, (1, 0, 2))       # (P, C, N)

    # Jacobian sparsity matrix
    jac_sparsity = make_jacobian_sparsity(
        C, P, N,
        intr_size=num_intrinsics,
        add_priors=(priors_weight > 0.0),
        shared=shared
    )

    # with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
        # with CallbackOutputStream(bar):
    result = least_squares(
        cost_func,
        x0,          # x0 contains all the optimisable variables
        verbose=2,
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=200,
        method='trf',
        loss='cauchy',
        f_scale=1.5,
        x_scale='jac',
        jac_sparsity=jac_sparsity,
        args=(
            # All these are passed as a fixed parameters
            points2d,
            visibility_mask,
            points3d_th,
            points_weights,
            priors_weight,
            rvecs_cam_init,
            tvecs_cam_init,
            simple_focal,
            simple_distortion,
            complex_distortion,
            shared
        )
    )

    K_mats_opt, dist_coeffs_opt, cam_rvecs_opt, cam_tvecs_opt, board_rvecs_opt, board_tvecs_opt = unflatten_params(
        result.x,
        C, P,
        simple_focal=simple_focal,
        simple_distortion=simple_distortion,
        complex_distortion=complex_distortion,
        shared=shared)

    ret_vals = {
        'K_opt': K_mats_opt,
        'D_opt': dist_coeffs_opt,
        'cam_r_opt': cam_rvecs_opt,
        'cam_t_opt': cam_tvecs_opt,
        'board_r_opt': board_rvecs_opt,
        'board_t_opt': board_tvecs_opt,
    }
    return result.success, ret_vals