import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Dict, Literal, Optional
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from mokap.utils import CallbackOutputStream
from mokap.utils.datatypes import DistortionModel
from alive_progress import alive_bar

from mokap.utils.geometry.projective import project_object_views_batched, reprojection_errors, distortion
from mokap.utils.geometry.transforms import invert_rtvecs, extrinsics_matrix

DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'full': 8, 'rational': 8}


def scale_params(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Calculate scaling factors from x and apply the transform """
    mean = jnp.mean(x)
    scale = jnp.std(x)
    scale = jnp.where(scale < 1e-8, 1.0, scale)
    x_scaled = (x - mean) / scale
    return x_scaled, mean, scale

def unscale_params(x_scaled: jnp.ndarray, mean: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """ Apply the inverse transform using given mean and scale """
    return x_scaled * scale + mean

def scale_bounds(lb, ub, mean, scale):
    """ Scales the bounds using given mean and scale """
    lb_scaled = (lb - mean) / scale
    ub_scaled = (ub - mean) / scale
    return lb_scaled, ub_scaled


def _get_parameter_spec(
        nb_cams: int, nb_frames: int, origin_idx: int,
        fix_focal_principal: bool, fix_distortion: bool,
        fix_extrinsics: bool, fix_board_poses: bool,
        fix_aspect_ratio: bool, shared_intrinsics: bool,
        distortion_model: DistortionModel
) -> Dict:
    """
    defines the structure of the optimization vector X
    (the size and offset for each block of parameters to optimise)
    """
    spec = {'config': locals()}
    spec['blocks'] = {}
    current_offset = 0

    is_shared = shared_intrinsics and nb_cams > 1
    num_intr_sets = 1 if is_shared else nb_cams

    # --- Focal Length and Principal Point ---
    if not fix_focal_principal:
        size_per_set = 3 if fix_aspect_ratio else 4
        size = size_per_set * num_intr_sets
        spec['blocks']['focal_principal'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # --- Distortion coefficients ---
    n_d = DIST_MODEL_MAP[distortion_model]
    spec['config']['n_d'] = n_d
    if not fix_distortion and n_d > 0:
        size = n_d * num_intr_sets
        spec['blocks']['distortion'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # --- Camera extrinsics ---
    if not fix_extrinsics:
        # We optimize for all cameras except the origin camera
        num_optim_cams = nb_cams - 1
        size = 6 * num_optim_cams  # 6 params (3 rvec, 3 tvec) per camera
        spec['blocks']['extrinsics'] = {'offset': current_offset, 'size': size}
        current_offset += size

    # --- Board Poses ---
    if not fix_board_poses:
        size = 6 * nb_frames
        spec['blocks']['board_poses'] = {'offset': current_offset, 'size': size}
        current_offset += size

    spec['total_size'] = current_offset
    return spec


def _pack_params(
        camera_matrices: jnp.ndarray,
        dist_coeffs:     jnp.ndarray,
        cam_rvecs:       jnp.ndarray,
        cam_tvecs:       jnp.ndarray,
        board_rvecs:     jnp.ndarray,
        board_tvecs:     jnp.ndarray,
        spec: Dict
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """ Packs parameters into an optimization vector X and a fixed_params dict """

    optim_parts = []
    fixed_params = {}
    cfg = spec['config']
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    # --- Intrinsics ---
    # Camera matrix  packing
    if 'focal_principal' in spec['blocks']:
        if cfg['fix_aspect_ratio']:
            f = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) * 0.5
            fp_block = jnp.column_stack([f, camera_matrices[:, 0, 2], camera_matrices[:, 1, 2]])
        else:
            fp_block = jnp.column_stack([camera_matrices[:, 0, 0], camera_matrices[:, 1, 1], camera_matrices[:, 0, 2],
                                         camera_matrices[:, 1, 2]])
        optim_parts.append(jnp.mean(fp_block, axis=0) if is_shared else fp_block.ravel())
    else:
        fixed_params['K'] = camera_matrices

    # distortion packing
    if 'distortion' in spec['blocks']:
        n_d = cfg['n_d']
        d_block = dist_coeffs[:, :n_d]
        optim_parts.append(jnp.mean(d_block, axis=0) if is_shared else d_block.ravel())
    else:
        fixed_params['D'] = dist_coeffs

    # --- Extrinsics ---
    # Always store the full initial arrays in fixed_params so _unpack_params always has a reference for shape and fixed values
    fixed_params['cam_r'] = cam_rvecs
    fixed_params['cam_t'] = cam_tvecs

    if 'extrinsics' in spec['blocks']:
        # If we are optimizing, we add the relevant parts to the optim_parts list.
        origin_idx = spec['config'].get('origin_idx', 0)
        cam_mask = jnp.arange(cfg['nb_cams']) != origin_idx

        optim_parts.append(cam_rvecs[cam_mask].ravel())
        optim_parts.append(cam_tvecs[cam_mask].ravel())

        # Also store the fixed origin pose separately for convenience in unpacking
        fixed_params['origin_r'] = cam_rvecs[origin_idx]
        fixed_params['origin_t'] = cam_tvecs[origin_idx]
    # if fix_extrinsics=True the full arrays remain in fixed_params, all good

    # --- Board Poses ---
    if 'board_poses' in spec['blocks']:
        optim_parts.append(board_rvecs.ravel())
        optim_parts.append(board_tvecs.ravel())
    else:
        fixed_params['board_r'] = board_rvecs
        fixed_params['board_t'] = board_tvecs

    x0 = jnp.concatenate(optim_parts) if optim_parts else jnp.array([])
    return x0, fixed_params


def _unpack_params(
        x:              jnp.ndarray,
        fixed_params:   Dict[str, jnp.ndarray],
        spec: Dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Reconstructs all parameters from the optimization vector X and fixed_params """

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # --- Intrinsics ---
    K_out = fixed_params.get('K', jnp.zeros((C, 3, 3), dtype=x.dtype))

    if 'focal_principal' in spec['blocks']:
        info = spec['blocks']['focal_principal']
        fp_flat = x[info['offset']: info['offset'] + info['size']]

        size_per_set = 3 if cfg['fix_aspect_ratio'] else 4
        fp_block = fp_flat.reshape(-1, size_per_set)
        if is_shared:
            fp_block = jnp.tile(fp_block, (C, 1))

        if cfg['fix_aspect_ratio']:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 0])
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 1])
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 2])
        else:
            K_out = K_out.at[:, 0, 0].set(fp_block[:, 0])
            K_out = K_out.at[:, 1, 1].set(fp_block[:, 1])
            K_out = K_out.at[:, 0, 2].set(fp_block[:, 2])
            K_out = K_out.at[:, 1, 2].set(fp_block[:, 3])
        K_out = K_out.at[:, 2, 2].set(1.0)

    if 'distortion' in spec['blocks']:
        info = spec['blocks']['distortion']
        n_d = cfg['n_d']
        d_flat = x[info['offset']: info['offset'] + info['size']]
        d_block = d_flat.reshape(-1, n_d)
        if is_shared:
            d_block = jnp.tile(d_block, (C, 1))

        D_out = jnp.zeros((C, 8), dtype=x.dtype)
        D_out = D_out.at[:, :n_d].set(d_block)
    else:
        D_out = fixed_params.get('D', jnp.zeros((C, 8), dtype=x.dtype))

    # --- Extrinsics ---
    if 'extrinsics' in spec['blocks']:
        origin_idx = cfg.get('origin_idx', 0)
        info = spec['blocks']['extrinsics']
        num_optim_cams = C - 1

        extr_flat = x[info['offset']: info['offset'] + info['size']]
        r_optim = extr_flat[:3 * num_optim_cams].reshape(num_optim_cams, 3)
        t_optim = extr_flat[3 * num_optim_cams:].reshape(num_optim_cams, 3)

        # Create placeholders for the full arrays
        cam_r_out = jnp.zeros_like(fixed_params['cam_r'])
        cam_t_out = jnp.zeros_like(fixed_params['cam_t'])

        # Insert the optimized parameters for cameras before the origin_idx
        cam_r_out = cam_r_out.at[:origin_idx].set(r_optim[:origin_idx])
        cam_t_out = cam_t_out.at[:origin_idx].set(t_optim[:origin_idx])

        # Insert the fixed origin pose
        cam_r_out = cam_r_out.at[origin_idx].set(fixed_params['origin_r'])
        cam_t_out = cam_t_out.at[origin_idx].set(fixed_params['origin_t'])

        # Insert the optimized parameters for cameras AFTER the origin_idx
        cam_r_out = cam_r_out.at[origin_idx + 1:].set(r_optim[origin_idx:])
        cam_t_out = cam_t_out.at[origin_idx + 1:].set(t_optim[origin_idx:])

    # --- Board Poses ---
    if 'board_poses' in spec['blocks']:
        info = spec['blocks']['board_poses']
        board_flat = x[info['offset']: info['offset'] + info['size']]
        board_r_out = board_flat[:3 * P].reshape(P, 3)
        board_t_out = board_flat[3 * P:].reshape(P, 3)
    else:
        board_r_out = fixed_params['board_r']
        board_t_out = fixed_params['board_t']

    return K_out, D_out, cam_r_out, cam_t_out, board_r_out, board_t_out


def _get_bounds(
        spec: Dict,
        images_sizes_wh: ArrayLike
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes lower and upper bounds for the optimization variables based on the spec
    """
    cfg = spec['config']
    C = cfg['nb_cams']
    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C

    # initialize with -inf, +inf for all parameters
    lower_bounds = np.full(spec['total_size'], -np.inf, dtype=np.float64)
    upper_bounds = np.full(spec['total_size'], np.inf, dtype=np.float64)

    # --- Set bounds for Intrinsics ---
    if 'focal_principal' in spec['blocks'] or 'distortion' in spec['blocks']:
        for i in range(num_intr_sets):
            cam_idx = 0 if is_shared else i
            w, h = images_sizes_wh[cam_idx]

            # --- Focal Length and Principal Point ---
            if 'focal_principal' in spec['blocks']:
                info = spec['blocks']['focal_principal']
                size_per_set = info['size'] // num_intr_sets
                offset = info['offset'] + i * size_per_set

                f_lo, f_hi = 100.0, 100000.0
                cx_lo, cx_hi = w / 2.0 - w * 0.2, w / 2.0 + w * 0.2
                cy_lo, cy_hi = h / 2.0 - h * 0.2, h / 2.0 + h * 0.2

                if cfg['fix_aspect_ratio']:
                    # params are [f, cx, cy]
                    lower_bounds[offset:offset + 3] = [f_lo, cx_lo, cy_lo]
                    upper_bounds[offset:offset + 3] = [f_hi, cx_hi, cy_hi]
                else:
                    # params are [fx, fy, cx, cy]
                    lower_bounds[offset:offset + 4] = [f_lo, f_lo, cx_lo, cy_lo]
                    upper_bounds[offset:offset + 4] = [f_hi, f_hi, cx_hi, cy_hi]

            # --- Distortion coefficients ---
            if 'distortion' in spec['blocks']:
                info = spec['blocks']['distortion']
                n_d = cfg['n_d']
                offset = info['offset'] + i * n_d

                # Define bounds for all 8 potential coefficients
                k_lo, k_hi = -1.5, 1.5
                p_lo, p_hi = -0.5, 0.5
                k_higher_order_lo, k_higher_order_hi = -0.5, 0.5  # Tighter bounds for higher order

                dist_bounds_map = [
                    (k_lo, k_hi), (k_lo, k_hi),         # k1, k2
                    (p_lo, p_hi), (p_lo, p_hi),         # p1, p2
                    (k_lo, k_hi),                       # k3
                    (k_higher_order_lo, k_higher_order_hi),     # k4
                    (k_higher_order_lo, k_higher_order_hi),     # k5
                    (k_higher_order_lo, k_higher_order_hi)      # k6
                ]

                lb_dist = [b[0] for b in dist_bounds_map[:n_d]]
                ub_dist = [b[1] for b in dist_bounds_map[:n_d]]

                lower_bounds[offset:offset + n_d] = lb_dist
                upper_bounds[offset:offset + n_d] = ub_dist

    # Extrinsics and board poses are left unbounded
    return jnp.array(lower_bounds), jnp.array(upper_bounds)


def residual_weights(
        pts2d:                      jnp.ndarray,  # (C, P, N, 2)
        visibility_mask:            jnp.ndarray,  # (C, P, N)
        camera_matrices:            jnp.ndarray,  # (C, 3, 3)
        reproj_error:               Optional[jnp.ndarray] = None,  # (C, P, N) or None
        distance_falloff_gamma:     float = 2.0
) -> jnp.ndarray:
    """
    Compute per-observation weights for BA residuals based on
        - Visibility
        - Distance from image center
        - Number of views per point
        - Reprojection error (optional)
    """

    C, P, N, _ = pts2d.shape

    # Distance to center
    cx = camera_matrices[:, 0, 2][:, None, None]  # (C, 1, 1)
    cy = camera_matrices[:, 1, 2][:, None, None]
    center = jnp.stack([cx, cy], axis=-1)  # (C, 1, 1, 2)

    dists = jnp.linalg.norm(pts2d - center, axis=-1)  # (C, P, N)
    max_dist = jnp.sqrt(cx[:, 0, 0] ** 2 + cy[:, 0, 0] ** 2)[:, None, None] + 1e-8  # (C,)

    dist_weight = 1.0 / (1.0 + (dists / max_dist) ** distance_falloff_gamma)  # (C, P, N)

    # Weighting with the number of cameras seeing each point
    nb_views = jnp.sum(visibility_mask, axis=0)  # (P, N)
    nb_views_weight = jnp.tile(nb_views[None, :, :], (C, 1, 1))  # (C, P, N)
    nb_views_weight = nb_views_weight / (1.0 + nb_views_weight)

    # Optional reprojection weight
    if reproj_error is not None:
        reproj_weight = 1.0 / (1.0 + reproj_error)
        reproj_weight = jnp.clip(reproj_weight, 0.1, 1.0)
    else:
        reproj_weight = jnp.ones_like(visibility_mask, dtype=jnp.float32)

    # combine
    weights = (
            visibility_mask.astype(np.float32) *
            dist_weight *
            nb_views_weight *
            reproj_weight
    )
    # norm
    weights /= (jnp.max(weights) + 1e-8)

    return weights


def make_jacobian_sparsity(
        spec:       Dict,
        use_priors: bool
) -> csr_matrix:
    """
    Creates the Jacobian sparsity matrix based on the optimization parameter specification
    """

    cfg = spec['config']
    C, P, N = cfg['nb_cams'], cfg['nb_frames'], cfg['nb_points']
    origin_idx = cfg.get('origin_idx', 0)  # Get the origin_idx

    num_residuals = 2 * P * C * N
    if use_priors and 'extrinsics' in spec['blocks']:
        num_residuals += 6 * C

    num_params = spec['total_size']
    S = lil_matrix((num_residuals, num_params), dtype=bool)     # awww lil matrix

    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C

    # --- Reprojection Error ---

    # Each observation (p, c, n) depends on
    # intrinsics of camera c
    # extrinsics of camera c
    # pose of board p

    # Create a mapping from camera index to its position in the optimization vector
    optim_cam_indices = np.delete(np.arange(C), origin_idx)
    cam_idx_to_optim_pos = {cam_idx: pos for pos, cam_idx in enumerate(optim_cam_indices)}

    for p in range(P):
        for c in range(C):
            for n in range(N):
                # row index for the residual of point n in frame p, camera c
                base_row = (p * C * N + c * N + n) * 2

                intr_set_idx = 0 if is_shared else c
                if 'focal_principal' in spec['blocks']:
                    info = spec['blocks']['focal_principal']
                    size_per_set = info['size'] // num_intr_sets
                    col_offset = info['offset'] + intr_set_idx * size_per_set
                    S[base_row:base_row + 2, col_offset: col_offset + size_per_set] = 1

                if 'distortion' in spec['blocks']:
                    info = spec['blocks']['distortion']
                    n_d = cfg['n_d']
                    size_per_set = info['size'] // num_intr_sets
                    col_offset = info['offset'] + intr_set_idx * size_per_set
                    S[base_row:base_row + 2, col_offset: col_offset + n_d] = 1

                if 'extrinsics' in spec['blocks']:
                    # Check if the current camera 'c' is the origin
                    if c != origin_idx:
                        # If not the origin, it's an optimizable parameter. Find its column.
                        optim_pos = cam_idx_to_optim_pos[c]
                        info = spec['blocks']['extrinsics']
                        col_offset = info['offset'] + optim_pos * 6
                        S[base_row:base_row + 2, col_offset: col_offset + 6] = 1
                    # If c is the origin, its parameters are fixed and not in the Jacobian so leaving that part of the row as zero

                if 'board_poses' in spec['blocks']:
                    info = spec['blocks']['board_poses']
                    col_offset = info['offset'] + p * 6
                    S[base_row:base_row + 2, col_offset: col_offset + 6] = 1

    # --- Priors ---
    # The prior residuals depend only on the camera extrinsics
    if use_priors and 'extrinsics' in spec['blocks']:
        base_row = 2 * P * C * N
        extr_info = spec['blocks']['extrinsics']
        for c in range(C):
            if c != origin_idx:  # Only for optimizable cameras
                row_start = base_row + c * 6
                optim_pos = cam_idx_to_optim_pos[c]
                col_start = extr_info['offset'] + optim_pos * 6
                S[row_start:row_start + 6, col_start:col_start + 6] = 1

    return S.tocsr()


def cost_function(
        params:             jnp.ndarray,  # The 1D optimization vector
        fixed_params:       Dict,
        spec:               Dict,
        points2d:           jnp.ndarray,
        visibility_mask:    jnp.ndarray,
        points3d_th_jnp:    jnp.ndarray,
        points_weights:     jnp.ndarray,
        priors_weight:      float,
        distortion_model:   DistortionModel,
) -> jnp.ndarray:

    Ks, Ds, cam_r, cam_t, board_r, board_t = _unpack_params(params, fixed_params, spec)

    # Reprojection residuals
    r_w2c, t_w2c = invert_rtvecs(cam_r, cam_t)

    reproj, valid_depth_mask = project_object_views_batched(
        points3d_th_jnp, r_w2c, t_w2c, board_r, board_t,
        Ks, Ds, distortion_model=distortion_model
    )

    resid = reproj - points2d

    # Combine the pre-computed weights with the dynamic depth-validity weight
    effective_weights = points_weights * valid_depth_mask

    # Apply the combined weights to the residual
    # The jnp.where is no longer needed because a weight of 0 achieves the same goal.
    weighted_reproj_resid = resid * effective_weights[..., None]

    num_weighted_points = jnp.sum(effective_weights > 0)
    total_sum_sq_err = jnp.sum(jnp.square(weighted_reproj_resid))

    # jax prints work in there?? woah
    rms_error = jnp.sqrt(total_sum_sq_err / jnp.maximum(1, 2 * num_weighted_points))
    jax.debug.print("Mean Reprojection Error (RMS): {x:.3f}px", x=rms_error)

    # Prior residuals
    rvecs_cam_init = fixed_params['cam_r']
    tvecs_cam_init = fixed_params['cam_t']
    rvec_resid = cam_r - rvecs_cam_init
    tvec_resid = cam_t - tvecs_cam_init
    weighted_prior_resid = jnp.concatenate([rvec_resid.ravel(), tvec_resid.ravel()]) * priors_weight

    return jnp.concatenate([weighted_reproj_resid.ravel(), weighted_prior_resid])

def run_bundle_adjustment(
        camera_matrices:        jnp.ndarray,
        distortion_coeffs:      jnp.ndarray,
        cam_rvecs:              jnp.ndarray,
        cam_tvecs:              jnp.ndarray,
        board_rvecs:            jnp.ndarray,
        board_tvecs:            jnp.ndarray,
        image_points2d:         jnp.ndarray,   # (C, P, N, 2)
        visibility_mask:        jnp.ndarray,   # (C, P, N)
        object_points3d:        jnp.ndarray,
        images_sizes_wh:        ArrayLike,
        origin_idx:             int = 0,
        priors_weight:          float = 0.0,
        radial_penalty:         float = 2.0,
        fix_focal_principal:    bool = False,
        fix_distortion:         bool = False,
        fix_extrinsics:         bool = False,
        fix_board_poses:        bool = False,
        fix_aspect_ratio:       bool = False,
        shared_intrinsics:      bool = False,
        distortion_model:       DistortionModel = 'standard',
        max_frames:             Optional[int] = None
) -> Tuple[bool, Dict]:

    C, P_full, N = visibility_mask.shape

    # if an override is given, use it (this avoids reshaping large arrays)
    P = max_frames if max_frames is not None else P_full

    images_sizes_wh = np.atleast_2d(images_sizes_wh)

    spec = _get_parameter_spec(
        nb_cams=C, nb_frames=P, origin_idx=origin_idx,
        fix_focal_principal=fix_focal_principal, fix_distortion=fix_distortion,
        fix_extrinsics=fix_extrinsics, fix_board_poses=fix_board_poses,
        fix_aspect_ratio=fix_aspect_ratio, shared_intrinsics=shared_intrinsics,
        distortion_model=distortion_model
    )
    # Add nb_points to spec for jacobian sparsity calculation
    spec['config']['nb_points'] = N

    # Prepare and pad distortion coefficients if necessary
    if 'distortion' in spec['blocks']:
        n_d = spec['config']['n_d']
        current_d = distortion_coeffs.shape[1]
        if current_d < n_d:
            padding_config = ((0, 0), (0, n_d - current_d))
            distortion_coeffs = jnp.pad(distortion_coeffs, padding_config, mode='constant')

    x0, fixed_params = _pack_params(
        camera_matrices,
        distortion_coeffs,
        cam_rvecs,
        cam_tvecs,
        board_rvecs,
        board_tvecs,
        spec
    )

    # --- Bounds and Scaling ---
    lb, ub = _get_bounds(spec, images_sizes_wh)
    x0 = jnp.clip(x0, lb, ub)
    x0_scaled, mean, scale = scale_params(x0)
    lb_scaled, ub_scaled = scale_bounds(lb, ub, mean, scale)

    # --- Prepare for scipy: Convert to numpy ---
    x0_scaled_np = np.asarray(x0_scaled)
    lb_scaled_np, ub_scaled_np = np.asarray(lb_scaled), np.asarray(ub_scaled)

    jac_sparsity = make_jacobian_sparsity(spec, use_priors=priors_weight > 0.0)

    # --- Setup static arguments for the residual function ---
    points_weights = residual_weights(
        pts2d=image_points2d,
        visibility_mask=visibility_mask,
        camera_matrices=camera_matrices,
        distance_falloff_gamma=radial_penalty
    )   # (C, P, N)

    # Create the partial function, baking in all static data
    residuals_fn_partial = partial(
        cost_function,
        fixed_params=fixed_params,
        spec=spec,
        points2d=image_points2d,
        visibility_mask=visibility_mask,
        points3d_th_jnp=object_points3d,
        points_weights=points_weights,
        priors_weight=priors_weight,
        distortion_model=distortion_model
    )

    jitted_cost_func = jax.jit(residuals_fn_partial)
    jitted_jac_func = jax.jit(jax.jacfwd(residuals_fn_partial))

    # --- Create Wrappers for SciPy ---
    def scipy_cost_wrapper(params_scaled_np):
        params_unscaled = unscale_params(jnp.asarray(params_scaled_np), mean, scale)
        residuals = jitted_cost_func(params_unscaled)
        return np.asarray(residuals).copy() # copy is necessary becasuse JAX returns a view otherwise

    def scipy_jac_wrapper(params_scaled_np):
        params_unscaled = unscale_params(jnp.asarray(params_scaled_np), mean, scale)
        jac_unscaled = jitted_jac_func(params_unscaled)

        # Jacobian of f(g(x)) is J(f)(g(x)) * J(g)(x)
        # ... and Jacobian of g(x) is just the scale
        jac_scaled = jac_unscaled * scale  # chain rule
        return np.asarray(jac_scaled).copy()  # copy necessaruy here too

    # --- Call the scipy solver ---
    # with alive_bar(title='Bundle adjustment...', length=20, force_tty=True) as bar:
    #     with CallbackOutputStream(bar, keep_stdout=False):
    result = least_squares(
        scipy_cost_wrapper,
        x0_scaled_np,
        verbose=2,
        bounds=(lb_scaled_np, ub_scaled_np),
        jac=scipy_jac_wrapper,
        jac_sparsity=jac_sparsity,
        method='trf',
        loss='cauchy', f_scale=2.5,
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        max_nfev=500
    )

    # --- Unscale and unpack results ---
    x_final_unscaled = unscale_params(jnp.asarray(result.x), mean, scale)

    K_opt, D_opt, cam_r_opt, cam_t_opt, board_r_opt, board_t_opt = _unpack_params(
        x_final_unscaled, fixed_params, spec
    )

    ret_vals = {
        'K_opt': K_opt, 'D_opt': D_opt,
        'cam_r_opt': cam_r_opt, 'cam_t_opt': cam_t_opt,
        'board_r_opt': board_r_opt, 'board_t_opt': board_t_opt,
    }

    return result.success, ret_vals
