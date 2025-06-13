import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple, Optional, Dict, Literal, Union
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from mokap.utils import CallbackOutputStream
from alive_progress import alive_bar

from mokap.utils.geometry.projective import project_object_views_batched
from mokap.utils.geometry.transforms import invert_rtvecs

DistortionModel = Literal['none', 'simple', 'standard', 'full']
DIST_MODEL_MAP = {'none': 0, 'simple': 4, 'standard': 5, 'full': 8}  # TODO: we might want to support rational model too


class Scaler:
    """ A (very) simple class to scale parameters for optimization """
    def __init__(self):
        self.mean = 0.0
        self.scale = 1.0

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """ Calculate scaling factors from x and apply the transform """
        self.mean = np.mean(x)
        self.scale = np.std(x)
        if self.scale < 1e-8:
            self.scale = 1.0
        return (x - self.mean) / self.scale

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        """ Apply the inverse transform using stored mean and scale """
        return x_scaled * self.scale + self.mean

    def transform_bounds(self, lb, ub):
        """ Scales the bounds using stored mean and scale """
        lb_scaled = (lb - self.mean) / self.scale
        ub_scaled = (ub - self.mean) / self.scale
        return lb_scaled, ub_scaled


def _get_parameter_spec(
        nb_cams: int, nb_frames: int,
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
        size = 6 * nb_cams
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
        camera_matrices: ArrayLike,
        dist_coeffs:     ArrayLike,
        cam_rvecs:       ArrayLike,
        cam_tvecs:       ArrayLike,
        board_rvecs:     ArrayLike,
        board_tvecs:     ArrayLike,
        spec: Dict
) -> Tuple[Union[jax.Array, np.ndarray], Dict[str, np.ndarray]]:
    """ Packs parameters into an optimization vector X and a fixed_params dict """

    # the input arrays can be np or jnp and we want to keep them on the device they belong to!
    is_jax = isinstance(camera_matrices, jax.Array)
    np_ = jnp if is_jax else np

    optim_parts = []
    fixed_params = {}
    cfg = spec['config']
    is_shared = cfg['shared_intrinsics'] and cfg['nb_cams'] > 1

    # --- Intrinsics ---
    if 'focal_principal' in spec['blocks']:
        if cfg['fix_aspect_ratio']:
            f = (camera_matrices[:, 0, 0] + camera_matrices[:, 1, 1]) * 0.5
            fp_block = np_.column_stack([f, camera_matrices[:, 0, 2], camera_matrices[:, 1, 2]])
        else:
            fp_block = np_.column_stack([camera_matrices[:, 0, 0], camera_matrices[:, 1, 1], camera_matrices[:, 0, 2],
                                         camera_matrices[:, 1, 2]])
        optim_parts.append(fp_block.mean(axis=0) if is_shared else fp_block.ravel())
    else:
        fixed_params['K'] = camera_matrices

    if 'distortion' in spec['blocks']:
        n_d = cfg['n_d']
        dc = np_.asarray(dist_coeffs)
        if dc.shape[1] < n_d:   # pad if necessary
            pad = np_.zeros((cfg['nb_cams'], n_d - dc.shape[1]), dtype=dc.dtype)
            dc = np_.hstack([dc, pad])
        d_block = dc[:, :n_d]
        optim_parts.append(d_block.mean(axis=0) if is_shared else d_block.ravel())
    else:
        fixed_params['D'] = dist_coeffs

    # --- Extrinsics ---
    if 'extrinsics' in spec['blocks']:
        optim_parts.append(cam_rvecs.ravel())
        optim_parts.append(cam_tvecs.ravel())
    else:
        fixed_params['cam_r'] = cam_rvecs
        fixed_params['cam_t'] = cam_tvecs

    # --- Board Poses ---
    if 'board_poses' in spec['blocks']:
        optim_parts.append(board_rvecs.ravel())
        optim_parts.append(board_tvecs.ravel())
    else:
        fixed_params['board_r'] = board_rvecs
        fixed_params['board_t'] = board_tvecs

    x0 = np_.concatenate(optim_parts) if optim_parts else np_.array([])
    return x0, fixed_params


def _unpack_params(
        x:              ArrayLike,
        fixed_params:   Dict[str, np.ndarray],
        spec: Dict
) -> Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]:
    """ Reconstructs all parameters from the optimization vector X and fixed_params """

    # the input arrays can be np or jnp and we want to keep them on the device they belong to!
    is_jax = isinstance(x, jax.Array)
    np_ = jnp if is_jax else np

    cfg = spec['config']
    C, P = cfg['nb_cams'], cfg['nb_frames']
    is_shared = cfg['shared_intrinsics'] and C > 1

    # --- Intrinsics ---
    K_fixed = fixed_params.get('K')
    if K_fixed is not None and is_jax:   # K in fixed_params might still be a numpy array!
        K_fixed = jnp.asarray(K_fixed)
    K_out = K_fixed if K_fixed is not None else np_.zeros((C, 3, 3), dtype=x.dtype)

    if 'focal_principal' in spec['blocks']:
        info = spec['blocks']['focal_principal']
        fp_flat = x[info['offset']: info['offset'] + info['size']]

        size_per_set = 3 if cfg['fix_aspect_ratio'] else 4
        fp_block = fp_flat.reshape(-1, size_per_set)
        if is_shared:
            fp_block = np_.tile(fp_block, (C, 1))

        # here we need to branch on whether we're using jax or numpy arrays
        if is_jax:
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
        else:
            # in numpy we can do in-place assignment
            if cfg['fix_aspect_ratio']:
                K_out[:, 0, 0] = K_out[:, 1, 1] = fp_block[:, 0]
                K_out[:, 0, 2], K_out[:, 1, 2] = fp_block[:, 1], fp_block[:, 2]
            else:
                K_out[:, 0, 0], K_out[:, 1, 1] = fp_block[:, 0], fp_block[:, 1]
                K_out[:, 0, 2], K_out[:, 1, 2] = fp_block[:, 2], fp_block[:, 3]
            K_out[:, 2, 2] = 1.0

    if 'distortion' in spec['blocks']:
        info = spec['blocks']['distortion']
        n_d = cfg['n_d']
        d_flat = x[info['offset']: info['offset'] + info['size']]
        d_block = d_flat.reshape(-1, n_d)
        if is_shared:
            d_block = np_.tile(d_block, (C, 1))

        D_out = np_.zeros((C, 8), dtype=x.dtype)

        # we need to branch for the assignment here too
        if is_jax:
            D_out = D_out.at[:, :n_d].set(d_block)
        else:
            D_out[:, :n_d] = d_block
    else:
        D_fixed = fixed_params.get('D')
        if D_fixed is not None and is_jax:  # D in fixed_params might still be a numpy array!
            D_fixed = jnp.asarray(D_fixed)
        D_out = D_fixed if D_fixed is not None else np_.zeros((C, 8), dtype=x.dtype)

    # --- Extrinsics ---
    if 'extrinsics' in spec['blocks']:
        info = spec['blocks']['extrinsics']
        extr_flat = x[info['offset']: info['offset'] + info['size']]
        cam_r_out = extr_flat[:3 * C].reshape(C, 3)
        cam_t_out = extr_flat[3 * C:].reshape(C, 3)
    else:
        cam_r_out = np_.asarray(fixed_params['cam_r'])
        cam_t_out = np_.asarray(fixed_params['cam_t'])

    # --- Board Poses ---
    if 'board_poses' in spec['blocks']:
        info = spec['blocks']['board_poses']
        board_flat = x[info['offset']: info['offset'] + info['size']]
        board_r_out = board_flat[:3 * P].reshape(P, 3)
        board_t_out = board_flat[3 * P:].reshape(P, 3)
    else:
        board_r_out = np_.asarray(fixed_params['board_r'])
        board_t_out = np_.asarray(fixed_params['board_t'])

    return K_out, D_out, cam_r_out, cam_t_out, board_r_out, board_t_out


def _get_bounds(
        spec: Dict,
        images_sizes_wh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

                k_lo, k_hi = -1.5, 1.5
                p_lo, p_hi = -0.5, 0.5
                dist_bounds_map = [
                    (k_lo, k_hi), (k_lo, k_hi), (p_lo, p_hi), (p_lo, p_hi),  # k1, k2, p1, p2
                    (k_lo, k_hi), (k_lo, k_hi), (k_lo, k_hi), (k_lo, k_hi)   # k3, k4, k5, k6
                ]

                lb_dist = [b[0] for b in dist_bounds_map[:n_d]]
                ub_dist = [b[1] for b in dist_bounds_map[:n_d]]

                lower_bounds[offset:offset + n_d] = lb_dist
                upper_bounds[offset:offset + n_d] = ub_dist

    # Extrinsics and board poses are left unbounded
    return lower_bounds, upper_bounds


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

    num_residuals = 2 * P * C * N
    if use_priors and 'extrinsics' in spec['blocks']:
        num_residuals += 6 * C

    num_params = spec['total_size']
    S = lil_matrix((num_residuals, num_params), dtype=bool)     # awww lil matrix?

    is_shared = cfg['shared_intrinsics'] and C > 1
    num_intr_sets = 1 if is_shared else C

    # --- Reprojection Error part ---

    # Each observation (p, c, n) depends on
    # intrinsics of camera c
    # extrinsics of camera c
    # pose of board p

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
                    info = spec['blocks']['extrinsics']
                    col_offset = info['offset'] + c * 6
                    S[base_row:base_row + 2, col_offset: col_offset + 6] = 1

                if 'board_poses' in spec['blocks']:
                    info = spec['blocks']['board_poses']
                    col_offset = info['offset'] + p * 6
                    S[base_row:base_row + 2, col_offset: col_offset + 6] = 1

    # --- Priors Part ---
    # The prior residuals depend only on the camera extrinsics
    if use_priors and 'extrinsics' in spec['blocks']:
        base_row = 2 * P * C * N
        extr_info = spec['blocks']['extrinsics']
        for c in range(C):
            row_start = base_row + c * 6 # (3 for rvec, 3 for tvec)
            # it depends on the extrinsics parameters for camera c
            col_start = extr_info['offset'] + c * 6
            S[row_start:row_start + 6, col_start:col_start + 6] = 1

    return S.tocsr()


def jax_cost_function(
        params,     # this is the only variable that'll changes
        fixed_params,
        spec,
        points2d:           jnp.ndarray,
        visibility_mask:    jnp.ndarray,
        points3d_th_jnp:    jnp.ndarray,
        points_weights:     jnp.ndarray,
        prior_weight:       float,
        rvecs_cam_init:     jnp.ndarray,
        tvecs_cam_init:     jnp.ndarray,
):
    # unpack
    Ks, Ds, cam_r, cam_t, board_r, board_t = _unpack_params(
        params, fixed_params, spec
    )

    # invert camera poses to get world-to-camera transforms
    r_w2c, t_w2c = invert_rtvecs(cam_r, cam_t)

    # Project all points (i.e. it projects object points for many different poses into many different cameras)
    reproj = project_object_views_batched(
        points3d_th_jnp,
        r_w2c, t_w2c,
        board_r, board_t,
        Ks, Ds
    )

    # Compute residuals
    resid = reproj - points2d  # Assumes points2d_jnp is (C, P, N, 2)
    weighted_resid = jnp.where(visibility_mask[..., None], resid * points_weights[..., None], 0.0)

    # Printing is a side effect, so we use have to use jax.debug.print
    # (it will only print when the function is executed, not during tracing)
    visible_error_magnitudes = jnp.where(visibility_mask, jnp.linalg.norm(resid, axis=-1), jnp.nan)
    mean_reprojection_error_px = jnp.nanmean(visible_error_magnitudes)
    jax.debug.print("Mean Reprojection Error: {x:.2f}px", x=mean_reprojection_error_px)

    all_residuals = [weighted_resid.ravel()]
    if prior_weight > 0.0 and 'extrinsics' in spec['blocks']:
        rvec_resid = cam_r - rvecs_cam_init
        tvec_resid = cam_t - tvecs_cam_init
        prior_resid = jnp.concatenate([rvec_resid.ravel(), tvec_resid.ravel()]) * prior_weight
        all_residuals.append(prior_resid)

    return jnp.concatenate(all_residuals)


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
        priors_weight:          float = 0.0,
        radial_penalty:         float = 2.0,
        fix_focal_principal:    bool = False,
        fix_distortion:         bool = False,
        fix_extrinsics:         bool = False,
        fix_board_poses:        bool = False,
        fix_aspect_ratio:       bool = False,
        shared_intrinsics:      bool = False,
        distortion_model:       DistortionModel = 'standard',
) -> Tuple[bool, Dict]:

    C, P, N = visibility_mask.shape
    images_sizes_wh = np.atleast_2d(images_sizes_wh)

    spec = _get_parameter_spec(
        nb_cams=C, nb_frames=P,
        fix_focal_principal=fix_focal_principal, fix_distortion=fix_distortion,
        fix_extrinsics=fix_extrinsics, fix_board_poses=fix_board_poses,
        fix_aspect_ratio=fix_aspect_ratio, shared_intrinsics=shared_intrinsics,
        distortion_model=distortion_model
    )
    # Add nb_points to spec for jacobian sparsity calculation
    spec['config']['nb_points'] = N

    x0, fixed_params = _pack_params(
        camera_matrices, distortion_coeffs,
        cam_rvecs,
        cam_tvecs,
        board_rvecs,
        board_tvecs,
        spec
    )

    # We need it as a numpy array, to pass to scipy
    x0 = np.asarray(x0)

    # --- Bounds ---
    lb, ub = _get_bounds(spec, images_sizes_wh) # bounds are also computed as np
    x0 = np.clip(x0, lb, ub)

    scaler = Scaler()
    x0_scaled = scaler.fit_transform(x0)
    lb_scaled, ub_scaled = scaler.transform_bounds(lb, ub)

    # Jac sparsity matrix
    jac_sparsity = make_jacobian_sparsity(spec, use_priors=priors_weight > 0.0)

    # Setup
    points_weights = residual_weights(
        pts2d=image_points2d,
        visibility_mask=visibility_mask,
        camera_matrices=camera_matrices,
        distance_falloff_gamma=radial_penalty
    )   # (C, P, N)

    rvecs_cam_init = cam_rvecs.copy() if priors_weight > 0.0 else None
    tvecs_cam_init = cam_tvecs.copy() if priors_weight > 0.0 else None

    # we define the "closure" for the cost function: wraps the JAX cost function with the data that is constant
    def cost_closure(params_unscaled):
        return jax_cost_function(
            params_unscaled,
            fixed_params, spec,
            image_points2d, visibility_mask,
            object_points3d,
            points_weights,
            priors_weight,
            rvecs_cam_init,
            tvecs_cam_init
        )

    # wrapped functions for SciPy: scaling and conversion back to np
    def scipy_cost_wrapper(params_scaled):
        params_unscaled = scaler.inverse_transform(params_scaled)
        residuals_jnp = cost_closure(params_unscaled)
        return np.array(residuals_jnp, copy=True)

    # we JIT-compile the Jacobian function
    # Note: it differentiates the closure clost func, which works on UNSCALED parameters
    jitted_jac_func = jax.jit(jax.jacfwd(cost_closure))

    def scipy_jac_wrapper(params_scaled):
        params_unscaled = scaler.inverse_transform(params_scaled)
        # sooo the Jacobian of the scaled function is related to the unscaled one by the chain rule:
        # J(f(g(x))) = J(f)(g(x)) * J(g)(x)

        # with  g(x) for the inverse_transform
        #       f(x) for cost_closure
        #       J(g)(x) for just a diagonal matrix of scaler.scale
        jac_unscaled = jitted_jac_func(params_unscaled)
        jac_scaled = np.array(jac_unscaled * scaler.scale, copy=True)  # apply chain rule
        return jac_scaled

    # --- Call the solver ---
    result = least_squares(
        scipy_cost_wrapper,              # Scipy expects numpy arrays so this wrapper just does that
        x0_scaled,                       # x0 is already numpy
        verbose=2,
        bounds=(lb_scaled, ub_scaled),   # bounds are just scaled once that's it
        jac_sparsity=jac_sparsity,       # sparsity matrix is also computed once and that's it

        jac=scipy_jac_wrapper,           # this just applies the gradien to our scaled parameters with the chain rule

        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        max_nfev=500,

        method='trf',
        loss='cauchy', f_scale=2.5,
        args=()
    )

    # --- And finally just unscale the result ---
    x_final_unscaled = scaler.inverse_transform(result.x)

    K_opt, D_opt, cam_r_opt, cam_t_opt, board_r_opt, board_t_opt = _unpack_params(
        x_final_unscaled, fixed_params, spec
    )

    ret_vals = {
        'K_opt': K_opt, 'D_opt': D_opt,
        'cam_r_opt': cam_r_opt, 'cam_t_opt': cam_t_opt,
        'board_r_opt': board_r_opt, 'board_t_opt': board_t_opt,
    }
    return result.success, ret_vals
