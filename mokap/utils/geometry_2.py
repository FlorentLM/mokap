from typing import List, Tuple, Optional
import jax.numpy as jnp
import jax
from mokap.utils import geometry_jax


# Pre-allocate identity quaternion constant and zero translation
ID_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
ZERO_T = jnp.zeros((3,), dtype=jnp.float32)


def pad_to_length(
    arr:        jnp.ndarray,
    target_len: int,
    axis:       int = 0,
    pad_value:  Optional[float] = 0.0,
) -> jnp.ndarray:
    """
    Pad an array arr along specified axis up to size target_len by concatenating
    a constant array of shape [..., pad_amt, ...] with pad_value.
    If arr.shape[axis] >= target_len, it returns arr unchanged
    """
    current = arr.shape[axis]
    pad_amt = target_len - current
    if pad_amt <= 0:
        return arr

    pad_shape = list(arr.shape)
    pad_shape[axis] = pad_amt

    filler = jnp.full(pad_shape, pad_value, dtype=arr.dtype)
    return jnp.concatenate([arr, filler], axis=axis)


def triangulation(
        points2d:           jnp.ndarray,
        visibility_mask:    jnp.ndarray,
        rvecs_world:        jnp.ndarray,
        tvecs_world:        jnp.ndarray,
        camera_matrices:    jnp.ndarray,
        dist_coeffs:        jnp.ndarray,
) -> jnp.ndarray:
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

    # TODO: we probably want to get rid of this wrapper and use P matrices anywhere we can

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


@jax.jit
def axisangle_to_quaternion(rvec: jnp.ndarray) -> jnp.ndarray:
    """
    Convert one axis–angle (Rodrigues) vector rvec ∈ ℝ³ into a unit quaternion [w,x,y,z].
    If ‖rvec‖ < eps, returns [1,0,0,0].
    """
    theta = jnp.linalg.norm(rvec)
    eps = 1e-8

    def small_angle_quat():
        return ID_QUAT

    def normal_quat():
        axis = rvec / theta
        half = 0.5 * theta
        w = jnp.cos(half)
        xyz = axis * jnp.sin(half)
        return jnp.concatenate([jnp.array([w], dtype=rvec.dtype), xyz], axis=0)

    return jax.lax.cond(theta < eps, small_angle_quat, normal_quat)

# Batch version maps over axis=0 of an (N, 3) array and returns an (N, 4)
axisangle_to_quaternion_batched = jax.jit(
    jax.vmap(axisangle_to_quaternion, in_axes=0, out_axes=0)
)


@jax.jit
def quaternion_to_axisangle (q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert one quaternion q = [w, x, y, z] to a rvec ∈ ℝ³ (axis–angle).
    If sin(theta/2) ~ 0, it returns [0, 0, 0].
    """
    w, x, y, z = q
    # Clamp w into [-1, 1] to avoid NaNs
    w_clamped = jnp.clip(w, -1.0, 1.0)
    theta = 2.0 * jnp.arccos(w_clamped)
    s2 = 1.0 - w_clamped * w_clamped
    s = jnp.sqrt(s2 + 1e-12)
    eps = 1e-8

    # For numeric stability, branch on whether sin(theta/2) is ~ zero
    def normal_case():
        axis = jnp.array([x, y, z]) / s
        return axis * theta

    def small_case():
        return jnp.zeros((3,), dtype=q.dtype)

    return jax.lax.cond(s2 < (eps * eps), small_case, normal_case)

# Batched version maps over axis=0 of an (N, 4) array and returns a (N, 3)
quaternion_to_axisangle_batched = jax.jit(
    jax.vmap(quaternion_to_axisangle , in_axes=0, out_axes=0)
)


@jax.jit
def _principal_ev4(M: jnp.ndarray, num_iters: int = 6) -> jnp.ndarray:
    """
    Compute the principal (largest) eigenvector of a 4x4 symmetric matrix M with power‐method
    """
    def body_fn(i_v):
        i, v = i_v
        v_next = M @ v
        v_next = v_next / (jnp.linalg.norm(v_next) + 1e-12)
        return (i + 1, v_next)

    def cond_fn(i_v):
        i, _ = i_v
        return i < num_iters

    v0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=M.dtype)
    _, v_star = jax.lax.while_loop(cond_fn, body_fn, (0, v0))
    return v_star


@jax.jit
def quaternion_average(quats: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """
    Compute the Markley‐style average of N unit quaternions via principal eigenvector.

    Args:
        quats: (N,4) array of unit quaternions [w, x, y, z]
        weights: optional (N,) array. If None, it assumes uniform weights = 1/N
    Returns:
        q_avg ∈ ℝ⁴ (unit quaternion)
    """
    # Build 4x4 symmetric matrix M = ∑ w_i * (q_i q_i^T)
    if weights is None:
        M = quats.T @ quats                 # (4, 4)
    else:
        W = jnp.diag(weights)               # (N, N)
        M = quats.T @ (W @ quats)           # (4, 4)

    # Compute principal eigenvector with power‐method
    top = _principal_ev4(M)                 # (4,)
    top_unit = top / (jnp.linalg.norm(top) + 1e-12)

    # Ensure scalar part ≥ 0
    top_unit = jax.lax.cond(top_unit[0] < 0.0,
                             lambda q: -q,
                             lambda q: q,
                             top_unit)
    return top_unit


@jax.jit
def huber_weight(residual_norm: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """
    Compute Huber weight for each ||error||

        w = 1                   if ||error|| <= delta
        w = delta / ||error||   if ||error|| > delta
    """
    return jnp.where(residual_norm <= delta,1.0, delta / (residual_norm + 1e-12))


@jax.jit
def robust_translation_mean(
        t_samples:  jnp.ndarray,
        num_iters:  int = 3,
        delta:      float = 1.0
    ) -> jnp.ndarray:
    """
    Compute a Huber‐weighted mean of M translation samples (M, 3) with IRLS.
    If M == 0, it returns [0, 0, 0].
    """
    M = t_samples.shape[0]

    def no_data():
        return jnp.zeros((3,), dtype=t_samples.dtype)

    def with_data():
        # Initialize t0 as component wise median
        t0 = jnp.median(t_samples, axis=0)          # (3,)

        def body_fn(i_t):
            i, t_curr = i_t
            res = t_samples - t_curr                # (M, 3)
            norms = jnp.linalg.norm(res, axis=1)    # (M,)
            w = huber_weight(norms, delta)    # (M,)
            w = w / (jnp.sum(w) + 1e-12)            # normalize to sum=1
            t_next = jnp.sum(w[:, None] * t_samples, axis=0)  # (3,)
            return i + 1, t_next

        def cond_fn(i_t):
            i, _ = i_t
            return i < num_iters

        _, t_final = jax.lax.while_loop(cond_fn, body_fn, (0, t0))
        return t_final

    return jax.lax.cond(M == 0, no_data, with_data)




def estimate_initial_poses(
    rt_stack_flat: jnp.ndarray,  # (C, M_max, 7): rt vector [q_w, q_x, q_y, q_z, t_x, t_y, t_z]
    lengths:       jnp.ndarray   # (C,) integers
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loops over each camera

    For each camera c:
        Take the first lengths[c] rows of rt_stack_flat[c]
        Split into quaternions (4) and translations (3)
        Compute Markley mean of quaternions
        Compute component wise median of tvecs

    Returns:
        q_estimates: np array (C, 4)
        t_estimates: np array (C, 3)
    """

    C, M_max, _ = rt_stack_flat.shape

    q_list = []
    t_list = []

    for c in range(C):
        M_c = int(lengths[c])       # slice length for camera c
        rt_flat = rt_stack_flat[c]  # (M_max, 7)

        if M_c == 0:
            # No valid samples so return identity quaternion + zero translation
            q_med = ID_QUAT
            t_med = ZERO_T
        else:
            # Normal case, take the first M_c rows
            valid = rt_flat[:M_c, :]    # (M_c, 7)

            # Split into quaternions and translations
            quats = valid[:, :4]        # (M_c, 4)
            tvecs = valid[:, 4:]        # (M_c, 3)

            q_med = quaternion_average(quats)   # (4,)
            t_med = jnp.median(tvecs, axis=0)   # (3,)

        q_list.append(q_med)
        t_list.append(t_med)

    q_estimates = jnp.stack(q_list, axis=0)     # (C, 4)
    t_estimates = jnp.stack(t_list, axis=0)     # (C, 3)
    return q_estimates, t_estimates