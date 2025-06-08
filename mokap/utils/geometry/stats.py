from typing import Tuple
import jax
import numpy as np
from jax import numpy as jnp

from mokap.utils.geometry.transforms import ZERO_T, ID_QUAT


@jax.jit
def compute_errors_jax(
    observed:           jnp.ndarray,
    reprojected:        jnp.ndarray,
    visibility_mask:    jnp.ndarray,
    points3d_world:     jnp.ndarray,
    points_3d_th:       jnp.ndarray,
    tri_idx:            Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the multi-view reprojection error (i.e. the error in 2D of the reprojected 3D triangulated points)
    for each of C cameras, and the 3D consistency error (i.e. the error in the distances between pairs of points in 3D)

    Args:
        observed: observed points (the full array with nans for missing) (C, N, 2)
        reprojected: the same points after reprojection (C, N, 2)
        visibility_mask: the visibility mask of the N points in the C cameras (C, N)
        points3d_world: the full set of N points in 3D world coordinates (N, 3)
        points_3d_th: the full set of N theoretical point coordinates (N, 3)
        tri_idx: two arrays of length K

    Returns:
        err2d: error in 2D of the reprojected 3D triangulated points (C, N, 2)
        err3d: error in the distances between pairs of points in 3D (K,)
    """

    # mask out missing 2D reprojected errors
    m3 = visibility_mask[..., None]  # (C, N, 1)
    err2d = jnp.where(m3 > 0, reprojected - observed,jnp.nan)  # (C, N, 2)

    # 3D consistency: distances between all pairs for theoretical vs measured
    d_w = jnp.linalg.norm(points3d_world[:, None, :] - points3d_world[None, :, :], axis=-1)     # (N, N)
    d_t = jnp.linalg.norm(points_3d_th[:, None, :] - points_3d_th[None, :, :], axis=-1)     # (N, N)
    i, j = tri_idx
    err3d = d_w[i, j] - d_t[i, j]

    return err2d, err3d


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
            t_med = robust_translation_mean(tvecs, num_iters=4, delta=1.0)   # (3,)

        q_list.append(q_med)
        t_list.append(t_med)

    q_estimates = jnp.stack(q_list, axis=0)     # (C, 4)
    t_estimates = jnp.stack(t_list, axis=0)     # (C, 3)
    return q_estimates, t_estimates


@jax.jit
def quaternion_average(quats: jnp.ndarray) -> jnp.ndarray:
    # Build M = ∑ q_i q_i^T
    M = quats.T @ quats
    # eigh returns (eigenvalues, eigenvectors) sorted ascending by value
    w, V = jnp.linalg.eigh(M)
    top = V[:, -1]  # principal eigenvector
    top = top / (jnp.linalg.norm(top) + 1e-12)
    # force w >= 0
    top = jax.lax.cond(top[0] < 0.0, lambda q: -q, lambda q: q, top)
    return top


def _angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the angle between two unit quaternions q1, q2
    """
    d = jnp.abs(jnp.dot(q1, q2))
    d = jnp.clip(d, -1.0, 1.0)
    return 2.0 * jnp.arccos(d)


def filter_rt_samples(
        rt_stack: jnp.ndarray,
        ang_thresh: float = np.pi / 6.0,    # 30 degrees
        trans_thresh: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Robustly averages a stack of (quaternion, translation) poses
    """

    length = rt_stack.shape[0]

    if length == 0:
        return ID_QUAT, ZERO_T

    quats = rt_stack[:, :4]
    trans = rt_stack[:, 4:]

    # provisional mean
    q_med0 = quaternion_average(quats)
    t_med0 = jnp.median(trans, axis=0)

    # Compute errors
    ang_errs = jax.vmap(lambda q: _angular_distance(q, q_med0))(quats)
    trans_errs = jnp.linalg.norm(trans - t_med0, axis=1)

    # Build inlier mask
    keep_mask = (ang_errs <= ang_thresh) & (trans_errs <= trans_thresh)

    if jnp.sum(keep_mask) > 0:
        # Recompute with inliers
        q_filt = quats[keep_mask]
        t_filt = trans[keep_mask]
        q_med1 = quaternion_average(q_filt)
        t_med1 = robust_translation_mean(t_filt, num_iters=4, delta=1.0)
        return q_med1, t_med1
    else:
        # Fallback to provisional mean
        return q_med0, t_med0
