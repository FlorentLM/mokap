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
def quaternion_average(quats: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """
    Computes the average of a set of quaternions using Markley's method
    Handles the q/-q ambiguity by aligning all quaternions to a reference
    Accepts optional weights for each quaternion
    """

    # If no weights provided, use uniform weights
    if weights is None:
        weights = jnp.ones(quats.shape[0], dtype=quats.dtype)

    # Normalize weights to sum to 1 to avoid numerical issues
    weights = weights / (jnp.sum(weights) + 1e-8)

    # Pick the quaternion with the highest weight as the reference
    q_ref = quats[jnp.argmax(weights)]

    # Align all other quaternions to the reference
    dots = jnp.einsum('i,ji->j', q_ref, quats)
    flip = jnp.sign(dots)
    quats_aligned = quats * flip[:, None]

    # Build the weighted M matrix: M = ∑ w_i * q_i * q_i^T
    # We can do this with broadcasting and einsum.
    # q_aligned is (N, 4). We want to compute an (N, 4, 4) matrix and sum over N
    M = jnp.einsum('i,ij,ik->jk', weights, quats_aligned, quats_aligned)

    _, V = jnp.linalg.eigh(M)
    avg_quat = V[:, -1]

    # Ensure w >= 0 for a canonical representation
    return jax.lax.cond(avg_quat[0] < 0.0, lambda q: -q, lambda q: q, avg_quat)


def _angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the angle between two unit quaternions q1, q2
    """
    d = jnp.abs(jnp.dot(q1, q2))
    d = jnp.clip(d, -1.0, 1.0)
    return 2.0 * jnp.arccos(d)


def filter_rt_samples(
        rt_stack: jnp.ndarray,
        ang_thresh: float = np.pi / 6.0,
        trans_thresh: float = 1.0,
        num_iters: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
    """
    Robustly averages a stack of (quaternion, translation) poses using IRLS
    """
    length = rt_stack.shape[0]

    def fail_case():
        return ID_QUAT, ZERO_T, False

    def success_case():
        quats = rt_stack[:, :4]
        trans = rt_stack[:, 4:]

        # Initial guess
        q_curr = quaternion_average(quats)
        t_curr = jnp.median(trans, axis=0)

        def body_fn(i, qt_curr):
            q_c, t_c = qt_curr
            # Compute errors from current estimate
            ang_errs = jax.vmap(lambda q: _angular_distance(q, q_c))(quats)
            trans_errs = jnp.linalg.norm(trans - t_c, axis=1)

            weights = (ang_errs <= ang_thresh) & (trans_errs <= trans_thresh)
            weights = weights.astype(jnp.float32)

            has_inliers = jnp.sum(weights) > 0

            def update_estimate():
                q_next = quaternion_average(quats, weights=weights)

                # we use the same weights for the translation mean
                w_norm = weights / jnp.sum(weights)
                t_next = jnp.sum(w_norm[:, None] * trans, axis=0)
                return q_next, t_next

            def keep_estimate():
                return q_c, t_c

            q_next, t_next = jax.lax.cond(has_inliers, update_estimate, keep_estimate)
            return q_next, t_next

        q_final, t_final = jax.lax.fori_loop(0, num_iters, body_fn, (q_curr, t_curr))

        # Final check for inliers to determine success
        ang_errs = jax.vmap(lambda q: _angular_distance(q, q_final))(quats)
        trans_errs = jnp.linalg.norm(trans - t_final, axis=1)
        final_inliers = jnp.sum((ang_errs <= ang_thresh) & (trans_errs <= trans_thresh))

        return q_final, t_final, final_inliers > 0

    return jax.lax.cond(length == 0, fail_case, success_case)
