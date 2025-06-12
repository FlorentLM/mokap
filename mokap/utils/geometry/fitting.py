from typing import Tuple
import jax
import numpy as np
from jax import numpy as jnp

from mokap.utils.geometry.transforms import ID_QUAT, ZERO_T


@jax.jit
def find_rigid_transform(
    points_A: jnp.ndarray,  # (M, 3)
    points_B: jnp.ndarray   # (M, 3)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates the rigid transformation (rotation R, translation t) between two
    sets of corresponding 3D points (A and B) using the Kabsch algorithm
    It finds T such that: B ~ T(A) = R @ A + t

    Args:
        points_A: Source points (M, 3)
        points_B: Destination points (M, 3)

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    # Find centroids
    centroid_A = jnp.mean(points_A, axis=0)
    centroid_B = jnp.mean(points_B, axis=0)

    # Center the points
    A_centered = points_A - centroid_A
    B_centered = points_B - centroid_B

    # Compute the covariance matrix H
    H = A_centered.T @ B_centered

    # Find the rotation using SVD
    U, S, Vt = jnp.linalg.svd(H)
    R = Vt.T @ U.T

    # If the determinant is -1, it's a reflection
    # We must flip the sign of the last column of U or V
    det_R = jnp.linalg.det(R)
    # diagonal matrix to flip the sign if needed
    correction = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, det_R]])
    R_corrected = Vt.T @ correction @ U.T

    # Compute translation
    t = centroid_B - R_corrected @ centroid_A

    return R_corrected, t

# batched version for multiple point sets
find_rigid_transform_batched = jax.jit(jax.vmap(find_rigid_transform))


@jax.jit
def find_affine_transform(
    points_A: jnp.ndarray, # (M, 3)
    points_B: jnp.ndarray  # (M, 3)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates the affine transformation (A, t) between two sets of corresponding 3D points
    Finds T such that: B ~ T(A) = A @ A.T + t

    Args:
        points_A: Source points (M, 3)
        points_B: Destination points (M, 3)

    Returns:
        A_mat: The affine transformation matrix (3, 3)
        t_vec: The translation vector (3,)
    """
    M, _ = points_A.shape
    ones = jnp.ones((M, 1), dtype=points_A.dtype)
    X = jnp.concatenate([points_A, ones], axis=1)  # (M, 4)

    # Solve X @ T = points_B
    T, *_ = jnp.linalg.lstsq(X, points_B, rcond=None)  # T is (4, 3)

    A_mat = T[:3, :].T  # (3, 3)
    t_vec = T[3, :]     # (3,)
    return A_mat, t_vec

# batched version for multiple sets of points
find_affine_transform_batched = jax.jit(jax.vmap(find_affine_transform, in_axes=(0, 0), out_axes=(0, 0)))


@jax.jit
def interpolate3d(
    points3d:               jnp.ndarray,
    visibility_mask:        jnp.ndarray,
    points3d_theoretical:   jnp.ndarray
) -> jnp.ndarray:
    """
    Use triangulated 3D points and theoretical point layout (e.g. the calibration grid) to interpolate missing points

    Args:
        points3d: the N 3D points (N, 3)
        visibility_mask: the visibility mask of the N points (N,)
        points3d_theoretical: the full set of N theoretical point coordinates (N, 3)

    Returns:
        filled: (N, 3) filled-in points
    """

    N = points3d.shape[0]
    mask = visibility_mask.astype(bool)  # (N,)

    # Build design matrix [X_th | 1]
    ones = jnp.ones((N, 1), dtype=points3d.dtype)
    A = jnp.concatenate([points3d_theoretical, ones], axis=1)  # (N, 4)

    # zeroify just the rows we did observe for the weighted least squares
    A_obs = jnp.where(mask[:, None], A, 0.0)  # (N, 4)
    Y_obs = jnp.where(mask[:, None], points3d, 0.0)  # (N, 3)

    # Solve A_obs @ T ~ Y_obs
    T, *_ = jnp.linalg.lstsq(A_obs, Y_obs, rcond=None)  # (4, 3)

    # Predict N with that T
    filled_all = A @ T  # (N, 3)

    # And only replace the originally missing rows
    return jnp.where(mask[:, None], points3d, filled_all)


@jax.jit
def huber_weight(residual_norm: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """
    Compute Huber weight for each ||error||

        w = 1                   if ||error|| <= delta
        w = delta / ||error||   if ||error|| > delta
    """
    return jnp.where(residual_norm <= delta,1.0, delta / (residual_norm + 1e-12))


@jax.jit
def translation_average(
        t_samples:  jnp.ndarray,
        num_iters:  int = 3,
        delta:      float = 1.0
    ) -> jnp.ndarray:
    """
    Compute a Huber‐weighted mean of M translation samples (M, 3) with IRLS
    If M == 0, it returns [0, 0, 0]
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


def quaternions_angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
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
            ang_errs = jax.vmap(lambda q: quaternions_angular_distance(q, q_c))(quats)
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
        ang_errs = jax.vmap(lambda q: quaternions_angular_distance(q, q_final))(quats)
        trans_errs = jnp.linalg.norm(trans - t_final, axis=1)
        final_inliers = jnp.sum((ang_errs <= ang_thresh) & (trans_errs <= trans_thresh))

        return q_final, t_final, final_inliers > 0

    return jax.lax.cond(length == 0, fail_case, success_case)
