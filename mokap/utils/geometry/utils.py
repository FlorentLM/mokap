from typing import Tuple, Optional
import jax
from jax import numpy as jnp


@jax.jit
def pad_dist_coeffs(dist_coeffs: jnp.ndarray) -> jnp.ndarray:
    # TODO: This one is kinda bad
    """
    Simple utility to always return 8 distortion coefficients.
    It pads with zeros if fewer than 8 are provided, and truncates if more are provided.

    Args:
        dist_coeffs: distortion coefficients (any length)
    Returns:
        coefs8:  distortion coefficients of shape (8,)
    """
    coeffs = jnp.atleast_1d(dist_coeffs).ravel()

    coefs8 = jnp.zeros(8, dtype=coeffs.dtype)
    num_to_copy = jnp.minimum(coeffs.shape[0], 8)

    # jax.lax.fori_loop to copy elements one by one
    # This is JIT-compatible because the loop bounds are defined
    def body_fn(i, val):
        return val.at[i].set(coeffs[i])

    coefs8 = jax.lax.fori_loop(0, num_to_copy, body_fn, coefs8)

    return coefs8


@jax.jit
def find_affine(
    points3d_A:     jnp.ndarray,  # (B, M, 3) or (M, 3)
    points3d_B:     jnp.ndarray   # (B, M, 3) or (M, 3)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates the affine transformation T between two sets of 3D points
    T = [x; 1] -> y
    either for a single (M, 3) or a batch (B, M, 3)

    Args:
        points3d_A: M points A (B, M, 3) or (M, 3)
        points3d_B: M points B (B, M, 3) or (M, 3)

    Returns:
        R: (B, 3, 3)
        t: (B, 3) or (3, 3), (3)
    """

    # we go via a batch dim by default
    points3d_A = jnp.atleast_3d(points3d_A)     # (B, M, 3)
    points3d_B = jnp.atleast_3d(points3d_B)     # (B, M, 3)

    B, M, _ = points3d_A.shape
    ones = jnp.ones((B, M, 1), dtype=points3d_A.dtype)
    X = jnp.concatenate([points3d_A, ones], axis=2)     # (B, M, 4)

    # solve batched lstsq: X @ T = points3d_B
    T, *_ = jnp.linalg.lstsq(X, points3d_B, rcond=None)    # T is (B, 4, 3)

    # we want (B, 3, 3) and (B, 3)
    R = T[:, :3, :].transpose(0, 2, 1)
    t = T[:,  3, :]

    # drop batch dim if it's superfluous
    R = R.squeeze(0)
    t = t.squeeze(0)
    return R, t


@jax.jit
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
def focal_point_3d(
    camera_centers:     jnp.ndarray,
    direction_vectors:  jnp.ndarray
) -> jnp.ndarray:
    """
    Estimate the 3D focal point of C cameras

    Args:
        camera_centers: C camera centers (C, 3)
        direction_vectors: C direction vectors (C, 3)

    Returns:
        focal_point: (3,)
    """

    # A_i = I - d d^T
    D = direction_vectors[..., :, None]     # (C, 3, 1)
    A = jnp.eye(3)[None, ...] - D @ D.transpose(0,2,1)  # (C, 3, 3)

    # b_i = A_i @ C_i
    C = camera_centers[..., :, None]        # (C, 3, 1)
    b = (A @ C)[..., 0]                     # (C, 3)

    # stack into a (3N, 3) system and solve it
    A_stack = A.reshape(-1, 3)    # (3*C, 3)
    b_stack = b.reshape(-1)           # (3*C,)
    focal_point, *_ = jnp.linalg.lstsq(A_stack, b_stack, rcond=None)  # (3,)
    return focal_point
