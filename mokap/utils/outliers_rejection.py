from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from mokap.utils import geometry_2


# Pre-allocate identity quaternion constant and zero translation
ID_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
ZERO_T = jnp.zeros((3,), dtype=jnp.float32)


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
        return geometry_2.ID_QUAT, geometry_2.ZERO_T

    quats = rt_stack[:, :4]
    trans = rt_stack[:, 4:]

    # provisional mean
    q_med0 = geometry_2.quaternion_average(quats)
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
        q_med1 = geometry_2.quaternion_average(q_filt)
        t_med1 = geometry_2.robust_translation_mean(t_filt, num_iters=4, delta=1.0)
        return q_med1, t_med1
    else:
        # Fallback to provisional mean
        return q_med0, t_med0