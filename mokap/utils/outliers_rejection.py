from typing import Tuple
import jax
import jax.numpy as jnp
from mokap.utils import geometry_jax, geometry_2


@jax.jit
def _angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the angle between two unit quaternions q1, q2 via
       alpha = 2*arccos(|<q1, q2>|).
    Returns alpha in radians.
    """
    # Ensure unit‐length invariance by taking absolute dot‐product
    d = jnp.abs(jnp.dot(q1, q2))
    # Clamp into [–1,1] for numerical safety
    d = jnp.clip(d, -1.0, 1.0)
    return 2.0 * jnp.arccos(d)


@jax.jit
def filter_rt_samples(
    rt_stack:   jnp.ndarray,  # shape: (M, 7)  = [q (4), t (3)]
    length:     int,          # actual # of valid rows in rt_stack (<= M)
    ang_thresh: float = jnp.deg2rad(30.0),
    trans_thresh: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given a stack of shape (M, 7), where M >= length, and only the first `length`
    rows are valid: split into quaternions (4) + translations (3),
    compute a provisional quaternion mean + translation mean, then
    discard any sample whose angular error > ang_thresh (radians)
    or whose translation error > trans_thresh (same units as t).
    Finally, re‐average the _remaining_ samples and return (q_med, t_med).

    Args:
      rt_stack:    (M, 7) array. Only indices [0 .. length-1] are valid;
                   rows [length .. M-1] should be ignored.
      length:      integer, how many valid samples are in rt_stack.
      ang_thresh:  drop any quaternion whose angle from the mean exceeds this.
      trans_thresh:drop any translation whose ‖t – t_mean‖ exceeds this.

    Returns:
      q_med:  (4,) unit quaternion
      t_med:  (3,) translation vector
    """
    # Slice off only the valid part:
    valid = rt_stack[:length, :]       # shape (length, 7)
    quats = valid[:, :4]               # shape (length, 4)
    trans = valid[:, 4:]               # shape (length, 3)

    # If there are zero valid samples, just return identity + zero:
    def no_data():
        return geometry_2.ID_QUAT, geometry_2.ZERO_T

    def with_data():
        # 1) provisional mean quaternion & translation
        q_med0 = geometry_2.quaternion_average(quats)     # (4,)
        # For translation, just take a component-wise median:
        t_med0 = jnp.median(trans, axis=0)                  # (3,)

        # 2) compute angular errors and translation errors
        #    vectorized over all `length` samples:
        ang_errs = jax.vmap(lambda q: _angular_distance(q, q_med0))(quats)  # (length,)
        trans_errs = jnp.linalg.norm(trans - t_med0, axis=1)               # (length,)

        # 3) build a Boolean mask of “keep” (not outlier)
        keep_quat = ang_errs <= ang_thresh                 # (length,)
        keep_trans = trans_errs <= trans_thresh            # (length,)
        keep_mask = keep_quat & keep_trans                 # (length,)

        # 4) If nothing remains, fall back to “provisional”:
        num_kept = jnp.sum(keep_mask)

        def fallback():
            return q_med0, t_med0

        def recompute():
            q_filt = quats[keep_mask, :]                   # (K, 4)
            t_filt = trans[keep_mask, :]                   # (K, 3)
            q_med1 = geometry_2.quaternion_average(q_filt)  # (4,)
            t_med1 = geometry_2.robust_translation_mean(t_filt, num_iters=4, delta=1.0)  # (3,)
            return q_med1, t_med1

        return jax.lax.cond(num_kept > 0, recompute, fallback)

    return jax.lax.cond(length == 0, no_data, with_data)
