from typing import Optional
import jax
from jax import numpy as jnp


def maybe_put(x):
    return jax.device_put(x) if x is not None else None


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
