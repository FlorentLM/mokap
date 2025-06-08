from functools import partial
from typing import Tuple, Union, Iterable
import jax
from jax import numpy as jnp

_eps = 1e-8
_AXIS_MAP = {
    'x': jnp.array([1.0, 0.0, 0.0]),
    'y': jnp.array([0.0, 1.0, 0.0]),
    'z': jnp.array([0.0, 0.0, 1.0]),
}
# Pre-allocate identity quaternion constant and zero translation
ID_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
ZERO_T = jnp.zeros((3,), dtype=jnp.float32)


@jax.jit
def rodrigues(rvec: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a rotation vector (or several) to a rotation matrix (or several)
    Analogous to cv2.Rodrigues (when a rvec is passed) but can be compiled with JAX

    Args:
        rvec: rotation vectors (..., 3)

    Returns:
        rotation matrices (..., 3, 3)
    """

    # magnitude of each vector
    theta = jnp.linalg.norm(rvec, axis=-1, keepdims=True)  # (..., 1)

    # unit axis
    # if theta ~ 0 we just get small numbers
    k = rvec / (theta + _eps)                          # (..., 3)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    # build skew‐symmetric K
    zeros = jnp.zeros_like(kx)
    K = jnp.stack([
        jnp.stack([zeros, -kz, ky], axis=-1),
        jnp.stack([kz, zeros, -kx], axis=-1),
        jnp.stack([-ky, kx, zeros], axis=-1),
    ], axis=-2)     # (..., 3, 3)

    # now apply Rodrigues: R = I + sinθ . K + (1 − cosθ) . K @ K
    # sinθ and cosθ need a singleton matrix dimension for broadcasting
    sin_t = jnp.sin(theta)[..., None]  # (..., 1, 1)
    cos_t = jnp.cos(theta)[..., None]  # (..., 1, 1)
    I = jnp.eye(3)

    R = I + sin_t * K + (1 - cos_t) * (K @ K)  # (..., 3, 3)
    return R


@jax.jit
def inverse_rodrigues(Rmat: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a rotation matrix (or several) to a rotation vector (or several)
    Analogous to cv2.Rodrigues (when a Rmat is passed) but can be compiled with JAX

    Args:
        Rmat: rotation matrices (..., 3, 3)

    Returns:
        rotation vectors (..., 3)
    """

    trace = jnp.trace(Rmat, axis1=-2, axis2=-1)
    costheta = (trace - 1) / 2
    theta = jnp.arccos(jnp.clip(costheta, -1, 1))
    sintheta = jnp.sin(theta)[..., None]

    # skew part
    rv = jnp.stack([
        Rmat[..., 2, 1] - Rmat[..., 1, 2],
        Rmat[..., 0, 2] - Rmat[..., 2, 0],
        Rmat[..., 1, 0] - Rmat[..., 0, 1]
    ], axis=-1)                             # (..., 3)

    axis = rv / (2 * (sintheta + _eps))     # (..., 3)
    # if theta ~ 0 we just use zero axis
    axis = jnp.where(theta[..., None] > _eps, axis, 0.0)

    return axis * theta[..., None]          # (..., 3)


@jax.jit
def extrinsics_matrix(
    rvec:   jnp.ndarray,
    tvec:   jnp.ndarray
) -> jnp.ndarray:
    """
    Converts rotation vectors and translation vectors to 4x4 extrinsics matrices E

                                  [ r00, r01, r02 ]             [ t0 ]
        E = [ R | t ]   with  R = [ r10, r11, r12 ]   and   t = [ t1 ]
                                  [ r20, r21, r22 ]             [ t2 ]
    Args:
        rvec: rotation vector(s) (..., 3)
        tvec: translation vector(s) (..., 3)

    Returns:
        E: Extrinsics matrix (or matrices) E (..., 4, 4)
    """

    # make sure we have array inputs
    rvec = jnp.asarray(rvec)
    tvec = jnp.asarray(tvec)

    # Convert rotation vector into rotation matrix (and jacobian)
    R = rodrigues(rvec)         # (..., 3, 3)
    t = tvec[..., None]         # (..., 3, 1)

    # build the 3x4 top part
    E_upper = jnp.concatenate([R, t], axis=-1)  # (..., 3, 4)

    # build the bottom row and broadcast it to match leading dims
    batch_shape = E_upper.shape[:-2]    # (...)
    bottom = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        batch_shape + (1, 4)
    )  # (..., 1, 4)

    E = jnp.concatenate([E_upper, bottom], axis=-2)  # (..., 4, 4)
    return E


@jax.jit
def extmat_to_rtvecs(
    E:  jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Converts Extrinsics matrix (or matrices) E to rotation vector(s) and translation vector(s)

    Args:
        E: Extrinsics matrices (..., 3, 4) or (..., 4, 4)

    Returns:
        rvec: rotation vectors (..., 3)
        tvec: translation vectors (..., 3)
    """

    R = E[..., :3, :3]              # (..., 3, 3)
    tvec = E[..., :3, 3]            # (..., 3)
    rvec = inverse_rodrigues(R)     # (..., 3)
    return rvec, tvec


@jax.jit
def projection_matrix(
    K:  jnp.ndarray,
    E:  jnp.ndarray
) -> jnp.ndarray:
    """
    Computes projection matrices P = K @ [R|t]
    Accepts K and E with matching leading dims

    The projection matrix maps 3D points represented in real-world, camera-relative coordinates (X, Y, Z, 1)
    to 2D points in the image plane represented in normalized camera-relative coordinates (u, v, 1)

    2d_point     matrix_K           matrix_E        3d_point
               (intrinsics)       (extrinsics)

                                                    [ X ]
    [ u ]   [ fx, 0, cx ]   [ r00, r01, r02, t0 ]   [ Y ]
    [ v ] = [ 0, fy, cy ] . [ r10, r11, r12, t1 ] . [ Z ]
    [ 1 ]   [ 0,  0,  1 ]   [ r20, r21, r22, t2 ]   [ 1 ]

    Args:
        K: Camera matrices (..., 3, 3)
        E: Extrinsics matrices (..., 3, 4) or (..., 4, 4)

    Returns:
        P: Projection matrices (..., 3, 4)
    """

    E3x4 = E[..., :3, :]    # make sure E is (..., 3, 4)
    P = jnp.einsum('...ij,...jk->...ik', K, E3x4)
    return P


@jax.jit
def fundamental_matrix(
    K_pair:     Tuple[jnp.ndarray, jnp.ndarray],
    r_pair:     Tuple[jnp.ndarray, jnp.ndarray],
    t_pair:     Tuple[jnp.ndarray, jnp.ndarray],
    rank2_tol:  float = 1e-10
) -> jnp.ndarray:
    """
    Computes the fundamental matrix between two cameras given their intrinsics and extrinsics

    Args:
        K_pair: the two K matrices, each (3, 3)
        r_pair: the two rvecs, each (3,)
        t_pair: the two tvecs, each (3,)
        rank2_tol: tolerance for enforcing rank‐2 consistency

    Returns:
        F: The fundamental matrix (3, 3)
    """

    K1 = K_pair[0]
    K2 = K_pair[1]

    r1 = r_pair[0]
    r2 = r_pair[1]

    t1 = t_pair[0]
    t2 = t_pair[1]

    R1 = rodrigues(r1)
    R2 = rodrigues(r2)

    # we want column vectors
    t1 = t1.reshape(3, 1)
    t2 = t2.reshape(3, 1)

    # relative motion (rotation and translation)
    R_rel = R2.T @ R1
    t_rel = R2.T @ (t1 - t2)

    # Skew-symmetric matrix for t_rel
    t_rel_skew = jnp.array([
        [0, -t_rel[2, 0], t_rel[1, 0]],
        [t_rel[2, 0], 0, -t_rel[0, 0]],
        [-t_rel[1, 0], t_rel[0, 0], 0]
    ])
    E = t_rel_skew @ R_rel  # essential matrix

    # fundamental matrix
    F = jnp.linalg.inv(K2).T @ E @ jnp.linalg.inv(K1)
    F /= F[2, 2]

    # enforce rank-2
    U, S, Vt = jnp.linalg.svd(F)
    det_F = jnp.linalg.det(F)

    def enforce_rank2(F_in):
        S2 = S.at[2].set(0.0)
        return U @ jnp.diag(S2) @ Vt

    F = jax.lax.cond(
        jnp.logical_and(~(S[2] < rank2_tol), jnp.abs(det_F) < rank2_tol),
        enforce_rank2,
        lambda F_in: F_in,  # return the input if rank-2 is OK
        F)
    return F

batched_fundamental_matrices = jax.jit(
    jax.vmap(
        fundamental_matrix,
        in_axes=(
            0,    # K_pair: shape (P, 2, 3, 3) where P is the number of camera pairs (NOT the number of cameras!)
            0,    # r_pair: shape (P, 2, 3)
            0,    # t_pair: shape (P, 2, 3)
            None  # rank2_tol: shared scalar
        )
    ),
    static_argnums=(3,)      # rank2_tol should be static, it won't change at runtime
)

@jax.jit
def invert_extrinsics(
    rvec:   jnp.ndarray,
    tvec:   jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Inverts extrinsics vectors (camera-space -> world space or vice versa) [ Method 1 ]
    (rvec, tvec) -> (rvec_inv, tvec_inv)

    Args:
        rvec: rotation vector(s) (..., 3)
        tvec: translation vector(s) (..., 3)

    Returns:
        rvec_inv: inverted rotation vector(s) (..., 3)
        tvec_inv: inverted rotation vector(s) (..., 3)
    """

    # invert rotations
    R_mat = rodrigues(rvec)             # (..., 3, 3)
    R_inv = jnp.linalg.inv(R_mat)       # (..., 3, 3)
    # or R_mat.T because the matrix is orthonormal

    # invert translations
    tvec = jnp.asarray(tvec)[..., None]     # (..., 3, 1)
    tvec_inv = (-R_inv @ tvec)[..., 0]      # (..., 3)

    # back to axis-angle rotation vector
    rvec_inv = inverse_rodrigues(R_inv)     # (..., 3)
    return rvec_inv, tvec_inv


@jax.jit
def invert_extrinsics_2(
    rvec:   jnp.ndarray,
    tvec:   jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Inverts extrinsics vectors (camera-space -> world space or vice versa) [ Method 2 ]
    (rvec, tvec) -> (rvec_inv, tvec_inv)

    Args:
        rvec: rotation vector(s) (..., 3)
        tvec: translation vector(s) (..., 3)

    Returns:
        rvec_inv: inverted rotation vector(s) (..., 3)
        tvec_inv: inverted rotation vector(s) (..., 3)
    """
    E = extrinsics_matrix(rvec, tvec)   # (..., 4, 4)
    E_inv = jnp.linalg.inv(E)           # (..., 4, 4)
    return extmat_to_rtvecs(E_inv)      # (..., 3), (..., 3)


@jax.jit
def invert_extrinsics_matrix(
    E:  jnp.ndarray
) -> jnp.ndarray:
    """
    Inverts extrinsics matrix (camera-space -> world space or vice versa)
    E -> inv_E

    Args:
        E: extrinsics matrix (or matrices) (..., 3, 4) or (..., 4, 4)

    Returns:
        inv_E: inverted extrinsics matrix (or matrices)  (..., 4, 4)
    """

    # Since E.shape is known when we trace the function, we can use plain Python if/else on E.shape[-2:]

    last2 = (E.shape[-2], E.shape[-1])
    if last2 == (3, 4):
        bottom = jnp.array([0., 0., 0., 1.], dtype=E.dtype)
        bottom = bottom.reshape((1, 4))
        # broadcast to match leading dims
        bottom = jnp.broadcast_to(bottom, E.shape[:-2] + (1, 4))
        E4 = jnp.concatenate([E, bottom], axis=-2)  # (..., 4, 4)
        return jnp.linalg.inv(E4)

    elif last2 == (4, 4):
        return jnp.linalg.inv(E)

    else:
        # catching bad shapes early
        raise ValueError(f"Expected shape (..., 3, 4) or (..., 4, 4), got {last2}")


@jax.jit
def remap_rtvecs(
    rvec:       jnp.ndarray,
    tvec:       jnp.ndarray,
    rvec_ref:   jnp.ndarray,
    tvec_ref:   jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Remaps extrinsics vectors to another origin's extrinsics vectors

    Args:
        rvec: rotation vector(s) (..., 3)
        tvec: translation vector(s) (..., 3)
        rvec_ref: rotation vector(s) (..., 3)
        tvec_ref: translation vector(s) (..., 3)

    Returns:
        rvec_remap: remapped rotation vector(s) (..., 3)
        tvec_remap: remapped translation vector(s) (..., 3)
    """

    E = extrinsics_matrix(rvec, tvec)               # (..., 4, 4)
    E_ref = extrinsics_matrix(rvec_ref, tvec_ref)   # (..., 4, 4)

    # E_new = E_ref @ E^-1
    E_inv = jnp.linalg.inv(E)
    E_new = jnp.einsum('...ij,...jk->...ik', E_ref, E_inv)

    rvec_remap, tvec_remap = extmat_to_rtvecs(E_new)
    return rvec_remap, tvec_remap


@jax.jit
def remap_extmat(
    E:      jnp.ndarray,
    E_ref:  jnp.ndarray
) -> jnp.ndarray:
    """
    Re-expresses an extrinsics matrix (or matrices) in the coordinate frame of another origin

    Args:
       E: extrinsics matrix (or matrices) (..., 3, 4) or (..., 4, 4)
       E_ref: reference extrinsics matrix (or matrices) (..., 3, 4) or (..., 4, 4)

    Returns:
       E_remap: remapped extrinsics matrix (or matrices) (..., 4, 4)
    """

    E_inv = invert_extrinsics_matrix(E)  # (..., 4, 4)

    # pad origin_mat if necessary
    def pad3x4(E3x4):
        bottom = jnp.array([0.0, 0.0, 0.0, 1.0])
        bottom = jnp.broadcast_to(bottom, E3x4.shape[:-2] + (1, 4))
        return jnp.concatenate([E3x4, bottom], axis=-2)

    is3x4 = (E_ref.shape[-2] == 3)
    E_ref4 = jax.lax.cond(is3x4, pad3x4, lambda E: E, E_ref)

    return E_ref4 @ E_inv


@jax.jit
def remap_points3d(
    points3d:   jnp.ndarray,
    rvec_ref:   jnp.ndarray,
    tvec_ref:   jnp.ndarray
) -> jnp.ndarray:
    """
    Transform 3D points under a new origin

    Args:
       points3d: points 3D coordinates to transform (..., 3)
       rvec_ref: rotation vector(s) (..., 3)
       tvec_ref: translation vector(s) (..., 3)


    Returns:
       points3d_remap: transformed points 3D coordinates  (..., 3)
    """

    E_ref = extrinsics_matrix(rvec_ref, tvec_ref)   # (..., 4, 4)
    E_inv = jnp.linalg.inv(E_ref)       # (..., 4, 4)

    # apply only R and t to each point
    R_inv = E_inv[..., :3, :3]      # (..., 3, 3)
    t_inv = E_inv[..., :3, 3]       # (..., 3)
    return jnp.einsum('...ij,...j->...i', R_inv, points3d) + t_inv


@partial(jax.jit, static_argnums=(1,))
def Rmat_from_angle(
        angle_degrees:  float,
        axis:           Union[str, Iterable[float]]
) -> jnp.ndarray:
    """
    Creates a Rotation matrix from the given angle

    Args:
        angle_degrees: the angle in degrees (scalar)
        axis: the axis of rotation, either a (3,) iterable or an axis name ('x', 'y' or 'z')

    Returns:
        R_mat: The rotation matrix (3, 3)
    """

    theta = jnp.deg2rad(angle_degrees)

    if isinstance(axis, str):
        a = axis.lower()
        if a not in _AXIS_MAP:
            raise ValueError("Axis must be 'x','y','z' or a 3-element vector.")
        rvec = _AXIS_MAP[a] * theta
    else:
        v = jnp.asarray(axis, dtype=jnp.float32)
        if v.shape != (3,):
            raise ValueError("Axis must be 'x','y','z' or a 3-element vector.")
        v = v / (jnp.linalg.norm(v) + _eps)
        rvec = v * theta
    return rodrigues(rvec)


@partial(jax.jit, static_argnames=['axis'])
def rotate_points3d(
        points3d: jnp.ndarray,
        angle_degrees: float,
        axis: Union[str, Iterable[float]] = 'y'
) -> jnp.ndarray:
    """
    Args:
        points3d: 3D points to rotate (..., 3)
        angle_degrees: the angle in degrees (scalar)
        axis: the axis of rotation, either a (3,) iterable or an axis name ('x', 'y' or 'z')

    Returns:
        points3d_rot: the rotated points 3D (..., 3)
    """

    R = Rmat_from_angle(angle_degrees, axis)    # (3, 3)
    points3d_rot = jnp.einsum('ij,...j->...i', R, points3d)     # (..., 3)
    return points3d_rot


@partial(jax.jit, static_argnames=['axis'])
def rotate_pose(
        rvecs:          jnp.ndarray,
        tvecs:          jnp.ndarray,
        angle_degrees:  float,
        axis:           Union[str, Iterable[float]] = 'y'
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        rvecs: rvecs (..., 3)
        tvecs: tvecs (..., 3)
        angle_degrees: the angle in degrees (scalar)
        axis: the axis of rotation, either a (3,) iterable or an axis name ('x', 'y' or 'z')

    Returns:
        rvecs_rot: rotated rvecs (..., 3)
        tvecs_rot: rotated tvecs (..., 3)
    """

    Rg = Rmat_from_angle(angle_degrees, axis)   # (3, 3)

    Rg_b = Rg[jnp.newaxis, ...]                 # (1, 3, 3) for broadcasting
    Rl = rodrigues(rvecs)                       # (..., 3, 3)
    R_comb = jnp.matmul(Rg_b, Rl)               # (...,3, 3)
    rvecs_rot = inverse_rodrigues(R_comb)       # (..., 3)
    tvecs_rot = jnp.einsum('ij,...j->...i', Rg, tvecs)  # (..., 3)
    return rvecs_rot, tvecs_rot


@partial(jax.jit, static_argnames=['axis'])
def rotate_extrinsics_matrix(
    E:              jnp.ndarray,
    angle_degrees:  float,
    axis:           Union[str, Iterable[float]] = 'y',
    hom:            bool = False
) -> jnp.ndarray:
    """
    Args:
        E: extrinsics matrix (3, 4) or (4, 4)
        angle_degrees: the angle in degrees (scalar)
        axis: the axis of rotation, either a (3,) iterable or an axis name ('x', 'y' or 'z')
        hom: whether the matrix should be returned as a homogeneous matrix

    Returns:
        E_rot: (3, 4) or (4, 4) rotated extrinsics matrix (or matrices)
    """

    Rg = Rmat_from_angle(angle_degrees, axis)   # (3, 3)
    R, t = E[:3, :3], E[:3, 3]
    R_new = Rg @ R
    t_new = Rg @ t

    E_rot = jnp.concatenate([R_new, t_new[..., None]], axis=1)  # (3, 4)
    if hom:
        bottom = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=E_rot.dtype)
        E_rot = jnp.vstack([E_rot, bottom])
    return E_rot


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


quaternion_to_axisangle_batched = jax.jit(
    jax.vmap(quaternion_to_axisangle , in_axes=0, out_axes=0)
)


@jax.jit
def quaternion_inverse(q: jnp.ndarray) -> jnp.ndarray:
    """
    Invert a unit quaternion q = [w, x, y, z].
    For a unit quaternion, q^{-1} = [w, -x, -y, -z].
    """
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z], dtype=q.dtype)


@jax.jit
def rotate_vector(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a 3D vector v by the unit‐quaternion q.
    """
    # 1) turn q→axis–angle
    rvec = quaternion_to_axisangle(q)
    # 2) build R via Rodrigues
    R = rodrigues(rvec)
    # 3) apply
    return R @ v
