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
    Analogous function to cv2.Rodrigues (when a rvec is passed)

    Args:
        rvec: rotation vectors (..., 3)

    Returns:
        rotation matrices (..., 3, 3)
    """

    # magnitude of each vector
    theta = jnp.linalg.norm(rvec, axis=-1, keepdims=True)  # (..., 1)

    # When theta is small, we use the Taylor expansion of R = I + [r_x]
    # to avoid division by theta and the eventual numerical instability
    is_small_angle = theta < _eps

    # Normal angle
    k = rvec / jnp.where(is_small_angle, 1.0, theta)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    zeros = jnp.zeros_like(kx)
    K = jnp.stack([
        jnp.stack([zeros, -kz, ky], axis=-1),
        jnp.stack([kz, zeros, -kx], axis=-1),
        jnp.stack([-ky, kx, zeros], axis=-1),
    ], axis=-2)

    sin_t = jnp.sin(theta)[..., None]
    cos_t = jnp.cos(theta)[..., None]
    I = jnp.eye(3)

    R_normal = I + sin_t * K + (1 - cos_t) * (K @ K)

    # Small angle (Taylor expansion)
    # R ~ I + [r_x]
    rx, ry, rz = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    zeros_r = jnp.zeros_like(rx)
    # Skew-symmetric matrix of the rvec itself
    R_skew = jnp.stack([
        jnp.stack([zeros_r, -rz, ry], axis=-1),
        jnp.stack([rz, zeros_r, -rx], axis=-1),
        jnp.stack([-ry, rx, zeros_r], axis=-1),
    ], axis=-2)
    R_small = I + R_skew

    # Choose which result to use based on the angle
    return jnp.where(is_small_angle[..., None], R_small, R_normal)


@jax.jit
def inverse_rodrigues(Rmat: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a rotation matrix (or several) to a rotation vector (or several)
    Analogous function to cv2.Rodrigues (when a Rmat is passed)

    Args:
        Rmat: rotation matrices (..., 3, 3)

    Returns:
        rotation vectors (..., 3)
    """

    trace = jnp.trace(Rmat, axis1=-2, axis2=-1)
    costheta = (trace - 1) / 2

    # Use acos safely
    theta = jnp.arccos(jnp.clip(costheta, -1, 1))

    # Skew part of the matrix
    rv_unscaled = jnp.stack([
        Rmat[..., 2, 1] - Rmat[..., 1, 2],
        Rmat[..., 0, 2] - Rmat[..., 2, 0],
        Rmat[..., 1, 0] - Rmat[..., 0, 1]
    ], axis=-1)

    # there's an issue when theta is close to 0 or pi
    # sin(theta) is in the denominator
    sintheta = jnp.sin(theta)

    # Condition for using approximation (theta ~ 0 or theta ~ pi)
    is_singular = (sintheta ** 2) < _eps

    # Normal angle
    scale = jnp.where(is_singular, 1.0, 0.5 * theta / sintheta)
    rvec_normal = rv_unscaled * scale[..., None] # (..., 3)

    # TODO: Singular angle (theta is near 0 or pi)

    return rvec_normal


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

    # Convert rotation vector into rotation matrix
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
        K_pair: Tuple[jnp.ndarray, jnp.ndarray],
        r_pair: Tuple[jnp.ndarray, jnp.ndarray],  # these are world -> camera rvecs
        t_pair: Tuple[jnp.ndarray, jnp.ndarray],  # these are world -> camera tvecs
) -> jnp.ndarray:
    """
    Computes the fundamental matrix between two cameras given their intrinsics
    and world-to-camera extrinsics
    """
    K1, K2 = K_pair
    r1, r2 = r_pair
    t1, t2 = t_pair

    R1 = rodrigues(r1)  # world -> camera 1 rotation
    R2 = rodrigues(r2)  # world -> camera 2 rotation

    # The relative transformation from camera 1's coordinate system to camera 2's is
    # T_c1_c2 = T_w_c2 * inv(T_w_c1)

    # Relative rotation (from C1 frame to C2 frame)
    R_c1_c2 = R2 @ R1.T

    # Relative translation (vector from C1's origin to C2's origin, expressed in C2's frame)
    # C1_world = -R1.T @ t1
    # C2_world = -R2.T @ t2
    # t_c1_c2 = R2 @ (C1_world - C2_world)

    # This simplifies to
    t_c1_c2 = t2 - R_c1_c2 @ t1

    # The essential matrix E relates a point x1 in C1 to a point x2 in C2
    # via the equation x2^T * E * x1 = 0
    # The formula is E = [t]_x R, where t is the translation vector from C1's
    # origin to C2's origin, and R is the rotation from C1 to C2
    t_skew = jnp.array([
        [0, -t_c1_c2[2], t_c1_c2[1]],
        [t_c1_c2[2], 0, -t_c1_c2[0]],
        [-t_c1_c2[1], t_c1_c2[0], 0]
    ])
    E = t_skew @ R_c1_c2
    # TODO: Maybe would be good to take the Essential matrix computation out in a dedicated function?

    # Fundamental matrix F = K2^-T * E * K1^-1
    F = jnp.linalg.inv(K2).T @ E @ jnp.linalg.inv(K1)

    # Enforce rank-2 constraint (Longuet-Higgins constraint)
    U, S, Vt = jnp.linalg.svd(F)
    S_corrected = S.at[2].set(0.0)
    F_corrected = U @ jnp.diag(S_corrected) @ Vt

    F_normalized = F_corrected / (F_corrected[2, 2] + 1e-8)

    return F_normalized

batched_fundamental_matrices = jax.jit(
    jax.vmap(
        fundamental_matrix,
        in_axes=(
            0,    # K_pair: shape (P, 2, 3, 3) where P is the number of camera pairs (NOT the number of cameras!)
            0,    # r_pair: shape (P, 2, 3)
            0    # t_pair: shape (P, 2, 3)
        )
    )
)


@jax.jit
def invert_rtvecs(
        rvec: jnp.ndarray,
        tvec: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Inverts extrinsics vectors (rvec, tvec) -> (rvec_inv, tvec_inv)
    """
    R_mat = rodrigues(rvec)
    R_inv = jnp.swapaxes(R_mat, -1, -2)
    tvec_inv = (-R_inv @ tvec[..., None])[..., 0]
    rvec_inv = inverse_rodrigues(R_inv)
    return rvec_inv, tvec_inv


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
def rotate_rtvecs(
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

axisangle_to_quaternion_batched = jax.jit(jax.vmap(axisangle_to_quaternion))


@jax.jit
def quaternion_to_axisangle(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert one quaternion q = [w, x, y, z] to a rvec ∈ ℝ³ (axis–angle)
    if sin(theta/2) ~ 0, it returns [0, 0, 0]
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

quaternion_to_axisangle_batched = jax.jit(jax.vmap(quaternion_to_axisangle))


@jax.jit
def quaternion_inverse(q: jnp.ndarray) -> jnp.ndarray:
    """
    Invert a unit quaternion q = [w, x, y, z]
    For a unit quaternion, q^{-1} = [w, -x, -y, -z]
    """
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z], dtype=q.dtype)

quaternion_inverse_batched = jax.jit(jax.vmap(quaternion_inverse))


@jax.jit
def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """ Multiplies two quaternions q1 * q2 """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])


@jax.jit
def rotate_vector_by_quat(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a 3D vector v by the unit-quaternion q using q * v * q_inv
    """
    # Represent vector v as a pure quaternion [0, x, y, z]
    v_quat = jnp.concatenate([jnp.array([0.0]), v])

    # Compute the rotated vector
    q_inv = quaternion_inverse(q)
    v_rot_quat = quaternion_multiply(q, quaternion_multiply(v_quat, q_inv))

    # Return the vector part
    return v_rot_quat[1:]

# vmap over v, but not q (to rotate multiple vectors by one quaternion)
rotate_vectors_by_quat = jax.jit(jax.vmap(rotate_vector_by_quat, in_axes=(None, 0)))
# vmap over both q and v (to rotate multiple vectors by multiple quaternions)
rotate_vectors_by_quats = jax.jit(jax.vmap(rotate_vector_by_quat, in_axes=(0, 0)))


def quaternions_angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the angle between two unit quaternions q1, q2
    """
    d = jnp.abs(jnp.dot(q1, q2))
    d = jnp.clip(d, -1.0, 1.0)
    return 2.0 * jnp.arccos(d)
