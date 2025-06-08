# import os
# os.environ['JAX_LOG_COMPILES'] = '1'
# from jax import config
# config.update("jax_log_compiles", True)
import jax
import jax.numpy as jnp
from typing import Union, Iterable, Optional, Tuple
from functools import partial

# All the projective geometry related functions used throughout the project

## --- Some utilities

_eps = 1e-8 # small epsilon to avoid div by zero

_AXIS_MAP = {
    'x': jnp.array([1.0, 0.0, 0.0]),
    'y': jnp.array([0.0, 1.0, 0.0]),
    'z': jnp.array([0.0, 0.0, 1.0]),
}


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
def distortion(
        x:              jnp.ndarray,
        y:              jnp.ndarray,
        dist_coeffs:    jnp.ndarray
)-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Given x and y coordinates, and distortion coefficients, compute the tangential and radial distortions

    Args:
        x: x coordinates
        y: y coordinates
        dist_coeffs: distortion coefficients (≤8,)

    Returns:
        radial: radial distortion
        dx: tangential distortion in x
        dy: tangential distortion in y
    """

    k1, k2, p1, p2, k3, k4, k5, k6 = pad_dist_coeffs(dist_coeffs)
    r2 = x * x + y * y
    radial = (1
              + k1 * r2
              + k2 * r2 ** 2
              + k3 * r2 ** 3
              + k4 * r2 ** 4
              + k5 * r2 ** 5
              + k6 * r2 ** 6)
    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return radial, dx, dy


## --- Various OpenCV functions replacements for full JAX compatibility


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
def project_points(
    object_points:  jnp.ndarray,
    rvec:           jnp.ndarray,
    tvec:           jnp.ndarray,
    camera_matrix:  jnp.ndarray,
    dist_coeffs:    jnp.ndarray
) -> jnp.ndarray:
    """
    Replacement for cv2.projectPoints

    Args:
        object_points: any leading batch dims but last dim 3 (..., 3)
        rvec: rotation vector (3,)
        tvec: translation vector (3,)
        camera_matrix: camera matrix (3, 3)
        dist_coeffs: distortion coefficients (≤8,)

    Returns:
        image_points: same leading dims as object_points but last dim 2 (..., 2)
    """

    R = rodrigues(rvec)

    Xc = jnp.einsum('ij,...j->...i', R, object_points) + tvec
    x, y = Xc[..., 0] / Xc[..., 2], Xc[..., 1] / Xc[..., 2]

    radial, dx, dy = distortion(x, y, dist_coeffs)
    x_d = x * radial + dx
    y_d = y * radial + dy

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return jnp.stack([fx * x_d + cx, fy * y_d + cy], axis=-1)


# Vectorized version for C cameras
#    object_points is shared among cameras so in_axes[0] is None
#    the other camera‐specific args vary
project_multiple = jax.jit(
    jax.vmap(project_points,
             in_axes=(None, # object_points is shared
                      0,    # each camera has its own rvec
                      0,    # its own tvec
                      0,    # its own camera_matrix
                      0),   # its own dist_coeffs
             out_axes=0),
    static_argnums=()
)


@jax.jit
def undistort_points(
    points2d:       jnp.ndarray,
    camera_matrix:  jnp.ndarray,
    dist_coeffs:    jnp.ndarray,
    R:              Optional[jnp.ndarray] = None,
    P:              Optional[jnp.ndarray] = None,
    max_iter:       int = 5
) -> jnp.ndarray:
    """
    Invert distortion & reprojection for a single camera (i.e. replacement for cv2.undistortPoints)

    Args:
        points2d: any leading batch dims but last dim 2 (..., 2)
        camera_matrix: camera matrix (3, 3)
        dist_coeffs: distortion coefficients (≤8,)
        R: rectification (usually identity matrix) (3, 3)
        P: new projection (usually camera matrix) (3, 3)
        max_iter: maximum number of iterations, default 5 (same as OpenCV) is typically enough

    Returns:
        undistorted_points: same leading dims as points2d but last dim 2 (..., 2)
    """

    # fallback to default R and P (runs in host code)
    if R is None:
        R = jnp.eye(3, dtype=camera_matrix.dtype)
    if P is None:
        P = camera_matrix

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    #normalize
    x_d = (points2d[..., 0] - cx) / fx
    y_d = (points2d[..., 1] - cy) / fy
    x_u, y_u = x_d, y_d

    # Newton iteration to invert distortion
    def newton(i, uv):
        x, y = uv
        radial, dx, dy = distortion(x, y, dist_coeffs)
        return ((x_d - dx) / (radial + _eps),
                (y_d - dy) / (radial + _eps))

    x_u, y_u = jax.lax.fori_loop(0, max_iter, newton, (x_u, y_u))

    # reproject through R and P
    ones = jnp.ones_like(x_u)
    pts_h = jnp.stack([x_u, y_u, ones], axis=-1)    # (..., 3)
    pts_r = pts_h @ R.T         # (..., 3)
    pts_p = pts_r @ P.T         # (..., 3)

    unsistorted_points = pts_p[..., :2]       # (..., 2)
    return unsistorted_points


# Vectorized version for C cameras
#    max_iter is shared among cameras so in_axes[-1] is None
#    the other camera‐specific args vary
undistort_multiple = jax.jit(
    jax.vmap(undistort_points,
             in_axes=(
                0,     # each camera has its own points2d
                0,     # its own camera_matrix
                0),     # its own dist_coeffs
             out_axes=0),
    static_argnums=()
)

## ---- Functions to compute various useful mnatrices


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


## ---- Some convenience manipilation of matrices (or rvec tvec) - Invert, change the reference, etc


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


@jax.jit
def back_projection(
        points2d: jnp.ndarray,
        depth: Union[float, jnp.ndarray],
        camera_matrix: jnp.ndarray,
        extrinsics_matrix: jnp.ndarray,
        dist_coeffs: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Back-project 2D points into 3D world coords at given depth
    """

    original_shape = points2d.shape

    points2d_flat = points2d.reshape((-1, 2))  # (N, 2)

    # undistort if needed
    if dist_coeffs is not None:
        points2d_flat = undistort_points(
            points2d_flat,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            R=jnp.eye(3),  # no rectification
            P=camera_matrix,  # reproject into same camera
        )

    # make homogeneous image coords [u, v, 1]
    ones = jnp.ones((points2d_flat.shape[0], 1), dtype=points2d_flat.dtype)  # (N, 1)
    hom2d = jnp.concatenate([points2d_flat, ones], axis=1)  # (N, 3)

    # normalized camera coords: K^-1 @ hom2d  (3, N).T -> (N, 3)
    invK = jnp.linalg.inv(camera_matrix)
    cam_dirs = (invK @ hom2d.T).T  # (N, 3)

    # apply depth (broadcast if scalar)
    # The depth should broadcast to the flattened shape
    depth_arr = jnp.asarray(depth)  # Let JAX handle broadcasting
    cam_pts = cam_dirs * depth_arr.reshape(-1, 1)  # Ensure depth is (N, 1) or (1, 1) for broadcasting

    # camera -> world
    E_inv = invert_extrinsics_matrix(extrinsics_matrix)  # (..., 4, 4)

    # build homogeneous cam_pts [Xc, 1]
    ones4 = jnp.ones((cam_pts.shape[0], 1), dtype=cam_pts.dtype)
    hom_cam = jnp.concatenate([cam_pts, ones4], axis=1)  # (N, 4)

    # E_inv @ hom_cam.T   (..., 4, N) → .T → (N, 4)
    world_h = (E_inv @ hom_cam.T).T  # (N, 4)
    world_pts = world_h[:, :3]  # (N, 3)

    # The final dimension is 3 (for x, y, z), the leading dimensions are from the original input
    out_shape = original_shape[:-1] + (3,)
    return world_pts.reshape(out_shape)

# Do the back projection over n cameras
back_projection_batched = jax.jit(
    jax.vmap(
        back_projection,
        in_axes=(0, None, 0, 0, 0),
        out_axes=0
    )
)

## ---- Main functions - Used in Bundle adjustment for calibration, and in most of the downstream tasks


@jax.jit
def triangulate_one(
    points2d:       jnp.ndarray,
    P_mats:         jnp.ndarray,
    weight:         float = 1.0,
    lambda_reg:     float = 0.0
) -> jnp.ndarray:
    """
    Triangulate one point from C 2D observations and their corresponding projection matrices

        For each i-th 2D point and its corresponding camera matrix, two rows are added to matrix A:

               [ u_1 * P_1_3 - P_1_1 ]
               [ v_1 * P_1_3 - P_1_2 ]
        A =    [ u_2 * P_2_3 - P_2_1 ]
               [ v_2 * P_2_3 - P_2_2 ]
               [          ...        ]
               [          ...        ]
               [          ...        ]

        where P_i_j denotes the j-th row of the i-th camera matrix

    We use SVD to solve the system AX=0. The solution X is the last row of V^t from SVD

    See https://people.math.wisc.edu/~chr/am205/g_act/svd_slides.pdf for more info and sources

    Args:
        points2d: the 2D observations from C cameras (C, 2)
        P_mats: C projection matrices (C, 3, 4)
        weight: weight for the 2D points confidence (C,) or None
        lambda_reg: Regularisation term for Tikhonov Regularisation

    Returns:
        point3d: the 3D coordinates for the point (3,)

    """

    w = jnp.atleast_2d(weight)[:, None]

    P0, P1, P2 = P_mats[:, 0, :], P_mats[:, 1, :], P_mats[:, 2, :]
    u, v = points2d[:, 0], points2d[:, 1]

    # build the (2*C, 4) A-matrix for a single point
    r1 = (u[:, None] * P2 - P0) * w               # (C, 4)
    r2 = (v[:, None] * P2 - P1) * w               # (C, 4)
    A = jnp.concatenate([r1, r2], axis=0)   # (2*C, 4)

    if lambda_reg != 0.0:
        ATA = A.T@A + lambda_reg * jnp.eye(4)
        _, _, Vt = jnp.linalg.svd(ATA, full_matrices=False)
    else:
        _, _, Vt = jnp.linalg.svd(A, full_matrices=False)

    X = Vt[-1]
    X = X / (X[-1] + _eps)
    point3d = X[:3]
    return point3d  # (3,)


@jax.jit
def triangulate_svd(
    points2d:   jnp.ndarray,
    P_mats:     jnp.ndarray,
    weights:    Optional[jnp.ndarray] = None,
    lambda_reg: float = 0.0
) -> jnp.ndarray:
    """
    Triangulate 3D points from multiple 2D points and their corresponding projection matrices

        For each i-th 2D point and its corresponding camera matrix, two rows are added to matrix A:

               [ u_1 * P_1_3 - P_1_1 ]
               [ v_1 * P_1_3 - P_1_2 ]
        A =    [ u_2 * P_2_3 - P_2_1 ]
               [ v_2 * P_2_3 - P_2_2 ]
               [          ...        ]
               [          ...        ]
               [          ...        ]

        where P_i_j denotes the j-th row of the i-th camera matrix

    We use SVD to solve the system AX=0. The solution X is the last row of V^t from SVD

    See https://people.math.wisc.edu/~chr/am205/g_act/svd_slides.pdf for more info and sources

    Args:
        points2d: N 2D points from C cameras (C, N, 2)
        P_mats: C projection matrices (C, 3, 4)
        weights: weights for 2D points confidences (C, N, 3) or None
        lambda_reg: Regularisation term for Tikhonov Regularisation

    Returns:
        points3d: N 3D points coordinates (N, 3)

    """

    # We want to get rid of invalid values
    valid_observations = jnp.isfinite(points2d[..., 0]) & jnp.isfinite(points2d[..., 1])  # (C, N)
    # how many cams saw each point
    n_obs = jnp.sum(valid_observations, axis=0)  # (N,)

    #so we zero-fill u and v
    u = jnp.where(valid_observations, points2d[..., 0], 0.0)  # (C, N)
    v = jnp.where(valid_observations, points2d[..., 1], 0.0)  # (C, N)
    if weights is None:
        w = valid_observations.astype(points2d.dtype)            # (C, N)
    else:
        weights = jnp.asarray(weights)
        w = jnp.where(valid_observations, weights, 0.0)  # and also zero out any weight where data was invalid

    # pull out the three rows of each Projection matrix and insert a dummy 'point' axis
    P0 = P_mats[:, 0, :][:, None, :]  # (C, 1, 4)
    P1 = P_mats[:, 1, :][:, None, :]  # (C, 1, 4)
    P2 = P_mats[:, 2, :][:, None, :]  # (C, 1, 4)

    u_exp = u[:, :, None]  # (C, N, 1)
    v_exp = v[:, :, None]  # (C, N, 1)
    w_exp = w[:, :, None]  # (C, N, 1)

    # Build all the A's at once
    # row1_i(p) = u[i,p] * P2[i] - P0[i]
    # row2_i(p) = v[i,p] * P2[i] - P1[i]
    # then each multiplied by w[i, p]
    r1 = (u_exp * P2 - P0) * w_exp
    r2 = (v_exp * P2 - P1) * w_exp

    # stack the u and v contributions, then transpose -> (N, 2*C, 4)
    A = jnp.concatenate([r1, r2], axis=0).transpose(1, 0, 2)

    # Batched SVD
    if lambda_reg != 0.0:
        # build A^T A + λI for each point
        ATA = jnp.einsum('pni,pnj->pij', A, A) + lambda_reg * jnp.eye(4)
        _, _, Vh = jnp.linalg.svd(ATA, full_matrices=False)
        Xh = Vh[:, -1, :]      # (N, 4)
    else:
        _, _, Vh = jnp.linalg.svd(A, full_matrices=False)
        Xh = Vh[:, -1, :]

    # Dehomogenize
    Xh = Xh / (Xh[:, 3:4] + _eps)
    points3d = Xh[:, :3]    # (N, 3)

    # we only trust points seen by 2 cameras or more so we set the unreliable ones back to nan
    reliable = n_obs >= 2  # (N,)
    reliable = reliable[:, None]
    points3d = jnp.where(reliable, points3d, jnp.nan)
    return points3d


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


## Other utils - used in 3D visualisation mostly


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


## Rotate stuff - used in 3D visualisation

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

# TODO - these 3 below need to be reworked

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
