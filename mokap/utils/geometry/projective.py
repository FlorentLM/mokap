from typing import Tuple, Union, Optional
import jax
from jax import numpy as jnp
from mokap.utils.geometry.transforms import rodrigues, extrinsics_matrix, invert_extrinsics_matrix, projection_matrix
from mokap.utils.geometry.utils import pad_dist_coeffs

_eps = 1e-8

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

    # TODO: We prob want to support the rational model too
    # # Rational model
    # numerator = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    # denominator = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    # radial = numerator / (denominator + _eps)

    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4 + k5*r2**5 + k6*r2**6

    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return radial, dx, dy


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


undistort_multiple = jax.jit(
    jax.vmap(undistort_points,
             in_axes=(
                0,     # each camera has its own points2d
                0,     # its own camera_matrix
                0),     # its own dist_coeffs
             out_axes=0),
    static_argnums=()
)


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


back_projection_batched = jax.jit(
    jax.vmap(
        back_projection,
        in_axes=(0, None, 0, 0, 0),
        out_axes=0
    )
)


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


def triangulation(
        points2d:           jnp.ndarray,
        visibility_mask:    jnp.ndarray,
        rvecs_world:        jnp.ndarray,
        tvecs_world:        jnp.ndarray,
        camera_matrices:    jnp.ndarray,
        dist_coeffs:        jnp.ndarray,
) -> jnp.ndarray:
    """
    Triangulate points 2D seen by C cameras

    Args:
        points2d: points 2D detected by the C cameras (C, N, 2)
        visibility_mask: visibility mask for points 2D (C, N)
        rvecs_world: C rotation vectors (C, 3)
        tvecs_world: C translation vectors (C, 3)
        camera_matrices: C camera matrices (C, 3, 3)
        dist_coeffs: C distortion coefficients (C, ≤8)

    Returns:
        points3d: N 3D coordinates (N, 3)

    """

    # TODO: we probably want to get rid of this wrapper and use P matrices anywhere we can

    # this is converted back to a float array because the triangulate_svd accepts actual weights
    # we can multiply the visibility weights by a confidence score
    visibility_mask = visibility_mask.astype(jnp.float32)

    # Recover camera-centric extrinsics matrices and compute the projection matrices
    E_all = extrinsics_matrix(rvecs_world, tvecs_world)    # (C, 4, 4)
    E_inv_all = invert_extrinsics_matrix(E_all)            # (C, 4, 4)
    P_all = projection_matrix(camera_matrices, E_inv_all)  # (C, 3, 4)

    pts2d_ud = undistort_multiple(points2d, camera_matrices, dist_coeffs)
    pts3d = triangulate_svd(pts2d_ud, P_all, weights=visibility_mask)  # (N, 3)

    return pts3d
