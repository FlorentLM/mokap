from typing import Tuple, Union, Optional
import jax
from jax import numpy as jnp
from mokap.utils.geometry.transforms import rodrigues, extrinsics_matrix, invert_extrinsics_matrix, projection_matrix
from mokap.utils.jax_utils import pad_to_length

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

    k1, k2, p1, p2, k3, k4, k5, k6 = pad_to_length(dist_coeffs.ravel(), 8)
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

# batched version for projecting a set of points into multiple cameras
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

# undistort a set of points in multiple cameras
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

# back-project a set of points into multiple cameras
back_projection_batched = jax.jit(
    jax.vmap(
        back_projection,
        in_axes=(0, None, 0, 0, 0),
        out_axes=0
    )
)


@jax.jit
def triangulate_points_from_projections(
        points2d: jnp.ndarray,  # (C, N, 2)
        P_mats: jnp.ndarray,  # (C, 3, 4)
        weights: Optional[jnp.ndarray] = None,  # (C, N)
        lambda_reg: float = 0.0
) -> jnp.ndarray:
    """
    Triangulates N 3D points from C 2D observations using their corresponding projection matrices
    This is the core batched triangulation function using SVD

    Args:
        points2d: N 2D points from C cameras (C, N, 2), NaN values are ignored
        P_mats: C projection matrices (C, 3, 4)
        weights: Optional confidence weights for each 2D observation (C, N)
                 If None, visibility is inferred from NaNs in points2d and assigned a weight of 1.0
                 A weight of 0 indicates an invalid/invisible point
        lambda_reg: Regularisation term for Tikhonov Regularisation.

    Returns:
        points3d: N 3D points coordinates (N, 3). Unreliable points (seen by < 2 cameras) are NaN
    """

    # We want to get rid of invalid values
    valid_observations = jnp.isfinite(points2d[..., 0])  # (C, N)
    n_obs = jnp.sum(valid_observations, axis=0)  # (N,)

    u = jnp.where(valid_observations, points2d[..., 0], 0.0)
    v = jnp.where(valid_observations, points2d[..., 1], 0.0)

    if weights is None:
        w = valid_observations.astype(points2d.dtype)
    else:
        weights = jnp.asarray(weights)
        w = jnp.where(valid_observations, weights, 0.0)

    P0 = P_mats[:, None, 0, :]  # (C, 1, 4)
    P1 = P_mats[:, None, 1, :]  # (C, 1, 4)
    P2 = P_mats[:, None, 2, :]  # (C, 1, 4)

    u_exp = u[..., None]  # (C, N, 1)
    v_exp = v[..., None]  # (C, N, 1)
    w_exp = w[..., None]  # (C, N, 1)

    r1 = (u_exp * P2 - P0) * w_exp
    r2 = (v_exp * P2 - P1) * w_exp

    A = jnp.concatenate([r1, r2], axis=0).transpose(1, 0, 2)  # (N, 2*C, 4)

    if lambda_reg != 0.0:
        # build A^T A + lambda I for each point
        ATA = jnp.einsum('pni,pnj->pij', A, A) + lambda_reg * jnp.eye(4)
        _, _, Vh = jnp.linalg.svd(ATA, full_matrices=False)
    else:
        _, _, Vh = jnp.linalg.svd(A, full_matrices=False)

    # Dehomogenize
    Xh = Vh[:, -1, :]  # (N, 4)
    Xh = Xh / (Xh[:, 3:4] + _eps)
    points3d = Xh[:, :3]

    reliable = (n_obs >= 2)[:, None]
    return jnp.where(reliable, points3d, jnp.nan)


@jax.jit
def triangulate(
        points2d: jnp.ndarray,  # (C, N, 2)
        camera_matrices: jnp.ndarray,  # (C, 3, 3)
        dist_coeffs: jnp.ndarray,  # (C, <=8)
        rvecs: jnp.ndarray,  # (C, 3)
        tvecs: jnp.ndarray,  # (C, 3)
        weights: Optional[jnp.ndarray] = None  # (C, N)
) -> jnp.ndarray:
    """
    Triangulates 3D points from 2D observations across C cameras

    High-level wrapper that handles undistortion, projection matrix calculation,
    and calls the core triangulation solver

    Args:
        points2d: Points 2D detected by the C cameras (C, N, 2)
        camera_matrices: C camera matrices (C, 3, 3)
        dist_coeffs: C distortion coefficients (C, <=8)
        rvecs: C rotation vectors (world-to-camera) (C, 3)
        tvecs: C translation vectors (world-to-camera) (C, 3)
        weights: Optional confidence weights for each 2D observation (C, N)
                 If None, visibility is inferred from NaNs in points2d and used as a binary weight
    Returns:
        points3d: N 3D coordinates (N, 3)
    """

    # Undistort points first
    pts2d_ud = undistort_multiple(points2d, camera_matrices, dist_coeffs)

    # if no mask is provided, infer it
    if weights is None:
        # undistortion might also introduce NaNs, so using the original points2d allows to be exhaustive
        weights = jnp.isfinite(points2d[..., 0])

    weights = weights.astype(jnp.float32)

    # Recover camera-centric extrinsics matrices and compute the projection matrices
    E_all = extrinsics_matrix(rvecs, tvecs)
    P_all = projection_matrix(camera_matrices, E_all)

    # Call the core, batched triangulation function
    pts3d = triangulate_points_from_projections(pts2d_ud, P_all, weights=weights)
    return pts3d


@jax.jit
def find_rays_intersection_3d(
    ray_origins:     jnp.ndarray, # (C, 3)
    ray_directions:  jnp.ndarray  # (C, 3)
) -> jnp.ndarray:
    """
    Finds the 3D point that minimizes the sum of squared distances to a set of rays
    Each ray is defined by an origin point and a direction vector

    Args:
        ray_origins: C origin points for C rays (C, 3)
        ray_directions: C unit direction vectors for C rays (C, 3)

    Returns:
        intersection_point: The point of closest intersection (3,)
    """

    D = ray_directions[..., :, None]
    A = jnp.eye(3)[None, ...] - D @ D.transpose(0, 2, 1)

    C = ray_origins[..., :, None]
    b = (A @ C)[..., 0]

    A_stack = A.reshape(-1, 3)
    b_stack = b.reshape(-1)
    intersection_point, *_ = jnp.linalg.lstsq(A_stack, b_stack, rcond=None)
    return intersection_point
