from typing import Tuple, Union, Optional
import jax
from jax import numpy as jnp
from mokap.utils.geometry.transforms import rodrigues, extrinsics_matrix, invert_extrinsics_matrix, projection_matrix, \
    extmat_to_rtvecs, invert_rtvecs


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

    current_len = dist_coeffs.shape[0]
    padding_needed = max(0, 8 - current_len)
    dist_coeffs_padded = jnp.pad(dist_coeffs, (0, padding_needed))

    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs_padded[:8]
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

    Fundamental projection function. Projects points from some coordinate
    system into the image plane of a camera.

    Principal application is to project from world coordinates (using world-to-cam rtvecs)

    Args:
        object_points: Points in the source coordinate system (..., 3)
        rvec: Rotation vector for transform from source system to camera system (3,)
        tvec: Translation vector for transform from source system to camera system (3,)
        camera_matrix: Camera intrinsics matrix (3, 3)
        dist_coeffs: Camera distortion coefficients (≤8,)

    Returns:
        image_points: Projected 2D points in the image plane (..., 2)
    """

    R = rodrigues(rvec)

    Xc = jnp.einsum('ij,...j->...i', R, object_points) + tvec

    z = Xc[..., 2]
    z_safe = jnp.maximum(z, _eps)
    x, y = Xc[..., 0] / z_safe, Xc[..., 1] / z_safe

    radial, dx, dy = distortion(x, y, dist_coeffs)
    x_d = x * radial + dx
    y_d = y * radial + dy

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return jnp.stack([fx * x_d + cx, fy * y_d + cy], axis=-1)


# batched version for projecting a single set of points into multiple cameras
project_to_multiple_cameras = jax.jit(
    jax.vmap(
        project_points,
        in_axes=(None, 0, 0, 0, 0) # same points to project to everyone, and map over camera params
    )
)

# projects multiple sets of world points into multiple cameras
project_multiple_to_multiple = jax.jit(
    jax.vmap( # vmap over different sets of points (e.g., from different poses P)
        project_to_multiple_cameras,
        in_axes=(0, None, None, None, None), # map over points, keep camera params the same
        out_axes=1 # put the new mapped axis (P) after the camera axis (C)
    )
)


@jax.jit
def project_object_to_camera(
    object_points:  jnp.ndarray,  # (N, 3) in local object frame
    r_w2c:          jnp.ndarray,  # (3,) world-to-camera rotation
    t_w2c:          jnp.ndarray,  # (3,) world-to-camera translation
    r_o2w:          jnp.ndarray,  # (3,) object-to-world rotation
    t_o2w:          jnp.ndarray,  # (3,) object-to-world translation
    camera_matrix:  jnp.ndarray,  # (3, 3)
    dist_coeffs:    jnp.ndarray   # (D,)
) -> jnp.ndarray:
    """
    Projects 3D points from an object's local frame into a camera view by
    composing object-to-world and world-to-camera poses
    Used during calibration and bundle adjustment
    """
    # Compose poses: world -> camera and object -> world  ==>  object -> camera
    E_w2c = extrinsics_matrix(r_w2c, t_w2c)
    E_o2w = extrinsics_matrix(r_o2w, t_o2w)
    E_o2c = E_w2c @ E_o2w

    # Get the combined rvec/tvec for the final projection
    r_o2c, t_o2c = extmat_to_rtvecs(E_o2c)                      # TODO all the projection functions should use ext mats

    return project_points(object_points, r_o2c, t_o2c, camera_matrix, dist_coeffs)


# double-vmap for projecting N points from P poses into C cameras
project_object_views_batched = jax.jit(
    jax.vmap(       # vmap over cameras (C)
        jax.vmap(   # vmap over object poses (P)
            project_object_to_camera,
            # vmap over object-to-world poses (r_o2w, t_o2w)
            in_axes=(None, None, None, 0, 0, None, None)
        ),
        # vmap over camera parameters (r_w2c, t_w2c, K, D)
        in_axes=(None, 0, 0, None, None, 0, 0)
    )
    # Expected inputs for a (C, P, N, 2) output:
    # object_points: (N, 3)
    # r_w2c: (C, 3)
    # t_w2c: (C, 3)
    # r_o2w: (P, 3)
    # t_o2w: (P, 3)
    # camera_matrix: (C, 3, 3)
    # dist_coeffs: (C, D)
)


@jax.jit
def reproject_and_compute_error(
    world_points:       jnp.ndarray,    # (P, N, 3) or (N, 3)
    camera_matrices:    jnp.ndarray,    # (C, 3, 3)
    dist_coeffs:        jnp.ndarray,    # (C, D)
    cams_rc2w:          jnp.ndarray,    # (C, 3)
    cams_tc2w:          jnp.ndarray,    # (C, 3)
    observed_pts2d:     jnp.ndarray,    # (C, P, N, 2)
    visibility_mask:    jnp.ndarray     # (C, P, N)
) -> jnp.ndarray:
    """
    Reprojects 3D world points into multiple cameras and computes the pixel error

    This is a general-purpose function that works with any set of 3D world points

    Args:
        world_points: 3D points in the world coordinate system.
                      Can be a single set (N, 3) or multiple sets (P, N, 3)
        camera_matrices: Camera intrinsic matrices for C cameras
        dist_coeffs: Distortion coefficients for C cameras
        cams_rc2w: Camera-to-world rotation vectors for C cameras
        cams_tc2w: Camera-to-world translation vectors for C cameras
        observed_pts2d: The observed 2D points in each camera's image plane
        visibility_mask: A boolean mask indicating if a point was observed

    Returns:
        errors: An array of shape (C, P, N) with reprojection errors in pixels
                Non-visible points are marked with NaN
    """

    # Ensure world_points has a "pose" dimension for broadcasting
    if world_points.ndim == 2:
        world_points = world_points[None, ...] # (1, N, 3)
        observed_pts2d = observed_pts2d[:, None, ...] if observed_pts2d.ndim == 3 else observed_pts2d
        visibility_mask = visibility_mask[:, None, ...] if visibility_mask.ndim == 2 else visibility_mask

    # get world -> camera transforms
    r_w2c, t_w2c = invert_rtvecs(cams_rc2w, cams_tc2w)

    # massive projection: all world points, from all poses, into all cameras
    reprojected_pts = project_multiple_to_multiple(world_points, r_w2c, t_w2c, camera_matrices, dist_coeffs) # (C, P, N, 2)

    errors = jnp.linalg.norm(observed_pts2d - reprojected_pts, axis=-1)
    return jnp.where(visibility_mask, errors, jnp.nan)


def _undistort_points(
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

    # fallback to default R and P (runs in host code)   # TODO: ummmmm????
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


undistort_points = jax.jit(_undistort_points)

# undistort a set of points in multiple cameras
undistort_multiple = jax.jit(
    jax.vmap(_undistort_points,
             in_axes=(
                0,     # each camera has its own points2d
                0,     # its own camera_matrix
                0),     # its own dist_coeffs
             out_axes=0),
    static_argnums=()
)


def _back_projection(
        points2d: jnp.ndarray,
        depth: Union[float, jnp.ndarray],
        camera_matrix: jnp.ndarray,
        E_w2c: jnp.ndarray,
        dist_coeffs: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Back-project 2D points into 3D world coords at given depth
    """

    points2d_flat = points2d.reshape((-1, 2))  # (N, 2)

    # undistort if needed
    if dist_coeffs is not None:
        points2d_flat = undistort_points(
            points2d_flat,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            R=jnp.eye(3),       # no rectification
            P=camera_matrix,    # reproject into same camera
        )

    # make homogeneous image coords [u, v, 1]
    ones = jnp.ones((points2d_flat.shape[0], 1), dtype=points2d_flat.dtype)  # (N, 1)
    hom2d = jnp.concatenate([points2d_flat, ones], axis=1)  # (N, 3)

    # normalized camera coords: K^-1 @ hom2d  (3, N).T -> (N, 3)
    invK = jnp.linalg.inv(camera_matrix)
    cam_dirs = (invK @ hom2d.T).T  # (N, 3)

    # apply depth (broadcast if scalar)
    # The depth should broadcast to the flattened shape
    depth_arr = jnp.asarray(depth)
    cam_pts = cam_dirs * depth_arr[..., None]

    # camera -> world
    E_c2w = invert_extrinsics_matrix(E_w2c)  # (..., 4, 4)

    # build homogeneous cam_pts [Xc, 1]
    ones4 = jnp.ones((cam_pts.shape[0], 1), dtype=cam_pts.dtype)
    hom_cam = jnp.concatenate([cam_pts, ones4], axis=1)  # (N, 4)

    # E_c2w @ hom_cam.T   (..., 4, N) → .T → (N, 4)
    world_h = (E_c2w @ hom_cam.T).T  # (N, 4)
    world_pts = world_h[:, :3]  # (N, 3)

    return world_pts


back_projection = jax.jit(_back_projection)

# back-project (the same number of) points to multiple cameras
back_projection_batched = jax.jit(
    jax.vmap(
        _back_projection,
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

