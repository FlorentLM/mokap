from typing import Optional, Any, Dict, Iterable, Sequence, Union
import matplotlib
import numpy as np
from mokap.utils.geometry.fitting import rays_intersection_3d
from mokap.utils.geometry.projective import back_projection_batched
from mokap.utils.geometry.transforms import extrinsics_matrix, invert_extrinsics_matrix
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from jax.typing import ArrayLike
import jax.numpy as jnp


CUSTOM_COLORS = ['#9B5DE5', '#EF476F', '#FFD166', '#00BBF9', '#00F5D4', '#118ab2', '#073b4c', '#ee6c4d']


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    # From https://stackoverflow.com/a/18926541
    import matplotlib.colors as colors
    return colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                                                    cmap(np.linspace(minval, maxval, n)))


def plot_box_3d(
        centre: ArrayLike,
        size:   ArrayLike,
        color:  str = 'k',
        alpha:  float = 0.1,
        ax:     Optional[Axes3D] = None,
) -> Axes3D:

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    coords = np.indices((2, 2, 2)).reshape(3, -1).T
    v = (coords - 0.5) * np.array(size) + np.array(centre)

    faces_idx = np.array([
        [0, 1, 3, 2],   # bottom
        [4, 5, 7, 6],   # top
        [0, 1, 5, 4],   # back
        [2, 3, 7, 6],   # front
        [0, 2, 6, 4],   # left
        [1, 3, 7, 5],   # right
    ])
    faces = v[faces_idx]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.1, edgecolors=color, alpha=alpha))

    return ax


def plot_ellipsoid_3d(
        centre: ArrayLike,
        size:   ArrayLike,
        color:  str = 'k',
        alpha:  float = 0.1,
        resolution: int = 30,
        ax:     Optional[Axes3D] = None,
) -> Axes3D:

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    centre = np.asarray(centre)
    radii = np.asarray(size) / 2.0

    # Generate the surface points of a unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale and translate to create the ellipsoid
    x = radii[0] * x + centre[0]
    y = radii[1] * y + centre[1]
    z = radii[2] * z + centre[2]

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=4, cstride=4, linewidth=0)

    return ax


def plot_cameras_3d(
        rvecs_c2w:          ArrayLike,
        tvecs_c2w:          ArrayLike,
        camera_matrices:    ArrayLike,
        dist_coeffs:        ArrayLike,
        imsizes:            ArrayLike = np.array([1440, 1080]),
        cameras_names:      Optional[Sequence[Any]] = None,
        depth:              Optional[Union[float, ArrayLike]] = None,
        depth_ratio:        float = 0.75,
        colors:             Optional[Sequence[str]] = None,
        trust_volume:       Optional[Dict[str, ArrayLike]] = None,
        ax:                 Optional[Axes3D] = None,
) -> Axes3D:

    """ Matplotlib 3D plot for viewing C cameras, with their frustums, and the global focal point """

    if rvecs_c2w.ndim != 2 or tvecs_c2w.ndim != 2 or camera_matrices.ndim != 3 or dist_coeffs.ndim != 2:
        raise ValueError('This function should be called for C cameras!')

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = CUSTOM_COLORS

    C = camera_matrices.shape[0]

    if cameras_names is None:
        cameras_names = [f'Cam #{c}' for c in range(C)]

    images_sizes = np.asarray(imsizes)
    if images_sizes.ndim == 1:
        images_sizes = np.vstack([images_sizes] * C)

    unit_coords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],         # need to repeat the first one for Poly3DCollection
        [0.5, 0.5],     # centre point
    ], dtype=np.float32)
    frustums_2d = (images_sizes[:, None, :] * unit_coords[None, :, :]).astype(np.float32)

    # Plot the axes arrows
    axes_length = 10
    ax.quiver(*[0, 0, 0], *[axes_length, 0, 0], color='r', alpha=0.5)
    ax.quiver(*[0, 0, 0], *[0, axes_length, 0], color='g', alpha=0.5)
    ax.quiver(*[0, 0, 0], *[0, 0, axes_length], color='b', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    E_c2w = extrinsics_matrix(rvecs_c2w, tvecs_c2w)

    # First, find the shared focal point. The calculation is independent of the initial depth
    # used for back-projection, we only need the normalized direction vectors
    # so we use a dummy depth of 1.0 to get the directions
    frustums_for_direction = back_projection_batched(
        frustums_2d, jnp.ones(C), camera_matrices, E_c2w, dist_coeffs, distortion_model='full'
    )

    directions = frustums_for_direction[:, -1] - tvecs_c2w
    directions_normalised = directions / np.linalg.norm(directions, axis=1)[:, None]
    focal_point = rays_intersection_3d(tvecs_c2w, directions_normalised)

    # Determine the plotting depths
    if depth is None:
        # Automatic mode: depth is 3/4 the distance from each camera to the focal point
        distances_to_focal = jnp.linalg.norm(tvecs_c2w - focal_point, axis=1)
        plot_depths = distances_to_focal * depth_ratio
    else:
        # Manual override: use the fixed depth for all cameras
        plot_depths = jnp.array([depth] * C)

    # Calculate the final frustums for plotting using the determined depths
    frustums_3d = back_projection_batched(
        frustums_2d,
        plot_depths,
        camera_matrices,
        E_c2w,
        dist_coeffs,
        distortion_model='full'
    )

    for n in range(C):
        col = colors[n]

        # Cameras positions (optical centres)
        ax.scatter(*tvecs_c2w[n], color=col, label=cameras_names[n], alpha=1.0)

        # Frustum plans
        ax.add_collection3d(
            Poly3DCollection([frustums_3d[n, :-1]], facecolors=col, edgecolors=col, linewidths=1, linestyles='-',
                             alpha=0.05))

        # Frustum lines
        for corner in frustums_3d[n, :-2]:
            ax.plot(*np.stack([tvecs_c2w[n], corner]).T, color=col, linestyle='-', linewidth=0.25, alpha=0.5)

        # Optical axis
        ax.plot(*np.stack([tvecs_c2w[n], frustums_3d[n, -1]]).T, color=col, linestyle='--', linewidth=1.0, alpha=0.5)

    ax.scatter(*focal_point, marker='*', color='k', s=25)

    if trust_volume is not None:
        # Check the format of the trust_volume dict to determine behavior
        first_value = next(iter(trust_volume.values()))

        if np.isscalar(first_value):
            # Case 1: Values are extents (sizes) so we center the box on the shared focal point
            volume_centre = focal_point
            volume_size = [trust_volume['x'], trust_volume['y'], trust_volume['z']]
        else:
            # Case 2: Values are ranges (min, max). Calculate the box's own center and size
            x_min, x_max = trust_volume['x']
            y_min, y_max = trust_volume['y']
            z_min, z_max = trust_volume['z']
            volume_centre = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            volume_size = [x_max - x_min, y_max - y_min, z_max - z_min]

        ax = plot_box_3d(centre=volume_centre, size=volume_size, ax=ax, color='green', alpha=0.15)
        # ax = plot_ellipsoid_3d(centre=volume_centre, size=volume_size, ax=ax, color='green', alpha=0.15)
        ax.scatter(*volume_centre, marker='s', color='green', s=25)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    return ax


def plot_points_3d(
        points3d:       ArrayLike,
        points_names:   Optional[Sequence[Any]] = None,
        errors:         Optional[ArrayLike] = None,
        color:          str = 'k',
        label:          str = '3D points',
        ax:             Optional[Axes3D] = None,
) -> Axes3D:
    """ Matplotlib 3D plot for points, their names and the associated errors """

    if points3d.ndim != 2:
        raise ValueError('This function should be called for N 3D points!')

    points3d = np.asarray(points3d)

    if errors is not None:
        errors = np.asarray(errors)
        assert points3d.shape[0] == errors.shape[0]

    if points_names is None or len(points_names) != points3d.shape[0]:
        points_names = [str(i) for i in range(points3d.shape[0])]

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = points3d.T

    if errors is not None:
        colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)
        pts_scatter = ax.scatter(xs, ys, zs,
                                 c=errors, cmap=colormap, norm=normalize,
                                 marker='o', label=label, alpha=0.5)
    else:
        pts_scatter = ax.scatter(xs, ys, zs,
                                 color=color,
                                 marker='o', label=label, alpha=0.5)

    for p, name in enumerate(points_names):
        if errors is not None:
            c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else color
        else:
            c = color
        ax.text(xs[p], ys[p], zs[p], f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.legend()

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def plot_object_3d(
        object_points:  ArrayLike,
        rvec_w:         ArrayLike,
        tvec_w:         ArrayLike,
        color:          str = 'blue',
        label:          str = 'Object Ground Truth',
        ax:             Optional[Axes3D] = None,
) -> Axes3D:
    """
    Plots a 3D object (like a calibration pattern) given its local points and its pose in the world

    Args:
        object_points: The (N, 3) points of the board in its own local coordinate system (often with z=0)
        rvec_w: The rotation vector (3,) that transforms the board from its local frame to the world frame
        tvec_w: The translation vector (3,) that transforms the board from its local frame to the world frame
        color: The color for the board points
        label: The legend label for the board points
        ax: Optional existing Matplotlib Axes3D object
    """

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    # Create the 4x4 transformation matrix from the board's local frame to the world frame
    board_pose_matrix = extrinsics_matrix(rvec_w, tvec_w)

    # Convert local points to homogeneous coordinates
    local_points_hom = np.hstack([
        np.asarray(object_points),
        np.ones((np.asarray(object_points).shape[0], 1))
    ])

    # Apply the transformation to get the points in world coordinates
    world_points_hom = (board_pose_matrix @ local_points_hom.T).T
    world_points_3d = world_points_hom[:, :3]

    # Use the existing point plotter to display the board
    ax = plot_points_3d(
        points3d=world_points_3d,
        color=color,
        label=label,
        ax=ax
    )

    return ax


def plot_points2d_3d(
        points2d:           ArrayLike,
        rvecs_c2w:          ArrayLike,
        tvecs_c2w:          ArrayLike,
        camera_matrices:    ArrayLike,
        dist_coeffs:        ArrayLike,
        depth:              float = 10.0,
        points_names:       Optional[Iterable[Any]] = None,
        errors:             Optional[ArrayLike] = None,
        colors:             Optional[str] = None,
        ax:                 Optional[Axes3D] = None,
) -> Axes3D:

    if points2d.ndim != 3 or rvecs_c2w.ndim != 2 or tvecs_c2w.ndim != 2 or camera_matrices.ndim != 3 or dist_coeffs.ndim != 2:
        raise ValueError('This function should be called for CxN 2D points!')

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    C = points2d.shape[0]

    if colors is None:
        colors = CUSTOM_COLORS

    E_c2w = extrinsics_matrix(rvecs_c2w, tvecs_c2w)
    points2d_3d = back_projection_batched(points2d, jnp.asarray([depth] * C), camera_matrices, E_c2w, dist_coeffs, distortion_model='full')

    for n in range(C):

        xs, ys, zs = points2d_3d[n].T

        if errors is not None:
            colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)

            pts_scatter = ax.scatter(xs, ys, zs, s=10,
                                     c=errors[n], cmap=colormap, norm=normalize,
                                     marker='.', label='3D points', alpha=0.5)
        else:
            pts_scatter = ax.scatter(xs, ys, zs, s=10,
                                     c=colors[n],
                                     marker='.', label='3D points', alpha=0.5)

        if points_names is not None:
            for p, name in enumerate(points_names):
                if errors is not None:
                    c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else colors[n]
                else:
                    c = colors[n]
                ax.text(xs[p], ys[p], zs[p], f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def plot_triangulation_scene(
        points3d:           ArrayLike,
        points2d:           ArrayLike,
        rvecs_c2w:          ArrayLike,
        tvecs_c2w:          ArrayLike,
        camera_matrices:    ArrayLike,
        dist_coeffs:        ArrayLike,
        visibility_mask:    Optional[ArrayLike] = None,
        points_names:       Optional[Sequence[Any]] = None,
        errors:             Optional[ArrayLike] = None,
        cameras_names:      Optional[Sequence[Any]] = None,
        imsizes:            ArrayLike = np.array([1440, 1080]),
        frustums_depth:     float = 0.9,
        detections_depth:   float = 0.95,
        colors:             Optional[Sequence[str]] = None,
        trust_volume:       Optional[Dict[str, ArrayLike]] = None,
        object_rvec_w:      Optional[ArrayLike] = None,
        object_tvec_w:      Optional[ArrayLike] = None,
        object_points:      Optional[ArrayLike] = None,
        ax:                 Optional[Axes3D] = None,
) -> Axes3D:
    """
    Comprehensive 3D plot of a triangulation scene

    This function orchestrates several plotting utilities to show:
    1. Cameras, their names, and their viewing frustums
    2. The final triangulated 3D points
    3. The 2D detections back-projected into 3D space
    4. Lines (rays) from camera centers to their corresponding back-projected points
    5. Optionally a ground-truth object

    The depth of back-projected points is dynamically calculated to be a factor of the
    distance from the camera to the final 3D point

    Args:
        points3d: Triangulated 3D points (N, 3)
        points2d: 2D detections for each camera (C, N, 2)
        rvecs_c2w, tvecs_c2w: Camera-to-world extrinsics (C, 3) and (C, 3)
        camera_matrices: Camera intrinsics (C, 3, 3)
        dist_coeffs: Distortion coefficients (C, D)
        visibility_mask: Boolean mask for valid 2D points (C, N)
        points_names: Optional names for the N points
        errors: Optional per-point errors for coloring
        cameras_names: Optional names for the C cameras
        imsizes: Image dimensions (width, height) for frustum plotting
        frustums_depth: Ratio for camera frustum depth relative to focal point (default: 90% of the way)
        detections_depth: Where to place back-projected points along the ray to the 3D point (default: 95% of the way)
        colors: Optional list of colors for cameras
        trust_volume: Optional dictionary defining a bounding box to plot (or ranges in the three axes)
        object_rvec_w: Optional rvec (3,) for the ground truth board's pose in the world
        object_tvec_w: Optional tvec (3,) for the ground truth board's pose in the world
        object_points: Optional local coordinates (N, 3) of the ground truth board points
        ax: Optional existing Matplotlib Axes3D object
    """

    if ax is None:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = CUSTOM_COLORS

    points2d_plot = np.asarray(points2d).copy()
    if visibility_mask is not None:
        points2d_plot[~np.asarray(visibility_mask)] = np.nan

    # Plot cameras and frustums
    ax = plot_cameras_3d(
        rvecs_c2w, tvecs_c2w, camera_matrices, dist_coeffs,
        cameras_names=cameras_names,
        imsizes=imsizes,
        depth_ratio=frustums_depth,
        trust_volume=trust_volume,
        colors=colors,
        ax=ax
    )

    # Plot final triangulated 3D points
    ax = plot_points_3d(
        points3d,
        points_names=points_names,
        errors=errors,
        color='black',
        label='Triangulated points',
        ax=ax
    )

    if all(arg is not None for arg in [object_rvec_w, object_tvec_w, object_points]):
        ax = plot_object_3d(
            object_points=object_points,
            rvec_w=object_rvec_w,
            tvec_w=object_tvec_w,
            color='blue',
            label='Ground truth',
            ax=ax
        )
    # ---

    # Calculate dynamic depth for back-projection
    # Vector from each camera center to each 3D point -> shape (C, N, 3)
    cam_to_point_vectors = jnp.asarray(points3d)[None, :, :] - jnp.asarray(tvecs_c2w)[:, None, :]

    # Distance from each camera to each 3D point -> shape (C, N)
    depths_to_3d_points = jnp.linalg.norm(cam_to_point_vectors, axis=2)

    # Scale these depths to plot the back-projected points slightly closer to the camera
    plot_depths = depths_to_3d_points * detections_depth

    # Back-project the 2D points using the calculated dynamic depths
    E_c2w = extrinsics_matrix(rvecs_c2w, tvecs_c2w)
    points2d_in_3d = back_projection_batched(
        jnp.asarray(points2d_plot), plot_depths, camera_matrices, E_c2w, dist_coeffs, distortion_model='full'
    )
    points2d_in_3d = np.asarray(points2d_in_3d)  # convert to numpy for plotting

    # Plot the back-projected points and the rays
    C, N, _ = points2d_in_3d.shape
    for c in range(C):
        # Plot the back-projected points as small dots
        ax.scatter(points2d_in_3d[c, :, 0], points2d_in_3d[c, :, 1], points2d_in_3d[c, :, 2],
                   c=colors[c], marker='.', alpha=0.7, s=20)

        # Plot the rays from camera center to back-projected point
        for n in range(N):
            start_point = tvecs_c2w[c]
            end_point = points2d_in_3d[c, n, :]
            if np.all(np.isfinite(end_point)):
                ax.plot(*np.stack([start_point, end_point]).T,
                        color=colors[c], linestyle=':', linewidth=0.7, alpha=0.6)

    # Manually create a legend entry for the back-projected points
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [Line2D([0], [0], marker='.', color='gray', label='Back-projected Detections',
                              markerfacecolor='gray', markersize=8, linestyle='None')]
    ax.legend(handles=handles + legend_elements)

    ax.set_aspect('equal')
    return ax