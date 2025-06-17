from typing import Optional, Any, Dict, Iterable, Sequence

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


CUSTOM_COLORS = ['#9B5DE5', '#EF476F', '#FFD166', '#00BBF9', '#00F5D4', '#118ab2', '#073b4c', '#ee6c4d']


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    # From https://stackoverflow.com/a/18926541
    import matplotlib.colors as colors
    return colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                                                    cmap(np.linspace(minval, maxval, n)))


def plot_volume_3d(
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
        [0, 1, 5, 4],  # Front
        [1, 2, 6, 5],  # Right
        [2, 3, 7, 6],  # Back
        [3, 0, 4, 7],  # Left
        [0, 1, 2, 3],  # Bottom
        [4, 5, 6, 7],  # Top
    ], dtype=int)
    faces = v[faces_idx]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.1, edgecolors=color, alpha=alpha))

    return ax


def plot_cameras_3d(
        rvecs_c2w:          ArrayLike,
        tvecs_c2w:          ArrayLike,
        camera_matrices:    ArrayLike,
        dist_coeffs:        ArrayLike,
        cameras_names:      Optional[Sequence[Any]] = None,
        imsizes:            ArrayLike = np.array([1440, 1080]),
        depth:              float = 130.0,
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
    E_w2c = invert_extrinsics_matrix(E_c2w)
    frustums_3d = back_projection_batched(frustums_2d, depth, camera_matrices, E_w2c, dist_coeffs)

    directions = frustums_3d[:, -1] - tvecs_c2w
    # directions_normalised = directions / np.linalg.norm(directions, axis=1)[:, None]
    directions_normalised = directions / depth

    focal_point = rays_intersection_3d(tvecs_c2w, directions_normalised)

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
        volume_ranges = [np.ptp(axis) for axis in trust_volume.values()]
        ax = plot_volume_3d(focal_point, size=volume_ranges, ax=ax)

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

    if errors is not None:
        colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)
        pts_scatter = ax.scatter(*points3d.T,
                                 c=errors, cmap=colormap, norm=normalize,
                                 marker='o', label='3D points', alpha=0.5)
    else:
        pts_scatter = ax.scatter(*points3d.T,
                                 c=color,
                                 marker='o', label='3D points', alpha=0.5)

    for p, name in enumerate(points_names):
        if errors is not None:
            c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else color
        else:
            c = color
        ax.text(*points3d[p], f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.legend()

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

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
    E_w2c = invert_extrinsics_matrix(E_c2w)
    points2d_3d = back_projection_batched(points2d, depth, camera_matrices, E_w2c, dist_coeffs)

    for n in range(C):
        if errors is not None:
            colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)

            pts_scatter = ax.scatter(*points2d_3d[n].T, s=5,
                                     c=errors[n], cmap=colormap, norm=normalize,
                                     marker='.', label='3D points', alpha=0.5)
        else:
            pts_scatter = ax.scatter(*points2d_3d.T, s=5,
                                     c=colors[n],
                                     marker='.', label='3D points', alpha=0.5)

        if points_names is not None:
            for p, name in enumerate(points_names):
                point = points2d_3d[n][p]
                if np.isfinite(point).all():
                    if errors is not None:
                        c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else colors[n]
                    else:
                        c = colors[n]
                    ax.text(*point, f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax