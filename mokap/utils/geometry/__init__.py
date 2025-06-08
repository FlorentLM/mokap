"""
This package provides a collection of JAX-accelerated functions for 3D geometry,
and numerical optimization, tailored for multi-camera systems

Modules:
- camera: Functions related to camera intrinsic parameters.
- projective: Core projective geometry operations like triangulation and point projection.
- transforms: Functions for rotation representations (quaternions, matrices) and pose transformations.
- stats: Robust statistical methods for outlier rejection and averaging of poses.
- utils: Miscellaneous utility functions.
"""

# import os
# os.environ['JAX_LOG_COMPILES'] = '1'
# from jax import config
# config.update("jax_log_compiles", True)

# from . import camera, undistort_points, quaternion_average
# from . import projective
# from . import stats
# from . import transforms
# from . import utils
#
# # Expose key functions for easier access throughout the application
# from .camera import estimate_camera_matrix
# from .projective import (
#     inverse_rodrigues,
#     project_points,
#     project_multiple,
#     undistort_multiple,
#     triangulate_svd,
#     triangulation,
#     compute_errors_jax,
#     interpolate3d,
#     back_projection,
#     back_projection_batched,
# )
# from .stats import (
#     huber_weight,
#     robust_translation_mean,
#     estimate_initial_poses,
#     filter_outliers,
#     filter_rt_samples,
# )
# from .transforms import (
#     ID_QUAT,
#     ZERO_T,
#     axisangle_to_quaternion,
#     axisangle_to_quaternion_batched,
#     quaternion_to_axisangle,
#     quaternion_to_axisangle_batched,
#     quaternion_inverse,
#     rotate_vector,
#     extrinsics_matrix,
#     invert_extrinsics_matrix,
#     extmat_to_rtvecs,
#     projection_matrix,
#     remap_rtvecs,
#     remap_extmat,
#     remap_points3d,
#     Rmat_from_angle,
#     rotate_points3d,
#     rotate_pose,
#     rotate_extrinsics_matrix, rodrigues,
# )
# from .utils import pad_to_length