import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mokap.utils.geometry.transforms import invert_rtvecs, rodrigues, inverse_rodrigues
from mokap.utils.geometry.projective import project_points, triangulate
from mokap.utils.visualisation import plot_cameras_3d, plot_points_3d

# Use CPU for this test to avoid any GPU memory issues with Matplotlib
jax.config.update('jax_platform_name', 'cpu')
# Use float64 for higher precision in the ground truth calculations
jax.config.update('jax_enable_x64', True)

print("--- Triangulation Sanity Check ---")
print(f"Running on JAX device: {jax.default_backend().upper()}")

def create_test_scene():
    """ Creates a ground-truth scene with a 3D object and multiple cameras """

    # --- Define a simple 3D object (a cube centered at the origin) ---
    points_3d_ground_truth = jnp.array([
        [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
    ], dtype=jnp.float64) * 5.0

    # --- Define Camera Intrinsics (perfect pinhole cameras) ---
    num_cameras = 4
    image_size = (1440, 1080)
    fx, fy = 2000.0, 2000.0
    cx, cy = image_size[0] / 2, image_size[1] / 2
    camera_matrices = jnp.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * num_cameras, dtype=jnp.float64)
    dist_coeffs = jnp.zeros((num_cameras, 8), dtype=jnp.float64)

    # --- Define Camera Extrinsics (Poses) ---
    camera_distance = 60.0

    # Camera 1: At (0, 0, 40). Must be rotated 180 deg around Y to look at the origin.
    tvec_c2w_1 = jnp.array([0.0, 0.0, camera_distance])
    rvec_c2w_1 = jnp.array([0.0, jnp.pi, 0.0], dtype=jnp.float64)

    # Camera 2: At (40, 0, 0). Must be rotated -90 deg around Y to look at the origin.
    tvec_c2w_2 = jnp.array([camera_distance, 0.0, 0.0])
    rvec_c2w_2 = jnp.array([0.0, -jnp.pi/2, 0.0], dtype=jnp.float64)

    # Camera 3: At (-40, 0, 0). Must be rotated +90 deg around Y to look at the origin.
    tvec_c2w_3 = jnp.array([-camera_distance, 0.0, 0.0])
    rvec_c2w_3 = jnp.array([0.0, jnp.pi/2, 0.0], dtype=jnp.float64)

    # Camera 4: At (0, -40, 0). Must be rotated -90 deg around X to look up at the origin.
    tvec_c2w_4 = jnp.array([0.0, -camera_distance, 0.0])
    rvec_c2w_4 = jnp.array([-jnp.pi/2, 0.0, 0.0], dtype=jnp.float64)

    rvecs_c2w = jnp.stack([rvec_c2w_1, rvec_c2w_2, rvec_c2w_3, rvec_c2w_4])
    tvecs_c2w = jnp.stack([tvec_c2w_1, tvec_c2w_2, tvec_c2w_3, tvec_c2w_4])

    # Most functions (project, triangulate) need world-to-camera (w2c) poses
    rvecs_w2c, tvecs_w2c = invert_rtvecs(rvecs_c2w, tvecs_c2w)

    return {
        "points_3d_gt": points_3d_ground_truth, "Ks": camera_matrices, "Ds": dist_coeffs,
        "rvecs_c2w": rvecs_c2w, "tvecs_c2w": tvecs_c2w,
        "rvecs_w2c": rvecs_w2c, "tvecs_w2c": tvecs_w2c,
        "image_sizes": jnp.array([image_size] * num_cameras)
    }

@jax.jit
def perturb_cameras(
    key: jax.random.PRNGKey,
    rvecs_c2w: jnp.ndarray,
    tvecs_c2w: jnp.ndarray,
    max_angle_deg: float,
    max_translation: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies a random perturbation to a set of camera-to-world poses

    Args:
        key: JAX random key
        rvecs_c2w: Original camera-to-world rotation vectors (C, 3)
        tvecs_c2w: Original camera-to-world translation vectors (C, 3)
        max_angle_deg: The maximum angle of rotational perturbation in degrees
        max_translation: The maximum magnitude of translational perturbation per axis

    Returns:
        A tuple of (perturbed_rvecs_c2w, perturbed_tvecs_c2w).
    """
    num_cameras = rvecs_c2w.shape[0]

    # Split the key for independent random operations
    key_rot, key_trans = jax.random.split(key)

    # Rotation Perturbation
    # To apply a small random rotation, we'll generate a random axis-angle vector
    # and compose it with the original rotation

    # Generate random rotation axes (unit vectors)
    key_axis, key_angle = jax.random.split(key_rot)
    random_axes = jax.random.normal(key_axis, shape=(num_cameras, 3))
    random_axes /= jnp.linalg.norm(random_axes, axis=-1, keepdims=True)

    # Generate random rotation angles
    max_angle_rad = jnp.deg2rad(max_angle_deg)
    random_angles = jax.random.uniform(key_angle, shape=(num_cameras, 1)) * max_angle_rad

    # Create the random perturbation rotation vectors
    rvecs_perturb = random_axes * random_angles

    # Compose the original rotation with the perturbation
    # R_new = R_perturb @ R_orig
    R_orig = rodrigues(rvecs_c2w)
    R_perturb = rodrigues(rvecs_perturb)
    R_new = R_perturb @ R_orig
    perturbed_rvecs = inverse_rodrigues(R_new)

    # Translation Perturbation
    # Add a random vector where each component is in [-max_translation, max_translation]
    tvecs_perturb = jax.random.uniform(
        key_trans,
        shape=(num_cameras, 3),
        minval=-max_translation,
        maxval=max_translation
    )
    perturbed_tvecs = tvecs_c2w + tvecs_perturb

    return perturbed_rvecs, perturbed_tvecs


def main():
    # GENERATE SYNTHETIC DATA
    scene = create_test_scene()
    print(f"Generated a scene with {scene['points_3d_gt'].shape[0]} points and {scene['Ks'].shape[0]} cameras.")

    # (Optional) Perturb the camera poses to simulate errors
    perturb_scene = True
    key = jax.random.PRNGKey(42)  # A key for reproducible randomness

    if perturb_scene:
        key, subkey = jax.random.split(key)

        # Define the magnitude of errors
        max_rot_error_deg = 5.0  # degrees
        max_trans_error_units = 1.0  # world units (mm or cm)

        print("\n--- Perturbing Scene ---")
        print(
            f"Applying random errors to camera poses (max {max_rot_error_deg}° rot, {max_trans_error_units} units trans).")

        # Apply perturbation to the c2w poses
        perturbed_r, perturbed_t = perturb_cameras(
            subkey,
            scene['rvecs_c2w'],
            scene['tvecs_c2w'],
            max_rot_error_deg,
            max_trans_error_units
        )

        # Update the scene dictionary with the new perturbed poses
        scene['rvecs_c2w'] = perturbed_r
        scene['tvecs_c2w'] = perturbed_t

        # We also must re-calculate the world-to-camera (w2c) poses
        # as they are derived from the c2w poses
        scene['rvecs_w2c'], scene['tvecs_w2c'] = invert_rtvecs(perturbed_r, perturbed_t)
        print("------------------------\n")

    # Project the 3D points into each camera to get the 2D observations
    project_vmapped = jax.vmap(project_points, in_axes=(None, 0, 0, 0, 0))
    points_2d_observed = project_vmapped(
        scene['points_3d_gt'],
        scene['rvecs_w2c'],
        scene['tvecs_w2c'],
        scene['Ks'],
        scene['Ds']
    )
    print(f"Projected 3D points to 2D. Shape of observed 2D points: {points_2d_observed.shape}")

    # Add noise to simulate real-world conditions
    add_noise = True
    noise_level_px = 0.5
    if add_noise:
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=points_2d_observed.shape) * noise_level_px
        points_2d_observed += noise
        print(f"Added Gaussian noise with stddev = {noise_level_px} px to 2D points.")

    # TRIANGULATE POINTS
    # Now use the triangulation function with the observed 2D points and camera parameters
    # to see if we can recover the original 3D points
    print("\nAttempting to triangulate 3D points from 2D observations...")
    points_3d_triangulated = triangulate(
        points2d=points_2d_observed,
        camera_matrices=scene['Ks'],
        dist_coeffs=scene['Ds'],
        rvecs_w2c=scene['rvecs_w2c'],
        tvecs_w2c=scene['tvecs_w2c'],
        weights=None,  # Let the function infer from non-NaN values
        distortion_model='standard'
    )
    print(f"Triangulation complete. Shape of result: {points_3d_triangulated.shape}")

    # ANALYZE AND VISUALIZE
    # Calculate the reconstruction error
    error = jnp.linalg.norm(scene['points_3d_gt'] - points_3d_triangulated, axis=1)
    mean_error = jnp.mean(error)
    print(f"\nReconstruction Analysis:")
    print(f"Mean Euclidean distance error: {mean_error:.6f}")
    print(f"Max Euclidean distance error:  {jnp.max(error):.6f}")

    # Convert to numpy for plotting
    np_points_3d_gt = np.array(scene['points_3d_gt'])
    np_points_3d_triangulated = np.array(points_3d_triangulated)

    # Create the 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cameras
    plot_cameras_3d(
        rvecs_c2w=np.array(scene['rvecs_c2w']),
        tvecs_c2w=np.array(scene['tvecs_c2w']),
        camera_matrices=np.array(scene['Ks']),
        dist_coeffs=np.array(scene['Ds']),
        cameras_names=[f"Cam {i + 1}" for i in range(scene['Ks'].shape[0])],
        imsizes=np.array(scene['image_sizes']),
        depth=30.0,
        ax=ax
    )

    # Plot the ground-truth points (in black)
    plot_points_3d(
        points3d=np_points_3d_gt,
        points_names=[f"P{i}" for i in range(np_points_3d_gt.shape[0])],
        color='black',
        ax=ax
    )

    # Plot the triangulated points (in red)
    plot_points_3d(
        points3d=np_points_3d_triangulated,
        points_names=None,  # Don't plot names again
        color='#EF476F',    # A bright red/pink
        ax=ax
    )
    # Add a specific label for the red points
    ax.scatter([], [], [], c='#EF476F', marker='o', label='Triangulated Points')
    ax.legend()

    ax.set_title("Triangulation Sanity Check\n(Red points should overlay black points)")
    plt.show()


if __name__ == "__main__":
    main()