import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import cv2
from scipy.linalg import svd

# All the projective geometry related functions used throughout the project

def extrinsics_matrix(rvec, tvec, hom=False):
    """
        Converts rotation vector and translation vector to a 3x4 extrinsics matrix E

                                    [ r00, r01, r02 ]             [ t0 ]
            E = [R|t]      with R = [ r10, r11, r12 ]   and   t = [ t1 ]
                                    [ r20, r21, r22 ]             [ t2 ]
    """
    rvec = np.asarray(rvec).squeeze()
    tvec = np.asarray(tvec).squeeze()

    # Convert rotation vector into rotation matrix (and jacobian)
    R_mat, jacob = cv2.Rodrigues(rvec)
    # Insert R mat into the Transform matrix and append translation vector to last column
    E = np.hstack([R_mat, np.atleast_2d(tvec).reshape(-1, 1)])

    if hom:
        return np.vstack([E, np.array([0, 0, 0, 1])])
    else:
        return E


def extmat_to_rtvecs(ext_mat):
    """
        Converts 3x4 (or 4x4) Extrinsics matrix to rotation vector and translation vector
    """
    rvec, jacob = cv2.Rodrigues(ext_mat[:3, :3])
    tvec = ext_mat[:3, 3]
    return rvec.squeeze(), tvec


def projection_matrix(intrinsics_mat, ext_mat):
    """
        Just a dot product of K o E to return the projection matrix P

           This matrix maps 3D points represented in real-world, camera-relative coordinates (X, Y, Z, 1)
           to 2D points in the image plane represented in normalized camera-relative coordinates (u, v, 1)

           2d_point     matrix_K           matrix_E         3d_point
                      (intrinsics)       (extrinsics)

                                                             [ X ]
             [ u ]   [ fx, 0, cx ]   [ r00, r01, r02, t0 ]   [ Y ]
             [ v ] = [ 0, fy, cy ] o [ r10, r11, r12, t1 ] o [ Z ]
             [ 1 ]   [ 0,  0,  1 ]   [ r20, r21, r22, t2 ]   [ 1 ]

                                KE = P
    """
    return np.dot(intrinsics_mat, ext_mat[:3, :])


def fundamental_matrix(camera_matrices, rvecs, tvecs, rank2_tol=1e-10):
    """
        Computes the fundamental matrix between two cameras given their parameters

        Parameters:
            cam_params1 (dict): Parameters for Camera 1.
            cam_params2 (dict): Parameters for Camera 2.

        Returns:
            F (np.ndarray): The 3x3 fundamental matrix.
    """

    K1, K2 = camera_matrices
    r1, r2 = rvecs
    t1, t2 = tvecs

    R1, _ = cv2.Rodrigues(r1)
    R2, _ = cv2.Rodrigues(r2)

    t1 = t1.reshape(3, 1)
    t2 = t2.reshape(3, 1)

    # Relative rotation and translation
    R_rel = R2.T @ R1
    t_rel = R2.T @ (t1 - t2)

    # Skew-symmetric matrix for t_rel
    t_rel_skew = np.array([
        [0, -t_rel[2, 0], t_rel[1, 0]],
        [t_rel[2, 0], 0, -t_rel[0, 0]],
        [-t_rel[1, 0], t_rel[0, 0], 0]
    ])

    # Essential Matrix
    E = t_rel_skew @ R_rel

    # Fundamental Matrix
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    F /= F[2, 2]

    # Ensure rank-2 consistency
    U, S, Vt = np.linalg.svd(F)
    det_F = np.linalg.det(F)
    if not S[2] < rank2_tol and abs(det_F) < rank2_tol:
        F = U @ np.diag(S) @ Vt

    return F


def invert_extrinsics(rvec, tvec):
    """
        Inverts extrinsics vectors (camera-space -> world space or vice versa) - Method 1
    """
    rvec = np.asarray(rvec).squeeze()
    tvec = np.asarray(tvec).squeeze()

    R_mat, _ = cv2.Rodrigues(rvec)

    R_inv = np.linalg.inv(R_mat)  # or R_mat.T because the matrix is orthonormal
    tvec_inv = -R_inv @ tvec

    rvec_inv, _ = cv2.Rodrigues(R_inv)

    return rvec_inv.squeeze(), tvec_inv


def invert_extrinsics_2(rvec, tvec):
    """
        Inverts extrinsics vectors (world-to-camera -> camera-to-world or vice versa) - Method 2
    """
    rt_mat = extrinsics_matrix(rvec, tvec, hom=True)
    rt_inv = np.linalg.inv(rt_mat)
    return extmat_to_rtvecs(rt_inv)


def invert_extrinsics_matrix(ext_mat):
    """
        Inverts extrinsics matrix (world-to-camera -> camera-to-world or vice versa) - Method 2
    """
    # NOTES for our use case:
    # If we have world-to-camera matrix (i.e. using the the output of cv2.solvePnP), there is no need to invert the rotation matrix.
    # If we have camera-to-world matrix then we need to invert R and adjust t.

    if ext_mat.shape[0] == 3:
        ext_mat = np.vstack([ext_mat, np.array([0, 0, 0, 1])])
        return np.linalg.inv(ext_mat)[:3, :]
    else:
        return np.linalg.inv(ext_mat)


def remap_rtvecs(rvec, tvec, orig_rvec, orig_tvec):
    """
        Remaps extrinsics vectors to another origin's extrinsics vectors
    """
    origin_mat = extrinsics_matrix(orig_rvec, orig_tvec, hom=True)
    ext_mat = extrinsics_matrix(rvec, tvec, hom=True)
    return extmat_to_rtvecs(remap_extmat(ext_mat, origin_mat))


def remap_extmat(ext_mat, origin_mat):
    """
        Remaps an extrinsics matrix to another origin's extrinsics matrix
    """
    return np.dot(origin_mat, np.linalg.inv(ext_mat))


def remap_points3d(points3d, orig_rvec, orig_tvec):
    origin_mat = extrinsics_matrix(orig_rvec, orig_tvec, hom=True)

    new_points3d = points3d - np.linalg.inv(origin_mat)[:3, 3]
    return np.dot(new_points3d, np.linalg.inv(origin_mat)[:3, :3])


def back_projection(points2d, depth, intrinsics_mat, ext_mat):
    """
    Performs back-projection from 2D image coordinates to 3D world coordinates.

        Parameters
        ----------
        points2d : 2D image coordinates (u, v)
        depth : The depth value (Z coordinate) at the given 2D image points
        intrinsics_mat : The intrinsics camera matrix K
        ext_mat : The extrinsics camera matrix [R|t]
        Returns: Array of the 3D world coordinates for given depth

    """
    # TODO - Add (optional) undistortion using dist_coeffs

    if not (isinstance(depth, int) or isinstance(depth, float)) and np.atleast_1d(depth).shape[0] != points2d.shape[0]:
        raise AssertionError('Depth vector length does not match 2D points array')

    # 2D image coordinates -> normalized camera coordinates
    if points2d.ndim == 1:
        homogeneous_2dcoords = np.array([*points2d, 1])
    else:
        homogeneous_2dcoords = np.c_[points2d, np.ones(points2d.shape[0])]

    normalised_coords = np.linalg.inv(intrinsics_mat) @ homogeneous_2dcoords.T

    # Depth
    normalised_coords *= depth

    R = ext_mat[:3, :3]
    tvec = ext_mat[:3, 3]

    if points2d.ndim == 1:
        # Convert normalized camera coordinates to world coordinates
        points3d = R @ normalised_coords + tvec
    else:
        points3d = (R @ normalised_coords + tvec[:, np.newaxis]).T

    return points3d


def triangulate_points_svd(points2d, projection_matrices, weights=None, lambda_reg=None):
    """
    Triangulate 3D point from multiple 2D points and their corresponding camera matrices

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

    Parameters
    ----------
    points2d:     Array or list of n 2D points from m cameras: m x n x 2 (u, v)
    projection_matrices: Array or List of n projection matrices: m x P (3 x 4)
    lambda_reg:   Regularisation term for Tikhonov Regularisation

    Returns: Array of n 3D points coordinates

    """
    points2d = np.asarray(points2d)
    projection_matrices = np.asarray(projection_matrices)

    if points2d.ndim == 2:
        points2d = points2d[:, np.newaxis, :]

    nb_views, nb_points = points2d.shape[:2]
    if nb_views != len(projection_matrices):
        raise ValueError("Number of 2D points series must match the number of projection matrices!")

    if weights is not None:
        weights = np.array(weights)
        if weights.shape[0] != nb_points:
            raise ValueError("Number of weights must match the number of 2D points!")
        weights[weights <= 0] = np.nan
    else:
        weights = np.ones(nb_points)

    points_3d = np.full((nb_points, 3), np.nan)

    for p in range(nb_points):

        A = np.zeros((nb_views * 2, projection_matrices.shape[-1]))

        for i, ii in enumerate(range(0, nb_views * 2, 2)):
            P = projection_matrices[i]
            u, v = points2d[i, p]
            A[ii, :] = (u * P[2, :] - P[0, :]) * weights[p]
            A[ii + 1, :] = (v * P[2, :] - P[1, :]) * weights[p]

        A = A[~np.isnan(A).any(axis=1), :]

        if A.shape[0] < 2:
            # Not enough views to triangulate this point
            points_3d[p, :] = np.nan
        else:
            if lambda_reg is not None:
                # Regularize by adding lambda * I to A^T * A
                ATA = A.T @ A + lambda_reg * np.eye(A.shape[1])
                U, s, Vt = svd(ATA, full_matrices=False)
            else:
                U, s, Vt = svd(A, full_matrices=False)

            X = Vt[-1]
            X /= X[-1]  # Normalize to ensure the homogeneous coordinate is 1

            points_3d[p, :] = X[:3]

    return points_3d


def find_affine(Ps, Ps_2):
    """
        Estimate the affine transformation between two sets of points - Method 1
    """

    n = Ps.shape[0]
    Ps_homogeneous = np.hstack([Ps, np.ones((n, 1))])

    # Solve for the transformation matrix using least squares
    A_h, res, rank, s = np.linalg.lstsq(Ps_homogeneous, Ps_2, rcond=None)

    # Extract rotation and translation components
    R = A_h[:3, :3]
    t = A_h[3, :]

    return R, t


def find_affine2(Ps, Ps_2):
    """
        Estimate the affine transformation between two sets of points - Method 2
    """
    Rt = cv2.estimateAffine3D(Ps, Ps_2, force_rotation=True)
    return Rt[:3, :3], Rt[:3]