import logging
import cv2
import numpy as np
from jax import numpy as jnp
from typing import Tuple, Optional, Literal, Union, Sequence
from mokap.utils.geometry.projective import project_points, reprojection_errors
from mokap.utils.geometry.fitting import generate_ambiguous_pose
from mokap.utils.datatypes import ChessBoard, CharucoBoard, CalibrateCameraResult, DistortionModel

logger = logging.getLogger(__name__)


def solve_pnp_robust(
        object_points:  np.ndarray,
        image_points:   np.ndarray,
        camera_matrix:  np.ndarray,
        dist_coeffs:    np.ndarray,
        refine_method:  Optional[Literal['VVS', 'LM', 'none']] = None
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    A robust wrapper for solvePnP that handles the ambiguity of planar targets
    It returns a single, physically plausible pose with the lowest reprojection error

    Strategy:
        Tries to use the IPPE algorithm which is designed for planar calibration boards
        Falls back to the robust SQPNP algorithm
        Falls back again to the lenient iterative algorithm
        Manually generates and checks ambiguous poses if the solver doesn't
        Optionally refines the final pose
    """

    obj_pts_np = np.asarray(object_points, dtype=np.float32)
    img_pts_np = np.asarray(image_points, dtype=np.float32)
    cam_mat_np = np.asarray(camera_matrix, dtype=np.float32)
    dist_np = np.asarray(dist_coeffs if dist_coeffs is not None else np.zeros(5), dtype=np.float32)

    # Shape validation
    if obj_pts_np.ndim != 2 or obj_pts_np.shape[1] != 3:
        raise ValueError(f"Object points must have shape (N, 3), but got {obj_pts_np.shape}")

    if img_pts_np.ndim != 2 or img_pts_np.shape[1] != 2:
        raise ValueError(f"Image points must have shape (N, 2), but got {img_pts_np.shape}")

    if obj_pts_np.shape[0] != img_pts_np.shape[0]:
        raise ValueError("Mismatch in number of object and image points.")

    if obj_pts_np.shape[0] < 4:
        # most PnP methods require at least 4 points
        return False, None, None, None

    if cam_mat_np.shape != (3, 3):
        raise ValueError(f"Camera matrix must have shape (3, 3), but got {cam_mat_np.shape}")

    best_rvec, best_tvec = None, None

    # Strategy 1: Try IPPE
    if hasattr(cv2, 'SOLVEPNP_IPPE'):
        try:
            nb, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj_pts_np, img_pts_np, cam_mat_np, dist_np,
                                                         flags=cv2.SOLVEPNP_IPPE)
            if nb > 0:
                valid_solutions = [{'rvec': r, 'tvec': t, 'error': e[0]} for r, t, e in zip(rvecs, tvecs, errs) if
                                   t[2] > 0]
                if valid_solutions:
                    best = min(valid_solutions, key=lambda x: x['error'])
                    best_rvec, best_tvec = best['rvec'], best['tvec']
        except cv2.error:
            pass  # continue to fallback

        # Strategies 2 and 3: Try SQPNP, or Iterative
        if best_rvec is None:
            candidate_rvec, candidate_tvec = None, None

            # First try to get a candidate pose from SQPNP
            try:
                nb, rvecs_cv, tvecs_cv, errs_cv = cv2.solvePnPGeneric(obj_pts_np, img_pts_np,
                                                                      cam_mat_np, dist_np,
                                                                      flags=cv2.SOLVEPNP_SQPNP)
                if nb > 0:
                    valid_solutions = [{'rvec': r, 'tvec': t, 'error': e[0]}
                                       for r, t, e in zip(rvecs_cv, tvecs_cv, errs_cv) if t[2] > 0]
                    if valid_solutions:
                        # get the best candidate from the list
                        best_candidate = min(valid_solutions, key=lambda x: x['error'])
                        candidate_rvec, candidate_tvec = best_candidate['rvec'], best_candidate['tvec']

            except cv2.error:
                pass  # continue to final fallback

            # if SQPNP failed, try to get a candidate from Iterative
            if candidate_rvec is None:
                try:
                    success, rvec, tvec = cv2.solvePnP(obj_pts_np, img_pts_np, cam_mat_np, dist_np,
                                                       flags=cv2.SOLVEPNP_ITERATIVE)
                    if success and tvec[2] > 0:
                        candidate_rvec, candidate_tvec = rvec, tvec
                except cv2.error:
                    pass  # all solvers failed

            # Manual disambiguation for the best candidate
            if candidate_rvec is not None and candidate_tvec is not None:
                rvec1, tvec1 = candidate_rvec, candidate_tvec

                rvec1_j, tvec1_j = jnp.asarray(rvec1.squeeze()), jnp.asarray(tvec1.squeeze())
                obj_pts_j, img_pts_j = jnp.asarray(obj_pts_np), jnp.asarray(img_pts_np)
                cam_mat_j, dist_j = jnp.asarray(cam_mat_np), jnp.asarray(dist_np)

                rvec2_j, tvec2_j = generate_ambiguous_pose(rvec1_j, tvec1_j)

                if tvec2_j[2] <= 0:
                    # The ambiguous pose is invalid, so the candidate is probably correct
                    best_rvec, best_tvec = rvec1, tvec1
                else:
                    # if both are valid, compare their errors
                    reproj1 = project_points(obj_pts_j, rvec1_j, tvec1_j, cam_mat_j, dist_j)
                    reproj2 = project_points(obj_pts_j, rvec2_j, tvec2_j, cam_mat_j, dist_j)

                    errors1 = reprojection_errors(img_pts_j, reproj1)
                    errors2 = reprojection_errors(img_pts_j, reproj2)

                    # Compare using the standard RMS error
                    if errors1['rms'] <= errors2['rms']:
                        best_rvec, best_tvec = rvec1, tvec1
                        best_error = errors1
                    else:
                        best_rvec, best_tvec = np.asarray(rvec2_j).reshape(3, 1), np.asarray(tvec2_j).reshape(3, 1)
                        best_error = errors2

        if best_rvec is None or best_tvec is None:
            # all methods failed to produce a valid pose
            return False, None, None, None

    # If we got here from IPPE, we haven't calculated the error dict yet
    # If we got here from manual disambiguation, best_error is already set
    needs_error_recalc = best_error is None

    # Optionally refine
    if refine_method and refine_method.lower() != 'none':
        try:
            refine_func_map = {'vvs': cv2.solvePnPRefineVVS, 'lm': cv2.solvePnPRefineLM}
            refine_func = refine_func_map[refine_method.lower()]

            best_rvec, best_tvec = refine_func(
                objectPoints=obj_pts_np, imagePoints=img_pts_np, cameraMatrix=cam_mat_np,
                distCoeffs=dist_np, rvec=best_rvec, tvec=best_tvec
            )
            # After refinement, the already-calculated error is invalid and must be recalculated
            needs_error_recalc = True
        except (cv2.error, AttributeError, KeyError):
            pass

    best_rvec = best_rvec.squeeze()
    best_tvec = best_tvec.squeeze()

    if needs_error_recalc:
        final_reproj = project_points(obj_pts_np, best_rvec, best_tvec, cam_mat_np, dist_np)
        final_errors = reprojection_errors(img_pts_np, final_reproj)
    else:
        # otherwise, the one we stored is the correct one
        final_errors = best_error

    return True, best_rvec, best_tvec, final_errors


def calibrate_camera_robust(
    board:                  Union[ChessBoard, CharucoBoard],
    image_points_stack:     Sequence[np.ndarray],
    image_ids_stack:        Sequence[np.ndarray],
    image_size_wh:          Sequence[int],
    initial_K:              Optional[np.ndarray] = None,
    initial_D:              Optional[np.ndarray] = None,
    distortion_model:       DistortionModel = 'standard',
    fix_aspect_ratio:       bool = False
) -> CalibrateCameraResult:
    """ A convenience wrapper for OpenCV's camera calibration functions """

    # Build calibration flags
    calib_flags = 0
    if initial_K is not None and initial_D is not None:
        calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        if fix_aspect_ratio:
            calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO

    # Set distortion flags based on the model
    if distortion_model == 'none':
        calib_flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
                        cv2.CALIB_FIX_TANGENT_DIST)

    elif distortion_model == 'simple':
        # Optimize for k1, k2, but fix others
        calib_flags |= (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

    elif distortion_model == 'rational':
        calib_flags |= cv2.CALIB_RATIONAL_MODEL

    # 'standard' and 'full' don't need special flags, they are the default behavior
    # when the corresponding CALIB_FIX_K* flags are not set

    try:

        if board.type == 'charuco':

            # calib_flags |= cv2.CALIB_USE_LU   # TODO: Should we use LU or QR? How 'worse' are they?

            (rms, K_new, D_new, rvecs, tvecs,
             std_intr, _, pve_opencv) = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=image_points_stack,
                charucoIds=image_ids_stack,
                board=board.to_opencv(),
                imageSize=image_size_wh,
                cameraMatrix=initial_K.copy() if initial_K is not None else None,
                distCoeffs=initial_D.copy() if initial_D is not None else None,
                flags=calib_flags
            )

        elif board.type == 'chessboard':

            # For chessboard, it's always all points, so we repeat
            object_points_stack = [board.object_points] * len(image_points_stack)

            (rms, K_new, D_new, rvecs, tvecs,
             std_intr, _, pve_opencv) = cv2.calibrateCameraExtended(
                objectPoints=object_points_stack,
                imagePoints=image_points_stack,
                imageSize=image_size_wh,
                cameraMatrix=initial_K.copy() if initial_K is not None else None,
                distCoeffs=initial_D.copy() if initial_D is not None else None,
                flags=calib_flags
            )
        else:
            return CalibrateCameraResult(success=False, error_message=f"Unsupported board type '{board.type}'.")

        # Check for invalid results
        invalid_vals = not (np.isfinite(K_new).all() and np.isfinite(D_new).all())
        negative_K_vals = (K_new < 0).any()
        invalid_central_point = (K_new[:2, 2] >= np.array(image_size_wh)).any()

        if invalid_vals or negative_K_vals or invalid_central_point:
            return CalibrateCameraResult(success=False, error_message="Calibration resulted in an invalid camera matrix.")

        # These limits are for standard rectilinear lenses using the Brown-Conrady model
        # They are NOT suitable for fisheye lenses, which use a different model and calibration pipeline (cv2.fisheye.calibrate)
        absurd_distortion = False
        reason = ''
        d_abs = np.abs(D_new)

        if len(d_abs) >= 4 and (d_abs[2] > 0.5 or d_abs[3] > 0.5):
            # Tangential distortion should always be small for a well-centered lens
            # so |p1| > 0.5 or |p2| > 0.5 is almost certainly wrong
            absurd_distortion = True
            reason = "Unplausible tangential distortion (p1, p2)"

        if not absurd_distortion and len(d_abs) >= 2:
            # A k1 or k2 value with an absolute magnitude > 2.0 is extremely rare for non-fisheye lenses
            if d_abs[0] > 1.5 or d_abs[1] > 2.0:
                absurd_distortion = True
                reason = "Unplausible radial distortion (k1, k2)"

        if not absurd_distortion and distortion_model in ['full', 'rational']:
            # Check higher-order terms for full or rational models
            if len(d_abs) >= 5 and np.any(d_abs[4:8] > 1.5):
                absurd_distortion = True
                reason = "Unplausible higher-order distortion (k3-k6)"

        if absurd_distortion:
            error_message = f"Calibration resulted in invalid distortion: {reason}. Values: {D_new.round(4)}"
            return CalibrateCameraResult(success=False, error_message=error_message)

        # Note:
        # -----
        #
        # The per-view reprojection errors as returned by calibrateCamera() is:
        #   the square root of the sum of the 2 means in x and y of the squared diff
        #       np.sqrt(np.sum(np.mean(sq_diff, axis=0)))
        #
        # These are NOT the same as the per-view RMS errors typically computed after solvePnP():
        #   this one is the square root of the mean of the squared diff over both x and y
        #        np.sqrt(np.mean(sq_diff, axis=(0, 1)))
        #
        # In other words, the first one is larger by a factor √(2)
        #
        # In addition, the global RMS error returned by calibrateCamera() is:
        #       np.sqrt(np.sum([sq_diff for view in stack]) / np.sum([len(view) for view in stack]))
        #

        # ...so we just divide it by √2 and we're consistent with the rest of mokap
        pve_rms = pve_opencv.squeeze() / np.sqrt(2.0)

        # Package into the result dataclass
        return CalibrateCameraResult(
            success=True,
            rms_error=rms,
            K_new=K_new.squeeze(),
            D_new=D_new.squeeze(),
            rvecs=np.array(rvecs).squeeze(),
            tvecs=np.array(tvecs).squeeze(),
            std_devs_intrinsics=std_intr,
            per_view_errors=pve_rms
        )

    except cv2.error as e:
        error_msg = f"OpenCV Error in calibrateCamera: {e}"
        logger.warning(error_msg)
        return CalibrateCameraResult(success=False, error_message=error_msg)