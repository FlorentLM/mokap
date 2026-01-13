import pickle
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

from lucida import CameraRig
from mokap.utils import fileio
from mokap.reconstruction.config import ReconstructorConfig
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.utils import prepare_reconstruction_input
from mokap.reconstruction.debug.visualisation import *

# ================= CONFIGURATION =================
# Options: "RAYS", "EPIPOLAR", "HYPOTHESIS", "SOUP", "RAW_SOUP", "TRACKLETS", "LINKED_TRACKS"
MODE = "HYPOTHESIS"

FOLDER = Path().home() / 'Desktop' / '3d_ant_data'
PREFIX = '240905-1616'
SESSION = 22
FRAME = 926  # frame to view

# Only needed for "EPIPOLAR" / "HYPOTHESIS" modes
DEBUG_KEYPOINT = 'neck'
DEBUG_CAM_I = 0
DEBUG_CAM_J = 3


# =================================================

def load_images(folder, prefix, session, cams, frame):
    """Helper to load images for background."""
    imgs = {}
    for cam in cams:
        path = next((folder / prefix / 'sources').glob(f"*{cam}*session{session}.mp4"), None)
        if path:
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, fr = cap.read()
            if ret: imgs[cam] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cap.release()
    return imgs


if __name__ == "__main__":

    input_dir = FOLDER / PREFIX / 'inputs' / 'tracking'
    rig_file = FOLDER / PREFIX / 'calibration' / 'camera_rig.toml'

    rig = CameraRig.load(rig_file)
    cam_names = [c.name for c in rig]

    keypoints, bones = fileio.load_skeleton_SLEAP(input_dir, indices=False)
    bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    rec = Reconstructor(rig, bounds, ReconstructorConfig(repro_thresh=10.0, min_views=2))

    if MODE == "TRACKLETS" or MODE == "LINKED_TRACKS":
        filename = f'tracklets_session{SESSION}.pkl' if MODE == "TRACKLETS" else f'linked_tracks_session{SESSION}.pkl'
        track_path = FOLDER / PREFIX / 'outputs' / filename
        print(f"Loading tracklets from {track_path}...")

        with open(track_path, 'rb') as f:
            data = pickle.load(f)

        frames_list = convert_track_centric_to_frame_centric(data)
        run_sequence_viewer(frames_list, bones, bounds)

    else:
        print(f"Loading detections for frame {FRAME}...")

        df = fileio.load_session(input_dir, session=SESSION, use_polars=True)
        df_frame = df.filter(pl.col('frame') == FRAME)
        inputs = prepare_reconstruction_input(df_frame, cam_names, keypoints)

        # Prepare raw data arrays for visualisation
        kp_idx = keypoints.index(DEBUG_KEYPOINT)
        mask = inputs['kp_type_ids'] == kp_idx

        raw_dets = []
        raw_confs = []

        curr_coords = inputs['coords'][mask]
        curr_cam_ids = inputs['cam_ids'][mask]
        curr_scores = inputs['scores'][mask]

        # mapping: flat_index -> (cam_id, local_index_within_camera)
        flat_to_local_map = {}

        current_cam_counts = {c: 0 for c in range(len(rig))}

        for _ in range(len(rig)):
            raw_dets.append([])
            raw_confs.append([])

        # Fill with data
        for i, (c_id, coord, score) in enumerate(zip(curr_cam_ids, curr_coords, curr_scores)):
            c_id = int(c_id)
            local_idx = current_cam_counts[c_id]
            flat_to_local_map[i] = (c_id, local_idx)

            raw_dets[c_id].append(coord)
            raw_confs[c_id].append(score)
            current_cam_counts[c_id] += 1

        for c in range(len(rig)):
            if len(raw_dets[c]) > 0:
                raw_dets[c] = np.array(raw_dets[c])
                raw_confs[c] = np.array(raw_confs[c])
            else:
                raw_dets[c] = np.empty((0, 2), dtype=np.float32)
                raw_confs[c] = np.array([], dtype=np.float32)

        images = load_images(FOLDER, PREFIX, SESSION, cam_names, FRAME)

        if MODE == "RAYS":
            print("Visualising rays...")
            plot_cameras_rays(rec, raw_dets)

        elif MODE == "EPIPOLAR":
            print(f"Visualising epipolar geometry ({cam_names[DEBUG_CAM_I]} -> {cam_names[DEBUG_CAM_J]})...")
            plot_epipolar_segments(
                rec,
                raw_dets[DEBUG_CAM_I], raw_dets[DEBUG_CAM_J],
                images.get(cam_names[DEBUG_CAM_J]),
                DEBUG_CAM_I, DEBUG_CAM_J
            )

        elif MODE == "HYPOTHESIS":
            print(f"Generating hypotheses for '{DEBUG_KEYPOINT}'...")

            groups = rec._group_detections(curr_coords, curr_cam_ids)

            if not groups:
                print("No hypotheses generated (grouping failed or min_views not met).")
            else:
                pts, view_counts, summed_confs, errors, valid_mask = rec._triangulate_hypotheses(
                    curr_coords, curr_cam_ids, curr_scores, groups
                )

                # Filter to valid ones only
                valid_idx = np.where(valid_mask)[0]
                pts = pts[valid_idx]
                errors = errors[valid_idx]
                valid_groups = [groups[i] for i in valid_idx]

                if len(pts) > 0:
                    best_idx = np.argmin(errors)
                    best_point = pts[best_idx]
                    best_group_flat_indices = valid_groups[best_idx]

                    # Map flat indices back to (cam, local) for the visualisation
                    mapped_group = [flat_to_local_map[idx] for idx in best_group_flat_indices]

                    print(f"Visualising best hypothesis (Error: {errors[best_idx]:.2f})...")
                    plot_reprojection(rec, best_point, mapped_group, raw_dets, images)
                else:
                    print("Hypotheses generated but all failed validation (e.g. high reprojection error).")

        elif MODE == "SOUP":
            print("Running full frame reconstruction...")
            soup = rec.reconstruct_batch(inputs, keypoints)
            print(f"Reconstructed {soup.num_points} points.")

            plot_reconstructed_frame(rec, soup, bones)

            plt.show()

        elif MODE == "RAW_SOUP":
            print("Running full frame reconstruction (raw points + rays)...")
            soup = rec.reconstruct_batch(inputs, keypoints)
            print(f"Stats: {soup.num_points} Triangulated points, {len(soup.ray_origins)} Orphan rays")
            view_soup_frame(soup, FRAME, rig=rig)