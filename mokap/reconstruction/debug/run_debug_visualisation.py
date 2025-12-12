import pickle
import polars as pl
from pathlib import Path
from lucida import CameraRig
from mokap.utils import fileio
from mokap.reconstruction.config import ReconstructorConfig
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.utils import prepare_reconstruction_input
from mokap.reconstruction.debug.visualisation import *


# ================= CONFIGURATION =================
# Options: "RAYS", "EPIPOLAR", "HYPOTHESIS", "SOUP", "RAW_SOUP", "TRACKLETS", "LINKED_TRACKS"
MODE = "RAW_SOUP"

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

        # Prepare raw data arrays
        kp_idx = keypoints.index(DEBUG_KEYPOINT)
        mask = inputs['kp_type_ids'] == kp_idx

        # Regroup by camera for plotting
        raw_dets = []
        raw_confs = []

        for c in range(len(rig)):
            c_mask = (inputs['cam_ids'][mask] == c)
            if np.any(c_mask):
                raw_dets.append(inputs['coords'][mask][c_mask])
                raw_confs.append(inputs['scores'][mask][c_mask])
            else:
                raw_dets.append(rec.NULL_POINT2D_XP)
                raw_confs.append(rec.EMPTY_F32_NP)

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

            flat_dets = xp.concatenate(raw_dets)
            flat_confs = xp.concatenate(raw_confs)

            # Map global index back to (cam, local)
            counts = [len(d) for d in raw_dets]
            offsets = np.cumsum([0] + counts[:-1])


            def unflatten_index(global_idx):
                cam_idx = np.searchsorted(offsets, global_idx, side='right') - 1
                local_idx = global_idx - offsets[cam_idx]
                return int(cam_idx), int(local_idx)


            pts, groups, _, _, errors = rec._generate_hypotheses(
                inputs['coords'][mask], inputs['cam_ids'][mask],
                flat_dets, flat_confs
            )

            if len(pts) > 0:
                best_idx = np.argmin(errors)
                flat_group = groups[best_idx]
                mapped_group = [unflatten_index(idx) for idx in flat_group]

                print(f"Visualising best hypothesis (Error: {errors[best_idx]:.2f})...")
                plot_reprojection(rec, pts[best_idx], mapped_group, raw_dets, images)
            else:
                print("No hypotheses generated.")

        elif MODE == "SOUP":
            print("Running full frame reconstruction...")
            soup = rec.reconstruct_batch(inputs, keypoints)
            print(f"Reconstructed {soup.num_points} points.")
            plot_reconstructed_frame(rec, soup, bones)
            import matplotlib.pyplot as plt

            plt.show()

        elif MODE == "RAW_SOUP":
            print("Running full frame reconstruction (raw points + rays)...")
            soup = rec.reconstruct_batch(inputs, keypoints)
            print(f"Stats: {soup.num_points} Triangulated points, {len(soup.ray_origins)} Orphan rays")
            view_soup_frame(soup, FRAME, rig=rig)