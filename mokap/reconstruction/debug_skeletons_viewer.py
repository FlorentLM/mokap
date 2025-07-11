import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from mokap.utils import fileio


def convert_track_centric_to_frame_centric(track_data: dict) -> list:
    """
    Converts a track-centric dictionary {track_id: [skeletons]} into a
    frame-centric list [{'frame_idx': ..., 'skeletons': [...]}] suitable for the viewer
    """

    # Invert the dictionary to group by frame_idx
    frames_dict = defaultdict(lambda: {'skeletons': []})
    for track_id, skeletons_list in track_data.items():
        for skel in skeletons_list:
            frame_idx = skel['frame_idx']
            frames_dict[frame_idx]['skeletons'].append(skel)
            frames_dict[frame_idx]['frame_idx'] = frame_idx

    if not frames_dict:
        return []

    # Create a sorted list of frame dictionaries, filling in any empty frames
    min_frame = min(frames_dict.keys())
    max_frame = max(frames_dict.keys())

    output_list = []
    for i in range(min_frame, max_frame + 1):
        if i in frames_dict:
            output_list.append(frames_dict[i])
        else:
            # Add an empty frame to keep the timeline consistent
            output_list.append({'frame_idx': i, 'skeletons': []})

    return output_list


def draw_skeletons(frame_data: dict, bones: list, ax: plt.Axes):
    """ Draws all skeletons in a given frame """

    skeletons = frame_data.get("skeletons", [])
    color_map = plt.get_cmap('tab20', 20)
    artists = []  # artists to be removed later

    for i, skel in enumerate(skeletons):
        # Check for smoothed keypoints first, then fall back to raw keypoints
        if 'keypoints_smoothed' in skel:
            keypoints = skel['keypoints_smoothed']
        elif 'keypoints' in skel:
            keypoints = skel['keypoints']
        else:
            # This skeleton has no displayable points, skip it
            continue

        track_id = skel.get('track_id', -1)
        color = color_map(track_id % color_map.N) if track_id != -1 else 'gray'

        # Bones
        for kp1_name, kp2_name in bones:
            if kp1_name in keypoints and kp2_name in keypoints:
                p1 = np.array(keypoints[kp1_name])
                p2 = np.array(keypoints[kp2_name])
                line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=1.0, lw=2)
                artists.append(line)

        # Keypoints
        points_arr = np.array(list(keypoints.values()))
        if points_arr.size > 0:
            scatter = ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], c=[color], s=10, alpha=1.0)
            artists.append(scatter)

        # Track ID
        # Use a central keypoint that is likely to exist
        central_kp_name = 'thorax' if 'thorax' in keypoints else next(iter(keypoints), None)
        if central_kp_name:
            pos = np.array(keypoints[central_kp_name])
            text = ax.text(pos[0], pos[1], pos[2], f"ID:{track_id}", color=color, fontsize='x-small')
            artists.append(text)

    return artists


def run_viewer(all_frame_data, bones, volume_bounds, floor_z=None):
    """ Main viewer function. Expects a list of frame dictionaries """

    try:
        plt.rcParams['keymap.forward'].remove('right')
        plt.rcParams['keymap.back'].remove('left')
    except ValueError:
        pass

    num_frames = len(all_frame_data)
    if num_frames == 0:
        print("No data to display.")
        return

    # One-time Setup of the Figure and Axes
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0, 0.1, 1, 0.9], projection='3d')
    slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(volume_bounds['x'])
    ax.set_ylim(volume_bounds['y'])
    ax.set_zlim(volume_bounds['z'])
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    z_range = abs(ax.get_zlim()[1] - ax.get_zlim()[0])

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.view_init(elev=20., azim=-70)

    if floor_z is not None:
        xx, yy = np.meshgrid(np.linspace(*volume_bounds['x'], 10), np.linspace(*volume_bounds['y'], 10))
        zz = np.full_like(xx, floor_z)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    # Slider and event handling
    frame_slider = Slider(ax=slider_ax, label='Frame', valmin=0, valmax=num_frames - 1, valinit=0, valstep=1)
    current_artists = []

    def update(val):
        nonlocal current_artists
        frame_idx = int(frame_slider.val)

        for artist in current_artists:
            artist.remove()
        current_artists.clear()

        frame_data = all_frame_data[frame_idx]
        current_artists = draw_skeletons(frame_data, bones, ax)

        ax.set_title(f"Frame {frame_data.get('frame_idx', '')}")
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    def on_key(event):
        current_val = frame_slider.val
        if event.key == 'right':
            frame_slider.set_val(min(current_val + 1, num_frames - 1))
        elif event.key == 'left':
            frame_slider.set_val(max(current_val - 1, 0))
        elif event.key == 'pageup':
            frame_slider.set_val(min(current_val + 50, num_frames - 1))
        elif event.key == 'pagedown':
            frame_slider.set_val(max(current_val - 50, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)
    update(0)
    plt.show()


if __name__ == '__main__':
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    skeleton_input_path = folder / prefix / 'inputs' / 'tracking'
    _, bones = fileio.load_skeleton_SLEAP(skeleton_input_path, indices=False)
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    data_to_view = folder / prefix / 'outputs' / f'tracklets_session{session}.pkl'
    # data_to_view = folder / prefix / 'outputs' / f'linked_tracks_session{session}.pkl'

    print(f"Loading data from: {data_to_view.name}")
    with open(data_to_view, 'rb') as f:
        loaded_data = pickle.load(f)

    # all_Zs = []
    # for frame in tracked_skeletons:
    #     for skel in frame['skeletons']:
    #         max_z = np.max(np.stack(list(skel['keypoints'].values()), axis=1), axis=1)[2]
    #         all_Zs.append(float(max_z))
    #
    # hist, bin_edges = np.histogram(all_Zs, bins=1000)
    # peak_bin_index = np.argmax(hist)
    # floor_z = 0.5 * (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1])

    # Check if the data needs to be converted from track-centric to frame-centric
    if isinstance(loaded_data, dict):
        # the format for tracklets
        frame_data_for_viewer = convert_track_centric_to_frame_centric(loaded_data)
    elif isinstance(loaded_data, list):
        # the format from the final smoothed tracks
        frame_data_for_viewer = loaded_data
    else:
        raise TypeError("Loaded data is not in a recognized format (dict or list).")

    run_viewer(frame_data_for_viewer, bones, volume_bounds, floor_z=None)