import pickle
from pathlib import Path
from joblib import Parallel, delayed
from mokap.reconstruction.reconstruction import Reconstructor, SoupPoint
from mokap.utils import fileio


def reconstruct_batch(batch_of_groups, keypoints, camera_parameters, volume_bounds, config):
    # each worker process must have its own Reconstructor instance
    # (because each process will compile the JAX functions it needs)
    reconstructor = Reconstructor(
        camera_parameters=camera_parameters,
        volume_bounds=volume_bounds,
        config=config
    )

    soup_batch = {}
    for ftuple, df_frame in batch_of_groups:
        reconstructed_points = reconstructor.reconstruct_frame(
            df_frame=df_frame,
            keypoint_names=keypoints
        )
        soup_batch[ftuple[0]] = reconstructed_points

    return soup_batch


if __name__ == '__main__':
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    df = fileio.load_session(folder / prefix / 'inputs' / 'tracking', session=session, use_polars=True)

    cal_data = fileio.read_parameters(folder / prefix / 'calibration')
    keypoints, bones = fileio.load_skeleton_SLEAP(folder / prefix / 'inputs' / 'tracking', indices=False)

    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    reconstructor_config = {
        'repro_thresh': 10.0,
        'cluster_radius': 2.0,
        'view_count_weight': 10.0,
        'repro_error_weight': 1.0
    }

    grouped_by_frame = df.group_by('frame', maintain_order=True)
    all_grouped_frames = list(grouped_by_frame)

    # Split the list of groups into batches
    N_JOBS = -1         # use all available CPU cores
    BATCH_SIZE = 200

    num_frames = len(all_grouped_frames)
    frame_batches = [
        all_grouped_frames[i: i + BATCH_SIZE]
        for i in range(0, num_frames, BATCH_SIZE)
    ]
    num_batches_actual = len(frame_batches)

    print(
        f"Cooking points soup with {len(frame_batches)} batches across {N_JOBS if N_JOBS != -1 else 'all'} cores...")

    # Use joblib to run reconstruction in parallel
    results_list = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(reconstruct_batch)(
            batch, keypoints, cal_data, volume_bounds, reconstructor_config
        ) for batch in frame_batches
    )

    points_by_frame = {}
    for batch_dict in results_list:
        points_by_frame.update(batch_dict)

    out_file = folder / prefix / 'outputs' / f'points_soup_session{session}.pkl'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'wb') as f:
        pickle.dump(points_by_frame, f)

    print(f"Reconstruction complete. Points soup saved to {out_file}")