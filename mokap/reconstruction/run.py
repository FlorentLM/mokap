import pickle
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict
from alive_progress import alive_bar

from mokap.utils import fileio
from mokap.reconstruction.config import PipelineConfig
from mokap.reconstruction.datatypes import SoupData
from mokap.reconstruction.utils import create_canonical_map, prepare_reconstruction_input
from mokap.reconstruction.reconstruction import Reconstructor
from mokap.reconstruction.anatomy import StatsBootstrapper, AnatomyLearner
from mokap.reconstruction.tracking import SkeletonAssembler, MultiObjectTracker
from mokap.reconstruction.linking import FragmentMerger, TrackletLinker, load_tracklets, combine_chains
from lucida import CameraRig


# Stage 1: Reconstruction
# =======================
def stage1_reconstruction(
        df: pl.DataFrame,
        keypoints: list,
        camera_rig: CameraRig,
        volume_bounds: dict,
        config: PipelineConfig,
        batch_size: int,
        output_file: Path
) -> Optional[SoupData]:

    print("\nStage 1: Cooking 3D point soup")

    reconstructor = Reconstructor(
        camera_parameters=cal_data,
        volume_bounds=volume_bounds,
        config=config.reconstruction
    )

    camera_names = sorted(list(cal_data.keys()))

    # Sort frames for slicing
    all_frame_indices = np.sort(df["frame"].unique().to_numpy())
    total_frames = len(all_frame_indices)

    soup_batches = []

    print(f"Processing {total_frames} frames in batches of {batch_size}...")

    # Iterate batches sequentially (JAX handles internal parallelism)
    for i in range(0, total_frames, batch_size):
        batch_frames = all_frame_indices[i: i + batch_size]
        min_f, max_f = batch_frames[0], batch_frames[-1]

        # Filter DF for this batch
        df_batch = df.filter((pl.col("frame") >= min_f) & (pl.col("frame") <= max_f))

        if df_batch.is_empty():
            continue

        inputs = prepare_reconstruction_input(df_batch, camera_names, keypoints)

        # Run reconstruction
        batch_soup = reconstructor.reconstruct_batch(inputs, keypoints)

        # Keep if valid
        if batch_soup.num_points > 0 or len(batch_soup.ray_origins) > 0:
            soup_batches.append(batch_soup)
            print(f"  Batch {min_f}-{max_f}: {batch_soup.num_points} pts, {len(batch_soup.ray_origins)} rays")

    if not soup_batches:
        print("No points reconstructed.")
        return None

    print("Concatenating batches...")
    full_soup = SoupData.concatenate(soup_batches)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(full_soup, f)

    print(f"Reconstruction complete. Saved {full_soup.num_points} points to {output_file}")
    return full_soup


# Stage 2: Tracking
# =================
def stage2_tracking(
        soup: SoupData,
        bones: list,
        symmetry: list,
        config: PipelineConfig,
        stats_prior_file: Optional[Path],
        stats_output_file: Path,
        tracklets_output_file: Path
) -> Dict[int, List[dict]]:

    print("\nStage 2: Assembling and tracking skeletons")

    # Bootstrap anatomy
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=stats_prior_file,
        bootstrap_data=soup,
        config=config.anatomy
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialise pipeline
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats, config=config.anatomy)

    assembler = SkeletonAssembler(
        bones_list=bones,
        bone_stats=bone_stats,
        assembler_config=config.assembler,
        tracker_config=config.tracker
    )

    tracker = MultiObjectTracker(assembler=assembler, config=config.tracker)

    # Run
    unique_frames = np.unique(soup.frame_indices)
    if len(unique_frames) == 0:
        print("Soup empty.")
        return {}

    min_frame, max_frame = int(unique_frames[0]), int(unique_frames[-1])
    tracklets_by_id = defaultdict(list)

    print(f"Tracking skeletons from frame {min_frame} to {max_frame}...")

    with alive_bar(total=(max_frame - min_frame + 1), force_tty=True) as bar:
        for frame_idx in range(min_frame, max_frame + 1):

            # Update anatomy
            current_stats = anatomy_learner.get_stats()
            assembler.update_bone_stats(current_stats)

            frame_soup = soup.get_frame_slice(frame_idx)

            if frame_soup.num_points > 0 or len(frame_soup.ray_origins) > 0:
                active_tracklets = tracker.update(frame_soup, frame_idx)
            else:
                active_tracklets = tracker.predict_only(frame_idx)

            # Store results
            for tracklet in active_tracklets:
                if tracklet.last_update_frame == frame_idx:
                    anatomy_learner.add_sample(tracklet.skeleton)

                # Export to Dict for serialization
                skel_dict = tracklet.skeleton.to_dict()
                skel_dict.update({
                    'frame_idx': frame_idx,
                    'track_idx': tracklet.track_idx,
                    'track_health': tracklet.health,
                    'track_anatomical_integrity': tracklet.anatomical_integrity,
                    'track_uncertainty_pos': tracklet.uncertainty['position'].tolist(),
                    'track_velocity': tracklet.kf.x[3:6].flatten().tolist(),
                    'track_predicted_pos': tracklet.predicted_position.tolist(),
                    'time_since_update': tracklet.time_since_update
                })
                tracklets_by_id[tracklet.track_idx].append(skel_dict)

            bar()

    print(f"Tracking complete. Generated {len(tracklets_by_id)} unique tracklets.")

    tracklets_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklets_output_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)

    # Save final stats
    with open(stats_output_file, 'w') as f:
        json.dump(anatomy_learner.get_stats(), f, indent=2)

    return dict(tracklets_by_id)


# Stage 3: Linking
# ================
def stage3_linking(
        all_tracked_data: dict,
        keypoints: list,
        bones: list,
        symmetry: list,
        config: PipelineConfig,
        stats_file: Path,
        final_output_file: Path
):

    print("\nStage 3: Merging and linking tracklets")

    if not all_tracked_data:
        print("No tracklets found to link.")
        return

    with open(stats_file, 'r') as f:
        bone_stats = json.load(f)
    canonical_map = create_canonical_map(keypoints, symmetry)

    # 3a Merge
    print("  Merging overlapping fragments...")
    initial_tracklets = load_tracklets(all_tracked_data, config=config.linker)
    merger = FragmentMerger(initial_tracklets, bone_stats, bones, config=config.merger)
    merged_tracklets = merger.merge_fragments()

    # 3b Link
    print("  Linking temporal gaps...")
    linker = TrackletLinker(merged_tracklets, canonical_map, config=config.linker)
    chains = linker.link_tracklets()

    # 3c Combine
    print("  Combining chains...")
    final_linked_tracks = combine_chains(chains, merged_tracklets)

    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_file, 'wb') as f:
        pickle.dump(final_linked_tracks, f)

    print(f"Pipeline complete! Saved {len(final_linked_tracks)} final linked tracks to '{final_output_file}'")


## =====================================================================================================================

if __name__ == '__main__':

    # Config
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22
    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    config = PipelineConfig()

    BATCH_SIZE = 500

    # Paths
    input_dir = folder / prefix / 'inputs' / 'tracking'
    output_dir = folder / prefix / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    points_soup_file = output_dir / f'points_soup_session{session}.pkl'
    stats_file = output_dir / f'bone_stats_session{session}.json'
    tracklets_file = output_dir / f'tracklets_session{session}.pkl'
    final_tracks_file = output_dir / f'linked_tracks_session{session}.pkl'

    # Load Metadata
    df = fileio.load_session(input_dir, session=session, use_polars=True)
    rig_file = folder / prefix / 'calibration' / 'camera_rig.toml'
    rig = CameraRig.load(rig_file)
    print(f"Loaded rig with {len(rig)} cameras.")
    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(input_dir, symmetry=True)

    # Execution

    # Reconstruction
    if not points_soup_file.exists():
        points_soup = stage1_reconstruction(
            df, keypoints, rig, volume_bounds, config, BATCH_SIZE, points_soup_file
        )
    else:
        print(f"Loading existing soup from {points_soup_file}")
        with open(points_soup_file, 'rb') as f:
            points_soup = pickle.load(f)

    # Tracking
    if not tracklets_file.exists():
        tracked_data = stage2_tracking(
            points_soup, bones, symmetry, config, None, stats_file, tracklets_file
        )
    else:
        print(f"Loading existing tracklets from {tracklets_file}")
        with open(tracklets_file, 'rb') as f:
            tracked_data = pickle.load(f)

    # Linking
    stage3_linking(tracked_data, keypoints, bones, symmetry, config, stats_file, final_tracks_file)