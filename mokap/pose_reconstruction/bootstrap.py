"""
Bootstrap skeleton statistics from 3D point soup.

AnatomyBootstrapper: Learns bone length statistics (ratios, variability)
DynamicsBootstrapper: Learns motion dynamics (process noise, association weights)
"""
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import polars as pl
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.stats import median_abs_deviation

from mokap.pose_reconstruction.configs import MIN_PROCESS_NOISE, MAX_PROCESS_NOISE
from mokap.pose_reconstruction.datatypes import PointSoup
from mokap.pose_reconstruction.skeleton import Bone, Skeleton, SkeletonStats, BoneStats, KeypointDynamics
from mokap.pose_reconstruction.utils import robust_stats


def _run_trackpy(
        soup: PointSoup,
        search_range: float,
        memory: int = 0,
        max_frames: int = 4000,
        keypoint_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run trackpy linking on point soup (per keypoint type independently).
    """
    tp.quiet()

    polars_df = soup.to_dataframe()

    # Min / max frames
    min_frame = polars_df['frame'].min()
    if (polars_df['frame'].max() - min_frame) > max_frames:
        polars_df = polars_df.filter(pl.col('frame') < (min_frame + max_frames))

    # Filter for reconstructed points only
    if "status" in polars_df.columns:
        polars_df = polars_df.filter(pl.col("status") == "reconstructed")

    if keypoint_filter is not None:
        polars_df = polars_df.filter(pl.col('keypoint').is_in(keypoint_filter))

    partitions = polars_df.partition_by("keypoint", as_dict=True)

    results = []

    # 4. Process chunks
    for kp_name, sub_pldf in partitions.items():

        pdf = sub_pldf.to_pandas()

        linked = tp.link_df(
            pdf,
            search_range=search_range,
            pos_columns=['x', 'y', 'z'],
            t_column='frame',
            memory=memory
        )

        if 'keypoint' not in linked.columns:
            linked['keypoint'] = kp_name

        results.append(linked)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


class AnatomyBootstrapper:
    """
    Bootstrap bone length statistics from 3D point soup.

    - Track individual keypoints across frames using trackpy
    - Find co-occurring tracklet pairs (same frames, spatially close)
    - Measure bone lengths within each pair
    - Compute intra-individual variance per pair, then pool across pairs
    - Use symmetry (canonical names) to increase sample size

    Produces a SkeletonStats object with learned bone ratios and variabilities.
    """

    def __init__(
            self,
            skeleton: Skeleton,
            default_variability: float = 0.1,
            min_samples: int = 10,
            max_bone_length: float = np.inf,
            reference_bone: Optional[Bone] = None,
            min_tracklet_length: int = 5,
            max_displacement: float = 1.0,
            store_debug_data: bool = False
    ):
        self._debug = store_debug_data

        self.skeleton = skeleton

        # Config
        self.default_variability = default_variability
        self.min_samples = min_samples
        self.max_bone_length = max_bone_length
        self.min_tracklet_length = min_tracklet_length
        self.max_displacement = max_displacement

        self.reference_bone = reference_bone or skeleton.central_bone
        print(f"[Anatomy] Reference bone: {self.reference_bone}")

        # Debug data storage
        self.debug_histograms: Dict[str, np.ndarray] = {}
        self.debug_intra_individual_mads: Dict[str, List[float]] = defaultdict(list)

    def process(self, soup: PointSoup, max_frames: int = 5000) -> SkeletonStats:
        """
        Process point soup and return learned SkeletonStats.

        - Track individual keypoints across frames
        - For each bone, find co-occurring tracklet pairs
        - Measure bone lengths within pairs, compute per-pair variance
        - Pool intra-individual variances using symmetry (canonical names)
        """
        df = _run_trackpy(
            soup=soup,
            search_range=self.max_displacement,
            memory=1,
            max_frames=max_frames
        )

        # Collect measurements grouped by canonical bone name
        canon_all_lengths: Dict[str, List[float]] = defaultdict(list)
        canon_pair_stats: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for bone in self.skeleton.bones:
            # Inner-join dataframe on 'frame' to find co-occurring detections
            df_k1 = df[df['keypoint'] == bone.k1][['frame', 'x', 'y', 'z', 'particle']]
            df_k2 = df[df['keypoint'] == bone.k2][['frame', 'x', 'y', 'z', 'particle']]
            pairs = df_k1.merge(df_k2, on='frame', suffixes=('_1', '_2'))

            if pairs.empty:
                continue

            # Calculate distances
            pos1 = pairs[['x_1', 'y_1', 'z_1']].values
            pos2 = pairs[['x_2', 'y_2', 'z_2']].values
            pairs['dist'] = np.linalg.norm(pos1 - pos2, axis=1)

            # Filter by max bone length
            pairs = pairs[pairs['dist'] < self.max_bone_length]

            # Canonical key for symmetry pooling
            canon_key = self.skeleton.canonical(bone)

            # Group by tracklet pairs to get intra-individual stats
            for (p1, p2), group in pairs.groupby(['particle_1', 'particle_2']):
                if len(group) < self.min_tracklet_length:
                    continue

                lengths = group['dist'].values
                median_len = float(np.median(lengths))
                mad = float(median_abs_deviation(lengths))

                canon_pair_stats[canon_key].append((median_len, mad))
                canon_all_lengths[canon_key].extend(lengths.tolist())

                if self._debug:
                    self.debug_intra_individual_mads[canon_key].append(mad)

        # Store debug histograms
        if self._debug:
            self.debug_histograms = {k: np.array(v) for k, v in canon_all_lengths.items()}

        # Calculate reference length
        ref_canon_key = self.skeleton.canonical(self.reference_bone)

        if (ref_canon_key in canon_all_lengths and len(canon_all_lengths[ref_canon_key]) >= self.min_samples):
            reference_length = float(np.median(canon_all_lengths[ref_canon_key]))
        else:
            # Fallback: median of all bone lengths
            all_lengths = [l for lengths in canon_all_lengths.values() for l in lengths]
            reference_length = float(np.median(all_lengths)) if all_lengths else 1.0
            print(f"[Anatomy] Warning: Reference bone has insufficient data. "
                  f"Using fallback: {reference_length:.3f}")

        if np.isnan(reference_length) or reference_length <= 0:
            reference_length = 1.0

        print(f"[Anatomy] Reference length: {reference_length:.3f}")

        # Build SkeletonStats
        stats = SkeletonStats(self.skeleton)
        stats.reference_bone = self.reference_bone
        stats.reference_length_world = reference_length

        # Populate bone stats
        for bone in self.skeleton.bones:
            canon_key = self.skeleton.canonical(bone)

            all_lengths = canon_all_lengths.get(canon_key, [])
            pair_data = canon_pair_stats.get(canon_key, [])

            if len(pair_data) < 2 or len(all_lengths) < self.min_samples:
                # Insufficient data: use defaults
                ratio = 1.0
                variability = self.default_variability
            else:
                ratio = float(np.median(all_lengths)) / reference_length
                # Pooled uncertainty: median of intra-individual MADs
                variability = float(np.median([mad for (_, mad) in pair_data])) / reference_length
                variability = max(variability, ratio * 0.001)  # floor

            stats.bone_stats[bone] = BoneStats(
                ratio_length=ratio,
                variability=variability,
                count=len(all_lengths),
                pairs=len(pair_data),
                length_world=ratio * reference_length
            )

        return stats


class DynamicsBootstrapper:
    """
    Bootstrap dynamics parameters from 3D point soup.

    Learns per-keypoint:
    - Process noise (from acceleration statistics)
    - Association weight (from velocity jitter)

    Uses graph distance as prior for keypoints with insufficient data.
    """

    def __init__(
            self,
            skeleton: Skeleton,
            fps: float = 30.0,
            max_displacement: float = 1.0,
            min_track_length: int = 15,
            reference_bone_length: float = 1.0,
            min_process_noise: float = MIN_PROCESS_NOISE,
            max_process_noise: float = MAX_PROCESS_NOISE,
            measurement_noise: float = 0.5,
            store_debug_data: bool = False
    ):
        self._debug = store_debug_data
        self.skeleton = skeleton

        # Config
        self.fps = fps
        self.max_displacement = max_displacement
        self.min_track_length = min_track_length
        self.reference_bone_length = reference_bone_length
        self.min_process_noise = min_process_noise
        self.max_process_noise = max_process_noise
        self.base_measurement_noise = measurement_noise

        # Debug storage
        self.debug_tracks: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        self.debug_velocities: Dict[str, List[float]] = defaultdict(list)

    def process(self, soup: PointSoup, max_frames: int = 4000) -> 'SkeletonStats':
        """
        Process point soup and return dynamics parameters per keypoint.

        Returns dict mapping keypoint name -> {process_noise, measurement_noise,
                                                association_weight, source}
        """
        df = _run_trackpy(
            soup=soup,
            search_range=self.max_displacement,
            memory=0,
            max_frames=max_frames
        )

        if self._debug:
            self.debug_velocities.clear()
            self.debug_tracks.clear()

        # Collect velocity/acceleration stats per canonical keypoint
        canon_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"vel": [], "acc": []}
        )

        for (name, particle), track in df.groupby(['keypoint', 'particle']):
            if particle == -1 or len(track) < self.min_track_length:
                continue

            track = track.sort_values('frame')

            # Split into contiguous segments (no frame jumps)
            frame_diffs = np.diff(track['frame'].values)
            jump_indices = np.where(frame_diffs > 1)[0] + 1
            segments = np.split(track[['x', 'y', 'z']].values, jump_indices)

            for seg_pos in segments:
                if len(seg_pos) < 4:
                    continue

                # Velocity (dx/dt)
                vel_vec = np.diff(seg_pos, axis=0)
                vel = np.linalg.norm(vel_vec, axis=1)

                # Acceleration (dv/dt)
                acc_vec = np.diff(vel_vec, axis=0)
                acc = np.linalg.norm(acc_vec, axis=1)

                canon_name = self.skeleton.canonical(name)
                canon_stats[canon_name]["vel"].extend(vel.tolist())
                canon_stats[canon_name]["acc"].extend(acc.tolist())

                if self._debug:
                    self.debug_velocities[canon_name].extend(vel.tolist())
                    if len(self.debug_tracks[canon_name]) < 200:
                        self.debug_tracks[canon_name].append(track)

        # Derive final parameters per keypoint

        stats = SkeletonStats(self.skeleton)

        for keypoint in self.skeleton.keypoints:

            canon_name = self.skeleton.canonical(keypoint)
            data = canon_stats.get(canon_name, {"vel": [], "acc": []})

            total_distance = np.sum(data["vel"]) # otal distance travelled by this keypoint in the dataset

            if len(data["vel"]) < 50 or total_distance < (self.reference_bone_length * 5.0):    # TODO: Maybe move these thresholds somewhere easily configurable
                # Insufficient data: use topology-based prior
                graph_dist = self.skeleton.graph_distance(keypoint)
                process_noise = max(self.min_process_noise, 0.1 * (1 + graph_dist))
                weight = 1.0 / (1 + graph_dist * 0.5)
                source = "topology_prior"
            else:
                median_vel, vel_mad = robust_stats(data["vel"])
                median_acc, acc_mad = robust_stats(data["acc"])

                # Process noise from acceleration (clipped)
                process_noise = float(np.clip(median_acc + 2.0 * acc_mad, self.min_process_noise, self.max_process_noise))

                # Weight: stable keypoints get higher weight
                jitter = vel_mad / max(self.reference_bone_length, 0.01)
                weight = float(np.clip(1.0 / (1.0 + (jitter * 10.0) ** 2), 0.1, 1.0))
                source = "data"

            stats.keypoint_dynamics[keypoint] = KeypointDynamics(
                process_noise=process_noise,
                measurement_noise=self.base_measurement_noise,
                association_weight=weight,
                source=source
            )

        return stats


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from mokap.pose_reconstruction.utils import plot_tracks_3d

    BASE_DIR = Path.home() / 'Desktop' / '3d_ant_data'
    PREFIX = '240905-1616'
    SESSION = 22
    DEBUG_PLOT = True

    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'

    soup_file = output_dir / f'soup_session{SESSION}.parquet'
    skel_file = output_dir / 'messor_skeleton.toml'
    stats_file = output_dir / 'skeleton_stats.json'

    skeleton = Skeleton.load(input_dir)
    soup = PointSoup.load(soup_file)

    print(f"Loaded point soup from {soup_file}")
    print(f"Loaded skeleton with {len(skeleton.keypoints)} keypoints, {len(skeleton.bones)} bones")

    anat = AnatomyBootstrapper(
        skeleton=skeleton,
        default_variability=0.1,
        min_samples=10,
        max_bone_length=2.5,
        reference_bone=None,  # auto-select
        min_tracklet_length=5,
        max_displacement=1.5,
        store_debug_data=DEBUG_PLOT
    )
    stats_anatomy = anat.process(soup)
    stats_anatomy.save(stats_file)

    # Run dynamics bootstrap
    dyn = DynamicsBootstrapper(
        skeleton=skeleton,
        fps=100.0,
        max_displacement=1.5,
        min_track_length=5,
        reference_bone_length=stats_anatomy.reference_length_world,
        min_process_noise=0.01,
        measurement_noise=0.1,
        store_debug_data=DEBUG_PLOT
    )
    stats_dynamics = dyn.process(soup, max_frames=4000)
    stats_dynamics.save(stats_file)

    print(f"Saved skeleton stats (anatomy + dynamics) to {stats_file}")


    ##

    if DEBUG_PLOT:
        ref_len = stats_anatomy.reference_length_world

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"Bootstrap results (reference length: {ref_len:.2f} mm)",
            fontsize=16
        )
        colors = ["r", "g", "b", "orange", "purple", "cyan"]

        # Plot 1: Reference bone length distribution
        ax1 = fig.add_subplot(331)
        ax1.set_title("Reference bone length distribution")
        ref_canon_key = anat.skeleton.canonical(anat.reference_bone)

        if ref_canon_key in anat.debug_histograms:
            lengths = anat.debug_histograms[ref_canon_key]
            ax1.hist(lengths, bins=50, color="gray", alpha=0.7)
            ax1.axvline(ref_len, color="red", linestyle="--", label="Median")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "Ref bone data not available", ha='center', transform=ax1.transAxes)
        ax1.set_xlabel("Length (mm)")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bone length ratios
        ax2 = fig.add_subplot(332)
        ax2.set_title("Bone length ratios (population)")
        for i, (bone_key, lengths) in enumerate(list(anat.debug_histograms.items())[:6]):
            ratios = np.asarray(lengths) / ref_len
            ax2.hist(
                ratios, bins=50, alpha=0.3, density=True,
                label=bone_key, color=colors[i % len(colors)]
            )
        ax2.legend(fontsize="x-small")
        ax2.set_xlabel("Ratio to reference")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Intra-individual MAD distribution
        ax3 = fig.add_subplot(333)
        ax3.set_title("Intra-individual MAD per tracklet")
        for i, (bone_key, mads) in enumerate(list(anat.debug_intra_individual_mads.items())[:6]):
            if len(mads) < 3:
                continue
            ax3.hist(
                mads, bins=30, alpha=0.4, density=True,
                label=bone_key, color=colors[i % len(colors)]
            )
            ax3.axvline(np.median(mads), color=colors[i % len(colors)], linestyle="--", alpha=0.8)
        ax3.set_xlabel("MAD within tracklet (mm)")
        ax3.legend(fontsize="x-small")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Velocity distributions
        ax4 = fig.add_subplot(334)
        ax4.set_title("Velocity distributions")
        for i, (kp, vels) in enumerate(dyn.debug_velocities.items()):
            if i >= len(colors) or len(vels) < 50:
                continue
            med = np.median(vels)
            ax4.hist(
                vels, bins=50, alpha=0.3, density=True,
                label=kp, color=colors[i % len(colors)]
            )
            ax4.axvline(med, color=colors[i % len(colors)], linestyle="--")
        ax4.set_xlabel("Speed (mm / frame)")
        ax4.legend(fontsize="x-small")
        ax4.grid(True, alpha=0.3)

        # Plot 5: Table of anatomy stats
        ax5 = fig.add_subplot(335)
        ax5.axis("off")
        ax5.set_title("Bone statistics (intra-individual MAD)")

        table_data = []
        seen_canon = set()

        for bone, bone_stats in stats_anatomy.bone_stats.items():
            canon_key = anat.skeleton.canonical(bone)
            if canon_key in seen_canon:
                continue
            seen_canon.add(canon_key)

            table_data.append([
                canon_key[:20],
                f"{bone_stats.ratio_length:.3f}",
                f"{bone_stats.variability:.4f}",
                f"{bone_stats.pairs}",
            ])
            if len(table_data) >= 10:
                break

        if table_data:
            table = ax5.table(
                cellText=table_data,
                colLabels=["Bone", "Med. Ratio", "Variability", "Pairs"],
                loc="center",
                cellLoc="center",
            )
            table.scale(1, 1.3)

        # Plot 6: Table of learned KF parameters
        ax6 = fig.add_subplot(336)
        ax6.axis("off")
        ax6.set_title("Learned KF parameters")

        table_data = []
        seen_canon = set()

        # Sort by association weight
        sorted_items = sorted(
            stats_dynamics.keypoint_dynamics.items(),
            key=lambda x: x[1].association_weight,
            reverse=True,
        )

        for name, params in sorted_items:
            canon_name = skeleton.canonical(name)
            if canon_name in seen_canon:
                continue
            seen_canon.add(canon_name)

            src = " (prior)" if params.source != "data" else ""
            table_data.append([
                canon_name[:15] + src,
                f"{params.process_noise:.3f}",
                f"{params.association_weight:.2f}",
            ])
            if len(table_data) >= 12:
                break

        if table_data:
            table = ax6.table(
                cellText=table_data,
                colLabels=["Keypoint", "Q (process noise)", "Weight"],
                loc="center",
                cellLoc="center",
            )
            table.scale(1, 1.2)

        # Plot 7: Example 3D tracks
        ax7 = fig.add_subplot(338, projection="3d")
        preferred = ["thorax", "neck", "head"]
        chosen = next(
            (p for p in preferred if p in dyn.debug_tracks),
            next(iter(dyn.debug_tracks), None),
        )
        if chosen and dyn.debug_tracks[chosen]:
            tracks_df = pd.concat(dyn.debug_tracks[chosen])
            plot_tracks_3d(ax7, tracks_df, f"Tracks: {chosen}")
        else:
            ax7.text(0, 0, 0, "No long tracks")

        plt.tight_layout()
        plt.show()