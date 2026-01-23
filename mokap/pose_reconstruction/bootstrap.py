import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from mokap.pose_reconstruction.datatypes import PointSoup
from mokap.pose_reconstruction.utils import create_canonical_map, plot_tracks_3d, robust_stats


# TODO: Trackpy linking should be done once as it is used by both classes


class AnatomyBootstrapper:
    """
    Bootstrap bone length statistics from 3D point soup.

    We need to separate population variation (different subject sizes) from
    intra-individual variation (measurement noise + articulation).

    1. For each bone, track both endpoint keypoints independently using trackpy
    2. Find co-occurring tracklet pairs (same frames, spatially close)
    3. Measure bone lengths within each co-occurring pair
    4. Compute variance within each pair, then pool across pairs
    5. Use symmetry to double the sample pool
    """

    def __init__(
            self,
            keypoint_names: List[str],
            bones: List[Tuple[str, str]],
            symmetry_pairs: Optional[List[Tuple[str, str]]] = None,
            MAD_ratio: float = 0.1,
            min_samples: int = 10,
            max_bone_length: float = np.inf,
            reference_bone: Optional[Sequence[str]] = None,
            min_tracklet_length: int = 5,
            tracklet_search_range: float = 2.0,
    ):
        self.keypoint_names = keypoint_names
        self.bones = [tuple(sorted(b)) for b in bones]
        self.canon_map = create_canonical_map(keypoint_names, symmetry_pairs)

        # Config
        self.MAD_ratio = MAD_ratio
        self.min_samples = min_samples
        self.max_bone_length = max_bone_length
        self.min_tracklet_length = min_tracklet_length
        self.tracklet_search_range = tracklet_search_range

        # Build skeleton graph for stability scoring
        self._skeleton_graph = nx.Graph()
        self._skeleton_graph.add_edges_from(self.bones)
        self._degrees = dict(self._skeleton_graph.degree())

        if reference_bone is not None:
            self.ref_bone = tuple(sorted(reference_bone))
        else:
            self.ref_bone = self._auto_select_ref_bone()
        print(f"[Anatomy] Reference bone: {self.ref_bone}")

        # Debug storage
        self.debug_histograms = {}
        self.debug_intra_individual_stds = defaultdict(list)
        self.debug_n_tracklet_pairs = defaultdict(int)

    def _auto_select_ref_bone(self) -> Tuple[str, str]:
        """Select the most stable bone based on graph connectivity."""

        return max(
            self._skeleton_graph.edges,
            key=lambda e: self._degrees[e[0]] + self._degrees[e[1]]
        )

    def process(self, soup: PointSoup, max_frames: int = 5000) -> Dict[str, Any]:
        """
        1. Track individual keypoints across frames
        2. For each bone find co-occurring tracklet pairs
        3. Measure bone lengths within pairs, compute per-pair variance
        4. Pool intra-individual variances (using symmetry)
        """

        import trackpy as tp
        tp.quiet()

        df = soup.to_df()

        print("[Anatomy] Tracking keypoints...")

        # Subset frames for speed if needed
        if df['frame'].nunique() > max_frames:
            f_min = df['frame'].min()
            df = df[df['frame'] < f_min + max_frames]

        # Link (per keypoint type of course)
        df['particle'] = -1
        for name, group in df.groupby('keypoint'):
            linked = tp.link_df(group, search_range=self.tracklet_search_range, pos_columns=['x', 'y', 'z'], memory=1)
            df.loc[linked.index, 'particle'] = linked['particle']

        # Bone measurement
        canon_all_lengths = defaultdict(list)
        canon_pair_stats = defaultdict(list)

        for u, v in self.bones:
            # inner-join the dataframe with itself on 'frame' to find pairs
            df_u = df[df['keypoint'] == u][['frame', 'x', 'y', 'z', 'particle']]
            df_v = df[df['keypoint'] == v][['frame', 'x', 'y', 'z', 'particle']]
            pairs = df_u.merge(df_v, on='frame', suffixes=('_u', '_v'))

            if pairs.empty:
                continue

            # Calculate distances for all pairs
            dist = np.linalg.norm(pairs[['x_u', 'y_u', 'z_u']].values - pairs[['x_v', 'y_v', 'z_v']].values, axis=1)
            pairs['dist'] = dist

            # Filter sane lengths
            pairs = pairs[pairs['dist'] < self.max_bone_length]

            # Group by unique tracklet pairs (particle_u + particle_v) to get intra-individual stats
            canon_key = ";".join(sorted((self.canon_map[u], self.canon_map[v])))

            for (p_u, p_v), group in pairs.groupby(['particle_u', 'particle_v']):
                if len(group) < self.min_tracklet_length:
                    continue

                lengths = group['dist'].values
                mad = median_abs_deviation(lengths)

                canon_pair_stats[canon_key].append((np.median(lengths), mad))
                canon_all_lengths[canon_key].extend(lengths)
                self.debug_intra_individual_stds[canon_key].append(mad)

        # Aggregation
        self.debug_histograms = {k: np.array(v) for k, v in canon_all_lengths.items()}
        ref_canon_key = ";".join(sorted([self.canon_map[k] for k in self.ref_bone]))

        # Calculate reference length
        if ref_canon_key in canon_all_lengths and len(canon_all_lengths[ref_canon_key]) >= self.min_samples:
            ref_length = float(np.median(canon_all_lengths[ref_canon_key]))
        else:
            all_lens = [l for lens in canon_all_lengths.values() for l in lens]
            ref_length = float(np.median(all_lens)) if all_lens else 1.0
            print(f"[Anatomy] Warning: Reference bone has insufficient data. Using fallback: {ref_length:.3f}")

        if np.isnan(ref_length) or ref_length <= 0:
            ref_length = 1.0

        print(f"[Anatomy] Reference length: {ref_length:.3f}")

        # Calculate ratios
        bones_ratios = {}
        for u, v in self.bones:
            out_key = ";".join(sorted((u, v)))
            canon_key = ";".join(sorted((self.canon_map[u], self.canon_map[v])))

            all_lengths = canon_all_lengths.get(canon_key, [])
            pair_data = canon_pair_stats.get(canon_key, [])

            if len(pair_data) < 2 or len(all_lengths) < self.min_samples:
                med_ratio, mad_ratio = 1.0, self.MAD_ratio

            else:
                med_ratio = float(np.median(all_lengths)) / ref_length
                # Pooled uncertainty: median of the intra-individual variances
                mad_ratio = float(np.median([mad for (_, mad) in pair_data])) / ref_length
                mad_ratio = max(mad_ratio, med_ratio * 0.001)  # floor

            bones_ratios[out_key] = {
                "median_ratio": med_ratio, "mad_ratio": mad_ratio,
                "count": len(all_lengths), "n_pairs": len(pair_data)
            }

        return {"reference_bone": list(self.ref_bone), "median_reference_length": ref_length,
                "bones_ratios": bones_ratios}

    def _fallback_stats(self) -> Dict[str, Any]:
        """Return fallback statistics when no data is available."""

        bones_ratios = {}
        for u, v in self.bones:
            out_key = ";".join(sorted((u, v)))
            bones_ratios[out_key] = {
                "median_ratio": 1.0,
                "mad_ratio": self.MAD_ratio,
                "count": 0,
                "n_pairs": 0,
            }

        return {
            "reference_bone": list(self.ref_bone),
            "median_reference_length": 1.0,
            "bones_ratios": bones_ratios,
        }


class DynamicsBootstrapper:
    """Bootstrap dynamics parameters (process noise, measurement noise) from 3D soup."""

    def __init__(
            self,
            keypoint_names: List[str],
            bones: List[Tuple[str, str]],
            symmetry_pairs: Optional[List[Tuple[str, str]]] = None,
            fps: float = 30.0,
            max_displacement: float = 5.0,
            min_track_length: float = 15,
            reference_bone_length: float = 1.0,
            min_process_noise: float = 0.01,
            measurement_noise: float = 0.5
    ):
        self.keypoint_names = keypoint_names
        self.canon_map = create_canonical_map(keypoint_names, symmetry_pairs)

        # Config
        self.fps = fps
        self.max_displacement = max_displacement
        self.min_track_length = min_track_length
        self.ref_bone_length = reference_bone_length
        self.min_q = min_process_noise
        self.base_measurement_noise = measurement_noise

        G = nx.Graph()
        G.add_edges_from(bones)
        try:
            self.centroid = max(G.degree, key=lambda x: x[1])[0]
            self.graph_dist = nx.single_source_shortest_path_length(G, self.centroid)
        except (ValueError, IndexError):
            self.centroid = keypoint_names[0]
            self.graph_dist = {k: 1 for k in keypoint_names}

        self.debug_tracks = defaultdict(list)
        self.debug_velocities = defaultdict(list)

    def process(self, soup: PointSoup) -> Dict[str, Any]:
        """
        1. Link detections into tracks.
        2. Vectorized velocity/acceleration calculation.
        3. Populate debug info for plotting.
        """
        import trackpy as tp
        tp.quiet()

        df = soup.to_df()

        # Ensure tracking is performed
        df['particle'] = -1
        for name, group in df.groupby('keypoint'):
            linked = tp.link_df(group, search_range=self.max_displacement, pos_columns=['x', 'y', 'z'], memory=0)
            df.loc[linked.index, 'particle'] = linked['particle']

        # Clear debug storage for new run
        self.debug_velocities.clear()
        self.debug_tracks.clear()

        canon_stats = defaultdict(lambda: {"vel": [], "acc": []})

        # Group by keypoint and particle (tracklet)
        for (name, particle), track in df.groupby(['keypoint', 'particle']):
            if particle == -1 or len(track) < self.min_track_length:
                continue

            track = track.sort_values('frame')

            # Split into contiguous segments (no jumps in frame indices)
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

                cname = self.canon_map.get(name, name)
                canon_stats[cname]["vel"].extend(vel)
                canon_stats[cname]["acc"].extend(acc)

                self.debug_velocities[cname].extend(vel)

                if len(self.debug_tracks[cname]) < 200:
                    self.debug_tracks[cname].append(track)

        # Derive final KF parameters
        final_params = {}
        for name in self.keypoint_names:
            cname = self.canon_map.get(name, name)
            data = canon_stats.get(cname, {"vel": [], "acc": []})

            if len(data["vel"]) < 50:
                dist = self.graph_dist.get(cname, 2)
                q, w, src = max(self.min_q, 0.5 * dist), 0.5, "topology_prior"
            else:
                vel_med, vel_mad = robust_stats(data["vel"])
                acc_med, acc_mad = robust_stats(data["acc"])
                q = max(self.min_q, float(acc_med + 2.0 * acc_mad))
                jitter = vel_mad / self.ref_bone_length
                w = float(1.0 / (1.0 + (jitter * 30.0) ** 2))
                src = "data"

            final_params[name] = {
                "process_noise": q,
                "measurement_noise": self.base_measurement_noise,
                "association_weight": w,
                "source": src
            }
        return final_params


if __name__ == "__main__":
    import pickle
    from mokap.utils import fileio

    BASE_DIR = Path.home() / 'Desktop' / '3d_ant_data'
    PREFIX = '240905-1616'
    SESSION = 22

    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'

    soup_file = output_dir / f"soup_session{SESSION}.pkl"
    bone_stats_file = output_dir / "skeleton_stats.json"

    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(input_dir, symmetry=True)

    with open(soup_file, "rb") as f:
        soup = pickle.load(f)

    # Anatomy
    anat = AnatomyBootstrapper(
        keypoint_names=soup.keypoints,
        bones=bones,
        symmetry_pairs=symmetry,
        MAD_ratio=0.1,
        min_samples=10,
        max_bone_length=2.5,
        reference_bone=None,  # auto-detected
        min_tracklet_length=5,
        tracklet_search_range=1.5,
    )
    anat_res = anat.process(soup)
    ref_len = anat_res["median_reference_length"]

    # Dynamics
    dyn = DynamicsBootstrapper(
        keypoint_names=soup.keypoints,
        bones=bones,
        symmetry_pairs=symmetry,
        fps=100.0,
        max_displacement=1.5,
        min_track_length=5,
        reference_bone_length=ref_len,
        min_process_noise=0.01,
        measurement_noise=0.1
    )
    dyn_res = dyn.process(soup)

    # Save

    combined_result = {
        "anatomy": anat_res,
        "dynamics": dyn_res
    }
    bone_stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(bone_stats_file, "w") as f:
        json.dump(combined_result, f, indent=2)
    print(f"Saved bootstrap stats to {bone_stats_file}")

    ##

    # Visualisation

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Bootstrap results (reference length: {ref_len:.2f} mm)",
        fontsize=16
    )

    # Plot 1: Reference bone length distribution
    ax1 = fig.add_subplot(331)
    ax1.set_title("Reference bone length distribution")
    ref_bone_canon = ";".join(sorted([anat.canon_map[k] for k in anat.ref_bone]))

    if ref_bone_canon in anat.debug_histograms:
        lengths = anat.debug_histograms[ref_bone_canon]
        ax1.hist(lengths, bins=50, color="gray", alpha=0.7)
        ax1.axvline(ref_len, color="red", linestyle="--", label="Median")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "Ref bone data not in hist", ha='center')
    ax1.set_xlabel("Length (mm)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bone length ratios
    ax2 = fig.add_subplot(332)
    ax2.set_title("Bone length ratios (population)")
    colors = ["r", "g", "b", "orange", "purple", "cyan"]
    for i, (bone, lengths) in enumerate(list(anat.debug_histograms.items())[:6]):
        ratios = np.asarray(lengths) / ref_len
        ax2.hist(
            ratios,
            bins=50,
            alpha=0.3,
            density=True,
            label=bone,
            color=colors[i % len(colors)],
        )
    ax2.legend(fontsize="x-small")
    ax2.set_xlabel("Ratio to reference")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Intra-individual MAD distribution
    ax3 = fig.add_subplot(333)
    ax3.set_title("Intra-individual MAD per tracklet")
    for i, (bone, mads) in enumerate(list(anat.debug_intra_individual_stds.items())[:6]):
        if len(mads) < 3:
            continue
        ax3.hist(
            mads,
            bins=30,
            alpha=0.4,
            density=True,
            label=bone,
            color=colors[i % len(colors)],
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
            vels,
            bins=50,
            alpha=0.3,
            density=True,
            label=kp,
            color=colors[i % len(colors)],
        )
        ax4.axvline(med, color=colors[i % len(colors)], linestyle="--")
    ax4.set_xlabel("Speed (mm / frame)")
    ax4.legend(fontsize="x-small")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Table of anatomy stats (showing intra-individual MAD)
    ax5 = fig.add_subplot(335)
    ax5.axis("off")
    ax5.set_title("Bone statistics (intra-individual MAD)")

    table_data = []
    seen_canon = set()

    for bone_key, stats in sorted(anat_res["bones_ratios"].items()):
        parts = bone_key.split(";")
        canon_key = ";".join(sorted([anat.canon_map.get(p, p) for p in parts]))

        if canon_key in seen_canon:
            continue
        seen_canon.add(canon_key)

        table_data.append([
            canon_key[:20],
            f"{stats['median_ratio']:.3f}",
            f"{stats['mad_ratio']:.4f}",
            f"{stats.get('n_pairs', stats.get('n_tracklets', 0))}",
        ])
        if len(table_data) >= 10:
            break

    table = ax5.table(
        cellText=table_data,
        colLabels=["Bone", "Med. Ratio", "MAD Ratio", "Pairs"],
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

    # Sort by confidence (weight)
    sorted_items = sorted(
        dyn_res.items(),
        key=lambda x: x[1]["association_weight"],
        reverse=True,
    )

    for name, p in sorted_items:
        cname = dyn.canon_map.get(name, name)
        if cname in seen_canon:
            continue
        seen_canon.add(cname)

        src = " (prior)" if p["source"] != "data" else ""
        table_data.append([
            cname[:15] + src,
            f"{p['process_noise']:.3f}",
            f"{p['association_weight']:.2f}",
        ])
        if len(table_data) >= 12:
            break

    table = ax6.table(
        cellText=table_data,
        colLabels=["Keypoint", "Q (Noise)", "Weight"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.2)

    # Plot 5: Example 3D tracks
    ax8 = fig.add_subplot(338, projection="3d")
    preferred = ["thorax", "neck", "head"]
    chosen = next(
        (p for p in preferred if p in dyn.debug_tracks),
        next(iter(dyn.debug_tracks), None),
    )
    if chosen:
        tracks_df = pd.concat(dyn.debug_tracks[chosen])
        plot_tracks_3d(ax8, tracks_df, f"Tracks: {chosen}")
    else:
        ax8.text(0, 0, 0, "No long tracks")

    plt.tight_layout()
    plt.show()