import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import networkx as nx
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from scipy.stats import median_abs_deviation
from mokap.reconstruction.config import AnatomyConfig
from mokap.reconstruction.datatypes import Bone, AssembledSkeleton, SoupData
from mokap.reconstruction.utils import create_canonical_map

logger = logging.getLogger(__name__)


class StatsBootstrapper:
    """ Handles the loading, creation, and standardization of anatomical statistics """

    def __init__(self,
                 output_path: Union[str, Path],
                 bones_list: List[Tuple[str, str]],
                 config: AnatomyConfig,
                 symmetry_map: Optional[List[Tuple[str, str]]] = None,
                 prior_stats_path: Optional[Union[str, Path]] = None,
                 bootstrap_data: Optional[SoupData] = None
                 ):

        self.config = config

        self.bones_list: List[Bone] = [frozenset(b) for b in bones_list]
        if not self.bones_list:
            raise ValueError("Cannot initialize StatsBootstrapper with an empty bone list.")

        keypoints = {kp for bone in self.bones_list for kp in bone}
        keypoints = sorted(list(keypoints))

        # Build skeleton graph for degree calculations
        self._skeleton_graph = nx.Graph()
        self._skeleton_graph.add_edges_from([tuple(b) for b in self.bones_list])
        self._degrees = dict(self._skeleton_graph.degree())

        self.output_path = Path(output_path)
        self.prior_path = Path(prior_stats_path) if prior_stats_path else None

        self.bootstrap_data = bootstrap_data

        self.json_delimiter_regex = re.compile(r'[-;, ]')
        self.canonical_map = create_canonical_map(keypoints, symmetry_map)

    def _symmetrise_bone_names(self, bone: Bone) -> Bone:
        """ Normalizes a bone's keypoint names using the canonical map """

        kp1_orig, kp2_orig = tuple(bone)
        kp1_canon = self.canonical_map.get(kp1_orig, kp1_orig)
        kp2_canon = self.canonical_map.get(kp2_orig, kp2_orig)

        return frozenset((kp1_canon, kp2_canon))

    def get_initial_stats(self) -> Dict:
        """
        Main method to get stats. Tries 3 ways to do it:
        1. Load from a user-provided prior stats file
        2. Load from a previously generated output stats file
        3. Bootstrap from 3D data if provided
        """

        # Try loading a user-provided prior file
        if self.prior_path and self.prior_path.exists():
            logger.info(f"Loading stats from provided prior file: '{self.prior_path.name}'")
            return self._load_and_process_file(self.prior_path)

        # Check if the designated output file already exists
        if self.output_path.exists():
            logger.info(f"Loading pre-existing stats file: '{self.output_path.name}'")
            with open(self.output_path, 'r') as f:
                stats = json.load(f)
            return self._validate_and_normalize_stats(stats)

        # Try to bootstrap from data
        if self.bootstrap_data is not None and self.bootstrap_data.num_points > 0:
            logger.info("No stats file provided or found. Bootstrapping from 3D data...")
            stats = self._bootstrap_from_data()
            self._save_stats(stats)
            return stats

        raise ValueError(
            '[ERROR] Could not obtain stats. No prior file provided, no existing stats file found, and no bootstrap data given.')

    def _get_most_stable_bone(self, available_bones: set[Bone]) -> Bone:
        """Selects the most stable bone (sum of degrees of keypoints)."""

        if not available_bones:
            raise ValueError("Cannot select a stable bone from an empty set.")

        best_bone = None
        max_score = -1

        for bone in available_bones:
            kp1, kp2 = tuple(bone)
            if kp1 not in self._degrees or kp2 not in self._degrees:
                continue
            score = self._degrees[kp1] + self._degrees[kp2]
            if score > max_score:
                max_score = score
                best_bone = bone

        if best_bone is None:
            # Fallback to the first available bone if no degrees match (disjoint graph)
            return list(available_bones)[0]

        return best_bone

    def _parse_bone_name(self, bone_name: str) -> Bone:
        parts = [str(p).strip() for p in self.json_delimiter_regex.split(bone_name) if str(p).strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Could not parse bone name '{bone_name}'. Expected two keypoints separated by a delimiter.")
        return frozenset(parts)

    def _load_and_process_file(self, file_path: Path) -> Dict:
        """Loads a JSON or CSV file, normalises it, and ensures std dev exists."""

        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            if not {'bone', 'length'}.issubset(df.columns):
                raise ValueError("CSV file must contain 'bone' and 'length' columns.")

            bone_data = df.set_index('bone')['length'].to_dict()

        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            if 'bones_ratios' in json_data:
                print("[INFO] File is already in final stats format. Validating...")
                return self._validate_and_normalize_stats(json_data)
            bone_data = json_data

        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}. Please use .csv or .json.")

        stats = self._normalize_lengths(bone_data)
        self._save_stats(stats)

        return stats

    def _normalize_lengths(self, bones_data: Dict[str, float]) -> Dict:
        """Normalises a dictionary of bone lengths using a reference bone."""

        parsed_bones_data = {self._parse_bone_name(name): length for name, length in bones_data.items()}

        if not parsed_bones_data:
            raise ValueError("Provided bones data is empty.")

        ref_bone = self._get_most_stable_bone(set(parsed_bones_data.keys()))
        ref_bone_len = parsed_bones_data[ref_bone]

        if ref_bone_len <= 0:
            raise ValueError(f"Reference bone '{' - '.join(ref_bone)}' has a non-positive length ({ref_bone_len}).")

        bones_ratios = {}
        generic_mad_ratio = self.config.BOOTSTRAP_GENERIC_MAD_RATIO
        for name, length in parsed_bones_data.items():
            median_ratio = length / ref_bone_len
            bone_key = ';'.join(sorted(list(name)))  # we want to serialize with consistent sorting
            bones_ratios[bone_key] = {
                'median_ratio': float(median_ratio),
                'mad_ratio': float(median_ratio * generic_mad_ratio)  # default MAD
            }

        logger.info(f"Using data-driven reference bone: '{'-'.join(ref_bone)}' (length: {ref_bone_len:.2f})")
        return {
            'reference_bone': sorted(list(ref_bone)),
            'median_reference_length': float(ref_bone_len),
            'bones_ratios': bones_ratios
        }

    def _validate_and_normalize_stats(self, stats: Dict) -> Dict:
        """Checks a fully-formatted stats dict and adds missing deviation."""

        generic_mad_ratio = self.config.BOOTSTRAP_GENERIC_MAD_RATIO
        updated = False

        # Standardize keys first (in case they come from a file with mixed delimiters)
        standardized_ratios = {
            ';'.join(sorted(list(self._parse_bone_name(k)))): v
            for k, v in stats['bones_ratios'].items()
        }
        stats['bones_ratios'] = standardized_ratios

        # Check for missing MAD values
        for bone_key, bone_stats in stats['bones_ratios'].items():
            if 'mad_ratio' not in bone_stats or not np.isfinite(bone_stats['mad_ratio']) or bone_stats[
                'mad_ratio'] == 0:
                bone_stats['mad_ratio'] = float(bone_stats['median_ratio'] * generic_mad_ratio)
                updated = True

        if updated:
            logger.debug(
                f"Standardized bone names and/or added missing MAD values using generic {generic_mad_ratio * 100:.1f}% ratio.")
            self._save_stats(stats)

        return stats

    def _greedy_assembler(self, frame_soup: SoupData) -> List[Dict[str, np.ndarray]]:
        """
        Greedy assembler for bootstrapping stats. Ignores Orphan Rays (requires high confidence).
        """

        if frame_soup.num_points == 0:
            return []

        skeletons = []
        used_indices = set()

        # Pre-group indices by keypoint type
        kp_to_indices = defaultdict(list)
        for i in range(frame_soup.num_points):
            kp_to_indices[frame_soup.kp_types[i]].append(i)

        # Iterate through every point as a potential seed
        for seed_kp_type_idx in kp_to_indices:
            seed_kp_name = frame_soup.keypoint_names[seed_kp_type_idx]

            for seed_idx in kp_to_indices[seed_kp_type_idx]:
                if seed_idx in used_indices:
                    continue

                seed_pos = frame_soup.positions[seed_idx]

                # Start a skeleton dict mapping Name -> Position
                skeleton_kps = {seed_kp_name: seed_pos}

                # Greedily find the closest available keypoint of each other type
                for other_kp_type_idx, candidate_indices in kp_to_indices.items():
                    if other_kp_type_idx == seed_kp_type_idx:
                        continue

                    other_kp_name = frame_soup.keypoint_names[other_kp_type_idx]

                    # Distance check
                    candidates_pos = frame_soup.positions[candidate_indices]
                    distances = np.linalg.norm(candidates_pos - seed_pos, axis=1)

                    best_local_idx = np.argmin(distances)
                    min_dist = distances[best_local_idx]

                    if min_dist < self.config.BOOTSTRAP_MAX_BONE_LEN:
                        skeleton_kps[other_kp_name] = candidates_pos[best_local_idx]

                if len(skeleton_kps) >= 2:
                    skeletons.append({'keypoints': skeleton_kps})

                # Note: this is bootstrapping so we don't care about 'using' points.
                # We just mark the seed to prevent identical iterations.
                used_indices.add(seed_idx)

        return skeletons

    def _bootstrap_from_data(self) -> Dict:
        """Performs data-driven bootstrapping (using symmetry)."""

        canonical_bone_lengths = defaultdict(list)
        canonical_to_original_map = defaultdict(set)

        unique_frames = np.unique(self.bootstrap_data.frame_indices)

        # Limit sampling if dataset is huge
        MAX_BOOTSTRAP_FRAMES = 1000 # TODO: Make this configurable
        if len(unique_frames) > MAX_BOOTSTRAP_FRAMES:
            unique_frames = np.random.choice(unique_frames, MAX_BOOTSTRAP_FRAMES, replace=False)
            logger.info(f"Bootstrapping from a random subset of {len(unique_frames)} frames.")

        with alive_bar(title='Gathering bone measurements...', length=20, total=len(unique_frames),
                       force_tty=True) as bar:
            for frame_idx in unique_frames:

                frame_soup = self.bootstrap_data.get_frame_slice(frame_idx)

                fragments = self._greedy_assembler(frame_soup)

                for frag in fragments:
                    kps = frag['keypoints']
                    for original_bone in self.bones_list:
                        if original_bone.issubset(kps):
                            kp1, kp2 = tuple(original_bone)
                            length = np.linalg.norm(kps[kp1] - kps[kp2])

                            if length > 1e-3:
                                canonical_bone = self._symmetrise_bone_names(original_bone)
                                canonical_bone_lengths[canonical_bone].append(length)
                                canonical_to_original_map[canonical_bone].add(original_bone)
                bar()

        # Calculate stats for canonical bones
        valid_canonical_bones = {b for b, lengths in canonical_bone_lengths.items()
                                 if len(lengths) >= self.config.BOOTSTRAP_MIN_SAMPLES}

        median_lengths = {b: np.median(canonical_bone_lengths[b]) for b in valid_canonical_bones}

        if not median_lengths:
            raise ValueError("Bootstrap failed: Not enough valid bone samples found in the data.")

        # Determine reference bone based on graph stability
        canonical_graph = nx.Graph()
        canonical_degrees = defaultdict(int)

        for bone in self.bones_list:
            canonical_bone = self._symmetrise_bone_names(bone)
            if len(canonical_bone) == 2:
                canonical_graph.add_edge(*tuple(canonical_bone))

        for kp, deg in self._degrees.items():
            if self.canonical_map and kp in self.canonical_map:
                canonical_degrees[self.canonical_map[kp]] += deg
            else:
                canonical_degrees[kp] += deg

        def get_best_canonical_ref(canonical_bones: set[Bone]) -> Bone:
            best_bone, max_score = None, -1
            for bone in canonical_bones:
                kp1, kp2 = tuple(bone)
                score = canonical_degrees.get(kp1, 0) + canonical_degrees.get(kp2, 0)
                if score > max_score:
                    max_score = score
                    best_bone = bone
            return best_bone if best_bone else list(canonical_bones)[0]

        canonical_ref_bone = get_best_canonical_ref(set(median_lengths.keys()))
        ref_len = median_lengths[canonical_ref_bone]

        # Fallback to arbitrary original bone for final name
        reference_bone = list(canonical_to_original_map[canonical_ref_bone])[0]

        logger.info(
            f"Bootstrap determined stability-driven canonical reference '{' - '.join(canonical_ref_bone)}'. "
            f"Saving final reference as '{' - '.join(reference_bone)}' (median length: {ref_len:.2f})"
        )

        # Calculate canonical ratios
        canonical_stats = {}
        for bone, med_len in median_lengths.items():
            ratios_dist = np.array(canonical_bone_lengths[bone]) / ref_len
            canonical_stats[bone] = {
                'median_ratio': float(med_len / ref_len),
                'mad_ratio': float(median_abs_deviation(ratios_dist))
            }

        # Map the stats back to the original bone names
        final_bones_ratios = {}
        for canonical_bone, stats in canonical_stats.items():
            original_bones = canonical_to_original_map[canonical_bone]
            for original_bone in original_bones:
                bone_key = ';'.join(sorted(list(original_bone)))
                final_bones_ratios[bone_key] = stats

        final_stats = {
            'reference_bone': sorted(list(reference_bone)),
            'median_reference_length': float(ref_len),
            'bones_ratios': final_bones_ratios
        }
        return self._validate_and_normalize_stats(final_stats)

    def _save_stats(self, stats: Dict):
        """Saves the generated stats to the specified JSON file."""
        print(f"[INFO] Saving processed stats to '{self.output_path}'...")

        with open(self.output_path, 'w') as f:
            json.dump(stats, f, indent=2)


class AnatomyLearner:
    """
    Online learner that refines bone lengths based on high-confidence tracked skeletons.
    """

    def __init__(self, initial_stats: dict, config: AnatomyConfig):

        self.reference_bone: Bone = frozenset(initial_stats['reference_bone'])
        self.config = config

        self.ref_lengths = []
        self.bones_ratios: Dict[Bone, List[float]] = defaultdict(list)

        self.current_stats = initial_stats
        self.is_stale = True
        self.measurements_count = 0

    def add_sample(self, skeleton: AssembledSkeleton):
        """Adds a new high-quality skeleton to the measurement pool."""

        if skeleton.score < self.config.LEARNER_MIN_SCORE_FOR_LEARNING:
            return

        if not self.reference_bone.issubset(skeleton.keypoints):
            return

        kp1_name, kp2_name = tuple(self.reference_bone)
        p1, p2 = skeleton.keypoints[kp1_name], skeleton.keypoints[kp2_name]
        ref_len = np.linalg.norm(p1 - p2)

        # small sanity check
        if not (self.config.LEARNER_MIN_REF_BONE_LEN < ref_len < self.config.LEARNER_MAX_REF_BONE_LEN):
            return

        self.ref_lengths.append(ref_len)

        for bone_str in self.current_stats['bones_ratios']:
            kp1, kp2 = bone_str.split(';')
            if kp1 in skeleton.keypoints and kp2 in skeleton.keypoints:
                bone_len = np.linalg.norm(skeleton.keypoints[kp1] - skeleton.keypoints[kp2])
                self.bones_ratios[frozenset((kp1, kp2))].append(bone_len / ref_len)

        self.measurements_count += 1
        if self.measurements_count >= self.config.LEARNER_MIN_SAMPLES_FOR_UPDATE:
            self.is_stale = True

    def get_stats(self) -> dict:
        """Returns the current best stats, re-computing them if enough new data has arrived."""

        if not self.is_stale:
            return self.current_stats

        new_bones_ratios = {
            ';'.join(sorted(list(b))): {'median_ratio': float(np.median(r)),
                                        'mad_ratio': float(median_abs_deviation(r))}
            for b, r in self.bones_ratios.items() if len(r) > self.config.LEARNER_MIN_SAMPLES_FOR_UPDATE
        }

        if not new_bones_ratios:  # not enough data yet so we stick to old stats
            self.is_stale = False
            return self.current_stats

        self.current_stats['bones_ratios'].update(new_bones_ratios)

        if self.ref_lengths:
            self.current_stats['median_reference_length'] = float(np.median(self.ref_lengths))

        self.is_stale = False
        self.measurements_count = 0

        logger.debug(f"Refinement complete. New reference length: {self.current_stats['median_reference_length']:.2f}")
        return self.current_stats
