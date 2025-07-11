import json
import pickle
import re
import logging
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional, Dict, List, FrozenSet
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from alive_progress import alive_bar
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import median_abs_deviation
from scipy.spatial import cKDTree
from mokap.utils import fileio, common_prefix_suffix
from mokap.utils.geometry.fitting import find_rigid_transform
from mokap.reconstruction.reconstruction import SoupPoint


logger = logging.getLogger(__name__)


# TODO: Profile the two MWIS solvers a bit more
#
# from pyscipopt import Model
# import cProfile
# import pstats
#
# def solve_mwis_SCIP(graph: nx.Graph) -> list[int]:
#     """ Solves the MWIS problem using the SCIP ILP solver """
#
#     model = Model("mwis")
#     model.hideOutput()
#
#     # Create a binary variable for each node in the graph
#     # The variable will be 1 if the node is in the solution, 0 otherwise
#     nodes = list(graph.nodes())
#     variables = {node: model.addVar(vtype="B", name=f"x_{node}") for node in nodes}
#
#     # Set the objective function: Maximize the sum of the weights of the selected nodes
#     objective_terms = [graph.nodes[node]['weight'] * variables[node] for node in nodes]
#     model.setObjective(sum(objective_terms), "maximize")
#
#     # Add constraints: For every edge (u, v) in the conflict graph, the two nodes
#     # cannot be chosen together. This is the "independent set" constraint
#     # x_u + x_v <= 1
#     for u, v in graph.edges():
#         model.addCons(variables[u] + variables[v] <= 1)
#
#     # Solve the model
#     model.optimize()
#
#     # Extract the solution
#     solution_nodes = []
#     if model.getStatus() == "optimal":
#         for node in nodes:
#             # Check if the variable is close to 1 in the solution
#             if model.getVal(variables[node]) > 0.99:
#                 solution_nodes.append(node)
#
#     return solution_nodes


def solve_mwis_networkx(graph: nx.Graph) -> list[int]:
    """ The basic MWIS solver using networkx """

    # TODO: move this to utils bc it is also used by the reconstructor

    if graph.number_of_nodes() == 0:
        return []

    # The MWC of the complement is equivalent to MWIS of the original
    complement_graph = nx.complement(graph)

    # Copy weights to the complement graph
    node_weights = nx.get_node_attributes(graph, 'weight')
    nx.set_node_attributes(complement_graph, node_weights, name='weight')

    winner_indices, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')
    return winner_indices

# TODO: move this to utils
def create_canonical_map(keypoint_names: List[str], symmetry_map: Optional[List[Tuple[str, str]]]) -> Dict[str, str]:
    """
    Creates a map from each keypoint name to a generalized "canonical type"
    This is used to assign per-type Kalman Filter parameters:
    {'leg_f_L1': 'legf1', 'leg_f_R2': 'legf2', 'thorax': 'thorax'}

    Args:
        keypoint_names: The full list of keypoint names
        symmetry_map: (Optional) a list of tuples, where each tuple is a symmetric pair of names

    Returns:
        A dictionary mapping each keypoint name to its canonical type string
    """
    canonical_map = {}
    names_delimiter_regex = re.compile(r'[-_. ]')

    # Process symmetric pairs first
    if symmetry_map:
        for name1, name2 in symmetry_map:
            prefix, suffix = common_prefix_suffix(name1, name2)

            # The part of the string that is different is the side identifier
            side1 = name1[len(prefix):len(name1) - len(suffix)]

            # The canonical name is the original name without the side identifier and cleaned up
            canonical_name = name1.replace(side1, '')
            canonical_name = names_delimiter_regex.sub('', canonical_name).lower()

            canonical_map[name1] = canonical_name
            canonical_map[name2] = canonical_name

    # any remaining non-symmetric keypoints
    for kp_name in keypoint_names:
        if kp_name not in canonical_map:
            # The canonical name is just the name itself, cleaned up
            canonical_name = names_delimiter_regex.sub('', kp_name).lower()
            canonical_map[kp_name] = canonical_name

    logger.debug("Generated Canonical Keypoint Map for Smoother:")
    unique_types = sorted(list(set(canonical_map.values())))
    logger.debug(f"  -> Found types: {unique_types}")
    example_kp = keypoint_names[2]
    logger.debug(f"  -> Example: '{example_kp}' maps to '{canonical_map.get(example_kp)}'")

    return canonical_map


# Type alias for clarity and type safety
Bone = FrozenSet[str]


@dataclass
class AssembledSkeleton:
    """ Represents a final assembled skeleton for a frame """
    keypoints: Dict[str, np.ndarray]
    score: float
    scale: float
    point_indices: Dict[str, int] = field(default_factory=dict)
    track_id: int = -1

    def to_dict(self) -> dict:
        return {
            'keypoints': self.keypoints,
            'score': self.score,
            'scale': self.scale,
            'point_indices': self.point_indices,
            'track_id': self.track_id
        }


@dataclass
class CandidateSkeleton:
    """ Assembler's internal representation for a potential skeleton during the assembly process """
    nodes: FrozenSet[Tuple[str, int]]
    scale: float
    competition_score: float
    anatomical_score: float
    constituent_indices: Optional[FrozenSet[int]] = None  # to track original fragments


CONFIG = {

    # TODO: okay this dict is getting ridiculously big lol

    # --- Stats Bootstrapper parameters ---
    "BOOTSTRAP_GENERIC_MAD_RATIO": 0.10,
    # Generic MAD ratio to apply when bootstrapping from a simple prior or if data-driven MAD is zero
    "BOOTSTRAP_MIN_SAMPLES": 20,  # Min samples needed to calculate data-driven stats for a bone
    "BOOTSTRAP_MAX_BONE_LEN": 4.0,  # Max bone length for the simple greedy assembler used in bootstrapping

    # --- Anatomy Learner parameters ---
    "LEARNER_MIN_SAMPLES_FOR_UPDATE": 30,
    # How many new high-quality skeleton measurements before re-calculating anatomy stats
    "LEARNER_MIN_SCORE_FOR_LEARNING": 5.0,  # Minimum score of a tracked skeleton to be used for learning
    "LEARNER_MIN_REF_BONE_LEN": 1.0,  # Min plausible length of reference bone for learning
    "LEARNER_MAX_REF_BONE_LEN": 15.0,  # Max plausible length of reference bone for learning

    # --- Assembler parameters ---
    "ASSEMBLER_MIN_KPS_FOR_SKELETON": 3,  # Min keypoints to be considered a valid skeleton fragment
    "ASSEMBLER_MIN_CENTRAL_ANCHORS": 2,  # Min keypoints to be considered anchors for seeding new skeletons
    "ASSEMBLER_BONE_SCORE_MAD_THRESH": 5.0,
    # How far a bone's length can deviate from expected (in MADs) before its score is zero
    "ASSEMBLER_BONE_SCORE_MAD_EPSILON": 0.05,  # Small constant added to MAD for numerical stability
    "ASSEMBLER_MIN_SANE_SCALE": 0.7,  # Min plausible scale estimate for a skeleton fragment
    "ASSEMBLER_MAX_SANE_SCALE": 1.5,  # Max plausible scale estimate for a skeleton fragment
    "ASSEMBLER_SCORE_DEBT_TOLERANCE": 10.0,  # How much of a score hit is it possible to take to add one more part

    "ASSEMBLER_MERGE_SCALE_TOLERANCE": 0.075,
    "ASSEMBLER_MERGE_LINKING_BONE_THRESHOLD": 90.0,
    "ASSEMBLER_MIN_BONE_SCORE_FOR_FRAGMENT": 70.0,
    "ASSEMBLER_HIGH_QUALITY_THRESHOLD": 90.0,
    "ASSEMBLER_QUALITY_BONUS_FACTOR": 1.5,  # 50% boost

    # --- Conflict solver parameters ---
    "CONFLICT_SOLVER_BROAD_RADIUS": 3.0,  # Skeletons with centroids further than this are assumed not to conflict
    "CONFLICT_SOLVER_SHARED_POINTS_TOLERANCE": 1,  # Max number of shared points before declaring a conflict
    "CONFLICT_SOLVER_PROXIMITY_RADIUS": 0.25,  # Max distance to consider two corresponding keypoints 'the same'
    "CONFLICT_SOLVER_JACCARD_THRESHOLD": 0.85,  # Jaccard proximity threshold to consider two skeletons 'clones'

    # --- Tracker parameters ---
    "TRACKER_MAX_TRACKELT_AGE": 15,  # How many frames a tracklet can coast without an update before being deleted
    "TRACKER_UNCERTAINTY_THRESHOLD": 100,
    # This is for variance of the position, so 100 is equivalent to a std dev of 10 units
    "TRACKER_MIN_KPS_FOR_INFERENCE": 3,  # Min shared KPs needed to infer a missing central keypoint via alignment
    "TRACKER_SCALE_LEARNING_RATE": 0.25,  # Learning rate for the tracklet's adapting scale estimate
    "TRACKER_ASSOCIATION_RADIUS": 1.0,  # Max distance between a tracklet's prediction and a candidate for association
    "TRACKER_ASSOCIATION_MIN_KPS": 3,  # Min shared keypoints to associate a tracklet with a candidate
    "TRACKER_CONTINUITY_BONUS": 500.0,  # Large bonus to a candidate's score if it matches an existing tracklet
    "TRACKER_ANATOMICAL_SCORE_ALPHA": 0.15,
    # Smoothing factor for the tracklet's score. 0 = no update, 1 = new value only
    "TRACKER_INFERRED_HEALTH_PENALTY": 0.05,  # Health reduction for an update based on an inferred point
    "TRACKER_HEALTH_DECAY_RATE": 0.98,  # Multiplicative decay of health per frame without an update

    # --- Cost function weights (for final assignment) ---
    "COST_POSE_DISTANCE_WEIGHT": 0.9,  # Weights for the Hungarian algorithm cost matrix. Lower cost = better
    "COST_SKELETON_SCORE_WEIGHT": -0.1,  # A higher intrinsic skeleton score should lower the cost (hence negative)

    # --- Kalman Filter parameters ---
    "KF_PROCESS_NOISE_POS": 0.1,  # Process noise for position (assumes random acceleration). Higher = less smooth
    "KF_PROCESS_NOISE_SCALE": 0.01,  # Process noise for scale
    "KF_MEASUREMENT_NOISE_POS": 5.0,  # Measurement noise for position (reflects 3D reconstruction uncertainty)
    "KF_MEASUREMENT_NOISE_SCALE": 0.25,  # Measurement noise for scale
    "KF_INIT_COV_VEL": 1.0,  # Initial covariance for velocity
    "KF_INIT_COV_SCALE": 1.0,  # Initial covariance for scale
    "KF_INFERENCE_UNCERTAINTY_FACTOR": 2.0,
    # Multiplier for measurement noise when a keypoint position is inferred, not measured
}


class StatsBootstrapper:
    """
    Handles the loading, creation, and standardization of anatomical statistics
    """

    def __init__(self,
            output_path:        Union[str, Path],
            bones_list:         List[Tuple[str, str]],
            symmetry_map:       Optional[List[Tuple[str, str]]] = None,
            prior_stats_path:   Optional[Union[str, Path]] = None,
            bootstrap_data:     Optional[Dict[int, List[SoupPoint]]] = None
         ):

        self.bones_list: List[Bone] = [frozenset(b) for b in bones_list]
        if not self.bones_list:
            raise ValueError("Cannot initialize StatsBootstrapper with an empty bone list.")

        keypoints = {kp for bone in self.bones_list for kp in bone}
        keypoints = sorted(list(keypoints))

        # Build skeleton graph for degree calculations
        # TODO: This is also done in the skeleton assembler, that's a bit redundant
        self._skeleton_graph = nx.Graph()
        self._skeleton_graph.add_edges_from([tuple(b) for b in self.bones_list])
        self._degrees = dict(self._skeleton_graph.degree())

        self.output_path = Path(output_path)
        self.prior_path = Path(prior_stats_path) if prior_stats_path else None
        self.bootstrap_data = bootstrap_data

        self.json_delimiter_regex = re.compile(r'[-;, ]')

        # Create the canonical map
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

        # try loading a user-provided prior file
        if self.prior_path and self.prior_path.exists():
            logger.info(f"Loading stats from provided prior file: '{self.prior_path.name}'")

            return self._load_and_process_file(self.prior_path)

        # check if the designated output file already exists
        if self.output_path.exists():
            logger.info(f"Loading pre-existing stats file: '{self.output_path.name}'")

            with open(self.output_path, 'r') as f:
                stats = json.load(f)

            return self._validate_and_normalize_stats(stats)  # always validate pre-existing files

        # try to bootstrap from data
        if self.bootstrap_data:
            logger.info("No stats file provided or found. Bootstrapping from 3D data...")

            stats = self._bootstrap_from_data()
            self._save_stats(stats)

            return stats

        raise ValueError(
            '[ERROR] Could not obtain stats. No prior file provided, no existing stats file found, and no bootstrap data given.')

    def _get_most_stable_bone(self, available_bones: set[Bone]) -> Bone:
        """
        Selects the most stable bone as defined by the sum of the degrees of the two keypoints
        forming it. This favors central well-connected bones.
        """

        if not available_bones:
            raise ValueError("Cannot select a stable bone from an empty set.")

        best_bone = None
        max_score = -1

        for bone in available_bones:
            kp1, kp2 = tuple(bone)
            if kp1 not in self._degrees or kp2 not in self._degrees:
                continue

            # Score is the sum of the degrees of the two keypoints
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
        """ Loads a JSON or CSV file, normalises it, and ensures std dev exists """

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
        """ Normalises a dictionary of bone lengths using a reference bone """

        parsed_bones_data = {self._parse_bone_name(name): length for name, length in bones_data.items()}

        if not parsed_bones_data:
            raise ValueError("Provided bones data is empty.")

        ref_bone = self._get_most_stable_bone(set(parsed_bones_data.keys()))
        ref_bone_len = parsed_bones_data[ref_bone]

        if ref_bone_len <= 0:
            raise ValueError(f"Reference bone '{' - '.join(ref_bone)}' has a non-positive length ({ref_bone_len}).")

        bones_ratios = {}
        generic_mad_ratio = CONFIG["BOOTSTRAP_GENERIC_MAD_RATIO"]
        for name, length in parsed_bones_data.items():
            median_ratio = length / ref_bone_len
            bone_key = ';'.join(sorted(list(name)))  # we want to serialize with consistent sorting
            bones_ratios[bone_key] = {
                'median_ratio': float(median_ratio),
                'mad_ratio': float(median_ratio * generic_mad_ratio)  # Default MAD
            }

        logger.info(f"Using data-driven reference bone: '{'-'.join(ref_bone)}' (length: {ref_bone_len:.2f})")
        return {
            'reference_bone': sorted(list(ref_bone)),
            'median_reference_length': float(ref_bone_len),
            'bones_ratios': bones_ratios
        }

    def _validate_and_normalize_stats(self, stats: Dict) -> Dict:
        """ Checks a fully-formatted stats dict and adds missing deviation """

        generic_mad_ratio = CONFIG["BOOTSTRAP_GENERIC_MAD_RATIO"]
        updated = False

        # Standardize keys first, in case they come from a file with mixed delimiters
        standardized_ratios = {
            ';'.join(sorted(list(self._parse_bone_name(k)))): v
            for k, v in stats['bones_ratios'].items()
        }
        stats['bones_ratios'] = standardized_ratios

        # Now check for missing MAD values
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

    def _greedy_assembler(self, frame_points: List[SoupPoint]) -> List[dict]:
        """ Simplified assembler for bootstrapping stats. Builds skeletons greedily from the central keypoint """

        if not frame_points:
            return []

        points_by_kp = defaultdict(list)
        for p in frame_points:
            points_by_kp[p.keypoint_type].append(p)

        skeletons = []
        used_point_ids = set()  # (kp_name, idx) of used points

        # Iterate through every point of every type as a potential seed
        for seed_kp, seed_points_list in points_by_kp.items():

            for seed_point in seed_points_list:
                seed_id = (seed_point.keypoint_type, seed_point.idx)

                if seed_id in used_point_ids:
                    continue

                center_pos = seed_point.position
                skeleton = {'keypoints': {seed_kp: center_pos}}

                # Greedily find the closest available keypoint of each other type
                for kp_name, points_list in points_by_kp.items():

                    if kp_name == seed_kp or not points_list:
                        continue

                    positions = np.array([p.position for p in points_list])
                    distances = np.linalg.norm(positions - center_pos, axis=1)
                    best_idx = np.argmin(distances)

                    if distances[best_idx] < CONFIG["BOOTSTRAP_MAX_BONE_LEN"]:
                        skeleton['keypoints'][kp_name] = points_list[best_idx].position

                if len(skeleton['keypoints']) >= 2:
                    skeletons.append(skeleton)

                # Mark all points used in this new skeleton so they aren't used as seeds again
                for kp, point_pos in skeleton['keypoints'].items():
                    # Find the original SoupPoint to get its ID
                    for p in points_by_kp[kp]:
                        if np.array_equal(p.position, point_pos):
                            used_point_ids.add((p.keypoint_type, p.idx))
                            break
        return skeletons

    def _bootstrap_from_data(self) -> Dict:
        """
        Performs data-driven bootstrapping, using symmetry to create robust stats
        """

        # Gather data under canonical bone names
        canonical_bone_lengths = defaultdict(list)
        canonical_to_original_map = defaultdict(set)

        with alive_bar(title='Gathering bone measurements...', length=20, total=len(self.bootstrap_data), force_tty=True) as bar:
            for frame_data in self.bootstrap_data.values():
                fragments = self._greedy_assembler(frame_data)
                for frag in fragments:
                    for original_bone in self.bones_list:
                        if original_bone.issubset(frag['keypoints']):
                            kp1, kp2 = tuple(original_bone)
                            length = np.linalg.norm(frag['keypoints'][kp1] - frag['keypoints'][kp2])
                            if length > 1e-3:
                                canonical_bone = self._symmetrise_bone_names(original_bone)
                                canonical_bone_lengths[canonical_bone].append(length)
                                # Store the mapping
                                canonical_to_original_map[canonical_bone].add(original_bone)
                bar()

        # Calculate stats for canonical bones
        valid_canonical_bones = {b for b, lengths in canonical_bone_lengths.items() if
                                 len(lengths) >= CONFIG["BOOTSTRAP_MIN_SAMPLES"]}
        median_lengths = {b: np.median(canonical_bone_lengths[b]) for b in valid_canonical_bones}

        if not median_lengths:
            raise ValueError("Bootstrap failed: Not enough valid bone samples found in the data.")

        # We use stability-based selection for the canonical reference
        canonical_graph = nx.Graph()
        canonical_degrees = defaultdict(int)
        for bone in self.bones_list:
            canonical_bone = self._symmetrise_bone_names(bone)
            # Only add edges if the canonical bone is valid (has 2 parts)
            if len(canonical_bone) == 2:
                canonical_graph.add_edge(*tuple(canonical_bone))

        # We need to map original degrees to canonical degrees
        # (a canonical kp's degree is the sum of degrees of its original constituent parts)
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

        # Pick any original bone that corresponds to the canonical reference
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

        # Map the canonical stats back to original bone names
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
        """ Saves the generated stats to the specified JSON file """
        print(f"[INFO] Saving processed stats to '{self.output_path}'...")

        with open(self.output_path, 'w') as f:
            json.dump(stats, f, indent=2)


class AnatomyLearner:
    def __init__(self, initial_stats: dict):

        self.reference_bone: Bone = frozenset(initial_stats['reference_bone'])

        # Store raw measurements to re-calculate stats on the fly
        self.ref_lengths = []
        self.bones_ratios: Dict[Bone, List[float]] = defaultdict(list)

        # Keep track of the current stats to avoid re-computing every frame
        self.current_stats = initial_stats
        self.is_stale = True  # Flag to indicate that stats need re-computing

        self.measurements_count = 0

        # Quality filter
        self.min_samples_for_update = CONFIG["LEARNER_MIN_SAMPLES_FOR_UPDATE"]
        self.min_score_for_learning = CONFIG["LEARNER_MIN_SCORE_FOR_LEARNING"]
        self.min_ref_len = CONFIG["LEARNER_MIN_REF_BONE_LEN"]
        self.max_ref_len = CONFIG["LEARNER_MAX_REF_BONE_LEN"]

    def add_sample(self, skeleton: AssembledSkeleton):
        """ Adds a new high-quality skeleton to the measurement pool """

        # Quality gate
        if skeleton.score < self.min_score_for_learning:
            return

        if not self.reference_bone.issubset(skeleton.keypoints):
            return

        # Add measurements
        kp1_name, kp2_name = tuple(self.reference_bone)
        p1, p2 = skeleton.keypoints[kp1_name], skeleton.keypoints[kp2_name]
        ref_len = np.linalg.norm(p1 - p2)

        # sanity check
        if not (self.min_ref_len < ref_len < self.max_ref_len):
            return

        self.ref_lengths.append(ref_len)

        for bone_str in self.current_stats['bones_ratios']:
            kp1, kp2 = bone_str.split(';')
            if kp1 in skeleton.keypoints and kp2 in skeleton.keypoints:
                bone_len = np.linalg.norm(skeleton.keypoints[kp1] - skeleton.keypoints[kp2])
                self.bones_ratios[frozenset((kp1, kp2))].append(bone_len / ref_len)

        self.measurements_count += 1
        if self.measurements_count >= self.min_samples_for_update:
            self.is_stale = True

    def get_stats(self) -> dict:
        """ Returns the current best stats, re-computing them if enough new data has arrived """

        if not self.is_stale:
            return self.current_stats

        # Recompute stats from the full pool of measurements
        new_bones_ratios = {
            ';'.join(sorted(list(b))): {'median_ratio': float(np.median(r)),
                                        'mad_ratio': float(median_abs_deviation(r))}
            for b, r in self.bones_ratios.items() if len(r) > 50
            # Only well-observed bones. TODO: maybe this could be added to config dict
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


class Tracklet:
    """
    Represents a single object (a skeleton) in a tracklet
    Manages state estimation (position, velocity, scale) using a Kalman Filter

    It can predict its future state and be updated with new measurements
    It also includes logic to infer the position of its central keypoint if it's occluded
    """

    def __init__(self,
            track_id:        int,
            initial_skeleton:   AssembledSkeleton,
            frame_idx:          int,
            central_kp:         str
        ):

        self.track_id = track_id
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Tracklet health and score metrics
        self.health = 1.0  # Running confidence metric (1.0 = high confidence)
        self.anatomical_integrity = initial_skeleton.score  # Exponential Moving Average of the skeleton score

        self.anatomical_score_alpha = CONFIG['TRACKER_ANATOMICAL_SCORE_ALPHA']
        self.inferred_health_penalty = CONFIG['TRACKER_INFERRED_HEALTH_PENALTY']
        self.health_decay_rate = CONFIG['TRACKER_HEALTH_DECAY_RATE']

        self.skeleton: AssembledSkeleton = initial_skeleton
        self.central_kp = central_kp

        # KF inference parameters
        self.inference_uncertainty_factor = CONFIG['KF_INFERENCE_UNCERTAINTY_FACTOR']
        self.min_kps_for_inference = CONFIG['TRACKER_MIN_KPS_FOR_INFERENCE']

        # Kalman Filter for 3D position (x, y, z), 3D velocity (vx, vy, vz), and scale (s)
        # State vector (dim_x = 7): [x, y, z, vx, vy, vz, s]
        # Measurement (dim_z = 4): [x, y, z, s]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        dt = 1.0  # Time step

        self.kf.F = np.array([[1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # Constant velocity and scale model

        # Measurement function
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # We measure position and scale

        # Process noise
        pos_vel_q = Q_discrete_white_noise(dim=2, dt=dt, var=CONFIG["KF_PROCESS_NOISE_POS"], block_size=3)
        scale_q = np.array([[CONFIG["KF_PROCESS_NOISE_SCALE"]]])
        self.kf.Q = block_diag(pos_vel_q, scale_q)

        # Measurement noise
        self.kf.R = np.diag([CONFIG["KF_MEASUREMENT_NOISE_POS"], CONFIG["KF_MEASUREMENT_NOISE_POS"],
                             CONFIG["KF_MEASUREMENT_NOISE_POS"], CONFIG["KF_MEASUREMENT_NOISE_SCALE"]])

        # Initial state covariance
        self.kf.P[3:6, 3:6] *= 1.0
        self.kf.P[6, 6] = 1.0

        # Initial state
        self.kf.x[:3] = self.skeleton.keypoints[self.central_kp].reshape(3, 1)
        self.kf.x[6] = self.skeleton.scale

    def predict(self, current_frame_idx: int):
        """ Predicts the state of the tracklet for the current frame """

        # The number of prediction steps is how many frames we've coasted
        steps_to_predict = current_frame_idx - self.last_update_frame

        for _ in range(steps_to_predict):
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
            # Decay health for each frame without a measurement update
            self.health *= self.health_decay_rate

    def update(self, skeleton: AssembledSkeleton, frame_idx: int):
        """
        Updates the tracklet's state with a new skeleton measurement

        If the central keypoint is missing, it attempts to infer its position using
        rigid alignment (Kabsch algorithm) before updating the Kalman Filter
        """

        update_skeleton = skeleton
        inferred = False

        # if the primary keypoint is missing, try to infer it
        if self.central_kp not in skeleton.keypoints:
            inferred_skeleton = self._infer_missing_central_kp(skeleton)
            if inferred_skeleton:
                # Use the completed skeleton for the update
                update_skeleton = inferred_skeleton
                inferred = True
            else:
                # if inference fails (too few points) we can't update the KF
                # We still update the tracklet's skeleton to the partial view, reset its age,
                # but do *not* update health or score_ema, as it was not a full KF update
                self.skeleton = skeleton
                self.score = skeleton.score
                self.time_since_update = 0
                self.last_update_frame = frame_idx
                return

        self.skeleton = update_skeleton

        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Create the measurement vector for the Kalman Filter
        measurement = np.array([*update_skeleton.keypoints[self.central_kp], update_skeleton.scale])

        # If the position was inferred, we are less certain about it...
        # so we tell this to the KF by temporarily increasing its measurement noise
        if inferred:
            original_R = self.kf.R.copy()
            self.kf.R[:3, :3] *= self.inference_uncertainty_factor  # only increase position uncertainty
            self.kf.update(measurement)
            self.kf.R = original_R  # and restore for the next update
        else:
            self.kf.update(measurement)

        # Update health and score metrics after a successful KF update

        # Update the smoothed anatomical score
        self.anatomical_integrity = self.anatomical_score_alpha * skeleton.score + (
                    1 - self.anatomical_score_alpha) * self.anatomical_integrity

        # Update the tracklet's health
        if inferred:
            # The update was based on an inferred point, so it's a bit less certain...
            # so restore health, but with a lil penalty
            self.health = 1.0 - self.inferred_health_penalty
        else:
            # The update was based on a direct measurement: this is a high-confidence event
            # so reset health to its maximum value
            self.health = 1.0

    def _infer_missing_central_kp(self, fragment: AssembledSkeleton) -> Optional[AssembledSkeleton]:
        """
        Infers the position of a missing central keypoint by aligning the last known
        full pose to the currently visible fragment using a rigid transform
        """

        prev_kps, curr_kps = self.skeleton.keypoints, fragment.keypoints
        common_names = list(prev_kps.keys() & curr_kps.keys())

        if len(common_names) < self.min_kps_for_inference:
            return None  # not enough information to infer a stable alignment

        # Point clouds for alignment: current fragment (A) to previous pose (B)
        points_A = np.array([prev_kps[name] for name in common_names])
        points_B = np.array([curr_kps[name] for name in common_names])

        # Use Kabsch algorithm to find R, t such that: B ~ R @ A + t
        R_mat, t_vec = find_rigid_transform(points_A, points_B)

        # Apply the found transform to the last known position of the central keypoint
        inferred_pos = np.array(R_mat) @ prev_kps[self.central_kp] + np.array(t_vec)

        # Return a new object to avoid modifying the input fragment
        completed_skeleton = AssembledSkeleton(
            keypoints=fragment.keypoints.copy(),
            score=fragment.score,
            scale=fragment.scale,
            point_indices=fragment.point_indices
        )
        completed_skeleton.keypoints[self.central_kp] = inferred_pos
        return completed_skeleton

    @property
    def predicted_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns a full skeleton pose based on the Kalman Filter's predicted position of the central keypoint
        """
        if self.central_kp not in self.skeleton.keypoints:
            return None

        translation = self.predicted_position - self.skeleton.keypoints[self.central_kp]

        return {kp_name: pos + translation for kp_name, pos in self.skeleton.keypoints.items()}

    @property
    def predicted_position(self) -> np.ndarray:
        """
        Returns the predicted 3D position from the KF state
        """
        return self.kf.x[:3].flatten()

    @property
    def predicted_scale(self) -> float:
        """
        Returns the predicted scale from the KF state
        """
        return self.kf.x[6, 0]

    @property
    def uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Returns the uncertainty (variance) of the state variables from the
        Kalman Filter's covariance matrix P
        """
        diag_P = self.kf.P.diagonal()
        return {
            'position': diag_P[0:3],
            'velocity': diag_P[3:6],
            'scale': diag_P[6]
        }


class SkeletonAssembler:
    """
    Assembles skeletons from a 'soup' of 3D reconstructed points for a single frame

    This is a two-stage process:
    1.  generate_candidates(): Generates all plausible skeleton candidates using a
        holistic, score-based growth algorithm. This creates redundancy on purpose
    2.  solve_conflicts(): Resolves the redundancy by building a conflict graph and
        finding the Maximum Weight Independent Set (MWIS), yielding the optimal set
        of non-overlapping skeletons
    """

    def __init__(self, bones_list: list, bone_stats: dict):

        self.bones_list: List[Bone] = [frozenset(bone) for bone in bones_list]
        self.update_bone_stats(bone_stats)

        # Build the graph to determine topology
        skeleton_graph = nx.Graph([tuple(b) for b in self.bones_list])
        if not skeleton_graph.nodes:
            raise ValueError("Cannot assemble skeletons from an empty bone list.")

        self._skeleton_graph = skeleton_graph

        # Get keypoint degrees to determine topology
        degrees = dict(self._skeleton_graph.degree())
        self.leaf_nodes = {node for node, degree in degrees.items() if degree == 1}

        non_leaf_nodes = set(degrees.keys()) - self.leaf_nodes
        self.central_anchors = set()
        self.secondary_anchors = set()

        if non_leaf_nodes:
            sorted_anchors = sorted(list(non_leaf_nodes), key=lambda node: degrees[node], reverse=True)
            min_central_count = CONFIG["ASSEMBLER_MIN_CENTRAL_ANCHORS"]
            self.central_anchors = set(sorted_anchors[:min_central_count])
            self.secondary_anchors = non_leaf_nodes - self.central_anchors

        logger.debug(f"Central Anchors: {sorted(list(self.central_anchors))}")
        logger.debug(f"Secondary Anchors: {sorted(list(self.secondary_anchors))}")
        logger.debug(f"Leaf Nodes: {sorted(list(self.leaf_nodes))}")

        self.central_kp = max(degrees, key=degrees.get)
        logger.debug(f"Assembler determined central keypoint: '{self.central_kp}'")

    # == PUBLIC METHODS ==

    def update_bone_stats(self, new_stats: dict):
        """ Allows the assembler's anatomical model to be updated on the fly """

        self.reference_bone: Bone = frozenset(new_stats['reference_bone'])
        self.median_ref_len = new_stats['median_reference_length']
        self.bones_ratios = {frozenset(k.split(';')): v for k, v in new_stats['bones_ratios'].items()}

    def assemble_frame(self, soup_points: List[SoupPoint]
        ) -> Tuple[List[CandidateSkeleton], Dict[Tuple[str, int], SoupPoint]]:
        """
        Main assembly entry point. Generates initial fragments and then creates all
        plausible merge hypotheses without committing to them
        """

        # Generate small, high-confidence initial fragments
        initial_fragments, points_map = self._generate_candidates(soup_points)
        if not initial_fragments:
            return [], {}

        # Generate all plausible merges between initial fragments (but keep initial fragments)
        merge_hypotheses = self._generate_merge_hypotheses(initial_fragments, points_map)

        # Return the complete set of hypotheses for the global solver
        all_hypotheses = initial_fragments + merge_hypotheses

        logger.debug(
            f"Assembler created {len(initial_fragments)} initial fragments and {len(merge_hypotheses)} merge hypotheses.")
        return all_hypotheses, points_map

    # == PRIVATE HELPER METHODS ==

    def _generate_merge_hypotheses(self,
            fragments: List[CandidateSkeleton],
            points_map: Dict[Tuple[str, int], SoupPoint]
        ) -> List[CandidateSkeleton]:
        """
        Generates all plausible merge hypotheses from a list of fragments without committing to any single merge.
        It is stateless (returns only the new, merged skeletons)
        """

        if len(fragments) < 2:
            return []

        # Each initial fragment is its own constituent part
        for i, frag in enumerate(fragments):
            # we use index i as a unique identifier for each initial fragment
            frag.constituent_indices = frozenset([i])

        merge_hypotheses = []

        # consider every possible pair of initial fragments for merging
        for i, j in combinations(range(len(fragments)), 2):
            skel_A, skel_B = fragments[i], fragments[j]

            # attempt to merge the two fragments
            merged_candidate = self._attempt_merge(skel_A, skel_B, points_map)

            if merged_candidate:
                # th emerged candidate's identity is the union of the parent fragments' IDs
                merged_candidate.constituent_indices = skel_A.constituent_indices.union(skel_B.constituent_indices)
                merge_hypotheses.append(merged_candidate)

        # TODO: could add a second layer of merging here by calling this function recursively on 'merge_hypotheses'...

        return merge_hypotheses

    def _generate_candidates(self,
            soup_points: List[SoupPoint]
        ) -> Tuple[List[CandidateSkeleton], Dict[Tuple[str, int], SoupPoint]]:
        """
        Generates all plausible skeleton candidates by exhaustively seeding from all anchor and leaf points.
        This creates the necessary redundancy for the downstream solver
        """

        if not soup_points:
            return [], {}

        points_map = {(p.keypoint_type, p.idx): p for p in soup_points}

        candidate_skeletons = []

        all_seed_nodes = list(points_map.keys())

        # used_as_seed is used to avoid starting a full growth from every single point in a large fragment
        # (which would be redundant). Only one point in a component should be used to seed
        used_as_seed = set()

        # Grow full skeletons from any anchor point
        all_anchor_types = self.central_anchors.union(self.secondary_anchors)
        anchor_seed_nodes = [node for node in all_seed_nodes if node[0] in all_anchor_types]

        for seed_node in anchor_seed_nodes:
            if seed_node in used_as_seed:
                continue

            candidate = self._grow_skeleton(
                seed_node,
                points_map,
                score_debt_tol=CONFIG["ASSEMBLER_SCORE_DEBT_TOLERANCE"]
            )

            if candidate:
                candidate_skeletons.append(candidate)
                # Mark all nodes in the discovered fragment as 'used for seeding'
                for node in candidate.nodes:
                    used_as_seed.add(node)

        # Find any remaining isolated 2-point leaf fragments that weren't part of a larger growth
        # TODO: should it be more than 2?
        leaf_seed_nodes = [node for node in all_seed_nodes if node[0] in self.leaf_nodes]
        for seed_node in leaf_seed_nodes:
            if seed_node in used_as_seed:
                continue

            fragment = self._find_leaf_fragment(seed_node, points_map)
            if fragment:
                candidate_skeletons.append(fragment)
                # mark these two nodes as used as well
                for node in fragment.nodes:
                    used_as_seed.add(node)

        logger.debug(f"Generated {len(candidate_skeletons)} initial candidates for this frame.")
        return candidate_skeletons, points_map

    def _attempt_merge(self,
           skel_A:      CandidateSkeleton,
           skel_B:      CandidateSkeleton,
           points_map:  Dict[Tuple[str, int], SoupPoint]
       ) -> Optional[CandidateSkeleton]:
        """
        Attempts to merge two disjoint fragments into a larger skeleton

        1. Ensures fragments are fully disjoint (no shared points or keypoint types)
        2. Checks for "clone" fragments using Jaccard similarity on keypoint positions
        3. Verifies scale consistency
        4. Finds the best high-quality 'linking' bone between fragments
        5. Validates this link using a 'shared neighbour' graph heuristic
        """

        # Pre-check 1: must be fully disjoint
        if not skel_A.nodes.isdisjoint(skel_B.nodes):
            return None

        kps_A = {node[0] for node in skel_A.nodes}
        kps_B = {node[0] for node in skel_B.nodes}
        if not kps_A.isdisjoint(kps_B):
            return None

        # Pre-check 2: handle clones with Jaccard score
        common_kps = kps_A.intersection(kps_B)
        if common_kps:
            # this block should theoretically not be reached due to the check above but it's a good safeguard
            proximal_intersection = 0
            kps_map_A = {node[0]: points_map[node].position for node in skel_A.nodes}
            kps_map_B = {node[0]: points_map[node].position for node in skel_B.nodes}

            for name in common_kps:
                if np.linalg.norm(kps_map_A[name] - kps_map_B[name]) < CONFIG["CONFLICT_SOLVER_PROXIMITY_RADIUS"]:
                    proximal_intersection += 1

            union_size = len(kps_A.union(kps_B))
            jaccard_prox = proximal_intersection / union_size if union_size > 0 else 0

            if jaccard_prox > CONFIG["CONFLICT_SOLVER_JACCARD_THRESHOLD"]:
                # they are not merge candidates, they are conflicting clones
                return None

        # Scale consistency
        if abs(skel_A.scale - skel_B.scale) > CONFIG["ASSEMBLER_MERGE_SCALE_TOLERANCE"]:
            return None
        combined_scale = (skel_A.scale + skel_B.scale) / 2.0

        # Find a high quailty linking bone
        best_link_score = -1.0
        best_linking_bone = None

        combined_nodes = skel_A.nodes.union(skel_B.nodes)
        combined_kps_map = {node[0]: points_map[node].position for node in combined_nodes}

        for kp_a in kps_A:
            for kp_b in kps_B:
                bone = frozenset((kp_a, kp_b))
                if bone in self.bones_ratios:
                    score = self._score_bone(bone, combined_kps_map, points_map, combined_nodes, combined_scale)
                    if score > best_link_score:
                        best_link_score = score
                        best_linking_bone = bone

        # Check if the best link we found is good enough to justify a merge
        if best_link_score < CONFIG["ASSEMBLER_MERGE_LINKING_BONE_THRESHOLD"]:
            return None

        # Shared neighbour check
        kp_a_name, kp_b_name = tuple(best_linking_bone)
        neighbors_of_a = set(self._skeleton_graph.neighbors(kp_a_name))
        neighbors_of_b = set(self._skeleton_graph.neighbors(kp_b_name))

        shared_neighbors = neighbors_of_a.intersection(neighbors_of_b)
        shared_neighbors.discard(kp_a_name)
        shared_neighbors.discard(kp_b_name)

        if not any(n in kps_A or n in kps_B for n in shared_neighbors):
            # island-to-island link with no existing anchor point
            # It's the signature of a pathological merge. Reject it.
            return None

        # Merge is valid
        merged_nodes = skel_A.nodes.union(skel_B.nodes)

        # Estimate the new average score for the merged object
        avg_score_A = skel_A.anatomical_score / len(skel_A.nodes) if skel_A.nodes else 0
        avg_score_B = skel_B.anatomical_score / len(skel_B.nodes) if skel_B.nodes else 0
        num_bones_A = max(1, len(skel_A.nodes) - 1)
        num_bones_B = max(1, len(skel_B.nodes) - 1)

        new_total_score = (avg_score_A * num_bones_A) + (avg_score_B * num_bones_B) + best_link_score
        new_num_bones = num_bones_A + num_bones_B + 1
        new_avg_score = new_total_score / new_num_bones

        # Use the standard factory to create the final candidate
        return self._create_candidate(merged_nodes, new_avg_score, combined_scale)

    def _create_candidate(self,
            nodes:      FrozenSet[Tuple[str, int]],
            avg_score:  float,
            scale:      float
        ) -> CandidateSkeleton:
        """ Factory function to create a CandidateSkeleton object """

        if avg_score <= 0:
            # this can happen if the fragment is nonsensical
            # so we return a candidate with a score of 0
            return CandidateSkeleton(
                nodes=nodes,
                competition_score=0.0,
                anatomical_score=0.0,
                scale=scale
            )

        num_nodes = len(nodes)

        size_bonus_multiplier = num_nodes
        base_score = avg_score * size_bonus_multiplier

        # quality-over-quantity boost for exceptionally well-formed skeletons
        # (his helps high-quality fragments to punch above their weight in the conflict resolution stage)
        quality_bonus = 0.0

        high_quality_thresh = CONFIG["ASSEMBLER_HIGH_QUALITY_THRESHOLD"]
        quality_bonus_factor = CONFIG["ASSEMBLER_QUALITY_BONUS_FACTOR"]

        if avg_score > high_quality_thresh:
            # The bonus is a percentage of the base score
            quality_bonus = base_score * (quality_bonus_factor - 1.0)

        # The final 'competition score' (used by MWIS) is the sum of the base score and all bonuses
        competition_score = base_score + quality_bonus

        # The 'anatomical_score' reflects the pure anatomical fit multiplied by size
        # This is useful for the AnatomyLearner and for diagnostics
        anatomical_score = avg_score * num_nodes

        return CandidateSkeleton(
            nodes=nodes,
            competition_score=competition_score,
            anatomical_score=anatomical_score,
            scale=scale
        )

    def _find_leaf_fragment(self,
            leaf_node:      Tuple[str, int],
            points_map:     Dict[Tuple[str, int], SoupPoint]
        ) -> Optional[CandidateSkeleton]:
        """ lightweight method to find the single best 2-point fragment starting from a leaf node """

        leaf_kp_name = leaf_node[0]

        # a leaf has only one neighbor in the skeleton graph
        try:
            parent_kp_name = next(self._skeleton_graph.neighbors(leaf_kp_name))
        except StopIteration:
            # this shouldn't happen if the skeleton graph is correct, but as a safeguard:
            return None

        # Find all available points of the parent type
        parent_candidates = [node for node in points_map if node[0] == parent_kp_name]
        if not parent_candidates:
            return None

        best_parent_node = None
        best_bone_score = -1.0

        # Iterate through potential parents to find the one that forms the best single bone
        for parent_cand_node in parent_candidates:
            kps = {leaf_kp_name: points_map[leaf_node].position, parent_kp_name: points_map[parent_cand_node].position}
            nodes = frozenset([leaf_node, parent_cand_node])

            # We need to estimate scale even for a 2-point fragment
            # This is a bit of a chicken-and-egg problem but we can use a quick estimate
            scale = self._get_skeleton_scale(kps)
            if not (CONFIG["ASSEMBLER_MIN_SANE_SCALE"] < scale < CONFIG["ASSEMBLER_MAX_SANE_SCALE"]):
                continue

            bone = frozenset((leaf_kp_name, parent_kp_name))
            score = self._score_bone(bone, kps, points_map, nodes, scale)

            if score > best_bone_score:
                best_bone_score = score
                best_parent_node = parent_cand_node

        # Check if the best connection we found is good enough
        if best_parent_node and best_bone_score > CONFIG["ASSEMBLER_MIN_BONE_SCORE_FOR_FRAGMENT"]:
            fragment_nodes = frozenset([leaf_node, best_parent_node])

            final_scale = self._get_skeleton_scale({n[0]: points_map[n].position for n in fragment_nodes})

            # Use the factory function to create the candidate with proper competition scoring
            return self._create_candidate(fragment_nodes, best_bone_score, final_scale)

        return None

    def _grow_skeleton(self,
           anchor_node:     Tuple[str, int],
           points_map:      Dict[Tuple[str, int], SoupPoint],
           score_debt_tol:  float
        ) -> Optional[CandidateSkeleton]:
        """ Grows a single skeleton candidate from an anchor point """

        # KDTree for fast spatial lookups
        all_nodes_list = list(points_map.keys())
        all_positions = np.array([points_map[node].position for node in all_nodes_list])
        if all_positions.shape[0] == 0: return None
        kdtree = cKDTree(all_positions)

        max_bone_len = self.median_ref_len * CONFIG["BOOTSTRAP_MAX_BONE_LEN"]

        # initialisation
        current_nodes = {anchor_node}
        current_kps = {anchor_node[0]: points_map[anchor_node].position}
        total_bone_score_sum = 0.0
        num_bones = 0

        while True:
            # Find all valid keypoints that could connect to the current skeleton
            # using topology (skeleton graph) and spatial proximity (KDTree)

            current_kp_names = {node[0] for node in current_nodes}
            nodes_to_evaluate = set()
            for node_in_skel in current_nodes:
                neighbor_kp_types = {n for n in self._skeleton_graph.neighbors(node_in_skel[0]) if
                                     n not in current_kp_names}
                if not neighbor_kp_types: continue

                center_pos = points_map[node_in_skel].position
                nearby_indices = kdtree.query_ball_point(center_pos, r=max_bone_len)
                for idx in nearby_indices:
                    cand_node = all_nodes_list[idx]
                    if cand_node[0] in neighbor_kp_types:
                        nodes_to_evaluate.add(cand_node)

            if not nodes_to_evaluate:
                break  # no more valid connections found

            # Calculate scale once per growth step
            current_step_scale = self._get_skeleton_scale(current_kps)
            if not (CONFIG["ASSEMBLER_MIN_SANE_SCALE"] < current_step_scale < CONFIG["ASSEMBLER_MAX_SANE_SCALE"]):
                # current skeleton has a bad scale so, abort
                break

            current_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0
            current_base_score = current_avg_score * len(current_nodes)
            current_quality_bonus = 0.0

            if current_avg_score > CONFIG["ASSEMBLER_HIGH_QUALITY_THRESHOLD"]:
                current_quality_bonus = current_base_score * (CONFIG["ASSEMBLER_QUALITY_BONUS_FACTOR"] - 1.0)
            current_growth_score = current_base_score + current_quality_bonus

            # Evaluate each potential extension
            best_extension = None
            best_new_growth_score = -float('inf')

            for cand_node in nodes_to_evaluate:

                # Calculate the score contribution of only the new bones this candidate would form
                cand_kp_name = cand_node[0]
                temp_kps = current_kps.copy()
                temp_kps[cand_kp_name] = points_map[cand_node].position
                new_scale = current_step_scale

                new_bone_score_sum = 0
                new_bone_count = 0
                for existing_node in current_nodes:
                    bone = frozenset((cand_node[0], existing_node[0]))
                    if bone in self.bones_ratios:
                        temp_nodes = frozenset(current_nodes | {cand_node})
                        score = self._score_bone(bone, temp_kps, points_map, temp_nodes, new_scale)
                        if score < -500:  # An impossible bone was formed.
                            new_bone_score_sum = -float('inf')
                            break
                        new_bone_score_sum += score
                        new_bone_count += 1

                if new_bone_count == 0 or new_bone_score_sum == -float('inf'):
                    continue

                # Calculate the potential new holistic competition score
                new_total_bones = num_bones + new_bone_count
                new_total_score_sum = total_bone_score_sum + new_bone_score_sum
                new_avg_score = new_total_score_sum / new_total_bones

                new_num_nodes = len(current_nodes) + 1
                new_base_score = new_avg_score * new_num_nodes

                # new_quality_bonus = 0.0
                # if new_avg_score > CONFIG["ASSEMBLER_HIGH_QUALITY_THRESHOLD"]:
                #     new_quality_bonus = new_base_score * (CONFIG["ASSEMBLER_QUALITY_BONUS_FACTOR"] - 1.0)

                # Calculate a smooth quality bonus
                bonus_factor = CONFIG["ASSEMBLER_QUALITY_BONUS_FACTOR"] - 1.0

                # Map average score to a 0-1 range, starting from a baseline (like 75) up to 100
                # Skeletons with avg score below 75 get no bonus.
                baseline_quality = 75.0
                quality_range = 100.0 - baseline_quality

                # Normalized quality (0 to 1, clamped)
                normalized_quality = max(0, (new_avg_score - baseline_quality) / quality_range)

                # The bonus is the base score times the bonus factor, scaled by the normalized quality
                new_quality_bonus = new_base_score * bonus_factor * normalized_quality

                # final score to compare for the growth decision
                new_growth_score = new_base_score + new_quality_bonus

                # Keep track of the best extension found so far
                if new_growth_score > best_new_growth_score:
                    best_new_growth_score = new_growth_score
                    best_extension = {
                        "node": cand_node,
                        "score_contribution": new_bone_score_sum,
                        "bone_count_increase": new_bone_count
                    }

            # Make the growth decision (ompare the best potential new score with the current score)
            if best_extension and best_new_growth_score > (current_growth_score - score_debt_tol):
                # Yep, it's a good move. Accept growth.
                best_node = best_extension["node"]
                current_nodes.add(best_node)
                current_kps[best_node[0]] = points_map[best_node].position
                total_bone_score_sum += best_extension["score_contribution"]
                num_bones += best_extension["bone_count_increase"]
            else:
                # Nope, best potential growth is not good enough. Stop.
                break

        # Finalise and return the grown skeleton
        if len(current_nodes) < CONFIG["ASSEMBLER_MIN_KPS_FOR_SKELETON"]:
            return None

        final_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0.0
        if final_avg_score <= 0:
            return None

        final_scale = self._get_skeleton_scale(current_kps)

        return self._create_candidate(frozenset(current_nodes), final_avg_score, final_scale)

    def _get_skeleton_scale(self,
            keypoints:  Dict[str, np.ndarray]
        ) -> float:
        """ Estimates the scale of a (possibly partial) skeleton relative to the reference stats """

        scales = []

        # Use primary reference bone first
        if self.reference_bone.issubset(keypoints) and self.median_ref_len > 1e-6:
            kp1, kp2 = tuple(self.reference_bone)
            scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / self.median_ref_len)

        # Use all other bones for a robust estimate
        for bone_type, stats in self.bones_ratios.items():
            kp1, kp2 = tuple(bone_type)
            if kp1 in keypoints and kp2 in keypoints:
                expected_len = self.median_ref_len * stats['median_ratio']
                if expected_len > 1e-6:
                    scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / expected_len)

        if not scales:
            return 1.0

        # Filter out absurd scale values before taking the median
        sane_scales = [s for s in scales if
                       CONFIG["ASSEMBLER_MIN_SANE_SCALE"] <= s <= CONFIG["ASSEMBLER_MAX_SANE_SCALE"]]

        return float(np.median(sane_scales)) if sane_scales else 1.0

    def _score_bone(self,
            bone:       Bone,
            keypoints:  Dict[str, np.ndarray],
            points_map: Dict[Tuple[str, int], SoupPoint],
            nodes:      FrozenSet[Tuple[str, int]],
            scale:      float
        ) -> float:
        """ Scores a single bone based on its length conformity to the stats (adjusted for scale) """

        if bone not in self.bones_ratios:
            return 0.0

        stats = self.bones_ratios[bone]
        kp1_name, kp2_name = tuple(bone)
        p1 = keypoints[kp1_name]
        p2 = keypoints[kp2_name]

        # Find confidences
        node1 = next(n for n in nodes if n[0] == kp1_name)
        node2 = next(n for n in nodes if n[0] == kp2_name)
        conf1 = points_map[node1].confidence
        conf2 = points_map[node2].confidence

        expected_length = self.median_ref_len * stats['median_ratio'] * scale

        # Scale the expected deviation
        expected_mad = self.median_ref_len * stats['mad_ratio'] * scale + CONFIG["ASSEMBLER_BONE_SCORE_MAD_EPSILON"]
        distance = np.linalg.norm(p1 - p2)

        # How many std dev (using MAD) the length is from the mean
        num_mads_away = abs(distance - expected_length) / (expected_mad + 1e-6)

        if num_mads_away > 2.0:  # Debug print for bones that are just a bit off
            logger.debug(f"      - Scoring bone {kp1_name}-{kp2_name}:\n"
                         f"          Scale: {scale:.2f}, Measured Dist: {distance:.2f}, Expected: {expected_length:.2f}"
                         f"          MADs away: {num_mads_away:.2f} (Thresh: {CONFIG['ASSEMBLER_BONE_SCORE_MAD_THRESH']})")

        if num_mads_away > CONFIG["ASSEMBLER_BONE_SCORE_MAD_THRESH"]:
            # impossible bone: large negative penalty that will poison the average score
            return -1000.0

        # Gaussian-like fall off for the length score
        length_score = np.exp(-0.5 * num_mads_away ** 2)

        # Factor in the confidence of the 2D detections that created the 3D points
        confidence_score = (conf1 + conf2) / 2.0

        return length_score * confidence_score


class MultiObjectTracker:
    """ Main class for tracking multiple skeletons over time """

    def __init__(self, assembler: SkeletonAssembler):

        self.assembler = assembler
        self.frame_idx = -1
        self.tracklets: List[Tracklet] = []
        self.next_track_id = 0
        self.max_age = CONFIG['TRACKER_MAX_TRACKELT_AGE']

    def update(self,
               soup_points: List[SoupPoint],
               frame_idx: int
        ) -> List[Tracklet]:
        """ Processes a single frame using 'Hypothesize-and-Solve' workflow """

        self.frame_idx = frame_idx

        # Predict forward state of existing tracklets
        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        # Generate all initial fragments from the 3D point soup and merge hypotheses from the assembler
        all_candidates, points_map = self.assembler.assemble_frame(soup_points)
        if not all_candidates:
            self.prune_tracklets()
            return self.get_active_tracklets()

        # Apply temporal guidance: boost scores of candidates that match tracklets
        bonuses = self._calculate_association_bonuses(all_candidates, points_map)
        for i, cand in enumerate(all_candidates):
            cand.competition_score += bonuses[i] * CONFIG["TRACKER_CONTINUITY_BONUS"]

        # Build a conflict graph that includes spatial conflicts and hierarchical conflicts (merge vs. its parts)
        conflict_graph = self._build_conflict_graph(all_candidates, points_map)

        # Solve for the globally optimal set of skeletons using MWIS
        # winner_indices = solve_mwis_SCIP(conflict_graph)
        winner_indices = solve_mwis_networkx(conflict_graph)

        winning_skeletons = [
            AssembledSkeleton(
                keypoints={node[0]: points_map[node].position for node in all_candidates[i].nodes},
                point_indices={node[0]: node[1] for node in all_candidates[i].nodes},
                score=all_candidates[i].anatomical_score,
                scale=all_candidates[i].scale
            )
            for i in winner_indices
        ]
        logger.debug(
            f"Conflict resolution selected {len(winner_indices)} skeletons from {len(all_candidates)} total candidates.")

        # Associate winning skeletons with tracklets and update state
        if self.tracklets and winning_skeletons:
            cost_matrix = self._build_final_assignment_cost_matrix(self.tracklets, winning_skeletons)
            tracklet_inds, winner_inds = linear_sum_assignment(cost_matrix)

            matched_winner_indices = set()
            for t_idx, w_idx in zip(tracklet_inds, winner_inds):
                if cost_matrix[t_idx, w_idx] < 1e9:
                    self.tracklets[t_idx].update(skeleton=winning_skeletons[w_idx], frame_idx=self.frame_idx)
                    matched_winner_indices.add(w_idx)
        else:
            matched_winner_indices = set()

        # Create new tracklets for unmatched skeletons
        for i, skel in enumerate(winning_skeletons):
            if i not in matched_winner_indices and self.assembler.central_kp in skel.keypoints:
                new_tracklet = Tracklet(self.next_track_id, skel, self.frame_idx, self.assembler.central_kp)
                self.tracklets.append(new_tracklet)
                self.next_track_id += 1

        # Prune old/lost tracklets
        self.prune_tracklets()

        return self.get_active_tracklets()

    def predict_only(self, frame_idx: int) -> List[Tracklet]:
        """ Handles frames with no detections by only running the prediction step """

        self.frame_idx = frame_idx

        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        self.prune_tracklets()

        return self.get_active_tracklets()

    def get_active_tracklets(self) -> List[Tracklet]:
        """ Returns a list of tracklets that have been updated in the current frame """
        return [t for t in self.tracklets if t.time_since_update == 0]

    def prune_tracklets(self):
        """ Removes tracklets that have been lost for too long or that are very uncertain """

        self.tracklets = [
            t for t in self.tracklets
            if t.time_since_update <= self.max_age and not np.sum(t.uncertainty['position']) > CONFIG[
                'TRACKER_UNCERTAINTY_THRESHOLD']
        ]

    def _build_conflict_graph(self,
            candidates: List[CandidateSkeleton],
            points_map: Dict[Tuple[str, int], SoupPoint]
        ) -> nx.Graph:
        """ Builds a conflict graph with both spatial and hierarchical conflicts """

        conflict_graph = nx.Graph()
        num_candidates = len(candidates)

        for i, cand in enumerate(candidates):
            weight = max(0, int(cand.competition_score * 100))
            conflict_graph.add_node(i, weight=weight)

        # Pre-calculate centroids for spatial checks
        centroids = np.array([
            np.mean([points_map[node].position for node in cand.nodes], axis=0) if cand.nodes else np.array([np.nan] * 3)
            for cand in candidates
        ])

        for i, j in combinations(range(num_candidates), 2):
            cand_i, cand_j = candidates[i], candidates[j]

            # Hierarchical conflict: if one candidate is composed of the other, they conflict
            if cand_i.constituent_indices and cand_j.constituent_indices:
                if cand_j.constituent_indices.issubset(cand_i.constituent_indices) or \
                        cand_i.constituent_indices.issubset(cand_j.constituent_indices):
                    conflict_graph.add_edge(i, j)
                    continue

            # Spatial Conflict (direct point sharing)
            shared_nodes = cand_i.nodes.intersection(cand_j.nodes)
            if len(shared_nodes) > CONFIG["CONFLICT_SOLVER_SHARED_POINTS_TOLERANCE"]:
                conflict_graph.add_edge(i, j)
                continue

            # Spatial Conflict (proximity)
            # Only check this if centroids are close
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)
            if dist_sq < CONFIG["CONFLICT_SOLVER_BROAD_RADIUS"] ** 2:
                kps_i = {node[0]: points_map[node].position for node in cand_i.nodes}
                kps_j = {node[0]: points_map[node].position for node in cand_j.nodes}
                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()
                if not union:
                    continue

                proximal_intersection = sum(
                    1 for name in common if
                    np.linalg.norm(kps_i[name] - kps_j[name]) < CONFIG["CONFLICT_SOLVER_PROXIMITY_RADIUS"]
                )
                if proximal_intersection / len(union) > CONFIG["CONFLICT_SOLVER_JACCARD_THRESHOLD"]:
                    conflict_graph.add_edge(i, j)

        return conflict_graph

    def _calculate_association_bonuses(self,
            candidates: List[CandidateSkeleton],
            points_map: Dict[Tuple[str, int], SoupPoint]
        ) -> np.ndarray:
        """
        Calculates a score bonus (0 to 1) for each candidate skeleton based on its
        best possible alignment with any existing tracklet's prediction
        """

        if not self.tracklets or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))

        for j, cand_skel in enumerate(candidates):
            skel_kps = {node[0]: points_map[node].position for node in cand_skel.nodes}

            if not skel_kps:
                continue

            max_bonus = 0.0
            for tracklet in self.tracklets:
                pred_pose = tracklet.predicted_pose

                if not pred_pose:
                    continue

                common_kps = pred_pose.keys() & skel_kps.keys()
                if len(common_kps) < CONFIG["TRACKER_ASSOCIATION_MIN_KPS"]:
                    continue

                # Calculate mean squared distance between common keypoints
                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel_kps[kp]) ** 2) for kp in common_kps) / len(common_kps)

                # Gaussian bonus falls off as distance increases
                bonus = np.exp(-0.5 * mean_dist_sq / (CONFIG["TRACKER_ASSOCIATION_RADIUS"] ** 2))
                if bonus > max_bonus:
                    max_bonus = bonus

            bonuses[j] = max_bonus

        return bonuses

    def _build_final_assignment_cost_matrix(self,
            tracklets:  List[Tracklet],
            skeletons:  List[AssembledSkeleton]
        ) -> np.ndarray:
        """ Builds the cost matrix for the final assignment via Hungarian algorithm """

        cost_matrix = np.full((len(tracklets), len(skeletons)), 1e9)

        for i, tracklet in enumerate(tracklets):
            pred_pose = tracklet.predicted_pose

            if not pred_pose:
                continue

            for j, skel in enumerate(skeletons):
                common_kps = pred_pose.keys() & skel.keypoints.keys()

                if len(common_kps) < CONFIG["TRACKER_ASSOCIATION_MIN_KPS"]:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel.keypoints[kp]) ** 2)
                                   for kp in common_kps) / len(common_kps)

                if mean_dist_sq > CONFIG["TRACKER_ASSOCIATION_RADIUS"] ** 2:
                    continue

                # The cost is a weighted sum of pose distance and the skeleton's intrinsic score
                # A good skeleton (high score) should reduce the cost
                cost = (CONFIG["COST_POSE_DISTANCE_WEIGHT"] * mean_dist_sq +
                        CONFIG["COST_SKELETON_SCORE_WEIGHT"] * skel.score)
                cost_matrix[i, j] = cost

        return cost_matrix


if __name__ == '__main__':

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    stats_output_file = folder / prefix / 'outputs' / f'bone_stats_session{session}.json'
    # prior_stats_file = Path().home() / 'Desktop' / 'bone_lengths.csv'
    prior_stats_file = None

    # Load data
    points_soup_file = folder / prefix / 'outputs' / f'points_soup_session{session}.pkl'
    skeleton_input_file = folder / prefix / 'inputs' / 'tracking'

    with open(points_soup_file, 'rb') as f:
        points_soup = pickle.load(f)

    if not points_soup:
        print("No points in the soup. Exiting.")
        exit()

    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(skeleton_input_file, symmetry=True)

    # Get Anatomical stats
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=prior_stats_file,
        bootstrap_data=points_soup
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialise the tracking pipeline
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats)
    assembler = SkeletonAssembler(bones_list=bones, bone_stats=bone_stats)
    tracker = MultiObjectTracker(assembler=assembler)

    # Run tracking pipeline
    frames_indices = sorted(points_soup.keys())
    min_frame, max_frame = frames_indices[0], frames_indices[-1]

    tracklets_by_id = defaultdict(list)

    with alive_bar(title='Tracking Skeletons...', length=20, total=(max_frame - min_frame + 1), force_tty=True) as bar:
        for frame_idx in range(min_frame, max_frame + 1):

            # The assembler's anatomical model is updated with the latest learned stats
            current_stats = anatomy_learner.get_stats()
            assembler.update_bone_stats(current_stats)

            # Run the tracker for the frame
            if frame_idx in points_soup:
                active_tracklets = tracker.update(points_soup[frame_idx], frame_idx)
            else:
                active_tracklets = tracker.predict_only(frame_idx)

            # Feed the results back into the learner
            for tracklet in active_tracklets:
                # We only use skeletons from the current frame for learning
                if tracklet.last_update_frame == frame_idx:
                    anatomy_learner.add_sample(tracklet.skeleton)

            # Convert dataclasses back to dicts for serialization
            for tracklet in active_tracklets:
                skel_dict = tracklet.skeleton.to_dict()
                skel_dict['track_id'] = tracklet.track_id
                skel_dict['track_health'] = tracklet.health
                skel_dict['track_anatomical_integrity'] = tracklet.anatomical_integrity
                skel_dict['track_uncertainty_pos'] = tracklet.uncertainty['position'].tolist()
                skel_dict['track_velocity'] = tracklet.kf.x[3:6].flatten().tolist()
                skel_dict['track_predicted_pos'] = tracklet.predicted_position.tolist()
                skel_dict['time_since_update'] = tracklet.time_since_update

                # add the frame_idx to the skeleton dict
                skel_dict['frame_idx'] = frame_idx

                tracklets_by_id[tracklet.track_id].append(skel_dict)

            bar()

    print("Tracking complete.")

    # Save final results (tracklet-centric format)
    output_file = folder / prefix / 'outputs' / f'tracklets_session{session}.pkl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)

    print(f"Results saved to '{output_file}'")
