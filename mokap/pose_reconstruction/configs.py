from dataclasses import dataclass

# TODO: Parameters need to be reorganised

MIN_PROCESS_NOISE = 0.01
MAX_PROCESS_NOISE = 2.0


@dataclass
class TrackletConfig:
    """Configuration for a Tracklet."""

    # Central KF
    central_process_noise_pos: float = 0.1
    central_process_noise_scale: float = 0.01
    central_measurement_noise_pos: float = 0.5 # 1.0  # measurement noise for central position
    central_measurement_noise_scale: float = 0.01 # 0.25

    centre_inference_uncertainty_factor: float = 3.0  # multiplier for measurement noise when a keypoint is inferred

    # Offset KFs
    offset_velocity_damping: float = 0.95  # Offsets tend to return to rest
    rigidity_factor: float = 0.3  # How much to pull offsets toward rest pose (0=free, 1=rigid)

    # Health/lifecycle
    health_decay_rate: float = 0.95
    inferred_health_penalty: float = 0.2
    max_coast_age: int = 10
    min_kps_for_inference: int = 3


@dataclass
class AssemblerConfig:
    """Configuration for the skeleton assembler class."""

    max_bone_len: float = 4.0
    min_kps_for_skeleton: int = 3  # min keypoints to be considered a valid skeleton fragment
    min_central_anchors: int = 2  # min number of most-connected keypoints to be primary anchors
    MAD_threshold: float = 5.0  # how far a bone's length can deviate (in mads) before its score is zero
    min_sane_scale: float = 0.7  # min plausible scale estimate for a skeleton fragment
    max_sane_scale: float = 1.5  # max plausible scale estimate for a skeleton fragment
    score_debt_tolerance: float = 10.0  # how much of a score hit is allowed to add one more part during growth
    merge_scale_tolerance: float = 0.075  # max relative difference in scale for two fragments to be merged
    merge_linking_bone_threshold: float = 90.0  # minimum score (0-100) for a bone connecting two fragments to be valid
    min_bone_score_for_fragment: float = 70.0  # minimum score for a simple 2-point leaf fragment to be created
    high_quality_threshold: float = 90.0  # score threshold above which a skeleton gets a quality bonus
    quality_bonus_factor: float = 1.5  # multiplicative bonus factor for high-quality skeletons (1.5 = 50% bonus)


@dataclass
class TrackerConfig:
    """Configuration for the multi object tracker and stateful tracklet classes."""

    # Tracker parameters
    max_tracklet_age: int = 15  # how many frames a tracklet can coast without an update before being deleted
    uncertainty_threshold: float = 100.0  # max position variance (mm^2) before a tracklet is pruned. (100 = 10mm std dev)
    min_kps_for_inference: int = 3  # min shared kps needed to infer a missing central keypoint via alignment
    # scale_learning_rate: float = 0.25  # learning rate for the tracklet's adapting scale estimate # TODO: add this
    association_radius: float = 1.0  # max distance (mm) between a tracklet's prediction and a candidate for association
    association_min_kps: int = 3  # min shared keypoints to associate a tracklet with a candidate
    continuity_bonus: float = 500.0  # large bonus to a candidate's score if it matches an existing tracklet
    anatomical_score_alpha: float = 0.15  # smoothing factor for the tracklet's score. 0=no update, 1=new value only
    inferred_health_penalty: float = 0.05  # health reduction for an update based on an inferred (not measured) point
    health_decay_rate: float = 0.98  # multiplicative decay of health per frame without an update

    # Scale gating
    scale_gate_hard_threshold: float = 0.3  # Reject if |Δscale|/scale > this
    scale_gate_soft_weight: float = 50.0  # Penalty weight for smaller deviations

    # Conflict solver parameters
    conflict_solver_broad_radius: float = 3.0  # skeletons with centroids further than this (mm) are assumed not to conflict
    conflict_solver_proximity_radius: float = 0.25  # max distance (mm) to consider two corresponding keypoints 'the same'
    conflict_solver_jaccard_threshold: float = 0.85  # jaccard proximity threshold to consider two skeletons 'clones'

    # Cost function weights (for final assignment)
    cost_pose_distance_weight: float = 0.9   # weight for pose distance in the hungarian assignment cost
    skeleton_score_bonus_weight: float = 0.1  # bonus for high anatomical scores