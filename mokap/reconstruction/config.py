"""
Centralized configuration for the multi-animal reconstruction and tracking pipeline

- ReconstructorConfig: For the initial 2D -> 3D point soup reconstruction
- AnatomyConfig: For bootstrapping and learning the animal's skeletal model
- AssemblerConfig: For assembling 3D points into skeleton candidates
- TrackerConfig: For tracking assembled skeletons over time with Kalman Filters
- MergerConfig: For merging skeleton fragments that overlap in time
- LinkerConfig: For linking tracklets across temporal gaps
- PipelineConfig: A master class to hold all other configurations

"""
from dataclasses import dataclass, field
from typing import Dict, Literal


# Stage 1: Cooking the soup (2D Detections -> 3D points)
# ======================================================

@dataclass
class ReconstructorConfig:
    """Configuration for the Reconstructor class."""

    T_epi: float = 15.0  # Epipolar distance threshold in pixels for considering a 2D point match.
    min_views: int = 2  # Minimum number of cameras that must see a point for it to be reconstructed.
    repro_thresh: float = 10.0  # Maximum average reprojection error (in pixels) for a 3D point to be considered valid.
    filter_method: Literal['average', 'best'] = 'average'  # Method to merge clustered 3D points: 'average' (weighted) or 'best' (highest score).
    cluster_radius: float = 5.0  # Radius (in mm) for DBSCAN to cluster duplicate 3D point hypotheses.
    view_count_weight: float = 10.0  # Weight of the number of views in the point's score function.
    detection_confidence_weight: float = 5.0  # Weight of the summed 2D detector confidences in the score function.
    repro_error_weight: float = -1.0 # Weight of the reprojection error in the score function (negative because it's a cost).
    softmax_temperature: float = 1.0  # Temperature for softmax weighting when merging points. 0=hard-max, inf=uniform
    jaccard_threshold_for_merge: float = 0.75  # Min Jaccard index of shared views for two points to be merged.
    enable_disjoint_merge: bool = False  # Whether to aggressively merge any nearby points, even if they don't share views.
    disjoint_merge_radius: float = 2.0  # Radius (in mm) for the aggressive disjoint merging.


# Stage 2: Assembly (3D point soup -> tracklets)
# ==============================================

@dataclass
class AnatomyConfig:
    """Configuration for anatomical model bootstrapping and learning."""

    # Stats Bootstrapper parameters
    BOOTSTRAP_GENERIC_MAD_RATIO: float = 0.10  # Generic MAD ratio to apply if data-driven MAD is zero.
    BOOTSTRAP_MIN_SAMPLES: int = 20  # Min samples needed to calculate data-driven stats for a bone.
    BOOTSTRAP_MAX_BONE_LEN: float = 4.0  # Max bone length (in mm) for the simple greedy assembler in bootstrapping.

    # Anatomy Learner parameters
    LEARNER_MIN_SAMPLES_FOR_UPDATE: int = 30  # How many new skeletons before re-calculating anatomy stats.
    LEARNER_MIN_SCORE_FOR_LEARNING: float = 5.0  # Minimum score of a tracked skeleton to be used for learning.
    LEARNER_MIN_REF_BONE_LEN: float = 1.0  # Min plausible length (mm) of reference bone for learning.
    LEARNER_MAX_REF_BONE_LEN: float = 15.0  # Max plausible length (mm) of reference bone for learning.


@dataclass
class AssemblerConfig:
    """Configuration for the SkeletonAssembler class."""

    # Assembler parameters
    MAX_BONE_LEN: float = 4.0
    MIN_KPS_FOR_SKELETON: int = 3  # Min keypoints to be considered a valid skeleton fragment.
    MIN_CENTRAL_ANCHORS: int = 2  # Min number of most-connected keypoints to be primary anchors.
    BONE_SCORE_MAD_THRESH: float = 5.0  # How far a bone's length can deviate (in MADs) before its score is zero.
    BONE_SCORE_MAD_EPSILON: float = 0.05  # Small constant added to MAD for numerical stability.
    MIN_SANE_SCALE: float = 0.7  # Min plausible scale estimate for a skeleton fragment.
    MAX_SANE_SCALE: float = 1.5  # Max plausible scale estimate for a skeleton fragment.
    SCORE_DEBT_TOLERANCE: float = 10.0  # How much of a score hit is allowed to add one more part during growth.
    MERGE_SCALE_TOLERANCE: float = 0.075  # Max relative difference in scale for two fragments to be merged.
    MERGE_LINKING_BONE_THRESHOLD: float = 90.0  # Minimum score (0-100) for a bone connecting two fragments to be valid.
    MIN_BONE_SCORE_FOR_FRAGMENT: float = 70.0  # Minimum score for a simple 2-point leaf fragment to be created.
    HIGH_QUALITY_THRESHOLD: float = 90.0  # Score threshold above which a skeleton gets a quality bonus.
    QUALITY_BONUS_FACTOR: float = 1.5  # Multiplicative bonus factor for high-quality skeletons (1.5 = 50% bonus).


@dataclass
class TrackerConfig:
    """Configuration for the MultiObjectTracker and stateful Tracklet classes."""

    # Tracker parameters
    MAX_TRACKLET_AGE: int = 15  # How many frames a tracklet can coast without an update before being deleted.
    UNCERTAINTY_THRESHOLD: float = 100.0  # Max position variance (mm^2) before a tracklet is pruned. (100 = 10mm std dev)
    MIN_KPS_FOR_INFERENCE: int = 3  # Min shared KPs needed to infer a missing central keypoint via alignment.
    SCALE_LEARNING_RATE: float = 0.25  # Learning rate for the tracklet's adapting scale estimate (not currently used by KF).
    ASSOCIATION_RADIUS: float = 1.0  # Max distance (mm) between a tracklet's prediction and a candidate for association.
    ASSOCIATION_MIN_KPS: int = 3  # Min shared keypoints to associate a tracklet with a candidate.
    CONTINUITY_BONUS: float = 500.0  # Large bonus to a candidate's score if it matches an existing tracklet.
    ANATOMICAL_SCORE_ALPHA: float = 0.15  # Smoothing factor for the tracklet's score. 0=no update, 1=new value only
    INFERRED_HEALTH_PENALTY: float = 0.05  # Health reduction for an update based on an inferred (not measured) point.
    HEALTH_DECAY_RATE: float = 0.98  # Multiplicative decay of health per frame without an update.

    # Conflict solver parameters
    CONFLICT_SOLVER_BROAD_RADIUS: float = 3.0  # Skeletons with centroids further than this (mm) are assumed not to conflict.
    CONFLICT_SOLVER_SHARED_POINTS_TOLERANCE: int = 1  # Max number of shared points before declaring a spatial conflict.
    CONFLICT_SOLVER_PROXIMITY_RADIUS: float = 0.25  # Max distance (mm) to consider two corresponding keypoints 'the same'.
    CONFLICT_SOLVER_JACCARD_THRESHOLD: float = 0.85  # Jaccard proximity threshold to consider two skeletons 'clones'.

    # Cost function weights (for final assignment)
    COST_POSE_DISTANCE_WEIGHT: float = 0.9  # Weight for pose distance in the Hungarian assignment cost.
    COST_SKELETON_SCORE_WEIGHT: float = -0.1  # Weight for skeleton score. Negative to reward high-score matches.

    # Kalman Filter parameters
    KF_PROCESS_NOISE_POS: float = 0.1  # Process noise for position (assumes random acceleration). Higher = less smooth
    KF_PROCESS_NOISE_SCALE: float = 0.01  # Process noise for scale.
    KF_MEASUREMENT_NOISE_POS: float = 1.0  # Measurement noise for position (reflects 3D reconstruction uncertainty).
    KF_MEASUREMENT_NOISE_SCALE: float = 0.25  # Measurement noise for scale.
    KF_INIT_COV_VEL: float = 1.0  # Initial covariance for velocity.
    KF_INIT_COV_SCALE: float = 1.0  # Initial covariance for scale.
    KF_INFERENCE_UNCERTAINTY_FACTOR: float = 2.0  # Multiplier for measurement noise when a keypoint is inferred.


# Stage 3: Linking (tracklets -> full tracks)
# ===========================================

@dataclass
class MergerConfig:
    """Configuration for the FragmentMerger class."""

    DEBUG: bool = True  # Enable verbose print statements for debugging the merging process.
    ANATOMY_MEAN_THRESHOLD: float = 70.0  # Min mean anatomical score (0-100) for a merge.
    ANATOMY_P90_THRESHOLD: float = 90.0  # Min 90th percentile score for a merge (ensures high quality).
    ANATOMY_CONFLICT_THRESHOLD: int = 2  # Max shared keypoints before an anatomy check is skipped.
    ANATOMY_BONE_MAD_THRESH: float = 5.0  # Max deviation (in MADs) for a bone to be considered valid.
    PROXIMITY_DIST_THRESH_MM: float = 2.5  # Max average distance for a proximity-based merge.
    PROXIMITY_CONFLICT_THRESHOLD: int = 1  # Max shared keypoints for a proximity check.
    CONFLICTING_FRAME_RATIO: float = 0.5  # Max ratio of overlapping frames that can be in conflict.
    PROXIMITY_MIN_INTEGRITY: float = 70.0  # Min integrity for a track to be considered for proximity merge.
    BONE_SCALE_TOLERANCE: float = 0.075  # Max relative scale difference between fragments for anatomy checks.
    MOTION_VELOCITY_THRESH_MM_S: float = 0.15  # Max median velocity diff (mm/s) for a merge.
    VELOCITY_COSINE_SIMILARITY_THRESHOLD: float = 0.8  # Min cosine similarity for velocities for a merge.

    # Pre-merge validation
    VALIDATION_MIN_INTEGRITY: float = 75.0  # Min integrity for a track to be part of a conflict check.
    VALIDATION_SHARED_KP_THRESH: int = 2  # Median shared KPs above this signals a strong conflict.


@dataclass
class LinkerConfig:
    """Configuration for the TrackletLinker class."""

    # Temporal linker parameters
    MIN_TRACKLET_LEN: int = 5  # Tracklets shorter than this are ignored for linking.
    MAX_FRAME_GAP: int = 20  # Max number of frames between tracklets to consider a link.
    COST_THRESHOLD: float = 15.0  # Max cost for a valid link.
    MIN_COMMON_KPS: int = 4  # Min shared keypoints needed for a stable alignment cost.
    TEMPLATE_FRAMES: int = 3  # Number of frames from start/end to create an average pose.

    # Linker cost function weights
    COST_W_MOTION: float = 1.5  # Weight for motion consistency (Mahalanobis distance).
    COST_W_SHAPE: float = 1.0  # Weight for shape consistency (rigid alignment error).
    COST_W_VELOCITY: float = 0.5  # Weight for velocity continuity across a gap.
    COST_W_QUALITY: float = -0.2  # Negative weight rewards linking high-quality tracklets.

    # Ambiguity parameters
    AMBIGUITY_THRESHOLD_RATIO: float = 0.20  # 2nd best link must be within 20% of best to be 'ambiguous'.
    AMBIGUITY_PENALTY: float = 10.0  # Cost penalty applied to ambiguous links.

    # Smoother/KF parameters for gap-filling
    SMOOTHER_KF_PROCESS_NOISE: Dict[str, float] = field(default_factory=lambda: {
        'default': 0.05, 'thorax': 0.01, 'neck': 0.01, 'eye': 0.01,
        'a0': 0.01, 'm0': 0.01, 'legf0': 0.01, 'legf1': 0.08, 'legf2': 0.15,
        'legm0': 0.01, 'legm1': 0.08, 'legm2': 0.15, 'a1': 0.1, 'a2': 0.2,
        'm1': 0.09,
    })


# Master configuration class
# ==========================

@dataclass
class PipelineConfig:
    reconstruction: ReconstructorConfig = field(default_factory=ReconstructorConfig)
    anatomy:        AnatomyConfig       = field(default_factory=AnatomyConfig)
    assembler:      AssemblerConfig     = field(default_factory=AssemblerConfig)
    tracker:        TrackerConfig       = field(default_factory=TrackerConfig)
    merger:         MergerConfig        = field(default_factory=MergerConfig)
    linker:         LinkerConfig        = field(default_factory=LinkerConfig)

