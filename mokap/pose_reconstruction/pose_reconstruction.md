# 3D pose extraction documentation

This pipeline reconstructs, assembles, and tracks multiple 3D skeletons from multi-view 2D detections in a bottom-up way.
It starts with a "3D point soup" of unassociated 3D candidates, statistically learns the skeletal structure and dynamics, and then performs guided greedy assembly with temporal smoothing.

### Data structures

*   **Point Soup:** A collection of 3D points and 3D rays derived from 2D inputs. It has no concept of "individuals", only isolated keypoint observations by keypoint type (left eye, thorax, etc).
*   **Skeleton Topology:** The static definition of the skeleton structure (bones, keypoint names, symmetry).
*   **Skeleton Stats:** Learned anatomical constraints (bone lengths, variabilities) and per-keypoint dynamics (process noise, measurement noise, association weights).
*   **Skeleton Hypothesis:** A candidate graph of connected nodes representing a potential (partial or full) skeleton in a time step.
*   **Tracklet:** A temporally consistent identity holding hierarchical state: a central Kalman filter (position, velocity, scale) plus per-keypoint offset filters for articulation.

---

## Phase 1: Point Soup Reconstruction (`soup.py`)

This module converts 2D multi-view detections into a unified set of 3D candidates without assigning them to specific individuals.

### 1.1. Clique detection and triangulation

For each keypoint type (batched across frames):

1.  **Epipolar graph construction:** For all detection pairs across different cameras in the same frame, compute the distance from point $p_2$ to the epipolar line of $p_1$. Pairs below `epipolar_threshold` form edges in an undirected graph.
2.  **Maximal clique enumeration:** Find all maximal cliques in the epipolar consistency graph. Each clique represents a set of 2D detections that are mutually consistent and likely correspond to the same 3D point.
3.  **Clique ranking:** Sort cliques by (number of views, total confidence score) in descending order.
4.  **Greedy acceptance:** Iterate through ranked cliques. For each clique:
    *   Skip if any constituent 2D detection has already been used.
    *   Triangulate using weighted DLT with all views in the clique.
    *   Compute reprojection RMSE, reject if above `reprojection_threshold`.
    *   Accept the 3D point and mark all constituent detections as used.

This approach avoids the blind merging problem: when two animals touch, their detections form separate cliques (not mutually epipolar-consistent) and are triangulated independently.

### 1.2. Orphan ray generation

Any 2D detection not consumed by a clique (such as visible in only one view) is converted into a 3D ray (origin + direction). These rays enable "virtual node" rescue during assembly.

---

## Phase 2: Bootstrapping (`bootstrap.py`)

Before tracking begins, the system learns anatomical and dynamic properties from the Point Soup in a self-supervised manner.

### 2.1. Anatomy (`AnatomyBootstrapper`)

Learns the expected length of bones and their allowable variance.

1.  **Linkage:** Uses `trackpy` to link isolated keypoints across time based on spatial proximity, creating crude single-node tracklets.
2.  **Pair mining:** Identifies tracklet pairs (e.g. a 'head' tracklet and 'neck' tracklet) that co-occur in the same frames.
3.  **Symmetry pooling:** Keypoints are mapped to their **canonical names** (`left_hand` $\to$ `hand`). Statistics are pooled across symmetric sides to increase sample size.
4.  **Stats calculation:**
    *   **Ratio:** Median length of the bone relative to a reference bone (usually the most central bone).
    *   **Variability:** Derived from the Median Absolute Deviation (MAD) of intra-individual lengths, pooled across tracklet pairs.

### 2.2. Dynamics (`DynamicsBootstrapper`)

Learns per-keypoint movement parameters.

1.  **Tracklet construction:** Links keypoints across frames using `trackpy`.
2.  **Segment extraction:** Splits tracklets at frame gaps to get contiguous motion segments.
3.  **Differentiation:** Calculates velocity and acceleration from position sequences.
4.  **Parameter derivation (per canonical keypoint):**
    *   **Process noise ($Q$):** Derived from acceleration statistics (median + 2×MAD, clipped). Reflects how erratic the keypoint's motion is.
    *   **Association weight:** Derived from velocity jitter. Stable keypoints (low jitter) receive higher weights and contribute more to tracklet association.
    *   **Measurement noise ($R$):** Set to a base value (configurable).
5.  **Fallback for sparse keypoints:** Keypoints with insufficient data use a topology-based prior: process noise and weight scale with graph distance from the central keypoint (extremities are assumed more erratic).

---

## Phase 3: Skeleton Assembly (`assembly.py`)

The `SkeletonAssembler` operates per-frame to construct `SkeletonHypothesis` instances from the Point Soup, optionally guided by tracklet predictions.

### 3.1. Guided assembly (prediction-driven)

When existing tracklets provide predictions:

1.  **Predicted anchor search:** For each tracklet's predicted central keypoint position, search the soup for nearby points of the correct type within `guided_search_radius`.
2.  **Best seed selection:** Pick the candidate closest to prediction (with slight confidence bonus).
3.  **Guided growth:** Grow the skeleton using the standard extension algorithm, but with an additional **prediction bonus** in scoring: candidates near their predicted positions receive a Gaussian-weighted score boost.
4.  **Track affinity tagging:** Guided hypotheses are tagged with their source tracklet ID for downstream association bonuses.

### 3.2. Blind assembly (bottom-up)

For points not consumed by guided assembly (new individuals entering the scene):

1.  **Anchor seeding:** Iterate through high-degree keypoints (thorax, etc.) and start candidate skeletons.
2.  **Leaf seeding:** Iterate through degree-1 keypoints with limited growth (fragments).

### 3.3. Extension (growth)

From a seed, the algorithm iteratively extends the skeleton:

1.  **Candidate search:** For each frontier node, find neighbouring keypoint types (defined by topology) within `max_search_radius`.
2.  **Real candidates:** Nearby 3D points from the soup.
3.  **Virtual candidates (ray rescue):** If no 3D point exists for a needed keypoint type, intersect orphan rays of that type with a sphere centered at the current node (radius = expected bone length). Both intersection solutions are registered as virtual candidates.
4.  **Extension scoring:** Each candidate is scored based on:
    *   Bone length deviation from expected (Gaussian in MAD units)
    *   Node confidence
    *   Prediction bonus (if guided assembly)
5.  **Scale consistency:** The skeleton's scale is re-estimated at each step. Growth terminates if scale becomes inconsistent.
6.  **Greedy selection:** The best-scoring extension is accepted if it improves the overall growth score (with tolerance for temporary score dips).

### 3.4. Fragment Merging

After independent growth, disjoint fragments may exist (e.g. a leg separated from torso):

1.  **Compatibility check:** Fragments must have disjoint keypoint sets and similar scales.
2.  **Link search:** Attempt to find a valid bone connection between fragments.
3.  **Merge hypotheses:** Compatible fragments are merged into combined hypotheses that compete in conflict resolution.

---

## Phase 4: Multi-individual tracking (`assembly.py`)

The `MultiObjectTracker` associates frame-level hypotheses with temporal identities using hierarchical Kalman filtering.

### 4.1. Hierarchical state estimation

Each `Tracklet` maintains two levels of Kalman filters:

**Central KF** (7D state):
$$\text{State} = [x, y, z, v_x, v_y, v_z, \text{scale}]$$
*   Constant velocity model for position
*   Scale tracked as a slowly-varying state
*   Observes: central keypoint position + skeleton scale

**Per-Keypoint Offset KFs** (6D state each):
$$\text{State} = [\delta x, \delta y, \delta z, \delta v_x, \delta v_y, \delta v_z]$$
*   Tracks each keypoint's offset from central position in body frame
*   Allows articulation: legs can move independently of thorax
*   Velocity damping prevents runaway drift
*   Process noise ($Q$) initialised from learned dynamics (erratic keypoints get higher $Q$)
*   Measurement noise ($R$) from learned dynamics

**Rigidity constraint:** Offset predictions are blended toward learned rest offsets via a configurable rigidity factor, preventing unrealistic deformations while allowing articulation.

### 4.2. Prediction

For each pending tracklet:
1.  Predict central KF forward (position, velocity, scale)
2.  Predict each offset KF forward
3.  Apply rigidity blending toward rest pose
4.  Transform offsets from body frame to world frame
5.  Return predicted world positions for all keypoints

### 4.3. Association

1.  **Cost matrix construction:** For each (tracklet, hypothesis) pair:
    *   **Scale gate (hard):** If scale ratio exceeds threshold → infinite cost (prevents queen↔worker switches)
    *   **Scale gate (soft):** Quadratic penalty for scale differences
    *   **Pose distance:** Weighted MSE of overlapping keypoints, using per-keypoint association weights from learned dynamics
    *   **Anatomical bonus:** Reward for high-quality skeleton scores
2.  **Assignment:** Solved via Linear Sum Assignment (Hungarian algorithm)
3.  **Guided affinity bonus:** Hypotheses that were grown from a tracklet's prediction receive an additional association bonus with that tracklet

### 4.4. Conflict resolution (MWIS)

Multiple hypotheses may explain the same observations (ghosts, fragments, merges):

1.  **Conflict graph construction:** Nodes = hypotheses, edges = conflicts:
    *   Shared 3D point indices
    *   Shared ray sources
    *   Merge provenance (a merge conflicts with its constituents)
    *   Spatial proximity with high keypoint overlap (clone detection)
2.  **Node weighting:** Competition score + temporal continuity bonus
3.  **Solution:** Maximum Weight Independent Set selects the highest-scoring non-conflicting skeleton set

### 4.5. Tracklet lifecycle

*   **Birth:** Unmatched hypotheses containing the central keypoint spawn new tracklets
*   **Update:** Matched tracklets update their KF states from the assigned hypothesis
    *   If central keypoint is missing: infer position via weighted average of (observed keypoint − predicted offset) across visible keypoints
*   **Coasting:** Unmatched tracklets predict forward without update, with health decay
*   **Death:** Tracklets exceeding `max_tracklet_age` or uncertainty threshold are pruned

### 4.6. Online statistics update

After each frame, high-quality tracklet observations update the global `SkeletonStats`:
*   Bone length ratios (EMA)
*   Reference length (EMA)

---

## Summary

1.  **Input:** Raw 2D detections from multiple cameras
2.  **`soup.py`:** 2D $\to$ 3D points via clique-based triangulation + orphan rays
3.  **`bootstrap.py`:** Points $\to$ `SkeletonStats` (learns anatomy + per-keypoint dynamics)
4.  **`assembly.py` / `SkeletonAssembler`:**
    *   Guided assembly from tracklet predictions (seeding + scoring bonus)
    *   Blind assembly for new individuals
    *   Ray-sphere rescue for single-view detections
    *   Score bones via Gaussian(length | stats)
    *   Merge compatible fragments
5.  **`assembly.py` / `MultiObjectTracker`:**
    *   Hierarchical KF prediction (central + per-keypoint offsets)
    *   Scale-gated weighted pose association
    *   MWIS conflict resolution
    *   Update central KF + offset KFs
6.  **Output:** Trajectories of articulated skeletons

---

## TODO

### Body frame orientation

~~Currently the local orientation `body_rotation` remains fixed throughout tracking. This assumes the animal never rotates, which causes offset predictions to drift when animals turn. Given the high framerate this is acceptable for now, but ideally the body frame should be updated each frame (Kabsch, or PCA-based orientation estimation from observed keypoints?)~~
Mostly done

### Tracklet storage

Pruned tracklets are currently deleted from the online state (they need to be extracted in the caller code).

### Online dynamics adaptation

Process noise ($Q$), measurement noise ($R$), and association weights are currently fixed after bootstrap. These could be adapted online based on tracking residuals (e.g., increase $Q$ for a keypoint that consistently has high innovation).

### Rest pose bootstrapping

Rest offsets are currently learned online via EMA during tracking, starting from the first discovered pose. Maybe they should be averaged across all individuals to get a true general rest pose?

### Ray intersection disambiguation

~~Ray-sphere intersection yields two solutions. Currently both are registered as candidates and scored equally. The tracklet's predicted offset could be used to disambiguate: prefer the intersection point closer to the predicted position for that keypoint.~~
Mostly done

### Scale freezing

The central KF tracks scale as a state variable. But that can drift... Scale could be frozen after initial estimation.