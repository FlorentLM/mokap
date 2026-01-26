# 3D pose extraction documentation

This pipeline reconstructs, assembles, and tracks multiple 3D skeletons from multi-view 2D detections in a bottom-up way.
It starts with a "3D point soup" of unassociated 3D candidates, statistically learning the skeletal structure, and then performing greedy assembly with temporal smoothing.

### Data structures
*   **Point Soup:** A collection of 3D points and 3D rays derived from 2D inputs. It has no concept of "individuals", only isolated keypoint observations by keypoint type (left eye, thorax, etc).
*   **Skeleton Topology:** The static definition of the skeleton structure (bones, keypoint names, symmetry).
*   **Skeleton Stats:** Learned anatomical constraints (bone lengths, variabilities) and dynamics.
*   **Skeleton Hypothesis:** A candidate graph of connected nodes representing a potential (partial or full) skeleton in a time step.
*   **Tracklet:** A temporally consistent identity holding a state (position, velocity, scale).

---

## Phase 1: Point soup reconstruction (`soup.py`)

This module converts 2D multi-view detections into a unified set of 3D candidates without assigning them to specific individuals.

### 1.1. Undistortion
All 2D input coordinates are undistorted using the camera rig's intrinsic parameters ($K, D$) before processing.

### 1.2. Pairwise epipolar checks
For each keypoint type (batched across frames):
1.  **Epipolar check:** Points are considered candidates for triangulation only if the distance from point $p_2$ to the epipolar line of $p_1$ is below `epipolar_threshold`.
2.  **Weighted (pairwise) triangulation:** Valid pairs are triangulated using weighted DLT.
3.  **Reprojection filter:** Points are discarded if their Root Mean Square Error (RMSE) upon reprojection exceeds `reprojection_threshold`.

### 1.3. Spatial merging and full triangulation
Because pairwise triangulation produces duplicate points for the same physical feature (Cam_1-Cam_2 and Cam_2-Cam_3 pairs), points are merged:
1.  **Clustering:** A KD-Tree queries neighbours within `merge_radius`. A Union-Find data structure clusters connected components.
2.  **Retriangulation:** All 2D observations contributing to a cluster are gathered. The point is re-triangulated using *all* available views.

### 1.4. Orphan ray generation
Any 2D detection that could not be triangulated (visible in only one view) is converted into a 3D ray (origin + direction). These are stored to allow for "virtual node" rescue during assembly.

---

## Phase 2: Bootstrapping (`bootstrap.py`)

Before tracking begins, the system learns the anatomical and dynamic properties of the subjects from the Point Soup in a self-supervised manner.

### 2.1. Anatomy
Learns the expected length of bones and their allowable variance.
1.  **Linkage:** Uses `trackpy` to link isolated keypoints across time based on spatial proximity, creating crude single-node tracklets.
2.  **Pair mining:** Identifies tracklet pairs (e.g. a 'head' tracklet and 'neck' tracklet) that co-occur in the same frames.
3.  **Symmetry pooling:** Keypoints are mapped to their **canonical names** (`left_hand` $\to$ `hand`). Statistics are pooled across symmetric sides to increase sample size.
4.  **Stats calculation:**
    *   **Ratio:** Median length of the bone relative to a reference bone (usually the most central bone).
    *   **Variability:** derived from the Median Absolute Deviation (MAD) of the intra-individual lengths.

### 2.2. Dynamics
Learns movement parameters.
1.  **Differentiation:** Calculates velocity and acceleration from the single-node tracklets.
2.  **Parameter derivation:**
    *   **Process noise ($Q$):** Derived from the acceleration distribution ($\to$ accounting for how erratic the movement is).
    *   **Association Weight:** Derived from velocity jitter ($\to$ stable points are trusted more during association).

---

## Phase 3: Skeleton sssembly (`assembly.py`)

This operates per-frame to construct `SkeletonHypothesis` instances from the Point Soup.

### 3.1. Assembly strategy
The assembler uses a **greedy growth** approach with **single observation rescue**.

1.  **Seeding:** The process iterates through _anchor_ keypoints (high-degree nodes like the thorax) and _leaf_ keypoints to start candidate skeletons.
2.  **Extension (growth):**
    *   From a seed, the algorithm looks for neighbours defined in the `SkeletonTopology`.
    *   **Real candidates:** Nearby 3D points from the soup.
    *   **Virtual candidates (ray casting):** If no 3D point exists, the system checks the _orphan rays_. It calculates the intersection between the ray and a sphere centered at the current node with radius = expected bone length.
3.  **Scoring & selection:**
    *   Candidates are scored based on deviation from expected bone lengths (in terms of MADs).
    *   The best scoring candidate is added to the hypothesis.
    *   **Scale estimation:** The skeleton scale is re-estimated using the median ratio of observed bones to learned statistics.

### 3.2. Fragment merging
After growth, the system may produce disjoint fragments (e.g. a left leg separated from the torso).
*   **Merge check:** Fragments are checked for compatibility (disjoint keypoint names, similar scale).
*   **Linking:** The system attempts to find a valid bone connection between fragments. If a high-enough scoring link exists, they are merged into a single hypothesis.

---

## Phase 4: Multi-individual tracking (`assembly.py`)

The tracker associates frame-level hypotheses with temporal identity.

### 4.1. State estimation (Kalman Filter)
Each `Tracklet` maintains a Kalman Filter with a 7D state:
$$ \text{State} = [x, y, z, v_x, v_y, v_z, \text{scale}] $$
*   **Constant velocity model:** Position and scale are observed; velocity is inferred.
*   **Scale tracking:** Scale is tracked as a state variable, allowing the model to work with different individual sizes.

### 4.2. Association
1.  **Prediction:** Existing tracklets predict their keypoint positions for the current frame.
2.  **Cost matrix:** Costs are calculated between _predicted tracklets_ and _assembled hypotheses_ based on:
    *   Mean Squared Error (MSE) of overlapping keypoints
    *   Anatomical score of the hypothesis
3.  **Assignment:** Solved via Linear Sum Assignment (Hungarian Algorithm).

### 4.3. Conflict resolution (MWIS)
Ideally, one hypothesis maps to one tracklet, but overlapping hypotheses ('ghosts') may exist.
1.  **Conflict graph:** A graph is built where:
    *   Nodes: Hypotheses
    *   Edges: Conflicts (spatial overlap: sharing the same source 3D point, or sharing the same source ray)
    *   Node weights: Hypothesis score + temporal continuity bonus
2.  **Solution:** The **Maximum Weight Independent Set (MWIS)** is solved (approximated via NetworkX or solved exactly via SCIP if available). This selects the highest-scoring set of non-overlapping skeletons.

### 4.4. Tracklet update
*   **Inference (Rigid Alignment):** If a tracklet is matched to a hypothesis that is missing the central keypoint (needed for the KF), the system infers the center using Kabsch rigid alignment between the previous and current pose.
*   **Lifecycle:**
    *   **Unmatched hypotheses:** Spawn new tracklets (if they contain the central keypoint).
    *   **Unmatched tracklets:** "Coast" (predict without update) for `max_tracklet_age` frames before deletion.

---

## TL;DR

1.  **Input:** raw 2D detections
2.  **`soup.py`:** 2D $\to$ 3D points + Rays
3.  **`bootstrap.py`:** Points $\to$ `SkeletonStats` (learns anatomy + dynamics)
4.  **`assembly.py` / `SkeletonAssembler`:**
    *   Build skeletons (using points + raycasting)
    *   Score bones via Gaussian(length | stats)
    *   Merge fragments
5.  **`assembly.py` / `MultiObjectTracker`:**
    *   Predict tracklets (KF)
    *   Calculate expected vs assembled pose distance
    *   Resolve spatial conflicts
    *   Update KF State (position, velocity, scale)
6.  **Output:** Trajectories of skeletons