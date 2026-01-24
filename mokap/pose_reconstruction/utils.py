import re
import numpy as np
import polars as pl
import pandas as pd
import networkx as nx
from typing import List, Optional, Tuple, Dict
from scipy.stats import median_abs_deviation
from mokap.utils import common_prefix_suffix


def solve_mwis_networkx(G: nx.Graph) -> List[int]:
    """ Solves the Maximum Weight Independent Set problem using NetworkX """

    if not G.nodes:
        return []

    # The MWC of the complement graph is equivalent to MWIS of the original graph
    complement_graph = nx.complement(G)

    # taking the complement does not copy weights so we have to do it explicitely
    node_weights = nx.get_node_attributes(G, 'weight')
    nx.set_node_attributes(complement_graph, node_weights, name='weight')

    winner_indices, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')
    return winner_indices


def solve_mwis_SCIP(G: nx.Graph) -> List[int]:
    """ Solves the Maximum Weight Independent Set problem using SCIP ILP solver """

    if not G.nodes:
        return []

    from pyscipopt import Model

    model = Model("mwis")
    model.hideOutput()

    # Create binary variable for each node in the graph
    # (1 if the node is in the solution, 0 if not)
    nodes = list(G.nodes())
    variables = {node: model.addVar(vtype="B", name=f"x_{node}") for node in nodes}

    # Objective function: maximize sum of weights of selected nodes
    objective_terms = [G.nodes[node]['weight'] * variables[node] for node in nodes]
    model.setObjective(sum(objective_terms), "maximize")

    # Constraints: for every edge (u, v) in the conflict graph, u and v cannot be chosen together
    # x_u + x_v <= 1
    for u, v in G.edges():
        model.addCons(variables[u] + variables[v] <= 1)

    model.optimize()

    solution_nodes = []
    if model.getStatus() == "optimal":
        for node in nodes:
            # Check if the variable is close to 1 in the solution
            if model.getVal(variables[node]) > 0.99:
                solution_nodes.append(node)

    return solution_nodes


def solve_mwis_greedy(G: nx.Graph) -> List[int]:
    """
    Approximation of MWIS: iteratively picks the highest weight node and removes its neighbours.
    Much faster, but might miss the optimal combination.
    """
    if not G.nodes:
        return []

    # Sort nodes by weight (highest first)
    nodes_sorted = sorted(G.nodes, key=lambda n: G.nodes[n].get('weight', 0), reverse=True)

    solution = []
    forbidden = set()

    for node in nodes_sorted:
        if node not in forbidden:
            solution.append(node)
            # Once a node is picked, its neighbuors can't be (conflict)
            forbidden.update(G.neighbors(node))

    return solution


def solve_mwis(G: nx.Graph, method='networkx') -> List[int]:
    if method == 'greedy':
        return solve_mwis_greedy(G)
    elif method == 'scip':
        return solve_mwis_SCIP(G)
    else:
        return solve_mwis_networkx(G)


##

def robust_stats(data: List[float], fallback_val: float = np.nan) -> Tuple[float, float]:
    """Returns median and MAD, safe for empty lists."""
    if len(data) == 0:
        return fallback_val, fallback_val
    arr = np.asarray(data)
    # scale 'normal' approximates std dev consistency
    return float(np.median(arr)), float(median_abs_deviation(arr, scale="normal"))


def plot_tracks_3d(ax, tracks_df: pd.DataFrame, title: str):
    ax.set_title(title)
    pids = tracks_df["particle"].unique()
    if len(pids) > 200:
        pids = np.random.choice(pids, 200, replace=False)
    for pid in pids:
        t = tracks_df[tracks_df["particle"] == pid].sort_values("frame")
        ax.plot(t.x, t.y, t.z, linewidth=0.5, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


# TODO: This will be removed once fileio is cleaned and uses polars for all disk-persistent data
def prepare_reconstruction_input(df: pl.DataFrame, cameras: List[str], keypoints: List[str]):
    """
    Converts Polars DataFrame to flat numpy arrays for the Reconstructor
    """

    df = df.sort(["frame", "keypoint", "camera"])

    cam_map = {cam_name: c for c, cam_name in enumerate(cameras)}
    kp_map = {kp_name: k for k, kp_name in enumerate(keypoints)}

    df = df.with_columns(
        pl.col("keypoint").replace(kp_map).cast(pl.Int16).alias("kp_type_id"),
        pl.col("camera").replace(cam_map).cast(pl.Int8).alias("cam_id"),
    ).sort(["frame", "kp_type_id", "cam_id", "score"], descending=[False, False, False, True])

    return {
        "frame_indices": df["frame"].to_numpy(),
        "kp_type_ids": df["kp_type_id"].to_numpy(),
        "cam_ids": df["cam_id"].to_numpy(),
        "coords": df.select(["x", "y"]).to_numpy(),
        "scores": df["score"].to_numpy()
    }