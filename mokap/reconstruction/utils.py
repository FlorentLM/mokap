import logging
import re
import polars as pl
import networkx as nx
from typing import List, Optional, Tuple, Dict
from mokap.utils import common_prefix_suffix


logger = logging.getLogger(__name__)


# TODO: Profile the two MWIS solvers a bit more (looks like NX is faster for many calls on small graphs)

def solve_mwis_networkx(graph: nx.Graph) -> List[int]:
    """ Solves the Maximum Weight Independent Set problem using NetworkX """

    if graph.number_of_nodes() == 0:
        return []

    # The MWC of the complement graph is equivalent to MWIS of the original graph
    complement_graph = nx.complement(graph)

    # taking the complement does not copy weights so we have to do it explicitely
    node_weights = nx.get_node_attributes(graph, 'weight')
    nx.set_node_attributes(complement_graph, node_weights, name='weight')

    winner_indices, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')
    return winner_indices


def solve_mwis_SCIP(graph: nx.Graph) -> List[int]:
    """ Solves the Maximum Weight Independent Set problem using SCIP ILP solver """

    from pyscipopt import Model

    model = Model("mwis")
    model.hideOutput()

    # Create a binary variable for each node in the graph
    # The variable will be 1 if the node is in the solution, 0 otherwise
    nodes = list(graph.nodes())
    variables = {node: model.addVar(vtype="B", name=f"x_{node}") for node in nodes}

    # Set the objective function: Maximize the sum of the weights of the selected nodes
    objective_terms = [graph.nodes[node]['weight'] * variables[node] for node in nodes]
    model.setObjective(sum(objective_terms), "maximize")

    # Add constraints: For every edge (u, v) in the conflict graph, the two nodes
    # cannot be chosen together. This is the "independent set" constraint
    # x_u + x_v <= 1
    for u, v in graph.edges():
        model.addCons(variables[u] + variables[v] <= 1)

    # Solve the model
    model.optimize()

    # Extract the solution
    solution_nodes = []
    if model.getStatus() == "optimal":
        for node in nodes:
            # Check if the variable is close to 1 in the solution
            if model.getVal(variables[node]) > 0.99:
                solution_nodes.append(node)

    return solution_nodes


def create_canonical_map(
        keypoint_names: List[str],
        symmetry_map:   Optional[List[Tuple[str, str]]]
) -> Dict[str, str]:
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
            # canonical name is just the name itself but cleaned up
            canonical_name = names_delimiter_regex.sub('', kp_name).lower()
            canonical_map[kp_name] = canonical_name

    logger.debug("Generated Canonical Keypoint Map for Smoother:")
    unique_types = sorted(list(set(canonical_map.values())))
    logger.debug(f"  -> Found types: {unique_types}")
    example_kp = keypoint_names[2]
    logger.debug(f"  -> Example: '{example_kp}' maps to '{canonical_map.get(example_kp)}'")

    return canonical_map


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
