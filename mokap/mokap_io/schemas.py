"""
Schema definitions and validation for Mokap data formats.

- Points2D: 2D keypoint detections (input to triangulation)
- Points3D: 3D point soup (output of reconstruction, input to assembly)
- Tracks3D: Tracked 3D poses (output of tracking)
"""
from dataclasses import dataclass
from typing import List, Any, Set
import numpy as np
import polars as pl


@dataclass(frozen=True)
class Column:
    """Definition of a single column in a schema."""

    name: str
    dtype: str
    required: bool = True
    default: Any = None
    description: str = ""

    @property
    def polars_dtype(self) -> pl.DataTypeClass:
        match self.dtype:
            case "Int32": return pl.Int32
            case "Int64": return pl.Int64
            case "Float32": return pl.Float32
            case "Float64": return pl.Float64
            case "Utf8": return pl.Utf8
            case "UInt64": return pl.UInt64
            case "Boolean": return pl.Boolean
            case _: raise TypeError(f'Unrecognized data type "{self.dtype}"')

    @property
    def numpy_dtype(self) -> type | np.dtype:
        match self.dtype:
            case "Int32": return np.int32
            case "Int64": return np.int64
            case "Float32": return np.float32
            case "Float64": return np.float64
            case "Utf8": return object
            case "UInt64": return np.uint64
            case "Boolean": return bool
            case _: raise TypeError(f'Unrecognized data type "{self.dtype}"')


# Schema definitions

# 2D detections: One row per 2D keypoint detection (input to triangulation)
POINTS_2D = [
    Column("frame", "Int32", True, None, "Frame index (0-based)"),
    Column("camera", "Utf8", True, None, "Camera name"),
    Column("keypoint", "Utf8", True, None, "Keypoint name"),
    Column("x", "Float32", True, None, "X coordinate (pixels)"),
    Column("y", "Float32", True, None, "Y coordinate (pixels)"),
    Column("score", "Float32", True, 1.0, "Detection confidence [0-1]"),
    Column("instance_id", "Utf8", False, "0", "Instance ID (multi-individual)"),
    Column("source", "Utf8", False, "unknown", "Detection source (sleap, dlc, catar, etc.)"),
]

# 3D point soup: One row per triangulated point or orphan ray
POINTS_3D = [
    Column("frame", "Int32", True, None, "Frame index (0-based)"),
    Column("keypoint", "Utf8", True, None, "Keypoint name"),
    Column("x", "Float32", True, None, "X coordinate (world units) or ray origin X"),
    Column("y", "Float32", True, None, "Y coordinate (world units) or ray origin Y"),
    Column("z", "Float32", True, None, "Z coordinate (world units) or ray origin Z"),
    Column("confidence", "Float32", True, 1.0, "Triangulation confidence [0-1]"),
    Column("reprojection_error", "Float32", False, None, "Reprojection RMSE (pixels)"),
    Column("camera_mask", "UInt64", False, 0, "Bitmask of contributing cameras"),
    Column("status", "Utf8", True, "reconstructed", "Point status: reconstructed, ray"),
    Column("instance_id", "Utf8", False, "0", "Instance ID (from 2D detections)"),
    Column("ray_dx", "Float32", False, None, "Ray direction X"),
    Column("ray_dy", "Float32", False, None, "Ray direction Y"),
    Column("ray_dz", "Float32", False, None, "Ray direction Z"),
]

# 3D tracked poses: One row per keypoint per frame per track
TRACKS_3D = [
    Column("track_id", "Int32", True, None, "Track ID"),
    Column("frame", "Int32", True, None, "Frame index (0-based)"),
    Column("keypoint", "Utf8", True, None, "Keypoint name"),
    Column("x", "Float32", True, None, "X coordinate (world units)"),
    Column("y", "Float32", True, None, "Y coordinate (world units)"),
    Column("z", "Float32", True, None, "Z coordinate (world units)"),
    Column("confidence", "Float32", True, 1.0, "Keypoint confidence [0-1]"),

    # Track-level metrics (replicated per keypoint)
    Column("scale", "Float32", False, 1.0, "Skeleton scale estimate"),
    Column("anatomical_score", "Float32", False, None, "Skeleton quality score"),
    Column("health", "Float32", False, 1.0, "Tracklet health [0-1]"),
    Column("integrity", "Float32", False, None, "Anatomical integrity score"),
    Column("uncertainty", "Float32", False, None, "Position uncertainty (covariance trace)"),
    Column("velocity_x", "Float32", False, None, "Velocity X component"),
    Column("velocity_y", "Float32", False, None, "Velocity Y component"),
    Column("velocity_z", "Float32", False, None, "Velocity Z component"),
]

SCHEMAS = {
    'Points2D': POINTS_2D,
    'Points3D': POINTS_3D,
    'Tracks3D': TRACKS_3D,
}

_SCHEMA_REQUIRED: dict[str, Set[str]] = {
    name: {c.name for c in cols if c.required}
    for name, cols in SCHEMAS.items()
}

_SCHEMA_OPTIONAL: dict[str, Set[str]] = {
    name: {c.name for c in cols if not c.required}
    for name, cols in SCHEMAS.items()
}

_RAY_COLUMNS = {"ray_dx", "ray_dy", "ray_dz"}   # ray columns must be all present or all absent


# Validation

class SchemaValidationError(Exception):
    """Raised when data does not conform to schema."""
    pass


def validate_dataframe(dataframe: pl.DataFrame, schema_name: str) -> List[str]:
    """
    Validate DataFrame against a schema.
    """
    # TODO: Add optional removal of unknown columns

    if schema_name not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema_name}")

    columns = set(dataframe.columns)
    required = _SCHEMA_REQUIRED[schema_name]
    optional = _SCHEMA_OPTIONAL[schema_name]

    missing = required - columns
    if missing:
        raise SchemaValidationError(f"{schema_name}: Missing required columns: {sorted(missing)}")

    warnings = []
    unknown = columns - required - optional
    if unknown:
        warnings.append(f"{schema_name}: Unknown columns (will be preserved): {sorted(unknown)}")

    if schema_name == 'Points3D':
        ray_present = _RAY_COLUMNS & columns

        if ray_present and ray_present != _RAY_COLUMNS:
            missing_ray = _RAY_COLUMNS - ray_present

            raise SchemaValidationError(
                f"Points3D: Partial ray columns."
                f" Present: {sorted(ray_present)},"
                f" Missing: {sorted(missing_ray)}."
                f" Must have all or none."
            )

    return warnings


def add_optional_columns(dataframe: pl.DataFrame, schema_name: str) -> pl.DataFrame:
    """
    Add missing optional columns with their default values.
    """
    if schema_name not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema_name}")

    existing = set(dataframe.columns)

    for col in SCHEMAS[schema_name]:
        if col.name not in existing and not col.required:
            dataframe = dataframe.with_columns(pl.lit(col.default).cast(col.polars_dtype).alias(col.name))
    return dataframe


def detect_schema(dataframe: pl.DataFrame) -> str | None:
    """
    Automatically identify the schema of a DataFrame based on its columns.

    Returns:
        The name of the schema ('Points2D', 'Points3D', 'Tracks3D') or None.
    """
    df_cols = set(dataframe.columns)
    for schema_name, required_cols in _SCHEMA_REQUIRED.items():
        if required_cols.issubset(df_cols):
            return schema_name
    return None


def empty_dataframe(schema_name: str) -> pl.DataFrame:
    """
    Create an empty DataFrame with the correct schema.
    """
    if schema_name not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema_name}")

    return pl.DataFrame(schema={c.name: c.polars_dtype for c in SCHEMAS[schema_name]})


def get_columns(schema_name: str, required_only: bool = False) -> List[str]:
    """
    Get column names for a schema.
    """
    if schema_name not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema_name}")

    if required_only:
        return [c.name for c in SCHEMAS[schema_name] if c.required]
    return [c.name for c in SCHEMAS[schema_name]]