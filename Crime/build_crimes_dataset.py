"""Build the Crimes mixed-response acquisition benchmark.

The target-year table is an offline replay oracle. The sampling algorithm must only
read rows selected by its acquisition policy. Knowledge artifacts use 2021 only.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.vq import kmeans2
from scipy.io import savemat
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config" / "construct.yaml").read_text(encoding="utf-8"))
FOR_DIR = ROOT / "data" / "processed"
UNIT_DIR = ROOT / "data" / "units"
GT_DIR = ROOT / "data" / "groundtruth"
KNOW_DIR = ROOT / "data" / "knowledge"
FIG_DIR = ROOT / "figs"
for directory in (FOR_DIR, UNIT_DIR, GT_DIR, KNOW_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RAW_COLUMNS = [
    "ID", "Case Number", "Date", "Primary Type", "Description", "Arrest",
    "Domestic", "Beat", "District", "Ward", "Community Area", "FBI Code",
    "Latitude", "Longitude", "Year",
]
RENAME = {
    "ID": "id", "Case Number": "case_number", "Date": "date",
    "Primary Type": "primary_type", "Description": "description",
    "Arrest": "arrest", "Domestic": "domestic", "Beat": "beat",
    "District": "district", "Ward": "ward", "Community Area": "community_area",
    "FBI Code": "fbi_code", "Latitude": "latitude", "Longitude": "longitude",
    "Year": "year",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mode_value(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def load_clean(year: int) -> tuple[pd.DataFrame, dict]:
    relative = CONFIG["raw"][str(year)] if str(year) in CONFIG["raw"] else CONFIG["raw"][year]
    path = ROOT / relative
    frame = pd.read_csv(path, sep="\t", usecols=RAW_COLUMNS, low_memory=False)
    raw_n = len(frame)
    frame = frame.rename(columns=RENAME)
    coordinate_present = frame[["latitude", "longitude"]].notna().all(axis=1)
    clean = CONFIG["cleaning"]
    in_bbox = (
        frame["latitude"].between(clean["latitude_min"], clean["latitude_max"])
        & frame["longitude"].between(clean["longitude_min"], clean["longitude_max"])
    )
    frame = frame.loc[coordinate_present & in_bbox].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y %I:%M:%S %p")
    frame["month"] = frame["date"].dt.month.astype(np.int8)
    frame["day"] = frame["date"].dt.day.astype(np.int8)
    frame["hour"] = frame["date"].dt.hour.astype(np.int8)
    frame["weekday"] = frame["date"].dt.weekday.astype(np.int8)
    frame["is_night"] = frame["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    frame["arrest"] = frame["arrest"].astype(bool)
    frame["domestic"] = frame["domestic"].astype(bool)
    violent = set(CONFIG["violent_primary_types"])
    frame["is_violent"] = frame["primary_type"].isin(violent)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = FOR_DIR / f"incidents_{year}_clean.csv.gz"
    frame.to_csv(out, index=False, compression="gzip")
    stats = {
        "raw_file": str(path.relative_to(ROOT)),
        "raw_sha256": sha256(path),
        "raw_rows": int(raw_n),
        "valid_rows": int(len(frame)),
        "removed_rows": int(raw_n - len(frame)),
        "coordinate_missing_rows": int((~coordinate_present).sum()),
        "out_of_bbox_with_coordinates": int((coordinate_present & ~in_bbox).sum()),
        "violent_rows": int(frame["is_violent"].sum()),
        "violent_share": float(frame["is_violent"].mean()),
        "processed_file": str(out.relative_to(ROOT)),
    }
    return frame, stats


def grid_coordinates(frame: pd.DataFrame, size: float) -> pd.DataFrame:
    grid = CONFIG["grid"]
    km_lat = float(grid["km_per_degree_latitude"])
    km_lon = km_lat * np.cos(np.deg2rad(float(grid["projection_reference_latitude"])))
    frame = frame.copy()
    frame["x_km"] = (frame["longitude"] - float(grid["origin_longitude"])) * km_lon
    frame["y_km"] = (frame["latitude"] - float(grid["origin_latitude"])) * km_lat
    frame["ix"] = np.floor(frame["x_km"] / size).astype(np.int32)
    frame["iy"] = np.floor(frame["y_km"] / size).astype(np.int32)
    return frame


def aggregate_units(frame: pd.DataFrame, year: int, size: float) -> pd.DataFrame:
    located = grid_coordinates(frame, size)
    keys = ["ix", "iy"]
    units = located.groupby(keys, as_index=False).agg(
        y1=("primary_type", "size"),
        violent_count=("is_violent", "sum"),
        arrest_count=("arrest", "sum"),
        night_count=("is_night", "sum"),
        incident_mean_latitude=("latitude", "mean"),
        incident_mean_longitude=("longitude", "mean"),
    )
    metadata = located.groupby(keys, as_index=False).agg(
        district=("district", mode_value),
        community_area=("community_area", mode_value),
    )
    units = units.merge(metadata, on=keys, how="left", validate="one_to_one")
    grid = CONFIG["grid"]
    km_lat = float(grid["km_per_degree_latitude"])
    km_lon = km_lat * np.cos(np.deg2rad(float(grid["projection_reference_latitude"])))
    units["cell_lat"] = float(grid["origin_latitude"]) + (units["iy"] + 0.5) * size / km_lat
    units["cell_lon"] = float(grid["origin_longitude"]) + (units["ix"] + 0.5) * size / km_lon
    clean = CONFIG["cleaning"]
    units["x"] = (units["cell_lon"] - clean["longitude_min"]) / (clean["longitude_max"] - clean["longitude_min"])
    units["y"] = (units["cell_lat"] - clean["latitude_min"]) / (clean["latitude_max"] - clean["latitude_min"])
    units["year"] = year
    units["grid_size_km"] = size
    tag = f"{int(round(size * 1000)):04d}"
    units["cell_id"] = "g" + tag + "_x" + units["ix"].astype(str) + "_y" + units["iy"].astype(str)
    units["violent_count"] = units["violent_count"].astype(np.int32)
    units["nonviolent_count"] = units["y1"] - units["violent_count"]
    units["violent_ratio"] = units["violent_count"] / units["y1"]
    units["y1_log1p"] = np.log1p(units["y1"])
    units["y2_a1_legacy"] = (units["violent_count"] > units["nonviolent_count"]).astype(np.int8)
    return units.sort_values(["iy", "ix"]).reset_index(drop=True)


def add_labels_and_truth(units: dict[int, pd.DataFrame], size: float, city_prior: float) -> dict:
    source = units[2021]
    target = units[2022]
    threshold = float(source["violent_ratio"].median())
    for year, frame in units.items():
        frame["y2"] = (frame["violent_ratio"] > threshold).astype(np.int8)
        frame["y2_threshold"] = threshold
        frame["y2_city_rate_legacy"] = (frame["violent_ratio"] > city_prior).astype(np.int8)
        q3 = float(frame["y1"].quantile(0.75))
        frame["y1_q3"] = q3
        frame["is_key_q3"] = (frame["y1"] > q3).astype(np.int8)
        for kappa in CONFIG["responses"]["smoothing"]["kappas"]:
            col = f"violent_ratio_smooth_k{kappa}"
            frame[col] = (frame["violent_count"] + kappa * city_prior) / (frame["y1"] + kappa)
    for kappa in CONFIG["responses"]["smoothing"]["kappas"]:
        col = f"violent_ratio_smooth_k{kappa}"
        smooth_threshold = float(source[col].median())
        for frame in units.values():
            frame[f"y2_smooth_k{kappa}"] = (frame[col] > smooth_threshold).astype(np.int8)
            frame[f"y2_smooth_k{kappa}_threshold"] = smooth_threshold

    lookup = target.set_index(["ix", "iy"])
    boundary = np.zeros(len(target), dtype=np.int8)
    component = np.full(len(target), -1, dtype=np.int32)
    for row_idx, row in enumerate(target.itertuples()):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            key = (row.ix + dx, row.iy + dy)
            if key in lookup.index and int(lookup.loc[key, "y2"]) != int(row.y2):
                boundary[row_idx] = 1
                break
    target["is_classification_boundary"] = boundary

    key_lookup = {(r.ix, r.iy): i for i, r in enumerate(target.itertuples()) if r.is_key_q3 == 1}
    visited = set()
    component_id = 0
    for start in key_lookup:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            component[key_lookup[current]] = component_id
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in key_lookup and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_id += 1
    target["key_component_id"] = component

    xy = target[["x", "y"]].to_numpy()
    _, labels = kmeans2(xy, CONFIG["spatial_blocks"]["n_blocks"], minit="++", seed=CONFIG["spatial_blocks"]["seed"])
    centroids = np.vstack([xy[labels == k].mean(axis=0) for k in range(CONFIG["spatial_blocks"]["n_blocks"])])
    target["spatial_block"] = labels.astype(np.int8)
    source_xy = source[["x", "y"]].to_numpy()
    _, source_labels = cKDTree(centroids).query(source_xy)
    source["spatial_block"] = source_labels.astype(np.int8)
    return {
        "classification_threshold": threshold,
        "high_crime_q3_2021": float(source["y1_q3"].iloc[0]),
        "high_crime_q3_2022": float(target["y1_q3"].iloc[0]),
        "high_crime_components_2022": int(component_id),
        "spatial_block_sizes_2022": {str(k): int((labels == k).sum()) for k in range(CONFIG["spatial_blocks"]["n_blocks"])},
    }


def nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def build_knowledge(source: pd.DataFrame, size: float) -> tuple[pd.DataFrame, dict, dict]:
    ix_values = np.arange(source["ix"].min(), source["ix"].max() + 1)
    iy_values = np.arange(source["iy"].min(), source["iy"].max() + 1)
    nx, ny = len(ix_values), len(iy_values)
    valid = np.zeros((ny, nx), dtype=bool)
    count = np.full((ny, nx), np.nan)
    ratio = np.full((ny, nx), np.nan)
    x_index = source["ix"].to_numpy() - ix_values.min()
    y_index = source["iy"].to_numpy() - iy_values.min()
    valid[y_index, x_index] = True
    count[y_index, x_index] = source["y1_log1p"]
    ratio[y_index, x_index] = source["violent_ratio"]

    sigma = float(CONFIG["knowledge"]["pi_hg"]["gaussian_sigma_cells"])
    numerator = gaussian_filter(np.nan_to_num(count) * valid, sigma=sigma)
    denominator = gaussian_filter(valid.astype(float), sigma=sigma)
    smooth = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    grad_y, grad_x = np.gradient(smooth, size, size)
    gradient = np.hypot(grad_x, grad_y)
    quantile = float(CONFIG["knowledge"]["pi_hg"]["normalization_quantile"])
    scale = float(np.quantile(gradient[valid], quantile))
    floor = float(CONFIG["knowledge"]["pi_hg"]["floor"])
    pi_hg_grid = np.clip(gradient / max(scale, 1e-12), floor, 1.0)

    threshold = float(source["violent_ratio"].median())
    mad = float(np.median(np.abs(source["violent_ratio"] - threshold)))
    tau = max(1.4826 * mad, 0.05)
    pi_bnd_grid = np.exp(-((ratio - threshold) ** 2) / (2 * tau ** 2))
    pi_bnd_grid = np.clip(pi_bnd_grid, floor, 1.0)
    pi_hg_filled = nearest_fill(pi_hg_grid, valid)
    pi_bnd_filled = nearest_fill(pi_bnd_grid, valid)

    knowledge = source[["cell_id", "ix", "iy", "cell_lat", "cell_lon", "x", "y"]].copy()
    knowledge["pi_hg"] = pi_hg_grid[y_index, x_index]
    knowledge["pi_bnd"] = pi_bnd_grid[y_index, x_index]
    knowledge["pi_fused_unweighted"] = np.maximum(knowledge["pi_hg"], knowledge["pi_bnd"])
    grid_data = {
        "ix": ix_values, "iy": iy_values, "valid_mask": valid,
        "log1p_count_smoothed": smooth, "pi_hg": pi_hg_filled,
        "pi_bnd": pi_bnd_filled, "grid_size_km": size,
        "classification_threshold": threshold, "tau": tau,
    }
    stats = {
        "gaussian_sigma_cells": sigma,
        "gradient_normalizer_q95": scale,
        "classification_threshold": threshold,
        "tau": tau,
        "pi_hg_mean": float(knowledge["pi_hg"].mean()),
        "pi_bnd_mean": float(knowledge["pi_bnd"].mean()),
        "pi_hg_nonflat_fraction": float((knowledge["pi_hg"] > 0.05).mean()),
        "pi_bnd_nonflat_fraction": float((knowledge["pi_bnd"] > 0.05).mean()),
    }
    return knowledge, grid_data, stats


def mat_frame(frame: pd.DataFrame) -> dict:
    result = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype == object:
            values = values.astype(str).astype(object)
        result[column] = values
    return result


def make_overview(units: dict[int, pd.DataFrame], knowledge: pd.DataFrame, size: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        (units[2021], "y1_log1p", "2021 log(1 + crime count)", "viridis"),
        (units[2022], "y1_log1p", "2022 log(1 + crime count)", "viridis"),
        (units[2022], "y2", "2022 high violent-ratio label", "coolwarm"),
        (units[2021], "violent_ratio", "2021 violent ratio", "magma"),
        (knowledge, "pi_hg", r"2021 $\pi_{hg}$: spatial difficulty", "cividis"),
        (knowledge, "pi_bnd", r"2021 $\pi_{bnd}$: class-boundary band", "cividis"),
    ]
    for ax, (frame, column, title, cmap) in zip(axes.flat, panels):
        scatter = ax.scatter(frame["cell_lon"], frame["cell_lat"], c=frame[column], s=18, marker="s", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(scatter, ax=ax, shrink=0.82)
    fig.suptitle(f"Crimes mixed-response data and 2021 knowledge ({size:g} km grid)")
    fig.savefig(FIG_DIR / f"data_knowledge_overview_{size:g}km.png", dpi=180)
    plt.close(fig)


def main() -> None:
    incidents, summary = {}, {"config": CONFIG, "years": {}, "scales": {}}
    for year in (2021, 2022):
        incidents[year], summary["years"][str(year)] = load_clean(year)
    city_prior = summary["years"]["2021"]["violent_share"]
    sizes = [float(CONFIG["grid"]["main_size_km"])] + [float(x) for x in CONFIG["grid"]["sensitivity_sizes_km"]]
    for size in sizes:
        units = {year: aggregate_units(incidents[year], year, size) for year in (2021, 2022)}
        scale_truth = add_labels_and_truth(units, size, city_prior)
        knowledge, grid_data, knowledge_stats = build_knowledge(units[2021], size)
        suffix = f"{size:g}km"
        paths = []
        for year, frame in units.items():
            path = UNIT_DIR / f"units_{year}_{suffix}.csv"
            frame.to_csv(path, index=False)
            paths.append(str(path.relative_to(ROOT)))
        gt_columns = [
            "cell_id", "ix", "iy", "cell_lat", "cell_lon", "x", "y", "y1",
            "y1_q3", "is_key_q3", "key_component_id", "y2", "y2_threshold",
            "is_classification_boundary", "spatial_block",
        ]
        gt_path = GT_DIR / f"groundtruth_2022_{suffix}.csv"
        units[2022][gt_columns].to_csv(gt_path, index=False)
        know_path = KNOW_DIR / f"knowledge_2021_{suffix}.csv"
        knowledge.to_csv(know_path, index=False)
        npz_path = KNOW_DIR / f"knowledge_grid_2021_{suffix}.npz"
        np.savez_compressed(npz_path, **grid_data)
        mat_path = UNIT_DIR / f"crimes_mixed_{suffix}.mat"
        savemat(mat_path, {
            "units_2021": mat_frame(units[2021]), "units_2022": mat_frame(units[2022]),
            "knowledge_2021": mat_frame(knowledge), "knowledge_grid": grid_data,
            "grid_size_km": size, "source_year": 2021, "target_year": 2022,
            "classification_threshold": scale_truth["classification_threshold"],
            "city_violent_rate_2021": city_prior,
        }, do_compression=True)
        if size == float(CONFIG["grid"]["main_size_km"]):
            make_overview(units, knowledge, size)

        scale_stats = {
            **scale_truth,
            "knowledge": knowledge_stats,
            "files": {
                "units": paths, "ground_truth": str(gt_path.relative_to(ROOT)),
                "knowledge": str(know_path.relative_to(ROOT)),
                "knowledge_grid": str(npz_path.relative_to(ROOT)),
                "matlab_bundle": str(mat_path.relative_to(ROOT)),
            },
        }
        for year, frame in units.items():
            scale_stats[str(year)] = {
                "occupied_cells": int(len(frame)),
                "y1_min_median_max": [int(frame["y1"].min()), float(frame["y1"].median()), int(frame["y1"].max())],
                "cells_below_10_incidents": int((frame["y1"] < 10).sum()),
                "q3_key_cells": int(frame["is_key_q3"].sum()),
                "y2_positive_cells": int(frame["y2"].sum()),
                "y2_positive_share": float(frame["y2"].mean()),
                "y2_a1_legacy_positive_share": float(frame["y2_a1_legacy"].mean()),
                "y2_city_rate_legacy_positive_share": float(frame["y2_city_rate_legacy"].mean()),
                "spatial_block_sizes": {str(k): int(v) for k, v in frame["spatial_block"].value_counts().sort_index().items()},
            }
            for kappa in CONFIG["responses"]["smoothing"]["kappas"]:
                scale_stats[str(year)][f"y2_smooth_k{kappa}_positive_share"] = float(frame[f"y2_smooth_k{kappa}"].mean())
        scale_stats["classification_boundary_cells_2022"] = int(units[2022]["is_classification_boundary"].sum())
        summary["scales"][suffix] = scale_stats

        for frame in units.values():
            assert not frame[["cell_id", "x", "y", "y1", "y2"]].isna().any().any()
            assert np.array_equal(frame["y1"], frame["violent_count"] + frame["nonviolent_count"])
            assert frame["cell_id"].is_unique
            assert 0.10 < frame["y2"].mean() < 0.90
        assert abs(units[2022]["is_key_q3"].mean() - 0.25) < 0.01
        assert knowledge[["pi_hg", "pi_bnd"]].notna().all().all()
        assert knowledge[["pi_hg", "pi_bnd"]].apply(lambda s: s.between(0, 1)).all().all()

    summary_path = ROOT / "data" / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary["scales"].items()}, ensure_ascii=False, indent=2))
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
