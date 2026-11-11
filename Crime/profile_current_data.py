from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
VIOLENT = {"HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ROBBERY", "BATTERY", "ASSAULT"}
BBOX = (41.60, 42.05, -87.95, -87.52)


def load_year(year: int) -> pd.DataFrame:
    path = RAW / f"Crimes_-_{year}_20231019.tsv"
    df = pd.read_csv(path, sep="\t", usecols=["Primary Type", "Latitude", "Longitude"], low_memory=False)
    raw_n = len(df)
    coord_ok = df[["Latitude", "Longitude"]].notna().all(axis=1)
    lat_ok = df["Latitude"].between(BBOX[0], BBOX[1])
    lon_ok = df["Longitude"].between(BBOX[2], BBOX[3])
    cleaned = df.loc[coord_ok & lat_ok & lon_ok].copy()
    cleaned["is_violent"] = cleaned["Primary Type"].isin(VIOLENT)
    lat0, lon0 = BBOX[0], BBOX[2]
    km_per_lat = 111.32
    km_per_lon = 111.32 * np.cos(np.deg2rad(cleaned["Latitude"].mean()))
    cleaned["iy"] = np.floor((cleaned["Latitude"] - lat0) * km_per_lat).astype(int)
    cleaned["ix"] = np.floor((cleaned["Longitude"] - lon0) * km_per_lon).astype(int)
    stats = {
        "raw_rows": int(raw_n),
        "valid_rows": int(len(cleaned)),
        "removed_rows": int(raw_n - len(cleaned)),
        "coordinate_missing_rows": int((~coord_ok).sum()),
        "out_of_bbox_among_coordinate_present": int((coord_ok & ~(lat_ok & lon_ok)).sum()),
        "violent_rows": int(cleaned["is_violent"].sum()),
        "violent_share": float(cleaned["is_violent"].mean()),
        "bbox_used_for_grid_origin": [float(lat0), float(lon0)],
    }
    return cleaned, stats


profiles = {}
units_by_year = {}
for year in (2021, 2022):
    incidents, stats = load_year(year)
    units = incidents.groupby(["ix", "iy"]).agg(
        y1=("Primary Type", "size"),
        violent=("is_violent", "sum"),
    ).reset_index()
    units["nonviolent"] = units["y1"] - units["violent"]
    units["y2"] = (units["violent"] > units["nonviolent"]).astype(int)
    units["key_q3"] = (units["y1"] > units["y1"].quantile(0.75)).astype(int)
    q3 = float(units["y1"].quantile(0.75))
    units["violent_ratio"] = units["violent"] / units["y1"]
    stats.update({
        "occupied_1km_cells": int(len(units)),
        "y1_min_median_q3_max": [int(units["y1"].min()), float(units["y1"].median()), q3, int(units["y1"].max())],
        "y1_q3_strict_key_cells": int(units["key_q3"].sum()),
        "y1_q3_nonstrict_key_cells": int((units["y1"] >= q3).sum()),
        "y2_positive_cells": int(units["y2"].sum()),
        "y2_positive_share": float(units["y2"].mean()),
        "alternative_label_positive_shares": {
            "violent_ratio_above_same_year_incident_rate": float((units["violent_ratio"] > stats["violent_share"]).mean()),
            "violent_ratio_above_same_year_cell_q3": float((units["violent_ratio"] > units["violent_ratio"].quantile(0.75)).mean()),
            "at_least_one_violent_incident": float((units["violent"] >= 1).mean()),
        },
        "zero_incident_cells_in_annual_occupied_table": 0,
    })
    profiles[str(year)] = stats
    units_by_year[year] = units

u21 = units_by_year[2021][["ix", "iy", "y1", "y2", "key_q3", "violent_ratio"]].rename(columns={"y1":"y1_2021", "y2":"y2_2021", "key_q3":"key_2021", "violent_ratio":"ratio_2021"})
u22 = units_by_year[2022][["ix", "iy", "y1", "y2", "key_q3", "violent_ratio"]].rename(columns={"y1":"y1_2022", "y2":"y2_2022", "key_q3":"key_2022", "violent_ratio":"ratio_2022"})
common = u21.merge(u22, on=["ix", "iy"], how="inner")
p0 = profiles["2021"]["violent_share"]
profiles["cross_year"] = {
    "common_cells": int(len(common)),
    "y1_spearman": float(common[["y1_2021", "y1_2022"]].corr(method="spearman").iloc[0, 1]),
    "y1_pearson_log1p": float(np.corrcoef(np.log1p(common["y1_2021"]), np.log1p(common["y1_2022"]))[0, 1]),
    "q3_key_recall_2022_from_2021": float(common.loc[common["key_2021"] == 1, "key_2022"].mean()),
    "q3_key_precision_2022_from_2021": float(common.loc[common["key_2022"] == 1, "key_2021"].mean()),
    "q3_key_overlap": int(((common["key_2021"] == 1) & (common["key_2022"] == 1)).sum()),
    "y2_agreement": float((common["y2_2021"] == common["y2_2022"]).mean()),
    "fixed_2021_incident_rate_threshold": p0,
    "label_2022_positive_share_above_fixed_2021_rate": float((units_by_year[2022]["violent_ratio"] > p0).mean()),
    "fixed_rate_label_year_agreement": float(((common["ratio_2021"] > p0) == (common["ratio_2022"] > p0)).mean()),
}

out = ROOT / "current_data_profile.json"
out.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(profiles, ensure_ascii=False, indent=2))
