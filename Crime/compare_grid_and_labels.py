from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VIOLENT = {"HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ROBBERY", "BATTERY", "ASSAULT"}
BBOX = (41.60, 42.05, -87.95, -87.52)


def load(year):
    p = ROOT / "data" / "raw" / f"Crimes_-_{year}_20231019.tsv"
    d = pd.read_csv(p, sep="\t", usecols=["Primary Type", "Latitude", "Longitude"], low_memory=False)
    d = d[d[["Latitude", "Longitude"]].notna().all(axis=1)]
    d = d[d["Latitude"].between(BBOX[0], BBOX[1]) & d["Longitude"].between(BBOX[2], BBOX[3])].copy()
    d["v"] = d["Primary Type"].isin(VIOLENT).astype(int)
    return d


def aggregate(d, size, lon_scale):
    d = d.copy()
    d["ix"] = np.floor((d["Longitude"] - BBOX[2]) * lon_scale / size).astype(int)
    d["iy"] = np.floor((d["Latitude"] - BBOX[0]) * 111.32 / size).astype(int)
    u = d.groupby(["ix", "iy"]).agg(y1=("v", "size"), violent=("v", "sum")).reset_index()
    u["ratio"] = u["violent"] / u["y1"]
    return u


data = {y: load(y) for y in (2021, 2022)}
p_city = data[2021]["v"].mean()
lon_scale = 111.32 * np.cos(np.deg2rad(data[2021]["Latitude"].mean()))
for size in (0.5, 1.0, 2.0):
    u = {y: aggregate(data[y], size, lon_scale) for y in (2021, 2022)}
    p_median = u[2021]["ratio"].median()
    p_q3 = u[2021]["ratio"].quantile(0.75)
    print(f"\nGRID={size} km; 2021 city rate={p_city:.6f}; 2021 cell median={p_median:.6f}; 2021 cell Q3={p_q3:.6f}")
    for y in (2021, 2022):
        a1 = (u[y]["violent"] > u[y]["y1"] - u[y]["violent"]).mean()
        print(y, {
            "cells": len(u[y]),
            "median_count": float(u[y]["y1"].median()),
            "cells_lt_10": int((u[y]["y1"] < 10).sum()),
            "cells_lt_10_share": round(float((u[y]["y1"] < 10).mean()), 4),
            "A1_positive": round(float(a1), 4),
            "above_2021_city_rate": round(float((u[y]["ratio"] > p_city).mean()), 4),
            "above_2021_cell_median": round(float((u[y]["ratio"] > p_median).mean()), 4),
            "above_2021_cell_Q3": round(float((u[y]["ratio"] > p_q3).mean()), 4),
        })
