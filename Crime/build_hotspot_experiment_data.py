"""Build isolated hotspot-response experiment artifacts; originals stay unchanged."""
from pathlib import Path

import pandas as pd

from analyze_hotspot_y2_candidate import four_neighbor_boundary

ROOT = Path(__file__).resolve().parent


def build_hotspot_units(units: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = units.copy()
    result["y2_violent_original"] = result["y2"]
    result["is_violent_boundary_original"] = result["is_classification_boundary"]
    result["y2"] = (result["y1"] > float(threshold)).astype("int8")
    result["y2_threshold"] = float(threshold)
    result["is_classification_boundary"] = four_neighbor_boundary(result, "y2")
    return result


def build_hotspot_knowledge(knowledge: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    result = knowledge.copy()
    selected = prior[["cell_id", "pi_bnd_hot"]]
    result = result.merge(selected, on="cell_id", how="left", validate="one_to_one")
    if result["pi_bnd_hot"].isna().any():
        raise ValueError("hotspot boundary prior does not cover the knowledge grid")
    result["pi_bnd_violent_original"] = result["pi_bnd"]
    result["pi_bnd"] = result["pi_bnd_hot"]
    return result


def main() -> None:
    threshold = 445.0
    source_units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    source_knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    prior = pd.read_csv(ROOT / "results" / "hotspot_y2_precheck_1km" / "boundary_prior_2021.csv")
    assert float(source_units["y1"].quantile(0.75)) == 504.0
    source_2021 = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    assert float(source_2021["y1"].quantile(0.75)) == threshold

    units = build_hotspot_units(source_units, threshold)
    knowledge = build_hotspot_knowledge(source_knowledge, prior)
    units_path = ROOT / "data" / "units" / "units_2022_1km_hotspot.csv"
    knowledge_path = ROOT / "data" / "knowledge" / "knowledge_2021_1km_hotspot.csv"
    units.to_csv(units_path, index=False)
    knowledge.to_csv(knowledge_path, index=False)
    print(f"UNITS={units_path}")
    print(f"KNOWLEDGE={knowledge_path}")
    print(f"POSITIVE={int(units['y2'].sum())}/{len(units)}")
    print(f"BOUNDARY={int(units['is_classification_boundary'].sum())}/{len(units)}")


if __name__ == "__main__":
    main()
