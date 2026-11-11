"""Report spatial alignment diagnostics for the rotated high-side smoke prior."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2022_1km_hotspot_oracle.csv")
units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
data = knowledge.merge(
    units[["cell_id", "y1_log1p", "y2"]], on="cell_id", how="inner", validate="one_to_one",
)
columns = {
    "aligned": "pi_bnd_highside_adjacent_2022",
    "rotated180": "pi_bnd_highside_rotated180_2022",
}
report = {
    "support_rows": int(len(knowledge)),
    "diagnostic_rows_with_2022_response": int(len(data)),
    "rotation": "pi_rot(x,y) = nearest_valid_grid_value(pi_aligned, (1-x,1-y))",
    "fields": {},
}
for name, column in columns.items():
    prior = data[column].to_numpy(dtype=float)
    top_n = int(np.ceil(0.2 * len(data)))
    top = data.iloc[np.argsort(-prior, kind="stable")[:top_n]]
    report["fields"][name] = {
        "min": float(prior.min()),
        "max": float(prior.max()),
        "nonzero_rows": int(np.count_nonzero(prior > 0)),
        "pearson_y1_log1p": float(pearsonr(prior, data["y1_log1p"]).statistic),
        "spearman_y1_log1p": float(spearmanr(prior, data["y1_log1p"]).statistic),
        "pearson_y2": float(pearsonr(prior, data["y2"]).statistic),
        "spearman_y2": float(spearmanr(prior, data["y2"]).statistic),
        "top20_rule": "Stable descending sort; ceil(0.2 * 649) = 130 rows",
        "top20_mean_y1_log1p": float(top["y1_log1p"].mean()),
        "top20_mean_y2": float(top["y2"].mean()),
    }
report["all_649_mean_y1_log1p"] = float(data["y1_log1p"].mean())
report["all_649_mean_y2"] = float(data["y2"].mean())
report["aligned_vs_rotated180_pearson"] = float(pearsonr(
    data[columns["aligned"]], data[columns["rotated180"]],
).statistic)
output = ROOT / "results" / "aggregated" / "hotspot_misleading2022_highside_rotated180_3seeds" / "prior_diagnostics.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
