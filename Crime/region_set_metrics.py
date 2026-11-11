"""Conditional Top-K box evaluation and grid-rank precision."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from region_recommender import rectangle_counts, rectangles_overlap
from spatial_validity import GridBoxValidity


def conditional_matched_references(boxes, n_reference, seed, box_valid_fn):
    """Generate same-size valid references; rank k avoids boxes 0 through k-1."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0 or n_reference <= 0:
        raise ValueError("boxes must be nonempty and n_reference must be positive")
    if not np.all(box_valid_fn(boxes)):
        raise ValueError("recommended boxes must lie entirely within the valid grid")
    rng = np.random.default_rng(seed)
    references = np.empty((len(boxes), n_reference, 4), dtype=float)
    references[:, :, 2:] = boxes[:, None, 2:]
    for index, box in enumerate(boxes):
        accepted = []
        n_accepted = 0
        attempts = 0
        while n_accepted < n_reference:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("could not generate enough conditional matched references")
            uniforms = rng.random((n_reference, 2))
            candidates = np.empty((n_reference, 4), dtype=float)
            candidates[:, 2:] = box[2:]
            candidates[:, :2] = box[2:] / 2 + uniforms * (1 - box[2:])
            valid = np.asarray(box_valid_fn(candidates), dtype=bool)
            for previous in boxes[:index]:
                valid &= ~rectangles_overlap(
                    candidates, np.repeat(previous[None, :], len(candidates), axis=0),
                )
            candidates = candidates[valid]
            accepted.append(candidates)
            n_accepted += len(candidates)
        references[index, :, :2] = np.concatenate(accepted, axis=0)[:n_reference, :2]
    return references


def conditional_region_metrics(
    boxes, incidents_xy, domain_width_km, n_reference, seed, box_valid_fn,
):
    """Evaluate each box against same-size references in its remaining feasible space."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    widths = np.asarray(domain_width_km, dtype=float)
    if widths.shape != (2,) or np.any(widths <= 0):
        raise ValueError("domain_width_km must contain two positive values")
    references = conditional_matched_references(boxes, n_reference, seed, box_valid_fn)
    reference_counts = rectangle_counts(
        incidents_xy, references.reshape(-1, 4),
    ).reshape(len(boxes), n_reference)
    counts = rectangle_counts(incidents_xy, boxes)
    reference_q3 = np.quantile(reference_counts, 0.75, axis=1, method="linear")
    pass_flags = counts > reference_q3
    return {
        "conditional_region_pass": float(np.mean(pass_flags)),
        "conditional_reference_q3": reference_q3.tolist(),
        "conditional_pass_flags": pass_flags.astype(int).tolist(),
        "conditional_reference_boxes": int(len(boxes) * n_reference),
    }


def topn_grid_precision(boxes, units, domain_width_km, cell_area_km2):
    """Fraction of covered cells whose average count rank is within area-equivalent top n."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        raise ValueError("at least one recommended box is required")
    validity = GridBoxValidity(units)
    if not validity(boxes).all():
        raise ValueError("recommended boxes must lie entirely within the valid grid")
    widths = np.asarray(domain_width_km, dtype=float)
    if widths.shape != (2,) or np.any(widths <= 0) or cell_area_km2 <= 0:
        raise ValueError("domain widths and cell area must be positive")
    total_area = float(np.sum(
        boxes[:, 2] * widths[0] * boxes[:, 3] * widths[1]
    ))
    n = int(np.rint(total_area / float(cell_area_km2)))
    n = min(max(n, 1), len(units))

    row_lookup = {
        (int(ix), intiy): row
        for row, (ix, intiy) in enumerate(zip(validity.ix, validity.iy))
    }
    covered = np.zeros(len(units), dtype=bool)
    for cx, cy, lx, ly in boxes:
        xs = validity._overlap_indices(cx - lx / 2, cx + lx / 2, validity.x0, validity.dx)
        ys = validity._overlap_indices(cy - ly / 2, cy + ly / 2, validity.y0, validity.dy)
        for ix in xs:
            for iy in ys:
                row = row_lookup.get((ix, iy))
                if row is not None:
                    covered[row] = True
    if not covered.any():
        raise RuntimeError("recommended boxes cover no valid grid cells")
    counts = np.asarray(units["y1"], dtype=float)
    if counts.shape != (len(units),) or not np.isfinite(counts).all():
        raise ValueError("units must contain one finite y1 count per cell")
    ranks = rankdata(-counts, method="average")
    hits = covered & (ranks <= n)
    return {
        "area_equivalent_n": n,
        "covered_grid_count": int(covered.sum()),
        "topn_hit_count": int(hits.sum()),
        "topn_grid_precision": float(hits.sum() / covered.sum()),
    }
