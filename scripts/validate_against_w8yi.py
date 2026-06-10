#!/usr/bin/env python
"""Validate a locally extracted UNI2-h H5 against a W8Yi reference H5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Tuple


def squeeze_coords(arr):
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape [N, 2], got {arr.shape}")
    return arr.astype("int64", copy=False)


def feature_dataset(h5):
    if "features" not in h5:
        raise KeyError("H5 file does not contain a 'features' dataset")
    ds = h5["features"]
    if ds.ndim == 3 and ds.shape[0] == 1:
        return ds, True
    if ds.ndim == 2:
        return ds, False
    raise ValueError(f"Expected features with shape [1, N, D] or [N, D], got {ds.shape}")


def read_feature_rows(ds, has_leading_axis: bool, indices):
    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    if has_leading_axis:
        data = ds[0, sorted_idx, :]
    else:
        data = ds[sorted_idx, :]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return data[inverse]


def quantiles(values):
    import numpy as np

    values = np.asarray(values)
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compare(reference_h5: Path, candidate_h5: Path, max_pairs: int, seed: int):
    import h5py
    import numpy as np

    with h5py.File(reference_h5, "r") as ref, h5py.File(candidate_h5, "r") as cand:
        ref_coords = squeeze_coords(ref["coords"][:])
        cand_coords = squeeze_coords(cand["coords"][:])

        ref_set = {tuple(x) for x in ref_coords.tolist()}
        cand_set = {tuple(x) for x in cand_coords.tolist()}
        common = sorted(ref_set & cand_set)
        ref_only = sorted(ref_set - cand_set)
        cand_only = sorted(cand_set - ref_set)

        coord_order_exact = (
            ref_coords.shape == cand_coords.shape and bool(np.all(ref_coords == cand_coords))
        )

        if common:
            if max_pairs and len(common) > max_pairs:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(len(common), size=max_pairs, replace=False)
                common_for_features = [common[i] for i in sorted(chosen.tolist())]
            else:
                common_for_features = common
        else:
            common_for_features = []

        ref_index = {tuple(coord): i for i, coord in enumerate(ref_coords.tolist())}
        cand_index = {tuple(coord): i for i, coord in enumerate(cand_coords.tolist())}
        ref_idx = np.asarray([ref_index[c] for c in common_for_features], dtype=np.int64)
        cand_idx = np.asarray([cand_index[c] for c in common_for_features], dtype=np.int64)

        metrics = {
            "reference_h5": str(reference_h5),
            "candidate_h5": str(candidate_h5),
            "reference_tiles": int(ref_coords.shape[0]),
            "candidate_tiles": int(cand_coords.shape[0]),
            "common_tiles": int(len(common)),
            "reference_only_tiles": int(len(ref_only)),
            "candidate_only_tiles": int(len(cand_only)),
            "common_over_reference": float(len(common) / ref_coords.shape[0])
            if ref_coords.shape[0]
            else None,
            "common_over_candidate": float(len(common) / cand_coords.shape[0])
            if cand_coords.shape[0]
            else None,
            "coord_order_exact": coord_order_exact,
            "feature_pairs_compared": int(len(common_for_features)),
        }

        if common_for_features:
            ref_ds, ref_leading = feature_dataset(ref)
            cand_ds, cand_leading = feature_dataset(cand)
            ref_feat = read_feature_rows(ref_ds, ref_leading, ref_idx).astype("float32")
            cand_feat = read_feature_rows(cand_ds, cand_leading, cand_idx).astype("float32")
            if ref_feat.shape != cand_feat.shape:
                raise ValueError(f"Feature shapes differ after matching: {ref_feat.shape} vs {cand_feat.shape}")
            dot = np.sum(ref_feat * cand_feat, axis=1)
            denom = np.linalg.norm(ref_feat, axis=1) * np.linalg.norm(cand_feat, axis=1)
            cosine = dot / np.maximum(denom, 1e-12)
            l2 = np.linalg.norm(ref_feat - cand_feat, axis=1)
            max_abs = np.max(np.abs(ref_feat - cand_feat), axis=1)
            metrics["cosine_similarity"] = quantiles(cosine)
            metrics["l2_distance"] = quantiles(l2)
            metrics["max_abs_difference"] = quantiles(max_abs)

        diff_rows = []
        for coord in ref_only[:10000]:
            diff_rows.append({"coord_x": coord[0], "coord_y": coord[1], "status": "reference_only"})
        for coord in cand_only[:10000]:
            diff_rows.append({"coord_x": coord[0], "coord_y": coord[1], "status": "candidate_only"})

    return metrics, diff_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-h5", type=Path, required=True, help="W8Yi reference H5")
    parser.add_argument("--candidate-h5", type=Path, required=True, help="Locally extracted H5")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--diff-csv", type=Path, default=None, help="Optional coordinate diff CSV")
    parser.add_argument("--max-feature-pairs", type=int, default=20000, help="Sample cap for feature comparisons")
    parser.add_argument("--seed", type=int, default=20260609, help="Sampling seed")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    metrics, diff_rows = compare(
        args.reference_h5.resolve(),
        args.candidate_h5.resolve(),
        args.max_feature_pairs,
        args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.diff_csv:
        write_csv(args.diff_csv.resolve(), diff_rows)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
