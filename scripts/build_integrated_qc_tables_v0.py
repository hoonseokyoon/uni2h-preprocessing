#!/usr/bin/env python3
"""Build v0 integrated WSI, RNA, and label QC tables.

This script consumes the source registry tables and writes metadata/pre-QC
tables. WSI image QC remains pending until the slide preprocessing pipeline
adds tissue tile and artifact metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "v0"
QC_RULE_VERSION = "integrated_qc_v0_2026-06-10"

TCGA_NEOADJUVANT_YES_CASES_V0 = {
    "TCGA-KIRC": {
        "TCGA-B0-4845",
        "TCGA-B0-4846",
        "TCGA-B2-5639",
        "TCGA-B8-5162",
        "TCGA-BP-4354",
        "TCGA-BP-4771",
        "TCGA-BP-4965",
        "TCGA-CJ-4900",
        "TCGA-CZ-4861",
    },
    "TCGA-LUAD": {
        "TCGA-50-5072",
        "TCGA-64-5775",
        "TCGA-73-4676",
    },
    "TCGA-UCEC": {
        "TCGA-B5-A0JN",
    },
}
TREATMENT_AUDIT_SOURCE_V0 = "cBioPortal TCGA PanCan Atlas HISTORY_NEOADJUVANT_TRTYN audit, 2026-06-10"

MAIN_BINARY_MIN_EXTERNAL = 20
MAIN_BINARY_WARN_EXTERNAL = 30
MAIN_BINARY_MIN_DEVELOPMENT = 20
MAIN_BINARY_WARN_DEVELOPMENT = 50
SURVIVAL_MIN_EVENT = 30
SURVIVAL_WARN_EVENT = 50

RNA_MIN_LIBRARY_SIZE = 1_000_000
RNA_MIN_DETECTED_GENES = 10_000
RNA_TPM_SUM_MIN = 900_000
RNA_TPM_SUM_MAX = 1_100_000
ROBUST_LOW_Z_THRESHOLD = -4.0

WSI_EXTRA_COLUMNS = [
    "qc_table_version",
    "qc_rule_version",
    "history_neoadjuvant_treatment_v0",
    "treatment_timing_qc_pass_v0",
    "treatment_timing_qc_reason_v0",
    "slide_type_code",
    "metadata_base_tumor_pass_v0",
    "main_strict_pre_qc_v0",
    "main_strict_pre_qc_reason_v0",
    "inclusive_tumor_pre_qc_v0",
    "inclusive_tumor_pre_qc_reason_v0",
    "single_best_pre_qc_rank_v0",
    "single_best_pre_qc_v0",
    "image_qc_status_v0",
    "tissue_tiles_20x",
    "usable_tiles_after_qc",
    "artifact_fraction",
    "blur_score",
    "pen_mark_fraction",
]

RNA_EXTRA_COLUMNS = [
    "qc_table_version",
    "qc_rule_version",
    "sample_type_code_v0",
    "history_neoadjuvant_treatment_v0",
    "treatment_timing_qc_pass_v0",
    "treatment_timing_qc_reason_v0",
    "rna_metadata_qc_pass_v0",
    "rna_metadata_qc_reason_v0",
    "rna_candidate_count_in_case_v0",
    "representative_rna_rank_v0",
    "representative_rna_v0",
    "rna_count_file_path_v0",
    "rna_count_qc_status_v0",
    "rna_qc_reason_v0",
    "library_size_unstranded",
    "detected_genes_count",
    "protein_coding_detected_genes_count",
    "tpm_sum",
    "rna_library_size_robust_z_v0",
    "rna_detected_genes_robust_z_v0",
]

LABEL_EXTRA_COLUMNS = [
    "qc_table_version",
    "qc_rule_version",
    "history_neoadjuvant_treatment_v0",
    "treatment_timing_qc_pass_v0",
    "treatment_timing_qc_reason_v0",
    "label_definition_qc_pass_v0",
    "label_definition_qc_reason_v0",
    "has_main_strict_wsi_pre_qc_v0",
    "has_representative_rna_qc_v0",
    "paired_main_pre_qc_eligible_v0",
    "usable_label_n",
    "paired_pre_qc_n",
    "positive_n",
    "negative_n",
    "minority_n",
    "endpoint_dataset_qc_status_v0",
    "endpoint_dataset_qc_reason_v0",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_missing(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "unknown", "not reported", "--"}:
        return None
    return text


def clean(value: Any) -> str:
    text = normalize_missing(value)
    return text if text is not None else ""


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def is_yes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().eq("yes")


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_csv_required(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path, dtype=str)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def extract_slide_type_code(file_name: Any) -> str:
    text = clean(file_name)
    match = re.search(r"-(DX|TS|BS|MS|FS)\d*", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def extract_rna_sample_type_code(sample_submitter_id: Any) -> str:
    text = clean(sample_submitter_id)
    if not text:
        return ""
    tcga_match = re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})", text, flags=re.IGNORECASE)
    if tcga_match:
        return tcga_match.group(1)
    cptac_match = re.search(r"-(\d{2})$", text)
    return cptac_match.group(1) if cptac_match else ""


def add_treatment_timing_qc(df: pd.DataFrame) -> None:
    df["history_neoadjuvant_treatment_v0"] = ""
    df["treatment_timing_qc_pass_v0"] = "yes"
    df["treatment_timing_qc_reason_v0"] = ""
    for idx, row in df[["dataset", "case_submitter_id"]].iterrows():
        dataset = clean(row.get("dataset"))
        case_id = clean(row.get("case_submitter_id"))
        if dataset.startswith("TCGA"):
            if case_id in TCGA_NEOADJUVANT_YES_CASES_V0.get(dataset, set()):
                df.at[idx, "history_neoadjuvant_treatment_v0"] = "yes"
                df.at[idx, "treatment_timing_qc_pass_v0"] = "no"
                df.at[idx, "treatment_timing_qc_reason_v0"] = "tcga_neoadjuvant_treatment_prior_to_resection"
            else:
                df.at[idx, "history_neoadjuvant_treatment_v0"] = "not_flagged"
                df.at[idx, "treatment_timing_qc_reason_v0"] = "tcga_neoadjuvant_not_flagged_or_missing"
        elif dataset.startswith("CPTAC"):
            df.at[idx, "history_neoadjuvant_treatment_v0"] = "assumed_no"
            df.at[idx, "treatment_timing_qc_reason_v0"] = "cptac_treatment_naive_cohort_assumed"
        else:
            df.at[idx, "history_neoadjuvant_treatment_v0"] = "unknown"
            df.at[idx, "treatment_timing_qc_reason_v0"] = "unsupported_dataset"


def derive_wsi_pre_qc(wsi: pd.DataFrame) -> pd.DataFrame:
    df = wsi.copy()
    df["qc_table_version"] = SCRIPT_VERSION
    df["qc_rule_version"] = QC_RULE_VERSION
    add_treatment_timing_qc(df)
    df["slide_type_code"] = df["slide_file_name"].map(extract_slide_type_code)

    ptn = to_number(df["percent_tumor_nuclei"])
    nec = to_number(df["percent_necrosis"])
    is_tcga = df["dataset"].fillna("").str.startswith("TCGA")
    is_cptac = df["dataset"].fillna("").str.startswith("CPTAC")
    base = (
        is_yes(df["in_scope_for_main"])
        & is_yes(df["planned_for_extraction_v0"])
        & is_yes(df["is_tumor_slide"])
    )
    treatment_pass = df["treatment_timing_qc_pass_v0"].eq("yes")
    tcga_main = is_tcga & df["sample_type"].fillna("").eq("Primary Tumor") & df["slide_type_code"].eq("DX")
    cptac_main = (
        is_cptac
        & df["specimen_type"].fillna("").eq("tumor_tissue")
        & df["tumor_segment_acceptable"].fillna("").eq("Yes")
        & (ptn.isna() | ptn.ge(30))
        & (nec.isna() | nec.le(70))
    )
    tcga_inclusive = (
        is_tcga
        & df["sample_type"].fillna("").eq("Primary Tumor")
        & df["slide_type_code"].isin(["DX", "TS", "BS"])
    )
    cptac_inclusive = is_cptac & df["specimen_type"].fillna("").eq("tumor_tissue")

    df["metadata_base_tumor_pass_v0"] = base.map(yes_no)
    df["main_strict_pre_qc_v0"] = (base & treatment_pass & (tcga_main | cptac_main)).map(yes_no)
    df["inclusive_tumor_pre_qc_v0"] = (base & treatment_pass & (tcga_inclusive | cptac_inclusive)).map(yes_no)
    df["main_strict_pre_qc_reason_v0"] = df.apply(wsi_main_reason, axis=1)
    df["inclusive_tumor_pre_qc_reason_v0"] = df.apply(wsi_inclusive_reason, axis=1)

    selected = df[df["main_strict_pre_qc_v0"].eq("yes")].copy()
    selected["_slide_type_priority"] = selected["slide_type_code"].map({"DX": 0, "TS": 1, "BS": 2, "MS": 3}).fillna(9)
    selected["_acceptable_priority"] = (~selected["tumor_segment_acceptable"].fillna("").eq("Yes")).astype(int)
    selected["_ptn_sort"] = to_number(selected["percent_tumor_nuclei"]).fillna(-1)
    selected["_nec_sort"] = to_number(selected["percent_necrosis"]).fillna(9999)
    selected = selected.sort_values(
        [
            "dataset",
            "case_submitter_id",
            "_acceptable_priority",
            "_slide_type_priority",
            "_ptn_sort",
            "_nec_sort",
            "slide_id",
            "wsi_id",
        ],
        ascending=[True, True, True, True, False, True, True, True],
    )
    selected["single_best_pre_qc_rank_v0"] = selected.groupby(["dataset", "case_submitter_id"]).cumcount() + 1
    rank_map = selected.set_index("wsi_id")["single_best_pre_qc_rank_v0"].to_dict()
    rank_values = df["wsi_id"].map(rank_map)
    df["single_best_pre_qc_v0"] = rank_values.eq(1).map(yes_no)
    df["single_best_pre_qc_rank_v0"] = rank_values.map(lambda value: "" if pd.isna(value) else str(int(value)))

    df["image_qc_status_v0"] = "pending"
    for col in ["tissue_tiles_20x", "usable_tiles_after_qc", "artifact_fraction", "blur_score", "pen_mark_fraction"]:
        df[col] = ""

    return df[list(wsi.columns) + WSI_EXTRA_COLUMNS]


def wsi_base_failures(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if clean(row.get("in_scope_for_main")) != "yes":
        reasons.append("outside_main_scope")
    if clean(row.get("planned_for_extraction_v0")) != "yes":
        reasons.append("not_planned_for_extraction")
    if clean(row.get("is_tumor_slide")) != "yes":
        reasons.append("not_tumor_slide")
    return reasons


def wsi_main_reason(row: pd.Series) -> str:
    base_reasons = wsi_base_failures(row)
    if base_reasons:
        return ";".join(base_reasons)
    treatment_reason = treatment_qc_failure_reason(row)
    if treatment_reason:
        return treatment_reason
    dataset = clean(row.get("dataset"))
    if dataset.startswith("TCGA"):
        reasons = []
        if clean(row.get("sample_type")) != "Primary Tumor":
            reasons.append("tcga_not_primary_tumor")
        if clean(row.get("slide_type_code")) != "DX":
            reasons.append("tcga_not_dx_slide")
        return "pass" if not reasons else ";".join(reasons)
    if dataset.startswith("CPTAC"):
        reasons = []
        if clean(row.get("specimen_type")) != "tumor_tissue":
            reasons.append("cptac_not_tumor_tissue")
        if clean(row.get("tumor_segment_acceptable")) != "Yes":
            reasons.append("cptac_tumor_segment_not_acceptable")
        ptn = pd.to_numeric(row.get("percent_tumor_nuclei"), errors="coerce")
        nec = pd.to_numeric(row.get("percent_necrosis"), errors="coerce")
        if pd.notna(ptn) and ptn < 30:
            reasons.append("cptac_low_percent_tumor_nuclei")
        if pd.notna(nec) and nec > 70:
            reasons.append("cptac_high_percent_necrosis")
        return "pass" if not reasons else ";".join(reasons)
    return "unsupported_dataset"


def wsi_inclusive_reason(row: pd.Series) -> str:
    base_reasons = wsi_base_failures(row)
    if base_reasons:
        return ";".join(base_reasons)
    treatment_reason = treatment_qc_failure_reason(row)
    if treatment_reason:
        return treatment_reason
    dataset = clean(row.get("dataset"))
    if dataset.startswith("TCGA"):
        reasons = []
        if clean(row.get("sample_type")) != "Primary Tumor":
            reasons.append("tcga_not_primary_tumor")
        if clean(row.get("slide_type_code")) not in {"DX", "TS", "BS"}:
            reasons.append("tcga_not_dx_ts_bs_slide")
        return "pass" if not reasons else ";".join(reasons)
    if dataset.startswith("CPTAC"):
        return "pass" if clean(row.get("specimen_type")) == "tumor_tissue" else "cptac_not_tumor_tissue"
    return "unsupported_dataset"


def treatment_qc_failure_reason(row: pd.Series) -> str:
    if clean(row.get("treatment_timing_qc_pass_v0")) == "yes":
        return ""
    return clean(row.get("treatment_timing_qc_reason_v0")) or "treatment_timing_qc_fail"


def build_count_file_index(count_dir: Path | None) -> dict[str, Path]:
    if count_dir is None:
        return {}
    if not count_dir.exists():
        raise FileNotFoundError(f"RNA count directory not found: {count_dir}")
    index: dict[str, Path] = {}
    for path in sorted(count_dir.rglob("*")):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def parse_star_counts(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, sep="\t", comment="#")
    required = {"gene_id", "unstranded", "tpm_unstranded"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing STAR-count columns: {missing}")

    gene_rows = df[df["gene_id"].astype(str).str.startswith("ENSG", na=False)].copy()
    if gene_rows.empty:
        raise ValueError("no ENSG gene rows found")

    counts = pd.to_numeric(gene_rows["unstranded"], errors="coerce")
    tpm = pd.to_numeric(gene_rows["tpm_unstranded"], errors="coerce")
    if counts.isna().any():
        raise ValueError("NaN in unstranded counts")
    if tpm.isna().any():
        raise ValueError("NaN in tpm_unstranded")
    if (counts < 0).any():
        raise ValueError("negative unstranded counts")
    if (tpm < 0).any():
        raise ValueError("negative tpm_unstranded")

    protein_coding_detected = ""
    if "gene_type" in gene_rows.columns:
        protein_coding_detected = int(((gene_rows["gene_type"] == "protein_coding") & counts.gt(0)).sum())

    return {
        "library_size_unstranded": float(counts.sum()),
        "detected_genes_count": int(counts.gt(0).sum()),
        "protein_coding_detected_genes_count": protein_coding_detected,
        "tpm_sum": float(tpm.sum()),
    }


def derive_rna_qc(rna: pd.DataFrame, count_dir: Path | None) -> pd.DataFrame:
    df = rna.copy()
    df["qc_table_version"] = SCRIPT_VERSION
    df["qc_rule_version"] = QC_RULE_VERSION
    df["sample_type_code_v0"] = df["sample_submitter_id"].map(extract_rna_sample_type_code)
    add_treatment_timing_qc(df)

    is_tcga = df["dataset"].fillna("").str.startswith("TCGA")
    tcga_primary_solid_tumor = ~is_tcga | df["sample_type_code_v0"].eq("01")
    metadata_pass = (
        is_yes(df["in_scope_for_main"])
        & is_yes(df["primary_tumor_candidate_v0"])
        & tcga_primary_solid_tumor
        & df["treatment_timing_qc_pass_v0"].eq("yes")
        & df["data_type"].fillna("").eq("Gene Expression Quantification")
        & df["experimental_strategy"].fillna("").eq("RNA-Seq")
        & df["workflow_type"].fillna("").eq("STAR - Counts")
        & df["data_format"].fillna("").eq("TSV")
        & df["access"].fillna("").eq("open")
    )
    df["rna_metadata_qc_pass_v0"] = metadata_pass.map(yes_no)
    df["rna_metadata_qc_reason_v0"] = df.apply(rna_metadata_reason, axis=1)

    count_index = build_count_file_index(count_dir)
    initialize_rna_count_columns(df, count_dir)
    if count_dir is not None:
        fill_rna_count_metrics(df, count_index)
        apply_rna_count_thresholds(df)

    rep_eligible = df["rna_metadata_qc_pass_v0"].eq("yes") & df["rna_count_qc_status_v0"].isin(["pending", "pass"])
    candidates = df[df["rna_metadata_qc_pass_v0"].eq("yes")].copy()
    candidate_counts = candidates.groupby(["dataset", "case_submitter_id"])["rna_id"].count().to_dict()
    df["rna_candidate_count_in_case_v0"] = [
        str(candidate_counts.get((row.dataset, row.case_submitter_id), 0)) for row in df[["dataset", "case_submitter_id"]].itertuples(index=False)
    ]

    ranked = df[rep_eligible].copy()
    ranked["_sample_type_priority"] = ranked["sample_type"].map(
        {"Primary Tumor": 0, "Additional - New Primary": 1, "Recurrent Tumor": 2}
    ).fillna(9)
    ranked["_file_size_num"] = to_number(ranked["file_size_bytes"])
    ranked = ranked.sort_values(
        ["dataset", "case_submitter_id", "_sample_type_priority", "sample_type_code_v0", "_file_size_num", "file_id"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    )
    ranked["representative_rna_rank_v0"] = ranked.groupby(["dataset", "case_submitter_id"]).cumcount() + 1
    rank_map = ranked.set_index("rna_id")["representative_rna_rank_v0"].to_dict()
    rank_values = df["rna_id"].map(rank_map)
    df["representative_rna_v0"] = rank_values.eq(1).map(yes_no)
    df["representative_rna_rank_v0"] = rank_values.map(lambda value: "" if pd.isna(value) else str(int(value)))

    return df[list(rna.columns) + RNA_EXTRA_COLUMNS]


def rna_metadata_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if clean(row.get("in_scope_for_main")) != "yes":
        reasons.append("outside_main_scope")
    if clean(row.get("primary_tumor_candidate_v0")) != "yes":
        reasons.append("not_primary_tumor_candidate")
    if clean(row.get("dataset")).startswith("TCGA") and clean(row.get("sample_type_code_v0")) != "01":
        reasons.append("tcga_rna_not_primary_solid_tumor_01")
    treatment_reason = treatment_qc_failure_reason(row)
    if treatment_reason:
        reasons.append(treatment_reason)
    if clean(row.get("data_type")) != "Gene Expression Quantification":
        reasons.append("not_gene_expression_quantification")
    if clean(row.get("experimental_strategy")) != "RNA-Seq":
        reasons.append("not_rna_seq")
    if clean(row.get("workflow_type")) != "STAR - Counts":
        reasons.append("not_star_counts")
    if clean(row.get("data_format")) != "TSV":
        reasons.append("not_tsv")
    if clean(row.get("access")) != "open":
        reasons.append("not_open_access")
    return "pass" if not reasons else ";".join(reasons)


def initialize_rna_count_columns(df: pd.DataFrame, count_dir: Path | None) -> None:
    df["rna_count_file_path_v0"] = ""
    if count_dir is None:
        df["rna_count_qc_status_v0"] = "pending"
        df["rna_qc_reason_v0"] = "rna_count_dir_not_provided"
    else:
        df["rna_count_qc_status_v0"] = "not_applicable"
        df["rna_qc_reason_v0"] = "metadata_qc_not_pass"
    for col in [
        "library_size_unstranded",
        "detected_genes_count",
        "protein_coding_detected_genes_count",
        "tpm_sum",
        "rna_library_size_robust_z_v0",
        "rna_detected_genes_robust_z_v0",
    ]:
        df[col] = ""


def fill_rna_count_metrics(df: pd.DataFrame, count_index: dict[str, Path]) -> None:
    metadata_mask = df["rna_metadata_qc_pass_v0"].eq("yes")
    for idx, row in df[metadata_mask].iterrows():
        file_name = clean(row.get("file_name"))
        path = count_index.get(file_name)
        if path is None:
            df.at[idx, "rna_count_qc_status_v0"] = "fail"
            df.at[idx, "rna_qc_reason_v0"] = "count_file_missing"
            continue
        df.at[idx, "rna_count_file_path_v0"] = str(path)
        try:
            metrics = parse_star_counts(path)
        except Exception as exc:
            df.at[idx, "rna_count_qc_status_v0"] = "fail"
            df.at[idx, "rna_qc_reason_v0"] = f"count_parse_error:{exc}"
            continue
        for key, value in metrics.items():
            df.at[idx, key] = value
        df.at[idx, "rna_count_qc_status_v0"] = "parsed"
        df.at[idx, "rna_qc_reason_v0"] = "parsed"


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median(skipna=True)
    mad = (numeric - median).abs().median(skipna=True)
    if pd.isna(median) or pd.isna(mad) or mad == 0:
        return pd.Series([0.0 if pd.notna(v) else pd.NA for v in numeric], index=values.index)
    return (numeric - median) / (1.4826 * mad)


def apply_rna_count_thresholds(df: pd.DataFrame) -> None:
    parsed_mask = df["rna_count_qc_status_v0"].eq("parsed")
    if not parsed_mask.any():
        return

    for dataset, index in df[parsed_mask].groupby("dataset").groups.items():
        lib_z = robust_z(pd.to_numeric(df.loc[index, "library_size_unstranded"], errors="coerce"))
        det_z = robust_z(pd.to_numeric(df.loc[index, "detected_genes_count"], errors="coerce"))
        df.loc[index, "rna_library_size_robust_z_v0"] = lib_z.round(4).astype(str)
        df.loc[index, "rna_detected_genes_robust_z_v0"] = det_z.round(4).astype(str)

    for idx, row in df[parsed_mask].iterrows():
        reasons: list[str] = []
        library_size = pd.to_numeric(row.get("library_size_unstranded"), errors="coerce")
        detected_genes = pd.to_numeric(row.get("detected_genes_count"), errors="coerce")
        tpm_sum = pd.to_numeric(row.get("tpm_sum"), errors="coerce")
        library_z = pd.to_numeric(row.get("rna_library_size_robust_z_v0"), errors="coerce")
        detected_z = pd.to_numeric(row.get("rna_detected_genes_robust_z_v0"), errors="coerce")

        if pd.isna(library_size) or library_size < RNA_MIN_LIBRARY_SIZE:
            reasons.append("low_library_size")
        if pd.isna(detected_genes) or detected_genes < RNA_MIN_DETECTED_GENES:
            reasons.append("low_detected_genes")
        if pd.isna(tpm_sum) or tpm_sum < RNA_TPM_SUM_MIN or tpm_sum > RNA_TPM_SUM_MAX:
            reasons.append("tpm_sum_out_of_range")
        if pd.notna(library_z) and library_z < ROBUST_LOW_Z_THRESHOLD:
            reasons.append("library_size_robust_low_outlier")
        if pd.notna(detected_z) and detected_z < ROBUST_LOW_Z_THRESHOLD:
            reasons.append("detected_genes_robust_low_outlier")

        if reasons:
            df.at[idx, "rna_count_qc_status_v0"] = "fail"
            df.at[idx, "rna_qc_reason_v0"] = ";".join(reasons)
        else:
            df.at[idx, "rna_count_qc_status_v0"] = "pass"
            df.at[idx, "rna_qc_reason_v0"] = "pass"


def derive_label_qc(labels: pd.DataFrame, wsi_qc: pd.DataFrame, rna_qc: pd.DataFrame) -> pd.DataFrame:
    df = labels.copy()
    df["qc_table_version"] = SCRIPT_VERSION
    df["qc_rule_version"] = QC_RULE_VERSION
    add_treatment_timing_qc(df)
    definition = df.apply(label_definition_qc, axis=1)
    df["label_definition_qc_pass_v0"] = definition.map(lambda item: item[0]).map(yes_no)
    df["label_definition_qc_reason_v0"] = definition.map(lambda item: item[1])

    wsi_cases = case_set(wsi_qc[wsi_qc["main_strict_pre_qc_v0"].eq("yes")])
    rna_cases = case_set(rna_qc[rna_qc["representative_rna_v0"].eq("yes")])
    df["has_main_strict_wsi_pre_qc_v0"] = [
        yes_no((row.dataset, row.case_submitter_id) in wsi_cases)
        for row in df[["dataset", "case_submitter_id"]].itertuples(index=False)
    ]
    df["has_representative_rna_qc_v0"] = [
        yes_no((row.dataset, row.case_submitter_id) in rna_cases)
        for row in df[["dataset", "case_submitter_id"]].itertuples(index=False)
    ]
    paired = (
        df["label_definition_qc_pass_v0"].eq("yes")
        & df["has_main_strict_wsi_pre_qc_v0"].eq("yes")
        & df["has_representative_rna_qc_v0"].eq("yes")
        & df["treatment_timing_qc_pass_v0"].eq("yes")
    )
    df["paired_main_pre_qc_eligible_v0"] = paired.map(yes_no)

    for col in [
        "usable_label_n",
        "paired_pre_qc_n",
        "positive_n",
        "negative_n",
        "minority_n",
        "endpoint_dataset_qc_status_v0",
        "endpoint_dataset_qc_reason_v0",
    ]:
        df[col] = ""
    fill_endpoint_dataset_counts(df)
    return df[list(labels.columns) + LABEL_EXTRA_COLUMNS]


def label_definition_qc(row: pd.Series) -> tuple[bool, str]:
    reasons: list[str] = []
    if clean(row.get("in_scope_for_main")) != "yes":
        reasons.append("outside_main_scope")
    if clean(row.get("label_status")) != "usable":
        reasons.append(f"label_status_{clean(row.get('label_status')) or 'missing'}")
    if not clean(row.get("mapping_rule_id")):
        reasons.append("missing_mapping_rule_id")

    task_type = clean(row.get("task_type"))
    if task_type == "survival":
        event = clean(row.get("event"))
        time_days = pd.to_numeric(row.get("time_days"), errors="coerce")
        if event not in {"0", "1"}:
            reasons.append("invalid_survival_event")
        if pd.isna(time_days) or time_days <= 0:
            reasons.append("invalid_survival_time")
    else:
        mapped = clean(row.get("mapped_label"))
        numeric = clean(row.get("label_numeric"))
        allowed = {clean(row.get("positive_class")), clean(row.get("negative_class"))}
        if not mapped or mapped not in allowed:
            reasons.append("mapped_label_not_in_declared_classes")
        if numeric not in {"0", "1"}:
            reasons.append("invalid_label_numeric")
    return (not reasons, "pass" if not reasons else ";".join(reasons))


def case_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["dataset"].astype(str), df["case_submitter_id"].astype(str)))


def fill_endpoint_dataset_counts(df: pd.DataFrame) -> None:
    for (endpoint, dataset), group in df.groupby(["endpoint_name", "dataset"]):
        idx = group.index
        usable = group[group["label_definition_qc_pass_v0"].eq("yes")]
        paired = group[group["paired_main_pre_qc_eligible_v0"].eq("yes")]
        task_type = clean(group["task_type"].iloc[0])
        cohort_role = clean(group["cohort_role"].iloc[0])

        if task_type == "survival":
            positive_n = int(paired["event"].eq("1").sum())
            negative_n = int(paired["event"].eq("0").sum())
            minority_n = min(positive_n, negative_n) if len(paired) else 0
            status, reason = survival_endpoint_status(positive_n)
        else:
            positive_class = clean(group["positive_class"].iloc[0])
            negative_class = clean(group["negative_class"].iloc[0])
            positive_n = int(paired["mapped_label"].eq(positive_class).sum())
            negative_n = int(paired["mapped_label"].eq(negative_class).sum())
            minority_n = min(positive_n, negative_n) if len(paired) else 0
            status, reason = binary_endpoint_status(minority_n, cohort_role)

        df.loc[idx, "usable_label_n"] = str(len(usable))
        df.loc[idx, "paired_pre_qc_n"] = str(len(paired))
        df.loc[idx, "positive_n"] = str(positive_n)
        df.loc[idx, "negative_n"] = str(negative_n)
        df.loc[idx, "minority_n"] = str(minority_n)
        df.loc[idx, "endpoint_dataset_qc_status_v0"] = status
        df.loc[idx, "endpoint_dataset_qc_reason_v0"] = reason


def binary_endpoint_status(minority_n: int, cohort_role: str) -> tuple[str, str]:
    if cohort_role == "external":
        if minority_n < MAIN_BINARY_MIN_EXTERNAL:
            return "fail", f"external_minority_n_lt_{MAIN_BINARY_MIN_EXTERNAL}"
        if minority_n < MAIN_BINARY_WARN_EXTERNAL:
            return "warn", f"external_minority_n_lt_{MAIN_BINARY_WARN_EXTERNAL}"
        return "pass", "pass"
    if minority_n < MAIN_BINARY_MIN_DEVELOPMENT:
        return "fail", f"development_minority_n_lt_{MAIN_BINARY_MIN_DEVELOPMENT}"
    if minority_n < MAIN_BINARY_WARN_DEVELOPMENT:
        return "warn", f"development_minority_n_lt_{MAIN_BINARY_WARN_DEVELOPMENT}"
    return "pass", "pass"


def survival_endpoint_status(event_n: int) -> tuple[str, str]:
    if event_n < SURVIVAL_MIN_EVENT:
        return "fail", f"event_n_lt_{SURVIVAL_MIN_EVENT}"
    if event_n < SURVIVAL_WARN_EVENT:
        return "warn", f"event_n_lt_{SURVIVAL_WARN_EVENT}"
    return "pass", "pass"


def summarize_counts_by_dataset(df: pd.DataFrame, mask_col: str, row_name: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    selected = df[df[mask_col].eq("yes")]
    for dataset, group in selected.groupby("dataset"):
        result[dataset] = {
            row_name: int(len(group)),
            "cases": int(group["case_submitter_id"].nunique()),
        }
    return result


def summarize_rna_counts(rna_qc: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    masks = {
        "metadata_pass": rna_qc["rna_metadata_qc_pass_v0"].eq("yes"),
        "representative": rna_qc["representative_rna_v0"].eq("yes"),
        "count_qc_pass": rna_qc["rna_count_qc_status_v0"].eq("pass"),
        "count_qc_pending": rna_qc["rna_count_qc_status_v0"].eq("pending"),
        "count_qc_fail": rna_qc["rna_count_qc_status_v0"].eq("fail"),
    }
    for label, mask in masks.items():
        result[label] = {}
        for dataset, group in rna_qc[mask].groupby("dataset"):
            result[label][dataset] = {
                "rna_files": int(len(group)),
                "cases": int(group["case_submitter_id"].nunique()),
            }
    return result


def summarize_label_status(labels: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (endpoint, dataset), group in labels.groupby(["endpoint_name", "dataset"]):
        result[f"{endpoint}|{dataset}"] = {
            "rows": int(len(group)),
            "label_status_counts": {
                str(k): int(v) for k, v in group["label_status"].value_counts(dropna=False).to_dict().items()
            },
            "label_definition_qc_pass": int(group["label_definition_qc_pass_v0"].eq("yes").sum()),
            "paired_main_pre_qc_eligible": int(group["paired_main_pre_qc_eligible_v0"].eq("yes").sum()),
            "endpoint_dataset_qc_status_v0": clean(group["endpoint_dataset_qc_status_v0"].iloc[0]),
            "endpoint_dataset_qc_reason_v0": clean(group["endpoint_dataset_qc_reason_v0"].iloc[0]),
        }
    return result


def summarize_paired_feasibility(label_qc: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (endpoint, dataset), group in label_qc.groupby(["endpoint_name", "dataset"]):
        paired = group[group["paired_main_pre_qc_eligible_v0"].eq("yes")]
        if clean(group["task_type"].iloc[0]) == "survival":
            counts = paired["event"].value_counts(dropna=False).to_dict()
        else:
            counts = paired["mapped_label"].value_counts(dropna=False).to_dict()
        result[f"{endpoint}|{dataset}"] = {
            "paired_pre_qc_cases": int(len(paired)),
            "paired_label_counts": {str(k): int(v) for k, v in counts.items()},
        }
    return result


def summarize_treatment_audit(wsi_qc: pd.DataFrame, rna_qc: pd.DataFrame, label_qc: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"source": TREATMENT_AUDIT_SOURCE_V0}
    for name, df, selected_col in (
        ("wsi", wsi_qc, "main_strict_pre_qc_v0"),
        ("rna", rna_qc, "representative_rna_v0"),
        ("labels", label_qc, "paired_main_pre_qc_eligible_v0"),
    ):
        result[name] = {}
        for dataset, group in df.groupby("dataset"):
            selected = group[group[selected_col].eq("yes")]
            excluded = group[group["treatment_timing_qc_pass_v0"].ne("yes")]
            result[name][dataset] = {
                "selected_cases": int(selected["case_submitter_id"].nunique()),
                "treatment_excluded_rows": int(len(excluded)),
                "treatment_excluded_cases": int(excluded["case_submitter_id"].nunique()),
                "history_neoadjuvant_treatment_counts": {
                    str(k): int(v)
                    for k, v in group["history_neoadjuvant_treatment_v0"].value_counts(dropna=False).to_dict().items()
                },
            }
    return result


def validate_outputs(wsi_qc: pd.DataFrame, rna_qc: pd.DataFrame, label_qc: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    for df, name, key in (
        (wsi_qc, "wsi_slide_pre_qc", "wsi_id"),
        (rna_qc, "rna_qc", "rna_id"),
        (label_qc, "label_qc", "label_id"),
    ):
        if df[key].isna().any() or df[key].astype(str).eq("").any():
            issues.append(f"{name}: empty {key}")
        duplicates = sorted(df.loc[df.duplicated(key, keep=False), key].dropna().astype(str).unique().tolist())
        if duplicates:
            issues.append(f"{name}: duplicate {key}: {duplicates[:5]}")

    single = wsi_qc[wsi_qc["single_best_pre_qc_v0"].eq("yes")]
    if single.duplicated(["dataset", "case_submitter_id"]).any():
        issues.append("wsi_slide_pre_qc: multiple single_best slides for a dataset/case")

    reps = rna_qc[rna_qc["representative_rna_v0"].eq("yes")]
    if reps.duplicated(["dataset", "case_submitter_id"]).any():
        issues.append("rna_qc: multiple representative RNA files for a dataset/case")

    invalid_paired = label_qc[
        label_qc["paired_main_pre_qc_eligible_v0"].eq("yes")
        & (
            label_qc["label_definition_qc_pass_v0"].ne("yes")
            | label_qc["has_main_strict_wsi_pre_qc_v0"].ne("yes")
            | label_qc["has_representative_rna_qc_v0"].ne("yes")
            | label_qc["treatment_timing_qc_pass_v0"].ne("yes")
        )
    ]
    if not invalid_paired.empty:
        issues.append(f"label_qc: invalid paired eligibility rows: {len(invalid_paired)}")

    return {"status": "pass" if not issues else "fail", "issues": issues}


def build_report(
    wsi_qc: pd.DataFrame,
    rna_qc: pd.DataFrame,
    label_qc: pd.DataFrame,
    *,
    started_at: str,
    finished_at: str,
    rna_count_dir: Path | None,
) -> dict[str, Any]:
    return {
        "script": "scripts/build_integrated_qc_tables_v0.py",
        "script_version": SCRIPT_VERSION,
        "qc_rule_version": QC_RULE_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "rna_count_dir": str(rna_count_dir) if rna_count_dir else "",
        "wsi": {
            "main_strict_pre_qc": summarize_counts_by_dataset(wsi_qc, "main_strict_pre_qc_v0", "slides"),
            "inclusive_tumor_pre_qc": summarize_counts_by_dataset(wsi_qc, "inclusive_tumor_pre_qc_v0", "slides"),
            "single_best_pre_qc": summarize_counts_by_dataset(wsi_qc, "single_best_pre_qc_v0", "slides"),
        },
        "rna": summarize_rna_counts(rna_qc),
        "labels": summarize_label_status(label_qc),
        "paired_endpoint_feasibility": summarize_paired_feasibility(label_qc),
        "treatment_timing_audit": summarize_treatment_audit(wsi_qc, rna_qc, label_qc),
        "validation": validate_outputs(wsi_qc, rna_qc, label_qc),
    }


def write_outputs(wsi_qc: pd.DataFrame, rna_qc: pd.DataFrame, label_qc: pd.DataFrame, report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_qc.to_csv(out_dir / "wsi_slide_pre_qc_table_v0.csv", index=False)
    rna_qc.to_csv(out_dir / "rna_qc_table_v0.csv", index=False)
    label_qc.to_csv(out_dir / "label_qc_table_v0.csv", index=False)
    (out_dir / "integrated_qc_report_v0.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v0 integrated WSI/RNA/label QC tables.")
    parser.add_argument("--wsi", type=Path, default=Path("manifests") / "wsi_slide_table_v0.csv")
    parser.add_argument("--rna", type=Path, default=Path("manifests") / "rna_sample_table_v0.csv")
    parser.add_argument("--labels", type=Path, default=Path("manifests") / "label_table_v0.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("manifests"))
    parser.add_argument(
        "--rna-count-dir",
        type=Path,
        default=None,
        help="Optional directory containing downloaded GDC STAR-counts TSV files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = utc_now()

    wsi = read_csv_required(args.wsi, ["wsi_id", "dataset", "case_submitter_id"])
    rna = read_csv_required(args.rna, ["rna_id", "dataset", "case_submitter_id"])
    labels = read_csv_required(args.labels, ["label_id", "dataset", "case_submitter_id", "endpoint_name"])

    wsi_qc = derive_wsi_pre_qc(wsi)
    rna_qc = derive_rna_qc(rna, args.rna_count_dir)
    label_qc = derive_label_qc(labels, wsi_qc, rna_qc)

    finished_at = utc_now()
    report = build_report(
        wsi_qc,
        rna_qc,
        label_qc,
        started_at=started_at,
        finished_at=finished_at,
        rna_count_dir=args.rna_count_dir,
    )
    write_outputs(wsi_qc, rna_qc, label_qc, report, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["validation"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
