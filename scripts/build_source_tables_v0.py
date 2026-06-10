#!/usr/bin/env python3
"""Build v0 WSI, RNA, and label source tables for the WSI-RNA fusion study.

This script creates source registries, not final analysis cohorts. It preserves
slide-level WSI records, RNA quantification file records, and long-format
case-endpoint labels so later scripts can define endpoint-specific case tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
CBIO_BASE = "https://www.cbioportal.org/api"

SCRIPT_VERSION = "v0"
SOURCE_METADATA_VERSION = "v0_2026-06-09"

PATHOGENIC_VARIANTS = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Region",
    "Splice_Site",
    "Start_Codon_Del",
    "Start_Codon_Ins",
    "Stop_Codon_Del",
    "Stop_Codon_Ins",
    "Translation_Start_Site",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    cohort_role: str
    cancer_context: str
    source_system: str
    gdc_project: str
    pathdb_tumor: str | None = None
    pathdb_disease_type_in_scope: str | None = None
    tcia_download_id: str | None = None
    tcia_download_slug: str | None = None
    tcia_download_size: str | None = None
    tcia_download_size_unit: str | None = None
    cbio_study: str | None = None
    cbio_mutation_profile: str | None = None
    cbio_sequenced_sample_list: str | None = None


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        dataset="TCGA-KIRC",
        cohort_role="development",
        cancer_context="KIRC_CCRCC",
        source_system="GDC",
        gdc_project="TCGA-KIRC",
        cbio_study="kirc_tcga_pan_can_atlas_2018",
        cbio_mutation_profile="kirc_tcga_pan_can_atlas_2018_mutations",
        cbio_sequenced_sample_list="kirc_tcga_pan_can_atlas_2018_sequenced",
    ),
    DatasetSpec(
        dataset="CPTAC-CCRCC",
        cohort_role="external",
        cancer_context="KIRC_CCRCC",
        source_system="PathDB",
        gdc_project="CPTAC-3",
        pathdb_tumor="CCRCC",
        pathdb_disease_type_in_scope="Clear Cell Renal Cell Carcinoma",
        tcia_download_id="44433",
        tcia_download_slug="cptac-ccrcc-da-path",
        tcia_download_size="190",
        tcia_download_size_unit="gb",
        cbio_study="rcc_cptac_gdc",
        cbio_mutation_profile="rcc_cptac_gdc_mutations",
        cbio_sequenced_sample_list="rcc_cptac_gdc_sequenced",
    ),
    DatasetSpec(
        dataset="TCGA-UCEC",
        cohort_role="development",
        cancer_context="UCEC",
        source_system="GDC",
        gdc_project="TCGA-UCEC",
        cbio_study="ucec_tcga_pan_can_atlas_2018",
        cbio_mutation_profile="ucec_tcga_pan_can_atlas_2018_mutations",
        cbio_sequenced_sample_list="ucec_tcga_pan_can_atlas_2018_sequenced",
    ),
    DatasetSpec(
        dataset="CPTAC-UCEC",
        cohort_role="external",
        cancer_context="UCEC",
        source_system="PathDB",
        gdc_project="CPTAC-3",
        pathdb_tumor="UCEC",
        tcia_download_id="46765",
        tcia_download_slug="cptac-ucec-da-path",
        tcia_download_size="154",
        tcia_download_size_unit="gb",
        cbio_study="uec_cptac_gdc",
        cbio_mutation_profile="uec_cptac_gdc_mutations",
        cbio_sequenced_sample_list="uec_cptac_gdc_sequenced",
    ),
    DatasetSpec(
        dataset="TCGA-LUAD",
        cohort_role="development",
        cancer_context="LUAD",
        source_system="GDC",
        gdc_project="TCGA-LUAD",
        cbio_study="luad_tcga_pan_can_atlas_2018",
        cbio_mutation_profile="luad_tcga_pan_can_atlas_2018_mutations",
        cbio_sequenced_sample_list="luad_tcga_pan_can_atlas_2018_sequenced",
    ),
    DatasetSpec(
        dataset="CPTAC-LUAD",
        cohort_role="external",
        cancer_context="LUAD",
        source_system="PathDB",
        gdc_project="CPTAC-3",
        pathdb_tumor="LUAD",
        tcia_download_id="44839",
        tcia_download_slug="cptac-luad-da-path",
        tcia_download_size="431.5",
        tcia_download_size_unit="gb",
        cbio_study="luad_cptac_gdc",
        cbio_mutation_profile="luad_cptac_gdc_mutations",
        cbio_sequenced_sample_list="luad_cptac_gdc_sequenced",
    ),
]

DATASET_BY_NAME = {spec.dataset: spec for spec in DATASET_SPECS}


ENDPOINTS = [
    {
        "endpoint_name": "kirc_ccrcc_grade",
        "analysis_set": "main",
        "endpoint_family": "grade",
        "task_type": "binary_classification",
        "datasets": ["TCGA-KIRC", "CPTAC-CCRCC"],
        "positive_class": "high",
        "negative_class": "low",
        "expected_regime_role": "WSI-dominant / morphology-driven",
        "mapping_rule_id": "grade_g1_g2_vs_g3_g4_v0",
    },
    {
        "endpoint_name": "kirc_ccrcc_stage",
        "analysis_set": "main",
        "endpoint_family": "stage",
        "task_type": "binary_classification",
        "datasets": ["TCGA-KIRC", "CPTAC-CCRCC"],
        "positive_class": "late",
        "negative_class": "early",
        "expected_regime_role": "clinical / complementary candidate",
        "mapping_rule_id": "stage_i_ii_vs_iii_iv_v0",
    },
    {
        "endpoint_name": "ucec_histologic_subtype",
        "analysis_set": "main",
        "endpoint_family": "histology",
        "task_type": "binary_classification",
        "datasets": ["TCGA-UCEC", "CPTAC-UCEC"],
        "positive_class": "serous",
        "negative_class": "endometrioid",
        "expected_regime_role": "WSI-dominant / morphology-driven",
        "mapping_rule_id": "ucec_endometrioid_vs_serous_v0",
    },
    {
        "endpoint_name": "ucec_tp53_mutation",
        "analysis_set": "main",
        "endpoint_family": "mutation",
        "task_type": "binary_classification",
        "datasets": ["TCGA-UCEC", "CPTAC-UCEC"],
        "gene": "TP53",
        "entrez_gene_id": 7157,
        "positive_class": "mutated",
        "negative_class": "wildtype",
        "expected_regime_role": "molecular / complementary candidate",
        "mapping_rule_id": "pathogenic_gene_mutation_v0",
    },
    {
        "endpoint_name": "luad_egfr_mutation",
        "analysis_set": "main",
        "endpoint_family": "mutation",
        "task_type": "binary_classification",
        "datasets": ["TCGA-LUAD", "CPTAC-LUAD"],
        "gene": "EGFR",
        "entrez_gene_id": 1956,
        "positive_class": "mutated",
        "negative_class": "wildtype",
        "expected_regime_role": "RNA-dominant candidate",
        "mapping_rule_id": "pathogenic_gene_mutation_v0",
    },
    {
        "endpoint_name": "luad_kras_mutation",
        "analysis_set": "main",
        "endpoint_family": "mutation",
        "task_type": "binary_classification",
        "datasets": ["TCGA-LUAD", "CPTAC-LUAD"],
        "gene": "KRAS",
        "entrez_gene_id": 3845,
        "positive_class": "mutated",
        "negative_class": "wildtype",
        "expected_regime_role": "RNA-dominant candidate",
        "mapping_rule_id": "pathogenic_gene_mutation_v0",
    },
    {
        "endpoint_name": "luad_stage",
        "analysis_set": "main",
        "endpoint_family": "stage",
        "task_type": "binary_classification",
        "datasets": ["TCGA-LUAD", "CPTAC-LUAD"],
        "positive_class": "late",
        "negative_class": "early",
        "expected_regime_role": "clinical / shift-sensitive candidate",
        "mapping_rule_id": "stage_i_ii_vs_iii_iv_v0",
    },
    {
        "endpoint_name": "kirc_ccrcc_os",
        "analysis_set": "supplementary",
        "endpoint_family": "survival",
        "task_type": "survival",
        "datasets": ["TCGA-KIRC", "CPTAC-CCRCC"],
        "positive_class": "event",
        "negative_class": "censored",
        "expected_regime_role": "weak-signal / shift-sensitive",
        "mapping_rule_id": "overall_survival_gdc_days_v0",
    },
    {
        "endpoint_name": "luad_os",
        "analysis_set": "supplementary",
        "endpoint_family": "survival",
        "task_type": "survival",
        "datasets": ["TCGA-LUAD", "CPTAC-LUAD"],
        "positive_class": "event",
        "negative_class": "censored",
        "expected_regime_role": "weak-signal / shift-sensitive",
        "mapping_rule_id": "overall_survival_gdc_days_v0",
    },
]


WSI_COLUMNS = [
    "wsi_id",
    "table_version",
    "cohort_role",
    "dataset",
    "cancer_context",
    "in_scope_for_main",
    "source_system",
    "source_project",
    "source_collection",
    "source_download_id",
    "source_download_slug",
    "source_download_size",
    "source_download_size_unit",
    "case_submitter_id",
    "case_uuid",
    "specimen_id",
    "sample_submitter_id",
    "sample_uuid",
    "slide_id",
    "slide_file_id",
    "slide_file_name",
    "expected_svs_filename",
    "data_category",
    "data_type",
    "experimental_strategy",
    "data_format",
    "access",
    "file_size_bytes",
    "md5sum",
    "sample_type",
    "sample_type_code",
    "is_tumor_slide",
    "specimen_type",
    "disease_type",
    "tumor",
    "tumor_site",
    "topographic_site",
    "tumor_histological_type",
    "tumor_segment_acceptable",
    "percent_tumor_nuclei",
    "percent_total_cellularity",
    "percent_necrosis",
    "embedding_medium",
    "pathology_id",
    "genomics_available",
    "genomics_case_id",
    "gdc_link",
    "proteomics_available",
    "proteomics_case_id",
    "pdc_link",
    "planned_for_extraction_v0",
    "source_metadata_version",
    "source_metadata_file",
    "source_retrieved_at",
    "notes",
]

RNA_COLUMNS = [
    "rna_id",
    "table_version",
    "cohort_role",
    "dataset",
    "cancer_context",
    "in_scope_for_main",
    "source_system",
    "source_project",
    "case_submitter_id",
    "case_uuid",
    "sample_submitter_id",
    "sample_uuid",
    "aliquot_submitter_id",
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "experimental_strategy",
    "workflow_type",
    "data_format",
    "access",
    "file_size_bytes",
    "md5sum",
    "sample_type",
    "is_tumor_sample",
    "primary_tumor_candidate_v0",
    "source_metadata_version",
    "source_retrieved_at",
    "notes",
]

LABEL_COLUMNS = [
    "label_id",
    "table_version",
    "analysis_set",
    "cohort_role",
    "dataset",
    "cancer_context",
    "case_submitter_id",
    "case_uuid",
    "in_scope_for_main",
    "endpoint_name",
    "endpoint_family",
    "task_type",
    "expected_regime_role",
    "raw_label_value",
    "mapped_label",
    "label_numeric",
    "event",
    "time_days",
    "time_months",
    "label_status",
    "exclusion_reason",
    "positive_class",
    "negative_class",
    "label_source_system",
    "label_source_field",
    "mapping_rule_id",
    "mapping_rule_version",
    "source_retrieved_at",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def request_json(
    method: str,
    url: str,
    *,
    json_payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> Any:
    headers = {"Content-Type": "application/json", "User-Agent": "path-rna-fusion-source-tables-v0"}
    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "POST":
                response = requests.post(url, json=json_payload, params=params, headers=headers, timeout=timeout)
            else:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("unreachable")


def gdc_paged(endpoint: str, payload: dict[str, Any], *, page_size: int = 2000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_payload = dict(payload)
        page_payload["size"] = page_size
        page_payload["from"] = offset
        data = request_json("POST", endpoint, json_payload=page_payload).get("data", {})
        hits = data.get("hits", [])
        rows.extend(hits)
        pagination = data.get("pagination", {})
        total = int(pagination.get("total", len(rows)))
        if not hits or len(rows) >= total:
            break
        offset += page_size
    return rows


def first_dict(values: Any) -> dict[str, Any]:
    if isinstance(values, list) and values:
        return values[0] or {}
    return {}


def normalize_missing(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "unknown", "not reported", "--", "[not available]"}:
        return None
    return text


def parse_gdc_case_uuid(link: Any) -> str | None:
    text = normalize_missing(link)
    if not text:
        return None
    match = re.search(r"/cases/([0-9a-fA-F-]{36})", text)
    return match.group(1) if match else None


def tcga_sample_type_code(sample_submitter_id: Any) -> str | None:
    text = normalize_missing(sample_submitter_id)
    if not text:
        return None
    match = re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})", text)
    return match.group(1) if match else None


def is_tumor_sample_type(sample_type: Any, sample_submitter_id: Any = None) -> bool | None:
    text = normalize_missing(sample_type)
    if text:
        if text.lower() in {"primary tumor", "recurrent tumor", "metastatic"}:
            return True
        if "normal" in text.lower():
            return False
    code = tcga_sample_type_code(sample_submitter_id)
    if code:
        return code in {"01", "02", "03", "05", "06", "07"}
    return None


def bool_text(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def query_gdc_files(
    *,
    project: str,
    data_type: str,
    case_submitter_ids: Iterable[str] | None = None,
    workflow_type: str | None = None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
        {"op": "=", "content": {"field": "data_type", "value": data_type}},
    ]
    if data_type == "Gene Expression Quantification":
        filters.extend(
            [
                {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
                {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
            ]
        )
    if workflow_type:
        filters.append({"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}})
    case_list = sorted({str(case_id) for case_id in (case_submitter_ids or []) if normalize_missing(case_id)})
    if case_list:
        filters.append({"op": "in", "content": {"field": "cases.submitter_id", "value": case_list}})

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "data_category",
        "data_type",
        "experimental_strategy",
        "analysis.workflow_type",
        "data_format",
        "access",
        "cases.submitter_id",
        "cases.case_id",
        "cases.project.project_id",
        "cases.primary_site",
        "cases.disease_type",
        "cases.samples.sample_type",
        "cases.samples.sample_id",
        "cases.samples.submitter_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
    ]
    payload = {
        "filters": {"op": "and", "content": filters},
        "fields": ",".join(fields),
        "format": "JSON",
    }
    return gdc_paged(GDC_FILES_ENDPOINT, payload, page_size=2000)


def query_gdc_cases(project: str, case_submitter_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    case_ids = sorted({str(case_id) for case_id in case_submitter_ids if normalize_missing(case_id)})
    if not case_ids:
        return {}
    fields = [
        "submitter_id",
        "case_id",
        "project.project_id",
        "primary_site",
        "disease_type",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.primary_diagnosis",
        "diagnoses.morphology",
        "diagnoses.tumor_grade",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.days_to_last_follow_up",
        "follow_ups.days_to_follow_up",
    ]
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "project.project_id", "value": project}},
                {"op": "in", "content": {"field": "submitter_id", "value": case_ids}},
            ],
        },
        "fields": ",".join(fields),
        "format": "JSON",
    }
    rows = gdc_paged(GDC_CASES_ENDPOINT, payload, page_size=2000)
    return {str(row.get("submitter_id")): row for row in rows if row.get("submitter_id")}


def flatten_gdc_file(file_info: dict[str, Any]) -> dict[str, Any]:
    case_info = first_dict(file_info.get("cases"))
    sample_info = first_dict(case_info.get("samples"))
    portion_info = first_dict(sample_info.get("portions"))
    analyte_info = first_dict(portion_info.get("analytes"))
    aliquot_info = first_dict(analyte_info.get("aliquots"))
    analysis_info = file_info.get("analysis") or {}
    return {
        "file_id": file_info.get("file_id") or file_info.get("id"),
        "file_name": file_info.get("file_name"),
        "file_size_bytes": file_info.get("file_size"),
        "md5sum": file_info.get("md5sum"),
        "data_category": file_info.get("data_category"),
        "data_type": file_info.get("data_type"),
        "experimental_strategy": file_info.get("experimental_strategy"),
        "workflow_type": analysis_info.get("workflow_type"),
        "data_format": file_info.get("data_format"),
        "access": file_info.get("access"),
        "case_submitter_id": case_info.get("submitter_id"),
        "case_uuid": case_info.get("case_id"),
        "source_project": (case_info.get("project") or {}).get("project_id"),
        "sample_type": sample_info.get("sample_type"),
        "sample_uuid": sample_info.get("sample_id"),
        "sample_submitter_id": sample_info.get("submitter_id"),
        "aliquot_submitter_id": aliquot_info.get("submitter_id"),
    }


def empty_string(value: Any) -> str:
    text = normalize_missing(value)
    return text if text is not None else ""


def build_tcga_wsi_rows(spec: DatasetSpec, retrieved_at: str) -> list[dict[str, Any]]:
    stderr(f"[WSI] Querying GDC slide images for {spec.dataset}")
    rows: list[dict[str, Any]] = []
    for file_info in query_gdc_files(project=spec.gdc_project, data_type="Slide Image"):
        flat = flatten_gdc_file(file_info)
        sample_code = tcga_sample_type_code(flat.get("sample_submitter_id"))
        tumor_flag = is_tumor_sample_type(flat.get("sample_type"), flat.get("sample_submitter_id"))
        file_name = empty_string(flat.get("file_name"))
        slide_id = file_name.removesuffix(".svs") if file_name else empty_string(flat.get("file_id"))
        row = {
            "wsi_id": f"{spec.dataset}:{flat.get('file_id')}",
            "table_version": SCRIPT_VERSION,
            "cohort_role": spec.cohort_role,
            "dataset": spec.dataset,
            "cancer_context": spec.cancer_context,
            "in_scope_for_main": "yes",
            "source_system": "GDC",
            "source_project": spec.gdc_project,
            "source_collection": spec.dataset,
            "source_download_id": "",
            "source_download_slug": "",
            "source_download_size": "",
            "source_download_size_unit": "",
            "case_submitter_id": flat.get("case_submitter_id"),
            "case_uuid": flat.get("case_uuid"),
            "specimen_id": "",
            "sample_submitter_id": flat.get("sample_submitter_id"),
            "sample_uuid": flat.get("sample_uuid"),
            "slide_id": slide_id,
            "slide_file_id": flat.get("file_id"),
            "slide_file_name": file_name,
            "expected_svs_filename": file_name,
            "data_category": flat.get("data_category"),
            "data_type": flat.get("data_type"),
            "experimental_strategy": flat.get("experimental_strategy"),
            "data_format": flat.get("data_format"),
            "access": flat.get("access"),
            "file_size_bytes": flat.get("file_size_bytes"),
            "md5sum": flat.get("md5sum"),
            "sample_type": flat.get("sample_type"),
            "sample_type_code": sample_code,
            "is_tumor_slide": bool_text(tumor_flag),
            "specimen_type": "",
            "disease_type": "",
            "tumor": spec.dataset.replace("TCGA-", ""),
            "tumor_site": "",
            "topographic_site": "",
            "tumor_histological_type": "",
            "tumor_segment_acceptable": "",
            "percent_tumor_nuclei": "",
            "percent_total_cellularity": "",
            "percent_necrosis": "",
            "embedding_medium": "",
            "pathology_id": "",
            "genomics_available": "",
            "genomics_case_id": "",
            "gdc_link": "",
            "proteomics_available": "",
            "proteomics_case_id": "",
            "pdc_link": "",
            "planned_for_extraction_v0": "yes",
            "source_metadata_version": SOURCE_METADATA_VERSION,
            "source_metadata_file": "GDC files API",
            "source_retrieved_at": retrieved_at,
            "notes": "",
        }
        rows.append(row)
    return rows


def title_yes(value: Any) -> str:
    text = normalize_missing(value)
    if not text:
        return ""
    return "Yes" if text.lower() == "yes" else text


def build_cptac_wsi_rows(spec: DatasetSpec, pathdb: pd.DataFrame, pathdb_path: Path, retrieved_at: str) -> list[dict[str, Any]]:
    assert spec.pathdb_tumor is not None
    stderr(f"[WSI] Reading PathDB slide metadata for {spec.dataset}")
    subset = pathdb[pathdb["Tumor"].astype(str).eq(spec.pathdb_tumor)].copy()
    rows: list[dict[str, Any]] = []
    for _, item in subset.iterrows():
        disease_type = empty_string(item.get("Disease_Type"))
        in_scope = "yes"
        notes = ""
        if spec.pathdb_disease_type_in_scope and disease_type != spec.pathdb_disease_type_in_scope:
            in_scope = "no"
            notes = "excluded_from_main_context_non_clear_cell"
        slide_id = empty_string(item.get("Slide_ID"))
        expected_name = f"{slide_id}.svs" if slide_id else ""
        specimen_type = empty_string(item.get("Specimen_Type"))
        is_tumor = None
        if specimen_type:
            if specimen_type == "tumor_tissue":
                is_tumor = True
            elif specimen_type == "normal_tissue":
                is_tumor = False
        row = {
            "wsi_id": f"{spec.dataset}:{slide_id}",
            "table_version": SCRIPT_VERSION,
            "cohort_role": spec.cohort_role,
            "dataset": spec.dataset,
            "cancer_context": spec.cancer_context,
            "in_scope_for_main": in_scope,
            "source_system": "PathDB/TCIA",
            "source_project": spec.gdc_project,
            "source_collection": spec.dataset,
            "source_download_id": spec.tcia_download_id or "",
            "source_download_slug": spec.tcia_download_slug or "",
            "source_download_size": spec.tcia_download_size or "",
            "source_download_size_unit": spec.tcia_download_size_unit or "",
            "case_submitter_id": item.get("Case_ID"),
            "case_uuid": parse_gdc_case_uuid(item.get("GDC_Link")),
            "specimen_id": item.get("Specimen_ID"),
            "sample_submitter_id": item.get("Specimen_ID"),
            "sample_uuid": "",
            "slide_id": slide_id,
            "slide_file_id": item.get("Pathology"),
            "slide_file_name": expected_name,
            "expected_svs_filename": expected_name,
            "data_category": "Biospecimen",
            "data_type": "Slide Image",
            "experimental_strategy": "Diagnostic Slide",
            "data_format": "SVS",
            "access": "open",
            "file_size_bytes": "",
            "md5sum": "",
            "sample_type": "",
            "sample_type_code": "",
            "is_tumor_slide": bool_text(is_tumor),
            "specimen_type": specimen_type,
            "disease_type": disease_type,
            "tumor": item.get("Tumor"),
            "tumor_site": item.get("Tumor_Site"),
            "topographic_site": item.get("Topographic_Site"),
            "tumor_histological_type": item.get("Tumor_Histological_Type"),
            "tumor_segment_acceptable": title_yes(item.get("Tumor_Segment_Acceptable")),
            "percent_tumor_nuclei": item.get("Percent_Tumor_Nuclei"),
            "percent_total_cellularity": item.get("Percent_Total_Cellularity"),
            "percent_necrosis": item.get("Percent_Necrosis"),
            "embedding_medium": item.get("Embedding_Medium"),
            "pathology_id": item.get("Pathology"),
            "genomics_available": item.get("Genomics_Available"),
            "genomics_case_id": item.get("Genomics"),
            "gdc_link": item.get("GDC_Link"),
            "proteomics_available": item.get("Proteomics_Available"),
            "proteomics_case_id": item.get("Proteomics"),
            "pdc_link": item.get("PDC_Link"),
            "planned_for_extraction_v0": "yes" if in_scope == "yes" else "no",
            "source_metadata_version": SOURCE_METADATA_VERSION,
            "source_metadata_file": str(pathdb_path),
            "source_retrieved_at": retrieved_at,
            "notes": notes,
        }
        rows.append(row)
    return rows


def build_wsi_table(pathdb_path: Path, retrieved_at: str) -> pd.DataFrame:
    pathdb = pd.read_csv(pathdb_path)
    rows: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        if spec.source_system == "GDC":
            rows.extend(build_tcga_wsi_rows(spec, retrieved_at))
        else:
            rows.extend(build_cptac_wsi_rows(spec, pathdb, pathdb_path, retrieved_at))
    df = pd.DataFrame(rows)
    return df.reindex(columns=WSI_COLUMNS).sort_values(["dataset", "case_submitter_id", "slide_id"]).reset_index(drop=True)


def build_rna_table(wsi_table: pd.DataFrame, retrieved_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    case_scope = (
        wsi_table.groupby("dataset")["case_submitter_id"]
        .apply(lambda values: sorted({str(v) for v in values.dropna().tolist() if normalize_missing(v)}))
        .to_dict()
    )
    in_scope_cases = {
        dataset: set(
            wsi_table.loc[
                (wsi_table["dataset"] == dataset) & (wsi_table["in_scope_for_main"] == "yes"),
                "case_submitter_id",
            ].dropna().astype(str)
        )
        for dataset in case_scope
    }
    case_uuid_map = (
        wsi_table.dropna(subset=["case_submitter_id"])
        .drop_duplicates(subset=["dataset", "case_submitter_id"])
        .set_index(["dataset", "case_submitter_id"])["case_uuid"]
        .to_dict()
    )
    for spec in DATASET_SPECS:
        case_ids = case_scope.get(spec.dataset, [])
        stderr(f"[RNA] Querying GDC STAR-counts files for {spec.dataset} ({len(case_ids)} WSI cases)")
        file_infos = query_gdc_files(
            project=spec.gdc_project,
            data_type="Gene Expression Quantification",
            case_submitter_ids=case_ids,
            workflow_type="STAR - Counts",
        )
        for file_info in file_infos:
            flat = flatten_gdc_file(file_info)
            case_id = empty_string(flat.get("case_submitter_id"))
            sample_type = flat.get("sample_type")
            tumor_sample = is_tumor_sample_type(sample_type, flat.get("sample_submitter_id"))
            row = {
                "rna_id": f"{spec.dataset}:{flat.get('file_id')}",
                "table_version": SCRIPT_VERSION,
                "cohort_role": spec.cohort_role,
                "dataset": spec.dataset,
                "cancer_context": spec.cancer_context,
                "in_scope_for_main": "yes" if case_id in in_scope_cases.get(spec.dataset, set()) else "no",
                "source_system": "GDC",
                "source_project": spec.gdc_project,
                "case_submitter_id": case_id,
                "case_uuid": flat.get("case_uuid") or case_uuid_map.get((spec.dataset, case_id), ""),
                "sample_submitter_id": flat.get("sample_submitter_id"),
                "sample_uuid": flat.get("sample_uuid"),
                "aliquot_submitter_id": flat.get("aliquot_submitter_id"),
                "file_id": flat.get("file_id"),
                "file_name": flat.get("file_name"),
                "data_category": flat.get("data_category"),
                "data_type": flat.get("data_type"),
                "experimental_strategy": flat.get("experimental_strategy"),
                "workflow_type": flat.get("workflow_type"),
                "data_format": flat.get("data_format"),
                "access": flat.get("access"),
                "file_size_bytes": flat.get("file_size_bytes"),
                "md5sum": flat.get("md5sum"),
                "sample_type": sample_type,
                "is_tumor_sample": bool_text(tumor_sample),
                "primary_tumor_candidate_v0": "yes" if tumor_sample is True else "no",
                "source_metadata_version": SOURCE_METADATA_VERSION,
                "source_retrieved_at": retrieved_at,
                "notes": "",
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.reindex(columns=RNA_COLUMNS).sort_values(["dataset", "case_submitter_id", "file_name"]).reset_index(drop=True)


def grade_low_high(raw_grade: Any) -> tuple[str | None, int | None, str]:
    grade = normalize_missing(raw_grade)
    if not grade:
        return None, None, "missing_grade"
    text = grade.upper()
    if text in {"G1", "GRADE 1"}:
        return "low", 0, ""
    if text in {"G2", "GRADE 2"}:
        return "low", 0, ""
    if text in {"G3", "GRADE 3"}:
        return "high", 1, ""
    if text in {"G4", "GRADE 4"}:
        return "high", 1, ""
    return None, None, f"unmapped_grade:{grade}"


def stage_early_late(raw_stage: Any) -> tuple[str | None, int | None, str]:
    stage = normalize_missing(raw_stage)
    if not stage:
        return None, None, "missing_stage"
    text = stage.upper().strip()
    for prefix in ("PATHOLOGIC", "PATHOLOGICAL", "STAGE", "FIGO"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.replace(" ", "")
    if not text or text in {"X", "NX", "MX", "0", "TIS", "IS"}:
        return None, None, f"excluded_stage:{stage}"
    if text.startswith("IV") or text.startswith("4") or text.startswith("III") or text.startswith("3"):
        return "late", 1, ""
    if text.startswith("II") or text.startswith("2") or text.startswith("I") or text.startswith("1"):
        return "early", 0, ""
    return None, None, f"unmapped_stage:{stage}"


def extract_first_diagnosis_value(case: dict[str, Any], field: str) -> Any:
    for diagnosis in case.get("diagnoses") or []:
        value = diagnosis.get(field)
        if normalize_missing(value):
            return value
    return None


def positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if pd.isna(numeric) or numeric <= 0:
        return None
    return numeric


def extract_os(case: dict[str, Any]) -> tuple[int | None, float | None, str]:
    demographic = case.get("demographic") or {}
    vital_status = normalize_missing(demographic.get("vital_status"))
    event: int | None = None
    if vital_status:
        status = vital_status.lower()
        if status == "dead":
            event = 1
        elif status == "alive":
            event = 0

    candidate_times: list[float] = []
    death_days = positive_float(demographic.get("days_to_death"))
    if death_days is not None:
        candidate_times.append(death_days)
    for diagnosis in case.get("diagnoses") or []:
        follow = positive_float(diagnosis.get("days_to_last_follow_up"))
        if follow is not None:
            candidate_times.append(follow)
    for followup in case.get("follow_ups") or []:
        follow = positive_float(followup.get("days_to_follow_up"))
        if follow is not None:
            candidate_times.append(follow)

    if event == 1 and death_days is not None:
        time_days = death_days
    elif candidate_times:
        time_days = max(candidate_times)
    else:
        time_days = None

    if event is None:
        return None, time_days, "missing_vital_status"
    if time_days is None:
        return event, None, "missing_positive_os_time"
    return event, time_days, ""


def cbio_samples(study_id: str, sample_list_id: str) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        f"{CBIO_BASE}/studies/{study_id}/samples",
        params={"projection": "SUMMARY", "sampleListId": sample_list_id},
        timeout=120,
    )


def cbio_mutated_patients(study_id: str, mutation_profile: str, sample_list_id: str, entrez_gene_id: int) -> set[str]:
    payload = {"sampleListId": sample_list_id, "entrezGeneIds": [entrez_gene_id]}
    rows = request_json(
        "POST",
        f"{CBIO_BASE}/molecular-profiles/{mutation_profile}/mutations/fetch",
        params={"projection": "DETAILED"},
        json_payload=payload,
        timeout=120,
    )
    mutated: set[str] = set()
    for row in rows:
        mutation_type = row.get("mutationType")
        patient_id = normalize_missing(row.get("patientId"))
        if patient_id and mutation_type in PATHOGENIC_VARIANTS:
            mutated.add(patient_id)
    return mutated


def cbio_clinical_values_by_patient(study_id: str, clinical_data_type: str, attribute_id: str) -> dict[str, set[str]]:
    rows = request_json(
        "GET",
        f"{CBIO_BASE}/studies/{study_id}/clinical-data",
        params={"clinicalDataType": clinical_data_type, "projection": "SUMMARY"},
        timeout=180,
    )
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("clinicalAttributeId") != attribute_id:
            continue
        patient_id = normalize_missing(row.get("patientId"))
        value = normalize_missing(row.get("value"))
        if patient_id and value:
            values[patient_id].add(value)
    return values


def ucec_histology_from_text(raw: Any) -> tuple[str | None, int | None, str]:
    text = normalize_missing(raw)
    if not text:
        return None, None, "missing_histology"
    upper = text.upper()
    has_endometrioid = "ENDOMETRIOID" in upper or "UTERINE ENDOMETRIOID" in upper
    has_serous = "SEROUS" in upper or "PAPILLARY SEROUS" in upper
    has_mixed = "MIXED" in upper
    has_other = any(term in upper for term in ["CLEAR CELL", "MUCINOUS", "CARCINOSARCOMA"])
    if has_serous and not has_endometrioid and not has_mixed:
        return "serous", 1, ""
    if has_endometrioid and not has_serous and not has_mixed and not has_other:
        return "endometrioid", 0, ""
    return None, None, f"excluded_histology:{text}"


def make_label_row(
    *,
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    case_id: str,
    case_uuid: str | None,
    in_scope_for_main: str,
    raw_label_value: Any,
    mapped_label: str | None,
    label_numeric: int | None,
    label_status: str,
    exclusion_reason: str,
    label_source_system: str,
    label_source_field: str,
    retrieved_at: str,
    event: int | None = None,
    time_days: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    time_months = None if time_days is None else time_days / 30.4375
    return {
        "label_id": f"{spec.dataset}:{case_id}:{endpoint['endpoint_name']}",
        "table_version": SCRIPT_VERSION,
        "analysis_set": endpoint["analysis_set"],
        "cohort_role": spec.cohort_role,
        "dataset": spec.dataset,
        "cancer_context": spec.cancer_context,
        "case_submitter_id": case_id,
        "case_uuid": case_uuid or "",
        "in_scope_for_main": in_scope_for_main,
        "endpoint_name": endpoint["endpoint_name"],
        "endpoint_family": endpoint["endpoint_family"],
        "task_type": endpoint["task_type"],
        "expected_regime_role": endpoint["expected_regime_role"],
        "raw_label_value": raw_label_value if raw_label_value is not None else "",
        "mapped_label": mapped_label or "",
        "label_numeric": label_numeric if label_numeric is not None else "",
        "event": event if event is not None else "",
        "time_days": round(float(time_days), 4) if time_days is not None else "",
        "time_months": round(float(time_months), 4) if time_months is not None else "",
        "label_status": label_status,
        "exclusion_reason": exclusion_reason,
        "positive_class": endpoint["positive_class"],
        "negative_class": endpoint["negative_class"],
        "label_source_system": label_source_system,
        "label_source_field": label_source_field,
        "mapping_rule_id": endpoint["mapping_rule_id"],
        "mapping_rule_version": SCRIPT_VERSION,
        "source_retrieved_at": retrieved_at,
        "notes": notes,
    }


def build_case_scope(wsi_table: pd.DataFrame) -> dict[str, dict[str, dict[str, str]]]:
    scope: dict[str, dict[str, dict[str, str]]] = {}
    for dataset, group in wsi_table.groupby("dataset"):
        cases: dict[str, dict[str, str]] = {}
        for _, row in group.iterrows():
            case_id = empty_string(row.get("case_submitter_id"))
            if not case_id:
                continue
            if case_id not in cases:
                cases[case_id] = {
                    "case_uuid": empty_string(row.get("case_uuid")),
                    "in_scope_for_main": "no",
                }
            if row.get("in_scope_for_main") == "yes":
                cases[case_id]["in_scope_for_main"] = "yes"
            if row.get("case_uuid") and not cases[case_id].get("case_uuid"):
                cases[case_id]["case_uuid"] = empty_string(row.get("case_uuid"))
        scope[dataset] = cases
    return scope


def build_gdc_clinical_cache(case_scope: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, dict[str, Any]]]:
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in DATASET_SPECS:
        case_ids = case_scope.get(spec.dataset, {}).keys()
        stderr(f"[Label] Querying GDC clinical cases for {spec.dataset}")
        cache[spec.dataset] = query_gdc_cases(spec.gdc_project, case_ids)
    return cache


def build_grade_labels(
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    cases: dict[str, dict[str, str]],
    clinical: dict[str, dict[str, Any]],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    rows = []
    for case_id, meta in sorted(cases.items()):
        raw = extract_first_diagnosis_value(clinical.get(case_id, {}), "tumor_grade")
        mapped, numeric, reason = grade_low_high(raw)
        status = "usable" if mapped and meta["in_scope_for_main"] == "yes" else "missing"
        exclusion = reason
        if meta["in_scope_for_main"] != "yes":
            status = "excluded"
            exclusion = "outside_main_disease_context"
        rows.append(
            make_label_row(
                endpoint=endpoint,
                spec=spec,
                case_id=case_id,
                case_uuid=meta.get("case_uuid"),
                in_scope_for_main=meta["in_scope_for_main"],
                raw_label_value=raw,
                mapped_label=mapped,
                label_numeric=numeric,
                label_status=status,
                exclusion_reason=exclusion,
                label_source_system="GDC",
                label_source_field="diagnoses.tumor_grade",
                retrieved_at=retrieved_at,
            )
        )
    return rows


def build_stage_labels(
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    cases: dict[str, dict[str, str]],
    clinical: dict[str, dict[str, Any]],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    rows = []
    for case_id, meta in sorted(cases.items()):
        raw = extract_first_diagnosis_value(clinical.get(case_id, {}), "ajcc_pathologic_stage")
        mapped, numeric, reason = stage_early_late(raw)
        status = "usable" if mapped and meta["in_scope_for_main"] == "yes" else "missing"
        exclusion = reason
        if meta["in_scope_for_main"] != "yes":
            status = "excluded"
            exclusion = "outside_main_disease_context"
        rows.append(
            make_label_row(
                endpoint=endpoint,
                spec=spec,
                case_id=case_id,
                case_uuid=meta.get("case_uuid"),
                in_scope_for_main=meta["in_scope_for_main"],
                raw_label_value=raw,
                mapped_label=mapped,
                label_numeric=numeric,
                label_status=status,
                exclusion_reason=exclusion,
                label_source_system="GDC",
                label_source_field="diagnoses.ajcc_pathologic_stage",
                retrieved_at=retrieved_at,
            )
        )
    return rows


def build_os_labels(
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    cases: dict[str, dict[str, str]],
    clinical: dict[str, dict[str, Any]],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    rows = []
    for case_id, meta in sorted(cases.items()):
        event, time_days, reason = extract_os(clinical.get(case_id, {}))
        mapped = "event" if event == 1 else "censored" if event == 0 else None
        status = "usable" if event is not None and time_days is not None and meta["in_scope_for_main"] == "yes" else "missing"
        exclusion = reason
        if meta["in_scope_for_main"] != "yes":
            status = "excluded"
            exclusion = "outside_main_disease_context"
        rows.append(
            make_label_row(
                endpoint=endpoint,
                spec=spec,
                case_id=case_id,
                case_uuid=meta.get("case_uuid"),
                in_scope_for_main=meta["in_scope_for_main"],
                raw_label_value=mapped,
                mapped_label=mapped,
                label_numeric=None,
                event=event,
                time_days=time_days,
                label_status=status,
                exclusion_reason=exclusion,
                label_source_system="GDC",
                label_source_field="demographic.vital_status;demographic.days_to_death;diagnoses.days_to_last_follow_up;follow_ups.days_to_follow_up",
                retrieved_at=retrieved_at,
            )
        )
    return rows


def build_mutation_labels(
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    cases: dict[str, dict[str, str]],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    assert spec.cbio_study and spec.cbio_mutation_profile and spec.cbio_sequenced_sample_list
    samples = cbio_samples(spec.cbio_study, spec.cbio_sequenced_sample_list)
    sequenced_patients = {normalize_missing(sample.get("patientId")) for sample in samples}
    sequenced_patients = {patient for patient in sequenced_patients if patient}
    mutated_patients = cbio_mutated_patients(
        spec.cbio_study,
        spec.cbio_mutation_profile,
        spec.cbio_sequenced_sample_list,
        int(endpoint["entrez_gene_id"]),
    )
    rows = []
    for case_id, meta in sorted(cases.items()):
        if case_id in sequenced_patients:
            mutated = case_id in mutated_patients
            mapped = "mutated" if mutated else "wildtype"
            numeric = 1 if mutated else 0
            status = "usable" if meta["in_scope_for_main"] == "yes" else "excluded"
            exclusion = "" if status == "usable" else "outside_main_disease_context"
            raw = "pathogenic_variant_present" if mutated else "no_pathogenic_variant_in_cbioportal_sequenced_samples"
        else:
            mapped = None
            numeric = None
            status = "missing" if meta["in_scope_for_main"] == "yes" else "excluded"
            exclusion = "not_in_cbioportal_sequenced_sample_list"
            if meta["in_scope_for_main"] != "yes":
                exclusion = "outside_main_disease_context"
            raw = ""
        rows.append(
            make_label_row(
                endpoint=endpoint,
                spec=spec,
                case_id=case_id,
                case_uuid=meta.get("case_uuid"),
                in_scope_for_main=meta["in_scope_for_main"],
                raw_label_value=raw,
                mapped_label=mapped,
                label_numeric=numeric,
                label_status=status,
                exclusion_reason=exclusion,
                label_source_system="cBioPortal",
                label_source_field=f"{spec.cbio_mutation_profile}:{endpoint['gene']}",
                retrieved_at=retrieved_at,
            )
        )
    return rows


def build_ucec_histology_labels(
    endpoint: dict[str, Any],
    spec: DatasetSpec,
    cases: dict[str, dict[str, str]],
    pathdb: pd.DataFrame,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    rows = []
    if spec.dataset == "TCGA-UCEC":
        values = cbio_clinical_values_by_patient(spec.cbio_study or "", "SAMPLE", "CANCER_TYPE_DETAILED")
        for case_id, meta in sorted(cases.items()):
            raw_values = sorted(values.get(case_id, set()))
            raw = " | ".join(raw_values)
            if len(raw_values) == 1:
                mapped, numeric, reason = ucec_histology_from_text(raw_values[0])
            elif len(raw_values) > 1:
                mapped, numeric, reason = None, None, f"conflicting_histology:{raw}"
            else:
                mapped, numeric, reason = None, None, "missing_histology"
            status = "usable" if mapped else "missing"
            rows.append(
                make_label_row(
                    endpoint=endpoint,
                    spec=spec,
                    case_id=case_id,
                    case_uuid=meta.get("case_uuid"),
                    in_scope_for_main=meta["in_scope_for_main"],
                    raw_label_value=raw,
                    mapped_label=mapped,
                    label_numeric=numeric,
                    label_status=status,
                    exclusion_reason=reason,
                    label_source_system="cBioPortal",
                    label_source_field="CANCER_TYPE_DETAILED",
                    retrieved_at=retrieved_at,
                )
            )
    else:
        hist_by_case = (
            pathdb[pathdb["Tumor"].astype(str).eq("UCEC")]
            .groupby("Case_ID")["Tumor_Histological_Type"]
            .apply(lambda series: sorted({str(v) for v in series.dropna().tolist() if normalize_missing(v)}))
            .to_dict()
        )
        for case_id, meta in sorted(cases.items()):
            raw_values = hist_by_case.get(case_id, [])
            raw = " | ".join(raw_values)
            mapped, numeric, reason = ucec_histology_from_text(raw)
            status = "usable" if mapped else "missing"
            rows.append(
                make_label_row(
                    endpoint=endpoint,
                    spec=spec,
                    case_id=case_id,
                    case_uuid=meta.get("case_uuid"),
                    in_scope_for_main=meta["in_scope_for_main"],
                    raw_label_value=raw,
                    mapped_label=mapped,
                    label_numeric=numeric,
                    label_status=status,
                    exclusion_reason=reason,
                    label_source_system="PathDB",
                    label_source_field="Tumor_Histological_Type",
                    retrieved_at=retrieved_at,
                )
            )
    return rows


def build_label_table(wsi_table: pd.DataFrame, pathdb_path: Path, retrieved_at: str) -> pd.DataFrame:
    pathdb = pd.read_csv(pathdb_path)
    case_scope = build_case_scope(wsi_table)
    clinical_cache = build_gdc_clinical_cache(case_scope)
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        for dataset in endpoint["datasets"]:
            spec = DATASET_BY_NAME[dataset]
            cases = case_scope.get(dataset, {})
            if endpoint["endpoint_family"] == "grade":
                rows.extend(build_grade_labels(endpoint, spec, cases, clinical_cache.get(dataset, {}), retrieved_at))
            elif endpoint["endpoint_family"] == "stage":
                rows.extend(build_stage_labels(endpoint, spec, cases, clinical_cache.get(dataset, {}), retrieved_at))
            elif endpoint["endpoint_family"] == "survival":
                rows.extend(build_os_labels(endpoint, spec, cases, clinical_cache.get(dataset, {}), retrieved_at))
            elif endpoint["endpoint_family"] == "mutation":
                rows.extend(build_mutation_labels(endpoint, spec, cases, retrieved_at))
            elif endpoint["endpoint_name"] == "ucec_histologic_subtype":
                rows.extend(build_ucec_histology_labels(endpoint, spec, cases, pathdb, retrieved_at))
            else:
                raise ValueError(f"Unsupported endpoint: {endpoint['endpoint_name']}")
    df = pd.DataFrame(rows)
    return df.reindex(columns=LABEL_COLUMNS).sort_values(["dataset", "endpoint_name", "case_submitter_id"]).reset_index(drop=True)


def summarize_tables(wsi: pd.DataFrame, rna: pd.DataFrame, labels: pd.DataFrame, started_at: str, finished_at: str) -> dict[str, Any]:
    def counts_by_dataset(df: pd.DataFrame, row_name: str) -> dict[str, Any]:
        if df.empty:
            return {}
        result: dict[str, Any] = {}
        for dataset, group in df.groupby("dataset"):
            entry = {row_name: int(len(group))}
            if "case_submitter_id" in group.columns:
                entry["cases"] = int(group["case_submitter_id"].nunique())
            if "in_scope_for_main" in group.columns:
                entry["in_scope_rows"] = int((group["in_scope_for_main"] == "yes").sum())
            result[dataset] = entry
        return result

    label_summary: dict[str, Any] = {}
    if not labels.empty:
        usable = labels[labels["label_status"] == "usable"].copy()
        for (endpoint, dataset), group in labels.groupby(["endpoint_name", "dataset"]):
            usable_group = usable[(usable["endpoint_name"] == endpoint) & (usable["dataset"] == dataset)]
            mapped_counts = usable_group["mapped_label"].value_counts(dropna=False).to_dict()
            if endpoint.endswith("_os"):
                event_counts = usable_group["event"].value_counts(dropna=False).to_dict()
                mapped_counts = {str(k): int(v) for k, v in event_counts.items()}
            label_summary[f"{endpoint}|{dataset}"] = {
                "rows": int(len(group)),
                "usable": int(len(usable_group)),
                "status_counts": {str(k): int(v) for k, v in group["label_status"].value_counts(dropna=False).to_dict().items()},
                "usable_label_counts": {str(k): int(v) for k, v in mapped_counts.items()},
            }
    return {
        "script": "scripts/build_source_tables_v0.py",
        "script_version": SCRIPT_VERSION,
        "source_metadata_version": SOURCE_METADATA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "wsi": counts_by_dataset(wsi, "slides"),
        "rna": counts_by_dataset(rna, "rna_files"),
        "labels": label_summary,
        "paired_endpoint_feasibility": summarize_paired_endpoint_feasibility(wsi, rna, labels),
        "validation": validate_tables(wsi, rna, labels),
    }


def summarize_paired_endpoint_feasibility(wsi: pd.DataFrame, rna: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    if wsi.empty or rna.empty or labels.empty:
        return {}

    wsi_scope = wsi[wsi["in_scope_for_main"] == "yes"]
    rna_scope = rna[(rna["in_scope_for_main"] == "yes") & (rna["primary_tumor_candidate_v0"] == "yes")]
    usable_labels = labels[(labels["in_scope_for_main"] == "yes") & (labels["label_status"] == "usable")]

    wsi_cases = set(zip(wsi_scope["dataset"].astype(str), wsi_scope["case_submitter_id"].astype(str)))
    rna_cases = set(zip(rna_scope["dataset"].astype(str), rna_scope["case_submitter_id"].astype(str)))

    result: dict[str, Any] = {}
    for (endpoint, dataset), group in usable_labels.groupby(["endpoint_name", "dataset"]):
        label_cases = set(zip(group["dataset"].astype(str), group["case_submitter_id"].astype(str)))
        paired = label_cases & wsi_cases & rna_cases
        paired_case_ids = {case_id for _, case_id in paired}
        paired_group = group[group["case_submitter_id"].astype(str).isin(paired_case_ids)]

        if endpoint.endswith("_os"):
            paired_counts = paired_group["event"].value_counts(dropna=False).to_dict()
        else:
            paired_counts = paired_group["mapped_label"].value_counts(dropna=False).to_dict()

        result[f"{endpoint}|{dataset}"] = {
            "usable_label_cases": int(len(label_cases)),
            "with_in_scope_wsi": int(len(label_cases & wsi_cases)),
            "with_primary_tumor_rna": int(len(label_cases & rna_cases)),
            "with_in_scope_wsi_and_primary_tumor_rna": int(len(paired)),
            "paired_label_counts": {str(k): int(v) for k, v in paired_counts.items()},
        }
    return result


def duplicate_values(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    duplicated = df[df.duplicated(column, keep=False)][column].dropna().astype(str).unique()
    return sorted(duplicated.tolist())


def validate_tables(wsi: pd.DataFrame, rna: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    for df, name, key in ((wsi, "wsi", "wsi_id"), (rna, "rna", "rna_id"), (labels, "labels", "label_id")):
        duplicates = duplicate_values(df, key)
        if duplicates:
            issues.append(f"{name}: duplicate {key}: {duplicates[:5]}")
    if not labels.empty:
        dup_cols = ["dataset", "case_submitter_id", "endpoint_name"]
        duplicated = labels[labels.duplicated(dup_cols, keep=False)]
        if not duplicated.empty:
            issues.append(f"labels: duplicate dataset/case/endpoint rows: {len(duplicated)}")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }


def write_outputs(wsi: pd.DataFrame, rna: pd.DataFrame, labels: pd.DataFrame, report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi.to_csv(out_dir / "wsi_slide_table_v0.csv", index=False)
    rna.to_csv(out_dir / "rna_sample_table_v0.csv", index=False)
    labels.to_csv(out_dir / "label_table_v0.csv", index=False)
    (out_dir / "source_table_build_report_v0.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v0 WSI/RNA/label source tables.")
    parser.add_argument(
        "--pathdb",
        type=Path,
        default=Path("manifests") / "cptac_metadata_07-09-2024.csv",
        help="PathDB CPTAC metadata CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("manifests"),
        help="Output directory for source tables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.pathdb.exists():
        raise FileNotFoundError(f"PathDB metadata CSV not found: {args.pathdb}")

    started_at = utc_now()
    stderr(f"[Start] build_source_tables_v0 at {started_at}")
    retrieved_at = started_at

    wsi = build_wsi_table(args.pathdb, retrieved_at)
    rna = build_rna_table(wsi, retrieved_at)
    labels = build_label_table(wsi, args.pathdb, retrieved_at)

    finished_at = utc_now()
    report = summarize_tables(wsi, rna, labels, started_at, finished_at)
    write_outputs(wsi, rna, labels, report, args.out_dir)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["validation"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
