"""Validation helpers for WSI UNI2-h artifact directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = (
    "features.h5",
    "thumbnail.jpg",
    "tissue_mask.png",
    "qc_preview.jpg",
    "manifest.json",
    "extract_stdout.log",
    "extract_stderr.log",
)


def validate_wsi_artifacts(root: str | Path, *, simulate_ok: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Validate one slide directory or an artifact root containing manifest files."""

    root_path = Path(root)
    if not root_path.exists():
        return {
            "artifact_root": str(root_path),
            "passed": False,
            "slide_count": 0,
            "passed_slides": 0,
            "failed_slides": 0,
            "slides": [],
            "issues": [f"artifact root does not exist: {root_path}"],
        }
    slide_dirs = discover_slide_dirs(root_path)
    if limit is not None:
        slide_dirs = slide_dirs[: max(0, limit)]
    slides = [validate_slide_dir(slide_dir, simulate_ok=simulate_ok) for slide_dir in slide_dirs]
    failed = [item for item in slides if not item["passed"]]
    issues = [] if slide_dirs else [f"no manifest.json files found under {root_path}"]
    return {
        "artifact_root": str(root_path),
        "passed": bool(slide_dirs) and not failed,
        "slide_count": len(slides),
        "passed_slides": len(slides) - len(failed),
        "failed_slides": len(failed),
        "slides": slides,
        "issues": issues,
    }


def discover_slide_dirs(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("manifest.json")})


def validate_slide_dir(slide_dir: Path, *, simulate_ok: bool = False) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_ARTIFACTS:
        path = slide_dir / name
        item = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}
        files[name] = item
        if not path.exists():
            issues.append(f"missing required artifact: {name}")
        elif name != "extract_stderr.log" and path.stat().st_size <= 0:
            issues.append(f"empty required artifact: {name}")

    manifest = read_manifest(slide_dir / "manifest.json", issues)
    if manifest:
        check_manifest_consistency(slide_dir, manifest, issues, warnings)

    features = slide_dir / "features.h5"
    if features.exists() and features.stat().st_size > 0:
        if simulate_ok and looks_like_simulated_features(features):
            warnings.append("features.h5 is simulated JSON content")
        else:
            check_h5_features(features, manifest, issues, warnings)

    if not simulate_ok:
        for image_name in ("thumbnail.jpg", "tissue_mask.png", "qc_preview.jpg"):
            path = slide_dir / image_name
            if path.exists() and path.stat().st_size > 0:
                check_image(path, issues)

    return {
        "slide_dir": str(slide_dir),
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "files": files,
    }


def read_manifest(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"manifest parse failed: {exc!r}")
        return {}
    if not isinstance(data, dict):
        issues.append("manifest is not a JSON object")
        return {}
    return data


def check_manifest_consistency(slide_dir: Path, manifest: dict[str, Any], issues: list[str], warnings: list[str]) -> None:
    features = slide_dir / "features.h5"
    if features.exists():
        expected_size = manifest.get("features_size_bytes")
        if expected_size not in {None, ""} and int(expected_size) != features.stat().st_size:
            issues.append(f"features_size_bytes mismatch: manifest={expected_size} actual={features.stat().st_size}")
        expected_md5 = str(manifest.get("features_md5") or "").strip().lower()
        if expected_md5 and expected_md5 != file_md5(features).lower():
            issues.append("features_md5 mismatch")
    for key, name in (
        ("thumbnail_jpg", "thumbnail.jpg"),
        ("tissue_mask_png", "tissue_mask.png"),
        ("qc_preview_jpg", "qc_preview.jpg"),
        ("extract_stdout_log", "extract_stdout.log"),
        ("extract_stderr_log", "extract_stderr.log"),
    ):
        manifest_path = str(manifest.get(key) or "")
        if not manifest_path:
            warnings.append(f"manifest missing optional path field: {key}")
            continue
        if Path(manifest_path).name != name:
            warnings.append(f"manifest {key} does not point to {name}: {manifest_path}")


def check_h5_features(path: Path, manifest: dict[str, Any], issues: list[str], warnings: list[str]) -> None:
    try:
        import h5py
    except ImportError:
        issues.append("h5py is required for real features.h5 validation")
        return
    try:
        with h5py.File(path, "r") as h5:
            dataset = first_existing_dataset(h5, ("features", "embedding", "embeddings", "x"))
            if dataset is None:
                issues.append("features.h5 has no feature dataset")
                return
            shape = tuple(int(item) for item in dataset.shape)
            if len(shape) not in {2, 3}:
                issues.append(f"feature dataset has unexpected shape: {shape}")
                return
            tile_count = shape[1] if len(shape) == 3 else shape[0]
            if tile_count < 0:
                issues.append("feature dataset has invalid tile count")
            attr_tiles = h5.attrs.get("num_tiles")
            if attr_tiles is not None and int(attr_tiles) != int(tile_count):
                issues.append(f"num_tiles attr mismatch: attr={attr_tiles} features={tile_count}")
            if manifest.get("num_tiles") not in {None, ""} and int(manifest["num_tiles"]) != int(tile_count):
                issues.append(f"manifest num_tiles mismatch: manifest={manifest['num_tiles']} features={tile_count}")
            if "coords" not in h5:
                warnings.append("features.h5 has no coords dataset")
    except Exception as exc:
        issues.append(f"features.h5 open/validation failed: {exc!r}")


def first_existing_dataset(h5: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in h5:
            return h5[name]
    return None


def check_image(path: Path, issues: list[str]) -> None:
    try:
        from PIL import Image
    except ImportError:
        issues.append("PIL is required for image artifact validation")
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        issues.append(f"{path.name} image validation failed: {exc!r}")


def looks_like_simulated_features(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    except OSError:
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(data, dict) and data.get("simulated") is True)


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()
