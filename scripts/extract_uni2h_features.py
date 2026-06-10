#!/usr/bin/env python
"""Extract W8Yi-style UNI2-h tile features from whole-slide images.

This script implements the public extraction specification described in
W8Yi/tcga-wsi-uni2h-features. It is intended for applying the same style of
processing to CPTAC slides and for producing candidate H5 files that can be
compared against W8Yi's TCGA release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "hf-hub:MahmoodLab/UNI2-h",
        "create_model_kwargs": {
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 5.33334,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": "timm.layers.SwiGLUPacked",
            "act_layer": "torch.nn.SiLU",
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        "output_dim": 1536,
    },
    "tiling": {
        "tile_size_20x": 256,
        "step_size_20x": 256,
        "input_size": 224,
        "objective_power_fallback": 20.0,
    },
    "tissue_filter": {
        "mask_max_dim": 2048,
        "saturation_min": 20,
        "value_max": 245,
        "median_filter_size": 3,
        "max_filter_size": 5,
        "min_filter_size": 5,
        "tissue_fraction_min": 0.15,
    },
    "preprocess": {
        "resize_resample": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "runtime": {
        "batch_size": 128,
        "precision": "fp16",
        "device": "auto",
    },
    "output": {
        "compression": None,
        "write_overlay": True,
        "write_thumbnail": True,
        "write_tissue_mask": True,
        "write_qc_preview": True,
        "overlay_alpha": 48,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read YAML configs. Install scripts/requirements-uni2h.txt."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def get_resample(name: str):
    from PIL import Image

    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    return Image.Resampling.BICUBIC


def nearest_common_objective(power: float) -> float:
    choices = [5.0, 10.0, 20.0, 40.0, 60.0]
    return min(choices, key=lambda x: abs(x - power))


def objective_power_from_slide(slide, fallback: float, override: float | None) -> float:
    if override is not None:
        return float(override)

    keys = [
        "openslide.objective-power",
        "aperio.AppMag",
        "hamamatsu.SourceLens",
        "mirax.OBJECTIVE_MAGNIFICATION",
    ]
    for key in keys:
        value = slide.properties.get(key)
        if value:
            try:
                return float(str(value).replace("x", "").replace("X", ""))
            except ValueError:
                pass

    for key in ("openslide.mpp-x", "aperio.MPP", "hamamatsu.PhysicalWidth"):
        value = slide.properties.get(key)
        if value:
            try:
                mpp = float(value)
                if mpp > 0:
                    return nearest_common_objective(10.0 / mpp)
            except ValueError:
                pass

    return float(fallback)


def build_tissue_mask(slide, cfg: Dict[str, Any]):
    import numpy as np
    from PIL import Image, ImageFilter

    tf = cfg["tissue_filter"]
    max_dim = int(tf["mask_max_dim"])
    thumb = slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
    hsv = thumb.convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8)
    mask = (arr[:, :, 1] >= int(tf["saturation_min"])) & (
        arr[:, :, 2] <= int(tf["value_max"])
    )

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if int(tf["median_filter_size"]) > 1:
        mask_img = mask_img.filter(ImageFilter.MedianFilter(int(tf["median_filter_size"])))
    if int(tf["max_filter_size"]) > 1:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(int(tf["max_filter_size"])))
    if int(tf["min_filter_size"]) > 1:
        mask_img = mask_img.filter(ImageFilter.MinFilter(int(tf["min_filter_size"])))

    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    return thumb, mask


def integral_image(mask):
    import numpy as np

    return np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def rect_sum(ii, x0: int, y0: int, x1: int, y1: int) -> int:
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def generate_tile_coords(slide, mask, tile_level0: int, step_level0: int, cfg: Dict[str, Any]):
    import numpy as np

    width, height = slide.dimensions
    mask_h, mask_w = mask.shape
    ii = integral_image(mask)
    keep_fraction = float(cfg["tissue_filter"]["tissue_fraction_min"])

    coords: List[Tuple[int, int]] = []
    for y in range(0, max(0, height - tile_level0 + 1), step_level0):
        my0 = int(math.floor(y * mask_h / height))
        my1 = int(math.ceil((y + tile_level0) * mask_h / height))
        my0 = max(0, min(mask_h - 1, my0))
        my1 = max(my0 + 1, min(mask_h, my1))
        for x in range(0, max(0, width - tile_level0 + 1), step_level0):
            mx0 = int(math.floor(x * mask_w / width))
            mx1 = int(math.ceil((x + tile_level0) * mask_w / width))
            mx0 = max(0, min(mask_w - 1, mx0))
            mx1 = max(mx0 + 1, min(mask_w, mx1))
            area = (mx1 - mx0) * (my1 - my0)
            if area <= 0:
                continue
            if rect_sum(ii, mx0, my0, mx1, my1) / area >= keep_fraction:
                coords.append((x, y))
    return np.asarray(coords, dtype=np.int64)


def center_crop(img, size: int):
    w, h = img.size
    if w < size or h < size:
        resample = get_resample("bicubic")
        img = img.resize((max(size, w), max(size, h)), resample)
        w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_tile(img, cfg: Dict[str, Any]):
    import numpy as np
    import torch

    input_size = int(cfg["tiling"]["input_size"])
    img = center_crop(img, input_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.asarray(cfg["preprocess"]["mean"], dtype=np.float32)
    std = np.asarray(cfg["preprocess"]["std"], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def read_tile(slide, coord: Tuple[int, int], tile_level0: int, cfg: Dict[str, Any]):
    target_size = int(cfg["tiling"]["tile_size_20x"])
    resample = get_resample(str(cfg["preprocess"]["resize_resample"]))
    x, y = int(coord[0]), int(coord[1])
    img = slide.read_region((x, y), 0, (tile_level0, tile_level0)).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), resample)
    return img


def load_model(cfg: Dict[str, Any], device: str):
    import timm
    import torch

    kwargs = dict(cfg["model"].get("create_model_kwargs") or {})
    kwargs = resolve_timm_kwargs(kwargs, timm=timm, torch=torch)
    model = timm.create_model(cfg["model"]["name"], **kwargs)
    model.eval()
    model.to(torch.device(device))
    return model


def resolve_timm_kwargs(kwargs: Dict[str, Any], *, timm, torch) -> Dict[str, Any]:
    resolved = dict(kwargs)
    symbol_map = {
        "timm.layers.SwiGLUPacked": timm.layers.SwiGLUPacked,
        "torch.nn.SiLU": torch.nn.SiLU,
    }
    for key, value in list(resolved.items()):
        if isinstance(value, str) and value in symbol_map:
            resolved[key] = symbol_map[value]
    return resolved


def normalize_model_output(out):
    import torch

    if isinstance(out, dict):
        for key in ("features", "embedding", "embeddings", "x"):
            if key in out:
                out = out[key]
                break
        else:
            out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Unsupported model output type: {type(out)!r}")
    if out.ndim == 3:
        out = out[:, 0, :]
    if out.ndim != 2:
        raise ValueError(f"Expected [B, D] model output, got shape {tuple(out.shape)}")
    return out


def embed_batch(model, batch, device: str, precision: str):
    import torch

    batch = batch.to(torch.device(device), non_blocking=True)
    use_amp = device.startswith("cuda") and precision.lower() in {"fp16", "bf16"}
    dtype = torch.float16 if precision.lower() == "fp16" else torch.bfloat16
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(batch)
        else:
            out = model(batch)
    return normalize_model_output(out).float().cpu().numpy()


def resolve_device(device_cfg: str) -> str:
    import torch

    if device_cfg and device_cfg != "auto":
        return device_cfg
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_overlay(thumb, coords, slide_dims, tile_level0: int, out_png: Path, alpha: int):
    make_overlay_image(thumb, coords, slide_dims, tile_level0, alpha).save(out_png)

def make_overlay_image(thumb, coords, slide_dims, tile_level0: int, alpha: int):
    from PIL import Image, ImageDraw

    thumb_rgba = thumb.convert("RGBA")
    overlay = Image.new("RGBA", thumb_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tw, th = thumb_rgba.size
    sw, sh = slide_dims
    fill = (255, 0, 0, int(alpha))
    outline = (255, 0, 0, min(255, int(alpha) * 2))
    for x, y in coords:
        x0 = int(round(x * tw / sw))
        y0 = int(round(y * th / sh))
        x1 = int(round((x + tile_level0) * tw / sw))
        y1 = int(round((y + tile_level0) * th / sh))
        draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), fill=fill, outline=outline)
    return Image.alpha_composite(thumb_rgba, overlay)


def write_tissue_mask(mask, out_png: Path) -> None:
    import numpy as np
    from PIL import Image

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_png)


def write_qc_preview(
    thumb,
    mask,
    coords,
    slide_dims,
    tile_level0: int,
    out_jpg: Path,
    *,
    alpha: int,
    slide_key: str,
    objective: float,
    elapsed_seconds: float,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    overlay = make_overlay_image(thumb, coords, slide_dims, tile_level0, alpha).convert("RGB")
    tissue_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB").resize(overlay.size)
    panel_height = 132
    width = overlay.size[0] * 2
    height = overlay.size[1] + panel_height
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(overlay, (0, 0))
    canvas.paste(tissue_mask, (overlay.size[0], 0))
    draw = ImageDraw.Draw(canvas)
    lines = [
        f"slide_key: {slide_key}",
        f"slide_dims: {slide_dims[0]} x {slide_dims[1]}",
        f"objective_power: {objective:.2f}",
        f"tile_size_level0: {tile_level0}",
        f"num_tiles: {int(coords.shape[0])}",
        f"elapsed_seconds: {elapsed_seconds:.1f}",
    ]
    y = overlay.size[1] + 10
    for line in lines:
        draw.text((12, y), line, fill=(0, 0, 0))
        y += 18
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_jpg, quality=90)


def extract(args: argparse.Namespace) -> None:
    try:
        import h5py
        import numpy as np
        import openslide
        import torch
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install scripts/requirements-uni2h.txt."
        ) from exc

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["runtime"]["batch_size"] = int(args.batch_size)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.no_overlay:
        cfg["output"]["write_overlay"] = False
    if args.no_qc_images:
        cfg["output"]["write_thumbnail"] = False
        cfg["output"]["write_tissue_mask"] = False
        cfg["output"]["write_qc_preview"] = False

    wsi_path = args.wsi.resolve()
    out_h5 = args.out.resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    slide_key = args.slide_id or wsi_path.stem

    slide = openslide.OpenSlide(str(wsi_path))
    objective = objective_power_from_slide(
        slide,
        fallback=float(cfg["tiling"]["objective_power_fallback"]),
        override=args.objective_power,
    )
    tile_level0 = int(round(float(cfg["tiling"]["tile_size_20x"]) * objective / 20.0))
    step_level0 = int(round(float(cfg["tiling"]["step_size_20x"]) * objective / 20.0))

    start_time = time.time()
    thumb, mask = build_tissue_mask(slide, cfg)
    coords = generate_tile_coords(slide, mask, tile_level0, step_level0, cfg)

    device = resolve_device(str(cfg["runtime"]["device"]))
    model = load_model(cfg, device)
    batch_size = int(cfg["runtime"]["batch_size"])
    precision = str(cfg["runtime"]["precision"])
    compression = cfg["output"].get("compression")
    if compression in ("", "null", "none"):
        compression = None

    first_write = True
    feature_ds = None
    expected_dim = int(cfg["model"].get("output_dim") or 0)

    with h5py.File(out_h5, "w") as h5:
        h5.attrs["slide_key"] = slide_key
        h5.attrs["source_wsi"] = str(wsi_path)
        h5.attrs["extractor"] = "extract_uni2h_features.py"
        h5.attrs["config_json"] = json.dumps(cfg, sort_keys=True)
        h5.attrs["config_hash"] = config_hash(cfg)
        h5.attrs["objective_power"] = float(objective)
        h5.attrs["tile_size_level0"] = int(tile_level0)
        h5.attrs["step_size_level0"] = int(step_level0)
        h5.attrs["tile_size_20x"] = int(cfg["tiling"]["tile_size_20x"])
        h5.attrs["step_size_20x"] = int(cfg["tiling"]["step_size_20x"])
        h5.attrs["slide_width"] = int(slide.dimensions[0])
        h5.attrs["slide_height"] = int(slide.dimensions[1])
        h5.create_dataset("coords", data=coords[None, :, :], dtype="int64", compression=compression)
        h5.create_dataset("coords_patching", data=coords, dtype="int64", compression=compression)
        h5.create_dataset(
            "annots",
            data=np.zeros((1, coords.shape[0], 1), dtype=np.int8),
            dtype="int8",
            compression=compression,
        )

        if coords.shape[0] == 0:
            dim = expected_dim if expected_dim > 0 else 1536
            h5.create_dataset("features", shape=(1, 0, dim), dtype="float32")
        else:
            iterator = range(0, coords.shape[0], batch_size)
            for start in tqdm(iterator, total=math.ceil(coords.shape[0] / batch_size), desc=slide_key):
                end = min(start + batch_size, coords.shape[0])
                tiles = [
                    preprocess_tile(read_tile(slide, tuple(coord), tile_level0, cfg), cfg)
                    for coord in coords[start:end]
                ]
                batch = torch.stack(tiles, dim=0)
                feats = embed_batch(model, batch, device, precision)
                if first_write:
                    dim = int(feats.shape[1])
                    chunk_n = min(max(1, batch_size), coords.shape[0])
                    feature_ds = h5.create_dataset(
                        "features",
                        shape=(1, coords.shape[0], dim),
                        dtype="float32",
                        chunks=(1, chunk_n, dim),
                        compression=compression,
                    )
                    first_write = False
                feature_ds[0, start:end, :] = feats.astype("float32", copy=False)

        h5.attrs["num_tiles"] = int(coords.shape[0])
        h5.attrs["elapsed_seconds"] = float(time.time() - start_time)

    elapsed = float(time.time() - start_time)

    if bool(cfg["output"].get("write_thumbnail", True)):
        thumbnail_path = args.thumbnail
        if thumbnail_path is None:
            thumbnail_path = out_h5.with_name(out_h5.stem + "__thumbnail.jpg")
        thumbnail_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        thumb.convert("RGB").save(thumbnail_path.resolve(), quality=90)

    if bool(cfg["output"].get("write_tissue_mask", True)):
        tissue_mask_path = args.tissue_mask
        if tissue_mask_path is None:
            tissue_mask_path = out_h5.with_name(out_h5.stem + "__tissue_mask.png")
        write_tissue_mask(mask, tissue_mask_path.resolve())

    if bool(cfg["output"].get("write_overlay", True)):
        overlay_path = args.overlay
        if overlay_path is None:
            overlay_path = out_h5.with_name(out_h5.stem + "__overlay.png")
        write_overlay(
            thumb,
            coords,
            slide.dimensions,
            tile_level0,
            overlay_path.resolve(),
            int(cfg["output"].get("overlay_alpha", 48)),
        )

    if bool(cfg["output"].get("write_qc_preview", True)):
        qc_preview_path = args.qc_preview
        if qc_preview_path is None:
            qc_preview_path = out_h5.with_name(out_h5.stem + "__qc_preview.jpg")
        write_qc_preview(
            thumb,
            mask,
            coords,
            slide.dimensions,
            tile_level0,
            qc_preview_path.resolve(),
            alpha=int(cfg["output"].get("overlay_alpha", 48)),
            slide_key=slide_key,
            objective=float(objective),
            elapsed_seconds=elapsed,
        )

    print(
        json.dumps(
            {
                "out_h5": str(out_h5),
                "slide_key": slide_key,
                "num_tiles": int(coords.shape[0]),
                "objective_power": float(objective),
                "tile_size_level0": int(tile_level0),
                "step_size_level0": int(step_level0),
                "device": device,
                "elapsed_seconds": round(elapsed, 3),
            },
            indent=2,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi", type=Path, required=True, help="Input WSI path, e.g. .svs")
    parser.add_argument("--out", type=Path, required=True, help="Output H5 path")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uni2h_w8yi_style.yaml"),
        help="YAML extraction config",
    )
    parser.add_argument("--slide-id", default=None, help="Slide key stored in H5 attrs")
    parser.add_argument("--objective-power", type=float, default=None, help="Override slide objective")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument("--overlay", type=Path, default=None, help="Optional overlay PNG path")
    parser.add_argument("--thumbnail", type=Path, default=None, help="Optional raw thumbnail JPG path")
    parser.add_argument("--tissue-mask", type=Path, default=None, help="Optional tissue mask PNG path")
    parser.add_argument("--qc-preview", type=Path, default=None, help="Optional QC preview JPG path")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay generation")
    parser.add_argument("--no-qc-images", action="store_true", help="Skip thumbnail, tissue mask, and QC preview")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
