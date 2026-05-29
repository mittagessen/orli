#!/usr/bin/env python3
"""Standalone orli inference and baseline visualisation — no kraken required.

Loads an orli checkpoint (.ckpt) or weights file (.safetensors), runs the
autoregressive decoder on one or more page images, and saves an overlay PNG
with the predicted baselines drawn on the original image.

Usage examples
--------------
From a Lightning checkpoint (training run):

    python scripts/plot_orli.py \\
        -w experiments/orli_regnetx/checkpoint_03-0.5800.ckpt \\
        -i page.jpg

From a converted safetensors file:

    python scripts/plot_orli.py \\
        -w best_0.5800.safetensors \\
        -i page.jpg

Override image size (must match what the model was trained on):

    python scripts/plot_orli.py \\
        -w checkpoint.ckpt \\
        -i page.jpg \\
        --image-size 1280 960

Save to a specific directory with RTL overlay colour:

    python scripts/plot_orli.py \\
        -w checkpoint.ckpt \\
        -i folio.jpg \\
        -o debug/ \\
        --line-color 0 180 255 \\
        --line-width 3

Multiple images at once:

    python scripts/plot_orli.py \\
        -w checkpoint.ckpt \\
        -i page1.jpg page2.jpg page3.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _overlay_baselines(
    image: Image.Image,
    baselines: list[list[tuple[int, int]]],
    color: tuple[int, int, int] = (255, 50, 50),
    width: int = 2,
    dot_radius: int = 4,
) -> Image.Image:
    # Force a clean RGB copy — avoids black output from palette/RGBA modes.
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for line in baselines:
        if len(line) < 2:
            continue
        pts = [(int(p[0]), int(p[1])) for p in line]
        draw.line(pts, fill=color, width=width)
        for x, y in (pts[0], pts[-1]):
            draw.ellipse(
                (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                fill=color,
            )
    return out


def _add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 4
    draw.rectangle((0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2), fill=(20, 20, 20))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


# ---------------------------------------------------------------------------
# Model loading — two paths: .ckpt and .safetensors
# ---------------------------------------------------------------------------

def _load_from_ckpt(ckpt_path: Path, device: str):
    """Load OrliModel directly from a Lightning checkpoint without kraken."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    cfg = ckpt.get("_module_config")
    if cfg is None:
        raise ValueError(
            "Checkpoint does not contain '_module_config'. "
            "Is this an orli checkpoint?"
        )
    dm_params = ckpt.get("datamodule_hyper_parameters", {})
    data_cfg = dm_params.get("data_config")
    # image_size in orli config is (H, W) — same convention as torchvision v2.Resize.
    image_size = getattr(data_cfg, "image_size", (1280, 960)) if data_cfg else (1280, 960)

    from orli.orli import OrliModel
    from orli.configs import MODEL_VARIANTS

    model_variant = getattr(cfg, "model_variant", "tiny")

    net = OrliModel(
        config={
            "anchors":                getattr(cfg, "anchors", None),
            "model_variant":          model_variant,
            "baseline_num_points":    getattr(cfg, "baseline_num_points", None),
            "curve_fourier_features": getattr(cfg, "curve_fourier_features", True),
            "anchor_embedding":       getattr(cfg, "anchor_embedding", True),
            "direct_point_regression": getattr(cfg, "direct_point_regression", False),
        },
        image_size=image_size,
    )

    # Strip the "net." prefix that Lightning adds.
    state = ckpt["state_dict"]
    net_state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    net.load_state_dict(net_state, strict=True)

    # Ensure user_metadata carries the correct image_size so that
    # prepare_for_inference → get_default_transforms uses the right resolution.
    net.user_metadata["image_size"] = image_size

    net.eval().to(device)
    return net, image_size


def _strip_uuid_prefix(state: dict) -> dict:
    """Strip the UUID namespace prefix that kraken's convert_models adds.

    kraken saves safetensors with keys like:
        ``c0c4e602-c923-4acc-bd66-2e9b0b3b8294.nn.encoder._stem.0.weight``
    We need plain:
        ``nn.encoder._stem.0.weight``
    """
    import re
    uuid_pat = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.'
    )
    stripped = {}
    for k, v in state.items():
        stripped[uuid_pat.sub('', k)] = v
    return stripped


def _detect_variant_from_state(state: dict) -> str:
    """Infer model_variant from state dict key names."""
    keys = list(state.keys())[:10]
    for k in keys:
        if '_stem' in k or '_block1' in k:
            return 'regnetx'
        if 'stages_0' in k or 'stem_0' in k:
            return 'regnetx'
        if 'convnextv2' in k or 'stages.0' in k:
            return 'tiny'
    # Fall back: check encoder key patterns more broadly
    encoder_keys = [k for k in state if 'encoder' in k]
    if encoder_keys:
        sample = encoder_keys[0]
        if '_stem' in sample or '_block' in sample:
            return 'regnetx'
    return 'tiny'  # safe default


def _load_from_safetensors(weights_path: Path, device: str, image_size: tuple[int, int]):
    """Load OrliModel from a safetensors weights file."""
    from safetensors.torch import load_file
    from orli.orli import OrliModel

    state = load_file(str(weights_path))

    # Strip UUID prefix added by kraken's convert_models.
    state = _strip_uuid_prefix(state)

    # Read metadata from the file header.
    import json
    from safetensors import safe_open
    meta = {}
    with safe_open(str(weights_path), framework="pt") as f:
        raw_meta = f.metadata() or {}
        for k, v in raw_meta.items():
            try:
                meta[k] = json.loads(v)
            except Exception:
                meta[k] = v

    stored_config = meta.get("config", {})
    if isinstance(stored_config, str):
        try:
            stored_config = json.loads(stored_config)
        except Exception:
            stored_config = {}

    # Auto-detect model_variant from key names — more reliable than metadata
    # because the metadata may be absent or stale after kraken conversion.
    model_variant = _detect_variant_from_state(state)
    print(f"    Detected model_variant: {model_variant}")

    anchors = stored_config.get("anchors") or meta.get("anchors")
    if anchors is None:
        import json as _json
        from importlib.resources import files
        with files("orli.assets").joinpath("anchors.json").open("r") as fp:
            anchors = tuple(tuple(row) for row in _json.load(fp))

    net = OrliModel(
        config={
            "anchors":                anchors,
            "model_variant":          model_variant,
            "baseline_num_points":    stored_config.get("baseline_num_points"),
            "curve_fourier_features": stored_config.get("curve_fourier_features", True),
            "anchor_embedding":       stored_config.get("anchor_embedding", True),
            "direct_point_regression": stored_config.get("direct_point_regression", False),
        },
        image_size=image_size,
    )
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    return net


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict(
    net,
    image_pil: Image.Image,
    image_size: tuple[int, int],
    device: str,
    max_lines: int = 768,
) -> list[list[tuple[int, int]]]:
    """Run orli autoregressive inference and return baselines in original pixel coords."""
    from orli.configs import OrliSegmentationInferenceConfig

    accelerator = "cuda" if "cuda" in device else "cpu"
    if accelerator == "cuda":
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        fabric_devices = [dev_idx]
    else:
        fabric_devices = 1

    inf_cfg = OrliSegmentationInferenceConfig(
        accelerator=accelerator,
        device=fabric_devices,
        precision="bf16-mixed" if accelerator == "cuda" else "32-true",
        text_direction="horizontal-lr",
        max_predicted_lines=max_lines,
        polygonize=False,
    )
    net.prepare_for_inference(inf_cfg)

    orig_w, orig_h = image_pil.size
    model_h, model_w = image_size
    print(f"    Original image: {orig_w}×{orig_h} px (W×H)")
    print(f"    Model input:    {model_w}×{model_h} px (W×H)")

    # Pass the ORIGINAL image to net.predict().
    # Internally it applies im_transforms (v2.Resize) for the forward pass,
    # then scales [0,1] output coords by (im.width, im.height) of the image
    # passed here.  Coordinates were normalised against original image dims
    # at compile time, so we must pass the original image — not a pre-resized
    # copy — to get correct scaling back to pixel space.
    seg = net.predict(image_pil)

    scale_x = 1.0  # already in original pixel space
    scale_y = 1.0

    baselines = []
    for line in seg.lines:
        if not line.baseline or len(line.baseline) < 2:
            continue
        baselines.append(list(line.baseline))
    return baselines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orli standalone inference — visualise predicted baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-w", "--weights", type=Path, required=True,
                   help="Path to a .ckpt checkpoint or .safetensors weights file.")
    p.add_argument("-i", "--image", type=Path, nargs="+", required=True,
                   help="Input page image(s).")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Output directory. Default: same directory as each image.")
    p.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"),
                   help="Model input size (H W). Read from checkpoint if omitted.")
    p.add_argument("--max-lines", type=int, default=768,
                   help="Maximum lines to generate per page (default 768).")
    p.add_argument("--line-color", nargs=3, type=int, default=[255, 50, 50],
                   metavar=("R", "G", "B"),
                   help="Baseline overlay colour (default: 255 50 50).")
    p.add_argument("--line-width", type=int, default=2,
                   help="Baseline line width in pixels (default 2).")
    p.add_argument("--device", default=None,
                   help="Device: cuda, cuda:0, cpu … (default: cuda if available).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suffix = args.weights.suffix.lower()

    print(f"Loading weights from {args.weights} on {device} …")

    if suffix == ".ckpt":
        net, image_size = _load_from_ckpt(args.weights, device)
    elif suffix in (".safetensors", ".st"):
        image_size = tuple(args.image_size) if args.image_size else (1280, 960)
        net = _load_from_safetensors(args.weights, device, image_size)
    else:
        sys.exit(f"Unsupported weights format: {suffix}. Use .ckpt or .safetensors")

    if args.image_size:
        image_size = tuple(args.image_size)

    print(f"Model ready. image_size (H, W) = {image_size}")

    color = tuple(args.line_color)

    for img_path in args.image:
        if not img_path.exists():
            print(f"  WARNING: image not found, skipping: {img_path}")
            continue

        print(f"  Processing {img_path.name} …")
        image_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        print(f"    Original image: {orig_w}×{orig_h} px (W×H)")
        print(f"    Model input:    {image_size[1]}×{image_size[0]} px (W×H)")

        baselines = _predict(net, image_pil, image_size, device, args.max_lines)
        print(f"    Detected {len(baselines)} baseline(s)")

        overlay = _overlay_baselines(
            image_pil, baselines, color=color, width=args.line_width
        )
        # Burn label into a small strip at the top so it doesn't obscure text.
        overlay = _add_label(overlay, f"{img_path.name}  —  {len(baselines)} lines detected")

        out_dir = args.output_dir or img_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_orli_baselines.png"
        overlay.save(out_path)
        print(f"    Wrote {out_path}")


if __name__ == "__main__":
    main()

