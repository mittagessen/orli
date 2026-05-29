#!/usr/bin/env python3
"""
Overlay baseline polylines from Orli/kraken ALTO-XML or PAGE-XML onto a copy of the page image.

Usage:
  python scripts/draw_baselines_from_xml.py BL_Or_2683-399_alto.xml
  python scripts/draw_baselines_from_xml.py page.xml --image-dir /path/to/images
  python scripts/draw_baselines_from_xml.py outputs/*.xml --output-dir overlays/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _parse_alto_baseline(attr: str) -> list[tuple[int, int]]:
    """ALTO BASELINE: space-separated x y x y ..."""
    coords = attr.split()
    if len(coords) < 4 or len(coords) % 2 != 0:
        return []
    pts = []
    for i in range(0, len(coords), 2):
        try:
            pts.append((int(float(coords[i])), int(float(coords[i + 1]))))
        except ValueError:
            continue
    return pts


def _parse_page_baseline(points_str: str) -> list[tuple[int, int]]:
    """PAGE points: x1,y1 x2,y2 ..."""
    pts = []
    for pair in points_str.split():
        parts = pair.split(",")
        if len(parts) != 2:
            continue
        try:
            pts.append((int(float(parts[0])), int(float(parts[1]))))
        except ValueError:
            continue
    return pts


def _xml_image_hint(xml_path: Path, root: etree._Element) -> str | None:
    ns = root.nsmap.get(None, "")
    if "loc.gov/standards/alto" in ns:
        for el in root.xpath(
            './/*[local-name()="fileName"]'
        ):
            if el.text and el.text.strip():
                return el.text.strip()
    for el in root.xpath('.//*[local-name()="Page"]'):
        name = el.get("imageFilename") or el.get("filename")
        if name:
            return name
    return None


def extract_baselines(xml_path: Path) -> tuple[list[list[tuple[int, int]]], str | None]:
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    ns = root.nsmap.get(None, "")
    polylines: list[list[tuple[int, int]]] = []

    if "loc.gov/standards/alto" in ns:
        for line in root.xpath('.//*[local-name()="TextLine"]'):
            baseline = line.get("BASELINE")
            if baseline:
                pts = _parse_alto_baseline(baseline)
                if len(pts) >= 2:
                    polylines.append(pts)
    else:
        for baseline_el in root.xpath('.//*[local-name()="Baseline"]'):
            points = baseline_el.get("points")
            if points:
                pts = _parse_page_baseline(points)
                if len(pts) >= 2:
                    polylines.append(pts)
        if not polylines:
            for line in root.xpath('.//*[local-name()="TextLine"]'):
                baseline_els = line.xpath('./*[local-name()="Baseline"]')
                if not baseline_els:
                    continue
                points = baseline_els[0].get("points")
                if points:
                    pts = _parse_page_baseline(points)
                    if len(pts) >= 2:
                        polylines.append(pts)

    return polylines, _xml_image_hint(xml_path, root)


def resolve_image_path(
    xml_path: Path,
    image_hint: str | None,
    image_dir: Path,
) -> Path | None:
    candidates: list[Path] = []
    stem = xml_path.name
    for suffix in ("_alto.xml", "_page.xml", ".alto.xml", ".xml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = xml_path.stem

    if image_hint:
        hint_path = Path(image_hint)
        if hint_path.is_file():
            candidates.append(hint_path)
        candidates.append(image_dir / hint_path.name)
        if hint_path.parent != Path("."):
            candidates.append(image_dir / hint_path.stem)

    for ext in IMAGE_EXTS:
        candidates.append(image_dir / f"{stem}{ext}")

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def draw_baselines(
    image_bgr: np.ndarray,
    polylines: list[list[tuple[int, int]]],
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    out = image_bgr.copy()
    for pts in polylines:
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [arr], False, color, thickness, lineType=cv2.LINE_AA)
    return out


def default_output_path(xml_path: Path, output_dir: Path | None) -> Path:
    stem = xml_path.stem
    if stem.endswith("_alto"):
        stem = stem[: -len("_alto")]
    base = output_dir if output_dir is not None else xml_path.parent
    return base / f"{stem}_baselines.jpg"


def process_one(
    xml_path: Path,
    image_dir: Path,
    output_path: Path | None,
    color: tuple[int, int, int],
    thickness: int,
) -> Path:
    polylines, hint = extract_baselines(xml_path)
    if not polylines:
        raise ValueError(f"No baselines found in {xml_path}")

    image_path = resolve_image_path(xml_path, hint, image_dir)
    if image_path is None:
        raise FileNotFoundError(
            f"Could not find image for {xml_path.name}. "
            f"Tried image_dir={image_dir} (hint={hint!r}). "
            f"Pass --image explicitly."
        )

    overlay = draw_baselines(load_image_bgr(image_path), polylines, color, thickness)
    out = output_path or default_output_path(xml_path, None)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), overlay):
        raise OSError(f"Failed to write {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Draw ALTO/PAGE-XML baselines in red on a copy of the page image.",
    )
    parser.add_argument(
        "xml",
        nargs="+",
        type=Path,
        help="ALTO-XML or PAGE-XML file(s)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root,
        help=f"Directory to search for page images (default: {project_root})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Explicit image path (only valid with a single XML file)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (single XML only; default: <stem>_baselines.jpg next to XML)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory when processing multiple XML files",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Baseline line thickness in pixels (default: 3)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="BGR color as R,G,B (default: 255,0,0 = red)",
    )
    args = parser.parse_args(argv)

    try:
        b, g, r = (int(x.strip()) for x in args.color.split(","))
        color = (b, g, r)
    except ValueError:
        parser.error("--color must be three comma-separated integers (BGR order)")

    xml_paths: list[Path] = []
    for p in args.xml:
        if p.is_dir():
            xml_paths.extend(sorted(p.glob("*.xml")))
        else:
            xml_paths.append(p)

    if args.image is not None and len(xml_paths) != 1:
        parser.error("--image requires exactly one XML file")

    for xml_path in xml_paths:
        xml_path = xml_path.resolve()
        if not xml_path.is_file():
            print(f"Skip (not found): {xml_path}", file=sys.stderr)
            continue

        out_path = args.output
        if len(xml_paths) > 1:
            out_path = None
        if args.output_dir is not None:
            stem = xml_path.stem.removesuffix("_alto")
            out_path = args.output_dir / f"{stem}_baselines.jpg"

        try:
            if args.image is not None:
                polylines, _ = extract_baselines(xml_path)
                overlay = draw_baselines(
                    load_image_bgr(args.image.resolve()),
                    polylines,
                    color,
                    args.thickness,
                )
                written = out_path or default_output_path(xml_path, args.output_dir)
                written.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(written), overlay)
            else:
                written = process_one(
                    xml_path,
                    args.image_dir.resolve(),
                    out_path,
                    color,
                    args.thickness,
                )
            n_lines, _ = extract_baselines(xml_path)
            print(f"{xml_path.name}: {len(n_lines)} baselines -> {written}")
        except (ValueError, FileNotFoundError, OSError) as err:
            print(f"Error ({xml_path.name}): {err}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
