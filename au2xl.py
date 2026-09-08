#!/usr/bin/env python3
"""
Convert Audacity labels/timing file (.txt) to xLights .xtiming file.

Usage:
    python audacity_to_xtiming.py input_labels.txt output.xtiming
    python audacity_to_xtiming.py input_labels.txt output.xtiming --name "Beats"
    python audacity_to_xtiming.py input_labels.txt output.xtiming --name "Phrases" --min-duration 50
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from my_utils import ANSI, good, error, warn, info, debug, memo, relpath, elapsed

APP_VERSION = "0.1.0"
#APP_NAME = "LightBench"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #Path(__file__).parent


def parse_audacity_labels(filepath: str) -> list[tuple[float, float, str]]:
    """
    Parse an Audacity labels file.
    Returns list of (start_sec, end_sec, label).
    """
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("\\"):  # skip empty / extended frequency lines
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                warn(f"Warning: skipping malformed line {line_num}: {line}")
                continue

            try:
                start = float(parts[0])
                end = float(parts[1]) if len(parts) > 1 else start
                label = parts[2].strip() if len(parts) > 2 else ""
            except ValueError:
                error(f"Warning: could not parse times on line {line_num}: {line}")
                continue

            if end < start:
                start, end = end, start  # swap if inverted

            labels.append((start, end, label))

    # Sort by start time
    labels.sort(key=lambda x: x[0])
    return labels


def create_xtiming(
    labels: list[tuple[float, float, str]],
    track_name: str = "Audacity Timing",
    min_duration_ms: int = 0,
) -> str:
    """
    Build a valid xLights .xtiming XML string.
    Times are converted from seconds → milliseconds.
    """
    # Root
    root = ET.Element("timings")

    timing = ET.SubElement(root, "timing")
    timing.set("name", track_name)
    # "fixed" or empty is fine for freeform/imported marks
    timing.set("type", "")

    effect_layer = ET.SubElement(timing, "EffectLayer")

    for start_sec, end_sec, label in labels:
        start_ms = int(round(start_sec * 1000))
        end_ms = int(round(end_sec * 1000))

        # Ensure a minimum visible duration for pure point labels
        if end_ms <= start_ms:
            end_ms = start_ms + max(min_duration_ms, 1)

        effect = ET.SubElement(effect_layer, "Effect")
        effect.set("label", label)
        effect.set("startTime", str(start_ms))
        effect.set("endTime", str(end_ms))
        # Optional attributes that some exporters include
        # effect.set("id", "0")
        # effect.set("protected", "0")

    # Pretty-print
    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding=None)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Audacity labels (.txt) to xLights .xtiming"
    )
    parser.add_argument("input", help="Audacity labels file (.txt)")
    parser.add_argument("output", help="Output .xtiming file")
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Name of the timing track inside the .xtiming file "
             "(default: derived from input filename)",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=0,
        help="Minimum duration in ms for point labels (default: 0 → 1 ms)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(error(f"Input file not found: {input_path}"))

    track_name = args.name or input_path.stem

    debug(f"Reading Audacity labels from: {input_path}")
    labels = parse_audacity_labels(str(input_path))
    good(f"Found {len(labels)} label(s)")

    if not labels:
        raise SystemExit(error("No valid labels found – aborting."))

    xml_content = create_xtiming(
        labels,
        track_name=track_name,
        min_duration_ms=args.min_duration,
    )

    # Add XML declaration
    full_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_content

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".xtiming":
        output_path = output_path.with_suffix(".xtiming")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_xml)

    good(f"Wrote {output_path}")
    info(f"Timing track name: \"{track_name}\"")
    memo("Import into xLights via: right-click timing area → Import Timing Track")
    debug("elapsed", elapsed())


if __name__ == "__main__":
    main()

#eof