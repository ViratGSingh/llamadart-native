#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--readelf", required=True, type=Path)
    parser.add_argument("--min-load-align", type=lambda value: int(value, 0), default=0x4000)
    return parser.parse_args()


def read_load_alignments(readelf: Path, library: Path) -> list[int]:
    result = subprocess.run(
        [str(readelf), "-lW", str(library)],
        capture_output=True,
        text=True,
        check=True,
    )
    alignments: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("LOAD"):
            continue
        fields = stripped.split()
        try:
            alignments.append(int(fields[-1], 16))
        except (IndexError, ValueError):
            continue
    return alignments


def main() -> None:
    args = parse_args()
    if not args.readelf.is_file():
        raise SystemExit(f"Missing llvm-readelf binary: {args.readelf}")
    if not args.out_dir.is_dir():
        raise SystemExit(f"Missing Android artifact directory: {args.out_dir}")

    libraries = sorted(args.out_dir.glob("*.so"))
    if not libraries:
        raise SystemExit(f"No shared libraries found under {args.out_dir}")

    errors: list[str] = []
    for library in libraries:
        alignments = read_load_alignments(args.readelf, library)
        if not alignments:
            errors.append(f"{library.name}: no LOAD program headers found")
            continue

        formatted = ", ".join(f"0x{alignment:x}" for alignment in alignments)
        print(f"{library.name}: LOAD alignments {formatted}")
        if any(alignment < args.min_load_align for alignment in alignments):
            errors.append(
                f"{library.name}: expected all LOAD alignments >= 0x{args.min_load_align:x}, got {formatted}"
            )

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
