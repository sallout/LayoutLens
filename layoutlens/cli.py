"""Command line: ``layoutlens IMAGE [--mode geometry|ocr|combined] [--out FILE] [--compact]``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from layoutlens import analyze


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="layoutlens", description="Structured geometry and text metadata from a UI screenshot.")
    ap.add_argument("image")
    ap.add_argument("--mode", choices=analyze.MODES, default="combined")
    ap.add_argument("--out", help="write the JSON here instead of standard output")
    ap.add_argument("--compact", action="store_true", help="one element or text per line (the form given to models)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    data = analyze.analyze(args.image, mode=args.mode)
    text = analyze.to_text(data) if args.compact else json.dumps(data, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"{args.mode}: {len(data.get('elements', []))} elements, {len(data.get('texts', []))} texts in {time.perf_counter() - t0:.2f} s -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
