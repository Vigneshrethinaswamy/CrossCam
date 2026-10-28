"""Generate a Markdown summary from evaluation and track diagnostic JSON files."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.report import write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/evaluation/summary.json")
    parser.add_argument("--diagnostics", type=Path,
                        default=ROOT / "outputs/diagnostics/diagnostics_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/crosscam_report.md")
    args = parser.parse_args()
    if not args.summary.is_file():
        print(f"Evaluation summary not found: {args.summary}", file=sys.stderr)
        return 2
    result = write_report(args.summary, args.output, args.diagnostics)
    print(f"Report written: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
