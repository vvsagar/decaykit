"""Command line interface for decaykit."""

import argparse
import os
import tempfile
from pathlib import Path

import polars as pl

from .decaystring import extract_decay, extract_decay_final, extract_decay_prompt

_MODES = {
    "final": extract_decay_final,
    "prompt": extract_decay_prompt,
    "decay": extract_decay,
}


def _sink_parquet(lf: pl.LazyFrame, path: Path) -> None:
    """Write atomically, so an in-place run cannot truncate the input."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(fd)
    try:
        lf.sink_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="decaykit",
        description="Add decay-string columns to a parquet file.",
    )
    parser.add_argument("input", type=Path, help="input parquet file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output parquet file (default: rewrite the input in place)",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=list(_MODES),
        help="variant to add, repeatable (default: all of final, prompt, decay)",
    )
    parser.add_argument(
        "--particle",
        default="B+",
        help="particle whose decay to extract (default: B+)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="parse in this many processes (default: 1)",
    )
    args = parser.parse_args(argv)

    modes = args.mode or list(_MODES)
    output = args.output or args.input

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.n_workers < 1:
        parser.error(f"--n-workers must be >= 1, got {args.n_workers}")

    # Stream instead of materializing the whole file: only the decay column
    # feeds the new expressions, and the other columns are passed through.
    lf = pl.scan_parquet(args.input).with_columns(
        *[
            _MODES[mode](particle=args.particle, n_workers=args.n_workers)
            for mode in modes
        ]
    )
    _sink_parquet(lf, output)
    print(f"wrote {output}")
