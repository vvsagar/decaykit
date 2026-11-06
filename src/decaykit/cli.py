"""Command line interface for decaykit."""

import argparse
import os
import tempfile
from pathlib import Path

import polars as pl
import polars_root as pr
import uproot

from .decaystring import extract_decay_final, extract_decay_prompt, extract_decay_raw

_MODES = {
    "final": extract_decay_final,
    "prompt": extract_decay_prompt,
    "raw": extract_decay_raw,
}

# Tree written by belle2 VariablesToNtuple. A ROOT input that already has it
# can be read without -t/--tree.
_DEFAULT_TREE = "variables"


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


def _source(
    input: Path, tree: str | None, parser: argparse.ArgumentParser
) -> tuple[pl.LazyFrame, Path]:
    """Return the lazy input together with the output it defaults to."""
    if input.suffix.lower() != ".root":
        if tree is not None:
            parser.error("-t/--tree applies only to .root inputs")
        # Stream instead of materializing the whole file: only the decay column
        # feeds the new expressions, and the other columns are passed through.
        return pl.scan_parquet(input), input

    if tree is None:
        # A .root input is converted, never rewritten: write its sibling .pq.
        with uproot.open(input) as f:
            if _DEFAULT_TREE not in f:
                parser.error(f"{input} has no '{_DEFAULT_TREE}' tree: pass -t/--tree")
        tree = _DEFAULT_TREE
    return pr.scan_root(str(input), tree), input.with_suffix(".pq")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="decaykit",
        description="Add decay-string columns to a parquet or ROOT file.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input parquet file (rewritten in place) or .root file "
        "(converted to a sibling .pq)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output parquet file (default: rewrite a parquet input in place; "
        "a .root input becomes <input>.pq)",
    )
    parser.add_argument(
        "-t",
        "--tree",
        help=f"ROOT tree to read; required for a .root input without "
        f"a '{_DEFAULT_TREE}' tree",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=list(_MODES),
        help="variant to add, repeatable (default: all of final, prompt, raw)",
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

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.n_workers < 1:
        parser.error(f"--n-workers must be >= 1, got {args.n_workers}")

    lf, default_output = _source(args.input, args.tree, parser)
    output = args.output or default_output

    lf = lf.with_columns(
        *[
            _MODES[mode](particle=args.particle, n_workers=args.n_workers)
            for mode in modes
        ]
    )
    _sink_parquet(lf, output)
    print(f"wrote {output}")
