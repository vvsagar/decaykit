"""Command line interface for decaykit."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import polars as pl
import polars_root as pr
import uproot

from .decaystring import add_decay_columns, count_decays

_MODES = ("final", "prompt", "raw")

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


def _confirm_overwrite(
    path: Path, assume_yes: bool, parser: argparse.ArgumentParser
) -> None:
    """Ask before clobbering an existing output, unless -y was given."""
    if assume_yes or not path.exists():
        return
    if sys.stdin is None or not sys.stdin.isatty():
        # No terminal to ask: refuse instead of silently clobbering.
        parser.error(f"output exists: {path} (pass -y to overwrite)")
    try:
        answer = input(f"{path} exists. Overwrite? [y/N] ")
    except EOFError:
        parser.error(f"output exists: {path} (pass -y to overwrite)")
    if answer.strip().lower() not in ("y", "yes"):
        parser.exit(1, "aborted\n")


def _source(
    input: Path, tree: str | None, parser: argparse.ArgumentParser
) -> tuple[pl.LazyFrame, Path]:
    """Return the lazy input together with the output it defaults to."""
    suffix = input.suffix.lower()
    if suffix == ".root":
        if tree is None:
            # A .root input is converted, never rewritten: write its sibling .pq.
            with uproot.open(input) as f:
                if _DEFAULT_TREE not in f:
                    parser.error(
                        f"{input} has no '{_DEFAULT_TREE}' tree: pass -t/--tree"
                    )
            tree = _DEFAULT_TREE
        # Converted, never rewritten: name the sibling after its tree as well.
        return pr.scan_root(str(input), tree), input.parent / f"{input.stem}_{tree}.pq"

    if tree is not None:
        parser.error("-t/--tree applies only to .root inputs")
    if suffix not in (".parquet", ".pq"):
        parser.error(
            f"unsupported input: {input} (expected .parquet, .pq or .root)"
        )
    # Stream instead of materializing the whole file: only the decay column
    # feeds the new expressions, and the other columns are passed through.
    return pl.scan_parquet(input), input


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
        "a .root input becomes <input>_<tree>.pq)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="overwrite an existing output file without asking",
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
        "--column",
        default="__MCDecayString__",
        help="decay-string column to read (default: __MCDecayString__)",
    )
    parser.add_argument(
        "--particle",
        default="B+",
        help="particle whose decay to extract (default: B+)",
    )
    parser.add_argument(
        "--stable",
        "--consider-as-stable",
        dest="stable",
        action="append",
        help="particle to keep as a leaf in final mode, repeatable "
        "(e.g. --stable pi0 --stable K_S0)",
    )
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=1,
        help="how many decays of the particle to extract (default: 1)",
    )
    parser.add_argument(
        "--keep-isr",
        action="store_false",
        dest="ignore_isr",
        help="keep gammaI (ISR) photons instead of stripping them",
    )
    parser.add_argument(
        "--keep-fsr",
        action="store_false",
        dest="ignore_fsr",
        help="keep gammaF (FSR) photons instead of stripping them",
    )
    parser.add_argument(
        "--keep-photos",
        action="store_false",
        dest="ignore_photos",
        help="keep gammaP (PHOTOS) photons instead of stripping them",
    )
    parser.add_argument(
        "--no-sort",
        action="store_false",
        dest="sort_daughters",
        help="keep daughter order from the decay string (default: sort by mass)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="also add '<particle>_count'",
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
    if args.max_occurrences < 1:
        parser.error(
            f"--max-occurrences must be >= 1, got {args.max_occurrences}"
        )
    if args.stable and "final" not in modes:
        parser.error("--stable requires the final mode")

    lf, default_output = _source(args.input, args.tree, parser)
    output = args.output or default_output
    # An in-place parquet rewrite (no -o) is the documented default: never ask.
    # Everything else -- -o with either input type, and every .root conversion --
    # writes a file that may already exist, so confirm first.
    in_place = output == args.input and args.input.suffix.lower() != ".root"
    if not in_place:
        _confirm_overwrite(output, args.yes, parser)

    if args.count:
        lf = lf.with_columns(
            count_decays(col_name=args.column, particle=args.particle)
        )
    lf = add_decay_columns(
        lf,
        col_name=args.column,
        particle=args.particle,
        modes=modes,
        n_workers=args.n_workers,
        ignore_isr=args.ignore_isr,
        ignore_fsr=args.ignore_fsr,
        ignore_photos=args.ignore_photos,
        sort_daughters=args.sort_daughters,
        consider_as_stable=args.stable,
        max_occurrences=args.max_occurrences,
    )
    _sink_parquet(lf, output)
    print(f"wrote {output}")
