"""Parse EvtGen decay strings and expose them as Polars expressions.

A decay string looks like ``[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]``:
brackets group a decay, ``->`` separates a parent from its daughters, and a
name is any run of characters that is neither a bracket nor whitespace.

The ``extract_decay*`` functions return a ``polars.Expr`` that can be dropped
straight into ``DataFrame.with_columns``.
"""

import re
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial

import polars as pl
from particle import Particle, ParticleNotFound
from particle.exceptions import MatchingIDNotFound

_CHUNK_SIZE = 50_000

# ``particle`` builds its EvtGen/PDGID lookup tables lazily, and those builds
# are not thread-safe. Polars evaluates expressions in parallel threads, so
# lookups must be serialized; results are cached, so the lock is only taken on
# the first lookup of each name.
_MASS_LOCK = threading.Lock()

# A decay string is a sequence of brackets and of names, where a name is a run
# of characters that is neither a bracket nor whitespace -- the same delimiters
# the character-by-character scanner used ('-', '>' and spaces only ever act as
# separators *between* names, so "pi-" stays one name). Tokenizing once per
# block with a C-level regex is much cheaper than indexing string[i] in Python,
# which allocates a one-character string per character.
_TOKEN_RE = re.compile(r"\[|\]|[^\[\]\s]+")

_SKIP = frozenset({"]", "->"})

_FSR = frozenset({"gammaF", "gammaP", "gammaI"})


@cache
def get_mass(evtgen_name: str, ignore_error: bool = True) -> float:
    """Return the mass of an EvtGen particle name, or ``0`` if it is unknown."""
    if evtgen_name == "gammaF":
        return 0

    try:
        with _MASS_LOCK:
            mass = Particle.from_evtgen_name(evtgen_name).mass
        if mass is None:
            return 0
        else:
            return mass
    except (ParticleNotFound, MatchingIDNotFound):
        if not ignore_error:
            print(f"mass not obtained for {evtgen_name}")
        return 0


def rearrange_daughters(decay_str: str, arrow_str: str = "-->") -> str:
    mother = decay_str.split(arrow_str)[0]
    daughters = decay_str.split(arrow_str)[1].split()
    daughters.sort(key=get_mass, reverse=True)

    return mother + arrow_str + " " + " ".join(daughters)


def _find_blocks(decay_str: str, particle: str, limit: int) -> list[str]:
    """Return the daughter blocks of the first ``limit`` decays, in string order."""
    blocks = []
    if not decay_str:
        return blocks
    marker = f"[{particle} -> "
    marker_len = len(marker)
    start = decay_str.find(marker)
    while start != -1 and len(blocks) < limit:
        pos = start + marker_len
        depth = 1
        # Jump from bracket to bracket instead of stepping over every character.
        while depth:
            close_at = decay_str.find("]", pos)
            if close_at == -1:  # unbalanced: no block for this marker
                pos = len(decay_str)
                break
            open_at = decay_str.find("[", pos)
            if open_at != -1 and open_at < close_at:
                depth += 1
                pos = open_at + 1
                continue
            depth -= 1
            pos = close_at + 1
            if depth == 0:
                blocks.append(decay_str[start + marker_len : close_at])
        start = decay_str.find(marker, pos)
    return blocks


def _skip_group(tokens: list[str], i: int) -> int:
    """Return the index just past the ``]`` closing the ``[`` at ``tokens[i]``."""
    depth = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == "[":
            depth += 1
        elif token == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _prompt_daughters(block: str) -> list[str]:
    """Return the direct daughters of ``block``, in string order."""
    tokens = _TOKEN_RE.findall(block)
    parts = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == "[":
            # direct daughter: its own name, its decay tree is not expanded
            if i + 1 < n and tokens[i + 1] not in ("[", "]", "->"):
                parts.append(tokens[i + 1])
            i = _skip_group(tokens, i)
        elif token in _SKIP:
            i += 1
        else:
            parts.append(token)
            i += 1
    return parts


def _final_daughters(block: str, stable: frozenset[str] = frozenset()) -> list[str]:
    """Return the leaves of the decay tree in ``block``, in string order."""
    tokens = _TOKEN_RE.findall(block)
    parts = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == "[":
            # a "stable" particle is a leaf here: keep it, drop its decay tree
            if i + 1 < n and tokens[i + 1] in stable:
                parts.append(tokens[i + 1])
                i = _skip_group(tokens, i)
                continue
            # otherwise the name belongs to an expanded decay: not a daughter
            i += 1
        elif token in _SKIP:
            i += 1
        else:
            # a name followed by '->' is a parent, so only leaves are daughters
            if i + 1 >= n or tokens[i + 1] != "->":
                parts.append(token)
            i += 1
    return parts


def _strip_fsr(result: str) -> str:
    """Remove the FSR/ISR photons from an assembled decay string."""
    return result.replace(" gammaF", "").replace(" gammaP", "").replace(" gammaI", "")


def _assemble(
    particle: str,
    daughters: list[str],
    ignore_fsr: bool,
    sort_daughters: bool,
) -> str:
    """Format "particle -> d1 d2 ..." straight from the daughter names.

    Equivalent to assembling the string and handing it to `rearrange_daughters`,
    without splitting and re-joining it: the masses are looked up while sorting
    the names, and the FSR photons are dropped from the list.
    """
    if ignore_fsr:
        # dropping the names is what the string-level " gammaF" replacements did:
        # every photon is removed, the space in front of it went with it, and the
        # daughters are single-spaced again when they are joined below
        daughters = [d for d in daughters if d not in _FSR]
    if sort_daughters and len(daughters) > 1:
        daughters = sorted(daughters, key=get_mass, reverse=True)
    return f"{particle} -> {' '.join(daughters)}"


def _apply_parser(
    decay_str: str,
    particle: str,
    mode: str,
    stable: frozenset[str],
    ignore_fsr: bool,
    sort_daughters: bool,
    max_occurrences: int,
) -> list[str | None]:
    results = []
    for block in _find_blocks(decay_str, particle, max_occurrences):
        if mode == "decay":
            # the block is kept verbatim, so photons are removed from the string
            # and a bracket-free block is still sorted the old way
            result = f"{particle} -> {block}"
            if ignore_fsr:
                result = _strip_fsr(result)
            if sort_daughters and "[" not in result:
                result = rearrange_daughters(result, arrow_str="->")
        else:
            daughters = (
                _prompt_daughters(block)
                if mode == "prompt"
                else _final_daughters(block, stable)
            )
            result = _assemble(particle, daughters, ignore_fsr, sort_daughters)
        results.append(result)
    return results + [None] * (max_occurrences - len(results))


def _parse_chunk(
    parser: Callable, particle: str, strings: list[str]
) -> list[list[str | None]]:
    return [parser(s, particle) for s in strings]


def _map_with_workers(
    series: pl.Series,
    parser: Callable,
    particle: str,
    n_workers: int,
    names: list[str],
) -> pl.Series:
    strings = series.to_list()
    if n_workers == 1:
        rows = [parser(s, particle) for s in strings]
    else:
        chunks = [
            strings[i : i + _CHUNK_SIZE] for i in range(0, len(strings), _CHUNK_SIZE)
        ]
        worker = partial(_parse_chunk, parser, particle)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            rows = []
            for chunk in executor.map(worker, chunks):
                rows.extend(chunk)
    return pl.DataFrame(
        {
            name: pl.Series(name, [row[k] for row in rows], dtype=pl.String)
            for k, name in enumerate(names)
        }
    ).to_struct(series.name)


def _expr(
    col_name: str,
    particle: str,
    mode: str,
    n_workers: int,
    ignore_fsr: bool,
    sort_daughters: bool,
    max_occurrences: int,
    stable: list[str] | None = None,
) -> pl.Expr:
    if max_occurrences < 1:
        raise ValueError(f"max_occurrences must be >= 1, got {max_occurrences}")
    name = f"{particle.replace('*', 'star')}_{mode}"
    names = [name] + [f"{name}_{k}" for k in range(2, max_occurrences + 1)]
    parser = partial(
        _apply_parser,
        mode=mode,
        stable=frozenset(stable or ()),
        ignore_fsr=ignore_fsr,
        sort_daughters=sort_daughters,
        max_occurrences=max_occurrences,
    )
    return (
        pl.col(col_name)
        .map_batches(
            partial(
                _map_with_workers,
                parser=parser,
                particle=particle,
                n_workers=n_workers,
                names=names,
            ),
            return_dtype=pl.Struct({n: pl.String for n in names}),
        )
        .struct.unnest()
    )


def count_decays(
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
) -> pl.Expr:
    """Return the number of decays of ``particle`` as '<particle>_count'."""
    return (
        pl.col(col_name)
        .str.count_matches(f"[{particle} -> ", literal=True)
        .fill_null(0)
        .alias(f"{particle.replace('*', 'star')}_count")
    )


def extract_decay(
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
    n_workers: int = 1,
    ignore_fsr: bool = True,
    sort_daughters: bool = True,
    max_occurrences: int = 1,
) -> pl.Expr:
    """Return 'particle -> <raw block>' preserving the full daughter string as-is.

    The first ``max_occurrences`` decays of ``particle`` go into '<particle>_decay',
    '<particle>_decay_2', ...; columns without a matching decay are None.
    """
    return _expr(
        col_name,
        particle,
        "decay",
        n_workers,
        ignore_fsr,
        sort_daughters,
        max_occurrences,
    )


def extract_decay_prompt(
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
    n_workers: int = 1,
    ignore_fsr: bool = True,
    sort_daughters: bool = True,
    max_occurrences: int = 1,
) -> pl.Expr:
    """Return 'particle -> d1 d2 ...' listing only direct daughters.

    Intermediate decays are collapsed to their parent name.
    The first ``max_occurrences`` decays of ``particle`` go into '<particle>_prompt',
    '<particle>_prompt_2', ...; columns without a matching decay are None.
    """
    return _expr(
        col_name,
        particle,
        "prompt",
        n_workers,
        ignore_fsr,
        sort_daughters,
        max_occurrences,
    )


def extract_decay_final(
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
    n_workers: int = 1,
    ignore_fsr: bool = True,
    sort_daughters: bool = True,
    consider_as_stable: list[str] | None = None,
    max_occurrences: int = 1,
) -> pl.Expr:
    """Return 'particle -> d1 d2 ...' listing the leaves of the decay tree.

    Recurses through all intermediate decays.
    Particles in ``consider_as_stable`` (e.g. ["pi0", "K_S0"]) are kept as-is
    and their decays are not expanded.
    The first ``max_occurrences`` decays of ``particle`` go into '<particle>_final',
    '<particle>_final_2', ...; columns without a matching decay are None.
    """
    return _expr(
        col_name,
        particle,
        "final",
        n_workers,
        ignore_fsr,
        sort_daughters,
        max_occurrences,
        stable=consider_as_stable,
    )
