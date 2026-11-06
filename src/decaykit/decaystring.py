"""Parse EvtGen decay strings and expose them as Polars expressions.

A decay string looks like ``Upsilon(4S) -> [B+ -> [J/psi -> e+ e-] K+ pi+ pi-]
[B- -> [D_s+ -> K+ K- pi+ [pi0 -> gamma gamma]] pi- K-]``:

brackets group a decay, ``->`` separates a parent from its daughters, and a
name is any run of characters that is neither a bracket nor whitespace.

``add_decay_columns`` parses the decay column of a ``DataFrame`` or
``LazyFrame`` and adds the requested variants. ``count_decays`` is available as
a ``polars.Expr`` for use in ``with_columns``.
"""

import multiprocessing
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial

import polars as pl
from particle import InvalidParticle, Particle, ParticleNotFound
from particle.exceptions import MatchingIDNotFound

_CHUNK_SIZE = 50_000

# A parser is ``_apply_parser_multi`` partially applied: it takes a decay
# string and a particle name and returns one row per (mode, occurrence).
type _Parser = Callable[[str, str], list[str | None]]

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

# A token that can never be the name of a decayed daughter.
_NON_NAME = frozenset({"[", "]", "->"})

# gammaI is ISR, gammaF is FSR and gammaP is PHOTOS. Their order here is also
# the order in which the three ignore toggles are zipped together.
_PHOTON_ORDER = ("gammaI", "gammaF", "gammaP")
_PHOTONS = frozenset(_PHOTON_ORDER)


@cache
def _get_mass(evtgen_name: str) -> float:
    """Return the mass of an EvtGen particle name, or ``0`` if it is unknown."""
    if evtgen_name in _PHOTONS:
        return 0

    try:
        with _MASS_LOCK:
            mass = Particle.from_evtgen_name(evtgen_name).mass
        if mass is None:
            return 0
        else:
            return mass
    except (InvalidParticle, ParticleNotFound, MatchingIDNotFound):
        return 0


def _rearrange_daughters(decay_str: str, arrow_str: str = "-->") -> str:
    mother = decay_str.split(arrow_str)[0]
    daughters = decay_str.split(arrow_str)[1].split()
    daughters.sort(key=_get_mass, reverse=True)

    return mother + arrow_str + " " + " ".join(daughters)


def _find_blocks(decay_str: str, particle: str, limit: int) -> list[str]:
    """Return the daughter blocks of the first ``limit`` decays, in string order."""
    blocks: list[str] = []
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


def _count_blocks(decay_str: str | None, particle: str) -> int:
    """Count the complete decays of ``particle`` in ``decay_str``.

    Mirrors the balanced-bracket scan of ``_find_blocks``: an occurrence of
    ``[particle -> ...]`` whose brackets don't all close is never counted, so
    the count always matches what ``add_decay_columns`` extracts.
    """
    if not decay_str:
        return 0
    count = 0
    marker = f"[{particle} -> "
    start = decay_str.find(marker)
    while start != -1:
        pos = start + len(marker)
        depth = 1
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
                count += 1
        start = decay_str.find(marker, pos)
    return count


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


def _both_daughters(
    tokens: list[str], stable: frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Return the prompt and final daughters in a single token walk.

    This is `_prompt_daughters` and `_final_daughters` over the same tokens, so
    extracting both modes pays for one walk instead of two.
    """
    prompt = []
    final = []
    depth = 0
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == "[":
            name = tokens[i + 1] if i + 1 < n else None
            # a direct daughter is the name of this group; its decay is not
            # expanded for the prompt output
            if depth == 0 and name is not None and name not in _NON_NAME:
                prompt.append(name)
            # a "stable" particle is a final-state leaf: keep it, drop its tree
            if name is not None and name in stable:
                final.append(name)
                i = _skip_group(tokens, i)
                continue
            depth += 1
            i += 1
        elif token == "]":
            depth -= 1
            i += 1
        elif token == "->":
            i += 1
        else:
            if depth == 0:
                prompt.append(token)
            # a name followed by '->' is a parent, so only leaves are final
            if i + 1 >= n or tokens[i + 1] != "->":
                final.append(token)
            i += 1
    return prompt, final


def _ignored_photons(
    ignore_isr: bool, ignore_fsr: bool, ignore_photos: bool
) -> frozenset[str]:
    """Return the photon names selected by the three ignore toggles."""
    flags = (ignore_isr, ignore_fsr, ignore_photos)
    return frozenset(
        name
        for name, ignore in zip(_PHOTON_ORDER, flags, strict=True)
        if ignore
    )


def _strip_photons(result: str, ignored_photons: frozenset[str]) -> str:
    """Remove the selected photons from an assembled decay string."""
    for name in _PHOTON_ORDER:
        if name in ignored_photons:
            result = result.replace(f" {name}", "")
    return result


def _assemble(
    particle: str,
    daughters: list[str],
    ignored_photons: frozenset[str],
    sort_daughters: bool,
) -> str:
    """Format "particle -> d1 d2 ..." straight from the daughter names.

    Equivalent to assembling the string and handing it to `_rearrange_daughters`,
    without splitting and re-joining it: the masses are looked up while sorting
    the names, and the ignored photons are dropped from the list.
    """
    if ignored_photons:
        # dropping the names is what the string-level " gammaX" replacements did:
        # every selected photon is removed, the space in front of it went with it,
        # and the daughters are single-spaced again when they are joined below
        daughters = [d for d in daughters if d not in ignored_photons]
    if sort_daughters and len(daughters) > 1:
        daughters = sorted(daughters, key=_get_mass, reverse=True)
    return f"{particle} -> {' '.join(daughters)}"


def _raw_result(
    particle: str,
    block: str,
    ignored_photons: frozenset[str],
    sort_daughters: bool,
) -> str:
    # the block is kept verbatim, so photons are removed from the string
    # and a bracket-free block is still sorted the old way
    result = f"{particle} -> {block}"
    if ignored_photons:
        result = _strip_photons(result, ignored_photons)
    if sort_daughters and "[" not in result:
        result = _rearrange_daughters(result, arrow_str="->")
    return result


def _apply_parser_multi(
    decay_str: str,
    particle: str,
    modes: tuple[str, ...],
    stable: frozenset[str],
    ignored_photons: frozenset[str],
    sort_daughters: bool,
    max_occurrences: int,
) -> list[str | None]:
    """Parse every requested mode, sharing the block and token walks.

    The result is mode-major and padded with ``None`` like a single-mode parse:
    all occurrences of the first mode, then all occurrences of the second, ...
    """
    blocks = _find_blocks(decay_str, particle, max_occurrences)
    missing = max_occurrences - len(blocks)
    want_prompt = "prompt" in modes
    want_final = "final" in modes
    want_raw = "raw" in modes
    columns: dict[str, list[str]] = {mode: [] for mode in modes}

    for block in blocks:
        if want_raw:
            columns["raw"].append(
                _raw_result(particle, block, ignored_photons, sort_daughters)
            )
        if want_prompt or want_final:
            prompt_parts: list[str] = []
            final_parts: list[str] = []
            if want_prompt and want_final:
                # one tokenization and one walk for both variants
                prompt_parts, final_parts = _both_daughters(
                    _TOKEN_RE.findall(block), stable
                )
            elif want_prompt:
                prompt_parts = _prompt_daughters(block)
            else:
                final_parts = _final_daughters(block, stable)
            if want_prompt:
                columns["prompt"].append(
                    _assemble(
                        particle, prompt_parts, ignored_photons, sort_daughters
                    )
                )
            if want_final:
                columns["final"].append(
                    _assemble(
                        particle, final_parts, ignored_photons, sort_daughters
                    )
                )

    results: list[str | None] = []
    for mode in modes:
        results.extend(columns[mode])
        results.extend([None] * missing)
    return results


def _parse_chunk(
    parser: _Parser, particle: str, strings: list[str]
) -> list[list[str | None]]:
    return [parser(s, particle) for s in strings]


def _map_with_workers(
    series: pl.Series,
    parser: _Parser,
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
        with ProcessPoolExecutor(
            max_workers=n_workers,
            # polars evaluates expressions in parallel threads, so the
            # interpreter is multi-threaded here: fork can deadlock the child
            # (e.g. on _MASS_LOCK), so fork fresh instead.
            mp_context=multiprocessing.get_context("forkserver"),
        ) as executor:
            rows = []
            for chunk in executor.map(worker, chunks):
                rows.extend(chunk)
    return pl.DataFrame(
        {
            name: pl.Series(name, [row[k] for row in rows], dtype=pl.String)
            for k, name in enumerate(names)
        }
    ).to_struct(series.name)


def _parse_expr(
    col_name: str,
    particle: str,
    modes: tuple[str, ...],
    n_workers: int,
    ignored_photons: frozenset[str],
    sort_daughters: bool,
    max_occurrences: int,
    stable: list[str] | None = None,
) -> tuple[pl.Expr, list[str]]:
    """Return the struct expression that parses all modes, and its field names."""
    if max_occurrences < 1:
        raise ValueError(f"max_occurrences must be >= 1, got {max_occurrences}")
    base = particle.replace("*", "star")
    names = []
    for mode in modes:
        name = f"{base}_{mode}"
        names.append(name)
        names.extend(f"{name}_{k}" for k in range(2, max_occurrences + 1))
    parser = partial(
        _apply_parser_multi,
        modes=modes,
        stable=frozenset(stable or ()),
        ignored_photons=ignored_photons,
        sort_daughters=sort_daughters,
        max_occurrences=max_occurrences,
    )
    expr = pl.col(col_name).map_batches(
        partial(
            _map_with_workers,
            parser=parser,
            particle=particle,
            n_workers=n_workers,
            names=names,
        ),
        return_dtype=pl.Struct({n: pl.String for n in names}),
    )
    return expr, names


def count_decays(
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
) -> pl.Expr:
    """Return the number of decays of ``particle`` as '<particle>_count'.

    The count is block-aware: it only counts occurrences of
    ``[particle -> ...]`` whose brackets balance, so malformed input yields 0,
    matching the ``None`` the parsed columns get.
    """

    def count(series: pl.Series) -> pl.Series:
        return pl.Series(
            [0 if s is None else _count_blocks(s, particle) for s in series],
            dtype=pl.UInt32,
        )

    return pl.col(col_name).map_batches(count, return_dtype=pl.UInt32).alias(
        f"{particle.replace('*', 'star')}_count"
    )


def _check_modes(modes: Sequence[str]) -> tuple[str, ...]:
    # a repeated mode would ask for the same column twice; keep the first
    modes = tuple(dict.fromkeys(modes))
    if not modes:
        raise ValueError("modes must not be empty")
    unknown = [mode for mode in modes if mode not in ("final", "prompt", "raw")]
    if unknown:
        raise ValueError(f"unknown mode(s): {', '.join(unknown)}")
    return modes


def add_decay_columns[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT,
    col_name: str = "__MCDecayString__",
    particle: str = "B+",
    modes: Sequence[str] = ("final", "prompt", "raw"),
    n_workers: int = 1,
    ignore_isr: bool = True,
    ignore_fsr: bool = True,
    ignore_photos: bool = True,
    sort_daughters: bool = True,
    consider_as_stable: list[str] | None = None,
    max_occurrences: int = 1,
) -> FrameT:
    """Add the requested decay variants, parsing each event only once.

    ``modes`` selects which of ``final``, ``prompt`` and ``raw`` to emit; the
    columns appear in that order. The first ``max_occurrences`` decays of
    ``particle`` go into '<particle>_<mode>', '<particle>_<mode>_2', ...;
    columns without a matching decay are None.

    Radiative photons are dropped by default: ``ignore_isr`` removes
    ``gammaI`` (ISR), ``ignore_fsr`` removes ``gammaF`` (FSR) and
    ``ignore_photos`` removes ``gammaP`` (PHOTOS). Set any of them to False to
    keep that photon kind.

    Particles in ``consider_as_stable`` (e.g. ["pi0", "K_S0"]) are kept as-is
    in the ``final`` variant instead of expanding their decays.

    Works on both ``DataFrame`` and ``LazyFrame``; a lazy query parses each
    batch exactly once.
    """
    expr, names = _parse_expr(
        col_name,
        particle,
        _check_modes(modes),
        n_workers,
        _ignored_photons(ignore_isr, ignore_fsr, ignore_photos),
        sort_daughters,
        max_occurrences,
        stable=consider_as_stable,
    )
    existing = set(
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )
    tmp = "__decaykit_parsed__"
    while tmp in existing:
        tmp = f"_{tmp}"
    parsed = frame.with_columns(expr.alias(tmp))
    fields = [pl.col(tmp).struct.field(name).alias(name) for name in names]
    # with_columns replaces columns that already exist, so rerunning on a frame
    # that already has these variants stays idempotent
    return parsed.with_columns(*fields).drop(tmp)
