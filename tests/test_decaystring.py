from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from decaykit import (
    count_decays,
    extract_decay,
    extract_decay_final,
    extract_decay_prompt,
    get_mass,
    rearrange_daughters,
)

COL = "__MCDecayString__"

SIMPLE = "[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"
NESTED = "[B0 -> [D*- -> [D0 -> K- pi+] pi-] pi+]"
TWO = "[B+ -> [K+ -> pi+ pi0] J/psi] [B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"
FSR = "[B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP gammaI]]"


def frame(*strings: str | None) -> pl.DataFrame:
    return pl.DataFrame({COL: list(strings)})


def column(strings: pl.Series | list[str | None], expr: pl.Expr) -> list[str | None]:
    return frame(*strings).select(expr).to_series().to_list()


def test_public_api_is_exported() -> None:
    import decaykit

    assert set(decaykit.__all__) == {
        "count_decays",
        "extract_decay",
        "extract_decay_final",
        "extract_decay_prompt",
        "get_mass",
        "rearrange_daughters",
    }


def test_count_decays_counts_occurrences() -> None:
    out = frame(SIMPLE, TWO, None).with_columns(count_decays())
    assert out["B+_count"].to_list() == [1, 2, 0]


def test_count_decays_other_particle() -> None:
    out = frame(NESTED).with_columns(count_decays(particle="B0"))
    assert out["B0_count"].to_list() == [1]


def test_extract_decay_keeps_the_block_verbatim() -> None:
    assert column([SIMPLE], extract_decay()) == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"
    ]


def test_extract_decay_sorts_a_bracket_free_block() -> None:
    decay = "[B+ -> pi+ K+]"
    assert column([decay], extract_decay()) == ["B+ -> K+ pi+"]
    assert column([decay], extract_decay(sort_daughters=False)) == ["B+ -> pi+ K+"]


def test_extract_decay_strips_fsr_by_default() -> None:
    assert column([FSR], extract_decay()) == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"
    ]


def test_extract_decay_keeps_fsr_when_asked() -> None:
    assert column([FSR], extract_decay(ignore_fsr=False)) == [
        "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP gammaI]"
    ]


def test_extract_decay_prompt_lists_direct_daughters() -> None:
    assert column([SIMPLE], extract_decay_prompt()) == ["B+ -> J/psi K+"]
    assert column([SIMPLE], extract_decay_prompt(sort_daughters=False)) == [
        "B+ -> K+ J/psi"
    ]


def test_extract_decay_final_lists_leaves() -> None:
    assert column([SIMPLE], extract_decay_final()) == ["B+ -> pi+ pi0 mu+ mu-"]


def test_extract_decay_final_keeps_stable_particles() -> None:
    final = extract_decay_final(particle="B0", consider_as_stable=["D0"])
    assert column([NESTED], final) == ["B0 -> D0 pi- pi+"]


def test_extract_decay_final_recurses_without_stable_particles() -> None:
    final = extract_decay_final(particle="B0")
    assert column([NESTED], final) == ["B0 -> K- pi+ pi- pi+"]


def test_max_occurrences_creates_columns_and_pads_with_none() -> None:
    out = frame(TWO, SIMPLE).with_columns(extract_decay(max_occurrences=2))
    assert out["B+_decay"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] J/psi",
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
    ]
    assert out["B+_decay_2"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
        None,
    ]


def test_max_occurrences_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_occurrences must be >= 1"):
        extract_decay(max_occurrences=0)


def test_no_match_and_null_input_give_none() -> None:
    out = frame("[B0 -> K+ pi-]", None).with_columns(extract_decay())
    assert out["B+_decay"].to_list() == [None, None]


def test_multiprocessing_matches_single_worker() -> None:
    strings = [SIMPLE, NESTED, TWO, FSR, None] * 4
    single = column(strings, extract_decay_final(n_workers=1))
    multi = column(strings, extract_decay_final(n_workers=2))
    assert single == multi


def test_rearrange_daughters_sorts_by_mass() -> None:
    assert rearrange_daughters("B+ --> pi+ K+") == "B+ --> K+ pi+"
    assert rearrange_daughters("B+ -> pi+ K+", arrow_str="->") == "B+ -> K+ pi+"


def test_get_mass() -> None:
    assert get_mass("K+") == pytest.approx(493.677, abs=0.001)
    assert get_mass("gammaF") == 0
    assert get_mass("not_a_particle") == 0


def test_get_mass_is_thread_safe() -> None:
    # particle's lazy lookup tables are not thread-safe, and polars evaluates
    # expressions in parallel; a race must not poison the cache with mass 0.
    get_mass.cache_clear()
    expected = {
        "J/psi": 3096.9,
        "K+": 493.677,
        "pi+": 139.570,
        "pi0": 134.977,
        "mu+": 105.658,
        "mu-": 105.658,
    }
    names = list(expected) * 100
    with ThreadPoolExecutor(max_workers=8) as executor:
        masses = list(executor.map(get_mass, names))
    for name, mass in zip(names, masses, strict=True):
        assert mass == pytest.approx(expected[name], abs=0.01)
