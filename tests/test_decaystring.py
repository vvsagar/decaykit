from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from decaykit import (
    add_decay_columns,
    count_decays,
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


def column(
    strings: pl.Series | list[str | None],
    mode: str = "raw",
    particle: str = "B+",
    **kwargs,
) -> list[str | None]:
    out = add_decay_columns(frame(*strings), modes=(mode,), particle=particle, **kwargs)
    return out[f"{particle.replace('*', 'star')}_{mode}"].to_list()


def test_public_api_is_exported() -> None:
    import decaykit

    assert set(decaykit.__all__) == {
        "add_decay_columns",
        "count_decays",
        "get_mass",
        "rearrange_daughters",
    }


def test_count_decays_counts_occurrences() -> None:
    out = frame(SIMPLE, TWO, None).with_columns(count_decays())
    assert out["B+_count"].to_list() == [1, 2, 0]


def test_count_decays_other_particle() -> None:
    out = frame(NESTED).with_columns(count_decays(particle="B0"))
    assert out["B0_count"].to_list() == [1]


def test_add_decay_columns_adds_all_modes() -> None:
    out = add_decay_columns(frame(SIMPLE))
    assert out.columns == [COL, "B+_final", "B+_prompt", "B+_raw"]
    assert out["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]
    assert out["B+_prompt"].to_list() == ["B+ -> J/psi K+"]
    assert out["B+_raw"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"
    ]


def test_add_decay_columns_keeps_the_block_verbatim() -> None:
    assert column([SIMPLE]) == ["B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"]


def test_add_decay_columns_sorts_a_bracket_free_block() -> None:
    decay = "[B+ -> pi+ K+]"
    assert column([decay]) == ["B+ -> K+ pi+"]
    assert column([decay], sort_daughters=False) == ["B+ -> pi+ K+"]


def test_add_decay_columns_strips_all_photons_by_default() -> None:
    assert column([FSR]) == ["B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"]


@pytest.mark.parametrize(
    ("ignore_isr", "ignore_fsr", "ignore_photos", "expected"),
    [
        (False, True, True, "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu- gammaI]"),
        (True, False, True, "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu-]"),
        (True, True, False, "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu- gammaP]"),
        (
            False,
            False,
            True,
            "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaI]",
        ),
        (
            True,
            False,
            False,
            "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP]",
        ),
        (
            False,
            True,
            False,
            "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu- gammaP gammaI]",
        ),
        (
            False,
            False,
            False,
            "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP gammaI]",
        ),
    ],
)
def test_add_decay_columns_keeps_only_the_selected_photons(
    ignore_isr, ignore_fsr, ignore_photos, expected
) -> None:
    assert column(
        [FSR],
        ignore_isr=ignore_isr,
        ignore_fsr=ignore_fsr,
        ignore_photos=ignore_photos,
    ) == [expected]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "B+ -> pi+ pi0 mu+ mu-"),
        ({"ignore_isr": False}, "B+ -> pi+ pi0 mu+ mu- gammaI"),
        ({"ignore_fsr": False}, "B+ -> pi+ pi0 mu+ mu- gammaF"),
        ({"ignore_photos": False}, "B+ -> pi+ pi0 mu+ mu- gammaP"),
        (
            {"ignore_isr": False, "ignore_fsr": False, "ignore_photos": False},
            "B+ -> pi+ pi0 mu+ mu- gammaF gammaP gammaI",
        ),
    ],
)
def test_photon_toggles_apply_to_final(kwargs, expected) -> None:
    assert column([FSR], "final", **kwargs) == [expected]


def test_photon_toggles_apply_to_prompt() -> None:
    # prompt mode collapses K+ and J/psi, so their photons never show up
    for kwargs in (
        {},
        {"ignore_isr": False},
        {"ignore_fsr": False},
        {"ignore_photos": False},
        {"ignore_isr": False, "ignore_fsr": False, "ignore_photos": False},
    ):
        assert column([FSR], "prompt", **kwargs) == ["B+ -> J/psi K+"]


def test_add_decay_columns_prompt_lists_direct_daughters() -> None:
    assert column([SIMPLE], "prompt") == ["B+ -> J/psi K+"]
    assert column([SIMPLE], "prompt", sort_daughters=False) == ["B+ -> K+ J/psi"]


def test_add_decay_columns_final_lists_leaves() -> None:
    assert column([SIMPLE], "final") == ["B+ -> pi+ pi0 mu+ mu-"]


def test_add_decay_columns_final_keeps_stable_particles() -> None:
    assert column(
        [NESTED], "final", particle="B0", consider_as_stable=["D0"]
    ) == ["B0 -> D0 pi- pi+"]


def test_add_decay_columns_final_recurses_without_stable_particles() -> None:
    assert column([NESTED], "final", particle="B0") == ["B0 -> K- pi+ pi- pi+"]


def test_max_occurrences_creates_columns_and_pads_with_none() -> None:
    out = add_decay_columns(frame(TWO, SIMPLE), modes=("raw",), max_occurrences=2)
    assert out["B+_raw"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] J/psi",
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
    ]
    assert out["B+_raw_2"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
        None,
    ]


def test_max_occurrences_is_mode_major() -> None:
    out = add_decay_columns(
        frame(TWO), modes=("raw", "prompt"), max_occurrences=2
    )
    assert out.columns == [
        COL,
        "B+_raw",
        "B+_raw_2",
        "B+_prompt",
        "B+_prompt_2",
    ]
    assert out["B+_prompt"].to_list() == ["B+ -> J/psi K+"]
    assert out["B+_prompt_2"].to_list() == ["B+ -> J/psi K+"]


def test_max_occurrences_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_occurrences must be >= 1"):
        add_decay_columns(frame(SIMPLE), modes=("raw",), max_occurrences=0)


def test_no_match_and_null_input_give_none() -> None:
    out = add_decay_columns(frame("[B0 -> K+ pi-]", None), modes=("raw",))
    assert out["B+_raw"].to_list() == [None, None]


def test_modes_must_be_known_and_non_empty() -> None:
    with pytest.raises(ValueError, match="modes must not be empty"):
        add_decay_columns(frame(SIMPLE), modes=())
    with pytest.raises(ValueError, match="unknown mode"):
        add_decay_columns(frame(SIMPLE), modes=("final", "nope"))


def test_duplicate_modes_are_ignored() -> None:
    out = add_decay_columns(frame(SIMPLE), modes=("final", "final"))
    assert out.columns == [COL, "B+_final"]


def test_add_decay_columns_replaces_existing_variants() -> None:
    df = frame(SIMPLE).with_columns(pl.lit("old").alias("B+_final"))
    out = add_decay_columns(df, modes=("final",))
    assert out.columns == [COL, "B+_final"]
    assert out["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]


def test_add_decay_columns_does_not_clobber_a_temporary_name() -> None:
    df = pl.DataFrame(
        {
            COL: [SIMPLE],
            "__decaykit_parsed__": [42],
        }
    )
    out = add_decay_columns(df, modes=("final",))
    assert out["__decaykit_parsed__"].to_list() == [42]
    assert out["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]


def test_multiprocessing_matches_single_worker() -> None:
    strings = [SIMPLE, NESTED, TWO, FSR, None] * 4
    single = column(strings, "final", n_workers=1)
    multi = column(strings, "final", n_workers=2)
    assert single == multi


def test_rearrange_daughters_sorts_by_mass() -> None:
    assert rearrange_daughters("B+ --> pi+ K+") == "B+ --> K+ pi+"
    assert rearrange_daughters("B+ -> pi+ K+", arrow_str="->") == "B+ -> K+ pi+"


def test_get_mass() -> None:
    assert get_mass("K+") == pytest.approx(493.677, abs=0.001)
    assert get_mass("gammaF") == 0
    assert get_mass("not_a_particle") == 0


def test_get_mass_handles_known_but_invalid_pdgid() -> None:
    # EvtGen's exotic X states map to PDGIDs that particle flags
    # is_valid=False, so from_pdgid raises InvalidParticle rather than
    # ParticleNotFound/MatchingIDNotFound; the lookup must still yield 0.
    assert get_mass("Xsu") == 0
    assert get_mass("anti-Xss") == 0
    assert column(["[B+ -> K+ Xsu]"], "prompt") == ["B+ -> K+ Xsu"]


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
