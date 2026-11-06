from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest
from decay_strings import COL, RADIATIVE, REAL, TWO_PI0, UNSORTED_D0

from decaykit import add_decay_columns, count_decays
from decaykit.decaystring import _get_mass, _rearrange_daughters


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

    assert set(decaykit.__all__) == {"add_decay_columns", "count_decays"}


def test_count_decays_counts_occurrences() -> None:
    out = frame(REAL, TWO_PI0, None).with_columns(count_decays(particle="pi0"))
    assert out["pi0_count"].to_list() == [1, 2, 0]


def test_count_decays_other_particle() -> None:
    out = frame(REAL).with_columns(count_decays(particle="B-"))
    assert out["B-_count"].to_list() == [1]


def test_count_decays_ignores_unbalanced_blocks() -> None:
    # an unterminated decay is not a complete block, so the count matches the
    # None the parse columns give instead of the raw number of markers
    unbalanced = "Upsilon(4S) -> [B+ -> K+ pi+ pi-"
    out = add_decay_columns(frame(unbalanced), modes=("raw",)).with_columns(
        count_decays(particle="B+")
    )
    assert out["B+_raw"].to_list() == [None]
    assert out["B+_count"].to_list() == [0]


def test_add_decay_columns_adds_all_modes() -> None:
    out = add_decay_columns(frame(REAL))
    assert out.columns == [COL, "B+_final", "B+_prompt", "B+_raw"]
    assert out["B+_final"].to_list() == ["B+ -> K+ pi+ pi- e+ e-"]
    assert out["B+_prompt"].to_list() == ["B+ -> J/psi K+ pi+ pi-"]
    assert out["B+_raw"].to_list() == ["B+ -> [J/psi -> e+ e-] K+ pi+ pi-"]


def test_add_decay_columns_keeps_the_block_verbatim() -> None:
    assert column([REAL]) == ["B+ -> [J/psi -> e+ e-] K+ pi+ pi-"]


def test_add_decay_columns_sorts_a_bracket_free_block() -> None:
    assert column([UNSORTED_D0], particle="D0") == ["D0 -> K+ pi- pi+ pi-"]
    assert column([UNSORTED_D0], particle="D0", sort_daughters=False) == [
        "D0 -> pi- K+ pi+ pi-"
    ]


def test_add_decay_columns_strips_all_photons_by_default() -> None:
    assert column([RADIATIVE]) == ["B+ -> [J/psi -> e+ e-] K+ pi+ pi-"]


@pytest.mark.parametrize(
    ("ignore_isr", "ignore_fsr", "ignore_photos", "expected"),
    [
        (False, True, True, "B+ -> [J/psi -> e+ e- gammaI] K+ pi+ pi-"),
        (True, False, True, "B+ -> [J/psi -> e+ e-] K+ pi+ pi- gammaF"),
        (True, True, False, "B+ -> [J/psi -> e+ e- gammaP] K+ pi+ pi-"),
        (
            False,
            False,
            True,
            "B+ -> [J/psi -> e+ e- gammaI] K+ pi+ pi- gammaF",
        ),
        (
            False,
            True,
            False,
            "B+ -> [J/psi -> e+ e- gammaP gammaI] K+ pi+ pi-",
        ),
        (
            True,
            False,
            False,
            "B+ -> [J/psi -> e+ e- gammaP] K+ pi+ pi- gammaF",
        ),
        (
            False,
            False,
            False,
            "B+ -> [J/psi -> e+ e- gammaP gammaI] K+ pi+ pi- gammaF",
        ),
    ],
)
def test_add_decay_columns_keeps_only_the_selected_photons(
    ignore_isr, ignore_fsr, ignore_photos, expected
) -> None:
    assert column(
        [RADIATIVE],
        ignore_isr=ignore_isr,
        ignore_fsr=ignore_fsr,
        ignore_photos=ignore_photos,
    ) == [expected]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "B+ -> K+ pi+ pi- e+ e-"),
        ({"ignore_isr": False}, "B+ -> K+ pi+ pi- e+ e- gammaI"),
        ({"ignore_fsr": False}, "B+ -> K+ pi+ pi- e+ e- gammaF"),
        ({"ignore_photos": False}, "B+ -> K+ pi+ pi- e+ e- gammaP"),
        (
            {"ignore_isr": False, "ignore_fsr": False, "ignore_photos": False},
            "B+ -> K+ pi+ pi- e+ e- gammaP gammaI gammaF",
        ),
    ],
)
def test_photon_toggles_apply_to_final(kwargs, expected) -> None:
    assert column([RADIATIVE], "final", **kwargs) == [expected]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "B+ -> J/psi K+ pi+ pi-"),
        ({"ignore_isr": False}, "B+ -> J/psi K+ pi+ pi-"),
        ({"ignore_fsr": False}, "B+ -> J/psi K+ pi+ pi- gammaF"),
        ({"ignore_photos": False}, "B+ -> J/psi K+ pi+ pi-"),
        (
            {"ignore_isr": False, "ignore_fsr": False, "ignore_photos": False},
            "B+ -> J/psi K+ pi+ pi- gammaF",
        ),
    ],
)
def test_photon_toggles_apply_to_prompt(kwargs, expected) -> None:
    # prompt mode collapses J/psi, so only the direct photon survives
    assert column([RADIATIVE], "prompt", **kwargs) == [expected]


def test_add_decay_columns_keeps_pi0_gammas() -> None:
    # gammaI/gammaF/gammaP are radiative; gamma from pi0 -> gamma gamma is signal
    assert column([REAL], "final", particle="B-") == [
        "B- -> K+ K- K- pi+ pi- gamma gamma"
    ]


def test_add_decay_columns_prompt_lists_direct_daughters() -> None:
    assert column([REAL], "prompt") == ["B+ -> J/psi K+ pi+ pi-"]
    assert column([REAL], "prompt", particle="B-") == ["B- -> D_s+ K- pi-"]
    assert column([REAL], "prompt", particle="B-", sort_daughters=False) == [
        "B- -> D_s+ pi- K-"
    ]


def test_add_decay_columns_final_lists_leaves() -> None:
    assert column([REAL], "final") == ["B+ -> K+ pi+ pi- e+ e-"]


def test_add_decay_columns_final_keeps_stable_particles() -> None:
    assert column([REAL], "final", particle="B-", consider_as_stable=["D_s+"]) == [
        "B- -> D_s+ K- pi-"
    ]


def test_add_decay_columns_final_recurses_without_stable_particles() -> None:
    assert column([REAL], "final", particle="B-") == [
        "B- -> K+ K- K- pi+ pi- gamma gamma"
    ]


def test_max_occurrences_creates_columns_and_pads_with_none() -> None:
    out = add_decay_columns(
        frame(TWO_PI0, REAL), particle="pi0", modes=("raw",), max_occurrences=2
    )
    assert out["pi0_raw"].to_list() == [
        "pi0 -> gamma gamma",
        "pi0 -> gamma gamma",
    ]
    assert out["pi0_raw_2"].to_list() == ["pi0 -> gamma gamma", None]


def test_max_occurrences_is_mode_major() -> None:
    out = add_decay_columns(
        frame(TWO_PI0), particle="pi0", modes=("raw", "prompt"), max_occurrences=2
    )
    assert out.columns == [
        COL,
        "pi0_raw",
        "pi0_raw_2",
        "pi0_prompt",
        "pi0_prompt_2",
    ]
    assert out["pi0_prompt"].to_list() == ["pi0 -> gamma gamma"]
    assert out["pi0_prompt_2"].to_list() == ["pi0 -> gamma gamma"]


def test_max_occurrences_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_occurrences must be >= 1"):
        add_decay_columns(frame(REAL), modes=("raw",), max_occurrences=0)


def test_no_match_and_null_input_give_none() -> None:
    no_b_plus = "Upsilon(4S) -> [B- -> [D_s+ -> K+ K- pi+ [pi0 -> gamma gamma]] pi- K-]"
    out = add_decay_columns(frame(no_b_plus, None), modes=("raw",))
    assert out["B+_raw"].to_list() == [None, None]


def test_modes_must_be_known_and_non_empty() -> None:
    with pytest.raises(ValueError, match="modes must not be empty"):
        add_decay_columns(frame(REAL), modes=())
    with pytest.raises(ValueError, match="unknown mode"):
        add_decay_columns(frame(REAL), modes=("final", "nope"))


def test_duplicate_modes_are_ignored() -> None:
    out = add_decay_columns(frame(REAL), modes=("final", "final"))
    assert out.columns == [COL, "B+_final"]


def test_add_decay_columns_replaces_existing_variants() -> None:
    df = frame(REAL).with_columns(pl.lit("old").alias("B+_final"))
    out = add_decay_columns(df, modes=("final",))
    assert out.columns == [COL, "B+_final"]
    assert out["B+_final"].to_list() == ["B+ -> K+ pi+ pi- e+ e-"]


def test_add_decay_columns_does_not_clobber_a_temporary_name() -> None:
    df = pl.DataFrame(
        {
            COL: [REAL],
            "__decaykit_parsed__": [42],
        }
    )
    out = add_decay_columns(df, modes=("final",))
    assert out["__decaykit_parsed__"].to_list() == [42]
    assert out["B+_final"].to_list() == ["B+ -> K+ pi+ pi- e+ e-"]


def test_multiprocessing_matches_single_worker() -> None:
    strings = [REAL, TWO_PI0, UNSORTED_D0, RADIATIVE, None] * 4
    single = column(strings, "final", n_workers=1)
    multi = column(strings, "final", n_workers=2)
    assert single == multi


def test_rearrange_daughters_sorts_by_mass() -> None:
    assert _rearrange_daughters("B+ --> pi+ K+") == "B+ --> K+ pi+"
    assert _rearrange_daughters("B+ -> pi+ K+", arrow_str="->") == "B+ -> K+ pi+"


def test_get_mass() -> None:
    assert _get_mass("K+") == pytest.approx(493.677, abs=0.001)
    for photon in ("gammaI", "gammaF", "gammaP"):
        assert _get_mass(photon) == 0
    assert _get_mass("not_a_particle") == 0


def test_get_mass_handles_known_but_invalid_pdgid() -> None:
    # EvtGen's exotic X states map to PDGIDs that particle flags
    # is_valid=False, so from_pdgid raises InvalidParticle rather than
    # ParticleNotFound/MatchingIDNotFound; the lookup must still yield 0.
    assert _get_mass("Xsu") == 0
    assert _get_mass("anti-Xss") == 0
    assert column(["[B+ -> K+ Xsu]"], "prompt") == ["B+ -> K+ Xsu"]


def test_get_mass_is_thread_safe() -> None:
    # particle's lazy lookup tables are not thread-safe, and polars evaluates
    # expressions in parallel; a race must not poison the cache with mass 0.
    _get_mass.cache_clear()
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
        masses = list(executor.map(_get_mass, names))
    for name, mass in zip(names, masses, strict=True):
        assert mass == pytest.approx(expected[name], abs=0.01)
