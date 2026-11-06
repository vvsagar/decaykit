import io

import numpy as np
import polars as pl
import pytest
import uproot

from decaykit.cli import main

COL = "__MCDecayString__"
SIMPLE = "[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"
NESTED = "[B0 -> [D*- -> [D0 -> K- pi+] pi-] pi+]"
TWO = "[B+ -> [K+ -> pi+ pi0] J/psi] [B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"
FSR = "[B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP gammaI]]"


def write_input(tmp_path):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [SIMPLE], "x": [1]}).write_parquet(path)
    return path


def write_root(tmp_path, tree="tree"):
    path = tmp_path / "in.root"
    with uproot.recreate(path) as f:
        f[tree] = {COL: np.array([SIMPLE]), "x": np.array([1])}
    return path


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_in_place_adds_all_modes(tmp_path):
    path = write_input(tmp_path)

    main([str(path)])

    df = pl.read_parquet(path)
    assert df.columns == [COL, "x", "B+_final", "B+_prompt", "B+_raw"]
    assert df["x"].to_list() == [1]
    assert df["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]
    assert df["B+_prompt"].to_list() == ["B+ -> J/psi K+"]
    assert df["B+_raw"].to_list() == ["B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"]


def test_output_keeps_input_untouched(tmp_path):
    path = write_input(tmp_path)
    before = path.read_bytes()
    out = tmp_path / "out.parquet"

    main([str(path), "-o", str(out)])

    assert path.read_bytes() == before
    assert "B+_final" in pl.read_parquet(out).columns


def test_mode_selects_variants(tmp_path):
    path = write_input(tmp_path)

    main([str(path), "--mode", "final", "--mode", "prompt"])

    df = pl.read_parquet(path)
    assert "B+_final" in df.columns
    assert "B+_prompt" in df.columns
    assert "B+_raw" not in df.columns


def test_particle_is_configurable(tmp_path):
    path = write_input(tmp_path)

    main([str(path), "--mode", "final", "--particle", "J/psi"])

    assert "J/psi_final" in pl.read_parquet(path).columns


def test_n_workers_matches_single_worker(tmp_path):
    path = write_input(tmp_path)

    main([str(path), "--n-workers", "2"])

    df = pl.read_parquet(path)
    assert df["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]
    assert df["B+_prompt"].to_list() == ["B+ -> J/psi K+"]


def test_n_workers_must_be_positive(tmp_path):
    path = write_input(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path), "--n-workers", "0"])


def test_root_input_writes_sibling_parquet(tmp_path):
    path = write_root(tmp_path)

    main([str(path), "-t", "tree"])

    df = pl.read_parquet(path.parent / f"{path.stem}_tree.pq")
    assert df.columns == [COL, "x", "B+_final", "B+_prompt", "B+_raw"]
    assert df[COL].to_list() == [SIMPLE]
    assert df["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]
    assert df["B+_prompt"].to_list() == ["B+ -> J/psi K+"]
    assert df["x"].to_list() == [1]


def test_root_input_can_choose_output(tmp_path):
    path = write_root(tmp_path)
    out = tmp_path / "out.parquet"

    main([str(path), "--tree", "tree", "-o", str(out)])

    assert "B+_final" in pl.read_parquet(out).columns
    assert not (path.parent / f"{path.stem}_tree.pq").exists()


def test_root_input_requires_a_tree(tmp_path):
    path = write_root(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path)])


def test_root_input_defaults_to_variables_tree(tmp_path):
    path = write_root(tmp_path, tree="variables")

    main([str(path)])

    assert (
        "B+_prompt"
        in pl.read_parquet(path.parent / f"{path.stem}_variables.pq").columns
    )


def test_tree_is_rejected_for_parquet_input(tmp_path):
    path = write_input(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path), "-t", "tree"])


def test_existing_output_needs_confirmation_without_a_tty(tmp_path):
    src = write_input(tmp_path)
    out = tmp_path / "out.parquet"
    out.write_bytes(b"old")

    with pytest.raises(SystemExit):
        main([str(src), "-o", str(out)])

    assert out.read_bytes() == b"old"


def test_in_place_run_needs_no_confirmation(tmp_path):
    path = write_input(tmp_path)

    # No -o: the rewrite is the documented default, so no prompt even
    # without a terminal to ask on.
    main([str(path)])

    assert "B+_final" in pl.read_parquet(path).columns


def test_root_default_output_asks_when_it_exists(tmp_path):
    path = write_root(tmp_path)
    out = path.parent / f"{path.stem}_tree.pq"
    out.write_bytes(b"old")

    with pytest.raises(SystemExit):
        main([str(path), "-t", "tree"])

    assert out.read_bytes() == b"old"


def test_yes_overwrites_without_asking(tmp_path):
    src = write_input(tmp_path)
    out = tmp_path / "out.parquet"
    out.write_bytes(b"old")

    main([str(src), "-o", str(out), "--yes"])

    assert "B+_final" in pl.read_parquet(out).columns


def test_interactive_overwrite_can_be_accepted(tmp_path, monkeypatch):
    src = write_input(tmp_path)
    out = tmp_path / "out.parquet"
    out.write_bytes(b"old")
    monkeypatch.setattr("sys.stdin", FakeTTY("y\n"))

    main([str(src), "-o", str(out)])

    assert "B+_final" in pl.read_parquet(out).columns


def test_interactive_overwrite_can_be_declined(tmp_path, monkeypatch):
    src = write_input(tmp_path)
    out = tmp_path / "out.parquet"
    out.write_bytes(b"old")
    monkeypatch.setattr("sys.stdin", FakeTTY("n\n"))

    with pytest.raises(SystemExit):
        main([str(src), "-o", str(out)])

    assert out.read_bytes() == b"old"


def test_column_selects_the_input_column(tmp_path):
    column = "DecayString"
    path = tmp_path / "in.parquet"
    pl.DataFrame({column: [SIMPLE], "x": [1]}).write_parquet(path)

    main([str(path), "--column", column, "--mode", "final"])

    df = pl.read_parquet(path)
    assert df.columns == [column, "x", "B+_final"]
    assert df["B+_final"].to_list() == ["B+ -> pi+ pi0 mu+ mu-"]


@pytest.mark.parametrize("flag", ["--stable", "--consider-as-stable"])
def test_stable_particles_are_kept(tmp_path, flag):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [NESTED]}).write_parquet(path)

    main([str(path), "--particle", "B0", "--mode", "final", flag, "D0"])

    assert pl.read_parquet(path)["B0_final"].to_list() == ["B0 -> D0 pi- pi+"]


def test_stable_requires_the_final_mode(tmp_path):
    path = write_input(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path), "--mode", "prompt", "--stable", "pi0"])


def test_max_occurrences_adds_columns(tmp_path):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [TWO, SIMPLE]}).write_parquet(path)

    main([str(path), "--mode", "raw", "--max-occurrences", "2"])

    df = pl.read_parquet(path)
    assert df["B+_raw"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] J/psi",
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
    ]
    assert df["B+_raw_2"].to_list() == [
        "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]",
        None,
    ]


def test_max_occurrences_must_be_positive(tmp_path):
    path = write_input(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path), "--max-occurrences", "0"])


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]"),
        (["--keep-isr"], "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu- gammaI]"),
        (["--keep-fsr"], "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu-]"),
        (["--keep-photos"], "B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu- gammaP]"),
        (
            ["--keep-isr", "--keep-fsr", "--keep-photos"],
            "B+ -> [K+ -> pi+ pi0 gammaF] [J/psi -> mu+ mu- gammaP gammaI]",
        ),
    ],
)
def test_photon_flags_select_individual_photon_kinds(tmp_path, flags, expected):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [FSR]}).write_parquet(path)

    main([str(path), "--mode", "raw", *flags])

    assert pl.read_parquet(path)["B+_raw"].to_list() == [expected]


def test_no_sort_preserves_daughter_order(tmp_path):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: ["[B+ -> pi+ K+]"]}).write_parquet(path)

    main([str(path), "--mode", "raw", "--no-sort"])

    assert pl.read_parquet(path)["B+_raw"].to_list() == ["B+ -> pi+ K+"]


def test_count_adds_a_count_column(tmp_path):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [TWO], "x": [1]}).write_parquet(path)

    main([str(path), "--mode", "final", "--count"])

    df = pl.read_parquet(path)
    assert df.columns == [COL, "x", "B+_count", "B+_final"]
    assert df["B+_count"].to_list() == [2]
