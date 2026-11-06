import numpy as np
import polars as pl
import pytest
import uproot

from decaykit.cli import main

COL = "__MCDecayString__"
SIMPLE = "[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"


def write_input(tmp_path):
    path = tmp_path / "in.parquet"
    pl.DataFrame({COL: [SIMPLE], "x": [1]}).write_parquet(path)
    return path


def write_root(tmp_path, tree="tree"):
    path = tmp_path / "in.root"
    with uproot.recreate(path) as f:
        f[tree] = {COL: np.array([SIMPLE]), "x": np.array([1])}
    return path


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

    df = pl.read_parquet(path.with_suffix(".pq"))
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
    assert not path.with_suffix(".pq").exists()


def test_root_input_requires_a_tree(tmp_path):
    path = write_root(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path)])


def test_root_input_defaults_to_variables_tree(tmp_path):
    path = write_root(tmp_path, tree="variables")

    main([str(path)])

    assert "B+_prompt" in pl.read_parquet(path.with_suffix(".pq")).columns


def test_tree_is_rejected_for_parquet_input(tmp_path):
    path = write_input(tmp_path)

    with pytest.raises(SystemExit):
        main([str(path), "-t", "tree"])
