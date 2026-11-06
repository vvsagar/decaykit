# decaykit

Parser for MC decay strings.

`decaykit` turns a column of `__MCDecayString__` values such as

```
[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]
```

into flat columns with the direct daughters, the final-state particles, or the
raw decay block.

## Install

```bash
pip install decaykit
# or: uv add decaykit --active
```

## Usage


### CLI:

```bash
decaykit mc.root                     # by default looks for 'variables' tree
decaykit mc.root -t tree_0           # read the ROOT tree, write mc_tree_0.pq
decaykit mc.root -t tree_0 -o out.pq # choose where the parquet goes
decaykit data.pq                     # rewrite data.pq in place
decaykit data.pq -o out.pq           # keep data.pq untouched; asks if out.pq exists
decaykit data.pq -o out.pq -y        # ... skip that confirmation
decaykit data.pq --mode final --mode prompt
decaykit data.pq --particle B0
decaykit data.pq --particle B0 --stable D0 --stable pi0  # keep D0 and pi0 as a final-state leaf
decaykit data.pq --particle B0 --count                   # add B0_count
decaykit data.pq --particle B0 --max-occurrences 2       # add _2 columns
decaykit data.pq --keep-isr --keep-fsr                   # keep gammaI and gammaF
decaykit data.pq --keep-photos                           # keep gammaP
decaykit data.pq --no-sort                               # keep the input daughter order
decaykit data.pq --n-workers 8             # parse in 8 processes
```

### library

```python
import polars as pl

from decaykit import add_decay_columns

df = pl.DataFrame({"__MCDecayString__": ["[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"]})

df = add_decay_columns(df, modes=("prompt", "final"), particle="B+")
# shape: (1, 3)
# ┌─────────────────────────────────┬────────────────┬───────────────────────┐
# │ __MCDecayString__               ┆ B+_prompt      ┆ B+_final              │
# ╞═════════════════════════════════╪════════════════╪═══════════════════════╡
# │ [B+ -> [K+ -> pi+ pi0] [J/psi… ┆ B+ -> J/psi K+ ┆ B+ -> pi+ pi0 mu+ mu- │
# └─────────────────────────────────┴────────────────┴───────────────────────┘
```

`add_decay_columns` works on both `DataFrame` and `LazyFrame`. It parses each
event once regardless of how many variants are requested.

Radiative photons are stripped by default. `gammaI` (ISR), `gammaF` (FSR) and
`gammaP` (PHOTOS) can each be kept independently:

```python
df = add_decay_columns(
    df,
    ignore_isr=False,     # keep gammaI
    ignore_fsr=False,     # keep gammaF
    ignore_photos=False,  # keep gammaP
)
```

