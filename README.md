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
decaykit data.pq                     # rewrite data.pq in place
decaykit data.pq -o out.pq           # keep data.pq untouched; asks if out.pq exists
decaykit data.pq -o out.pq -y        # ... skip that confirmation
decaykit data.pq --mode final --mode prompt
decaykit data.pq --particle B0
decaykit data.pq --n-workers 8       # parse in 8 processes
decaykit mc.root -t tree_0           # read the ROOT tree, write mc_tree_0.pq
decaykit mc.root -t tree_0 -o out.pq # choose where the parquet goes
decaykit vars.root                   # a 'variables' tree needs no -t
```

### library

```python
import polars as pl

from decaykit import extract_decay_final, extract_decay_prompt

df = pl.DataFrame({"__MCDecayString__": ["[B+ -> [K+ -> pi+ pi0] [J/psi -> mu+ mu-]]"]})

df.with_columns(
    extract_decay_prompt(particle="B+"),  # direct daughters
    extract_decay_final(particle="B+"),  # leaves of the decay tree
)
# shape: (1, 3)
# ┌─────────────────────────────────┬────────────────┬───────────────────────┐
# │ __MCDecayString__               ┆ B+_prompt      ┆ B+_final              │
# ╞═════════════════════════════════╪════════════════╪═══════════════════════╡
# │ [B+ -> [K+ -> pi+ pi0] [J/psi… ┆ B+ -> J/psi K+ ┆ B+ -> pi+ pi0 mu+ mu- │
# └─────────────────────────────────┴────────────────┴───────────────────────┘
```

