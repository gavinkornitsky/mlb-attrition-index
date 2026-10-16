# MLB Attrition Index

Measures how much each batter drives up pitch counts per out.

## Reproduce
Run everything from the repo root.

1. `python src/data_pull.py` (writes `statcast_raw.parquet` and `pa_table.parquet`, both gitignored)
2. `python src/baseline.py` (writes `batter_baseline.csv`)
3. Any of, in any order:
   - `python src/decomposition.py`
   - `python src/decomp_zone.py`
   - `python src/reliability.py`
