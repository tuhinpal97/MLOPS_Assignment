# Dataset

This project uses the UCI Heart Disease Cleveland data. The data is reproduced from the original source by the project itself rather than requiring a manual download.

Generate the raw and cleaned files from a clean checkout with:

```bash
python scripts/download_data.py
```

This creates:

- `data/raw/processed.cleveland.data` - downloaded Cleveland source data.
- `data/processed/heart_clean.csv` - cleaned 13-feature binary-classification dataset.

The download logic first tries the UCI source URL and then a public mirror fallback. Missing values represented by `?` are parsed as missing data; the original `num` outcome is converted to binary `target` (`0` => no disease, `1-4` => disease present). Feature imputation is intentionally deferred to the fitted scikit-learn preprocessing pipeline so cross-validation remains leakage-safe.
