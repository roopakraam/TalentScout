# Data

The raw and processed data are not committed (the CSV is about 96 MB).

## Getting the raw data

```bash
python scripts/download_data.py
```

This downloads the public Kaggle dataset
[EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
(Stefano Leone) and extracts `male_players.csv` into `data/raw/`. No Kaggle account is needed.
You can also download the archive by hand from the Kaggle page and unzip it there.

| File | Rows | Columns | Content |
| --- | --- | --- | --- |
| `data/raw/male_players.csv` | 180,021 | 109 | Every male player in FIFA 15 to EA Sports FC 24, one row per player per version |

## Processed files

Created by `notebooks/01_data_and_label.ipynb`:

| File | Content |
| --- | --- |
| `data/processed/labelled.parquet` | Eligible rows (age 21 or under, overall below 80, FIFA 15 to 21) with the label, the candidate features and the `set` column |
| `data/processed/split_ids.csv` | `player_id`, `fifa_version`, `set` for every row of the main analysis |
