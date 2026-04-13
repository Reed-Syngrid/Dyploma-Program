# Battery SoH Cross-Dataset Project

State-of-Health (SoH) modeling across three battery datasets:

- NASA
- Oxford
- Warwick

The project includes two modeling tracks:

1. **Random Forest baseline** for cross-dataset validation.
2. **LSTM transfer-learning pipeline** with few-shot Oxford adaptation and target scaling.

## Repository Structure

- `nasa_extractor.py` - NASA raw `.mat` -> processed features
- `oxford_extractor.py` - Oxford raw `.mat` -> processed features
- `warwick_extractor.py` - Warwick raw `.csv` -> processed features
- `data_merger.py` - merges processed datasets into `output/MASTER_processed.csv`
- `cross_val.py` - Random Forest cross-dataset experiments
- `lstm_cross_val.py` - LSTM few-shot transfer experiment
- `main.py` - primary model entrypoint utilities

## Key Features

- Physics-aware normalization with `resistance_ratio = R / R0`
- Domain/context features:
  - `nominal_capacity`
  - `form_factor`
- Dataset source tagging and merged master table
- Cross-dataset evaluation (OOD)

## Data Outputs

Generated CSVs:

- `output/NASA_processed.csv`
- `output/OXFORD_processed.csv`
- `output/WARWICK_processed.csv`
- `output/MASTER_processed.csv`

Generated plots:

- `cross_val_plot.png`
- `triple_cross_val_plot.png`
- `lstm_transfer_plot.png`

## Quick Start

From project root:

```bash
python nasa_extractor.py
python oxford_extractor.py
python warwick_extractor.py
python data_merger.py
```

Run Random Forest validation:

```bash
python cross_val.py
```

Run LSTM transfer validation:

```bash
python lstm_cross_val.py
```

## Version Notes

- **v1.0**: Random Forest baseline and R/R0 normalization.
- **v1.1**: LSTM transfer-learning pipeline + physics context columns.

