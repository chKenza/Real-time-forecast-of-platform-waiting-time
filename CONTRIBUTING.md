# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

macOS only: LightGBM needs the OpenMP runtime, not installed by default:

```bash
brew install libomp
```

## Getting the data

`data/` is gitignored. Download `x_train`, `y_train`, `x_test`, `y_sample` from the
[ENS Data Challenge](https://challengedata.ens.fr/participants/challenges/166/)
and place them in `data/` under their original file names.

## Running the project

```bash
# Generate EDA figures + a data-quality summary
python -m src.eda

# Train a model (see src/model.py's MODEL_REGISTRY for available --model choices)
python -m src.train --model lightgbm --tune --output models/lightgbm.joblib --predict-test

# Compare trained models' validation performance
python -m src.evaluate --model-path models/lightgbm.joblib --labels lightgbm

# Run the app locally
streamlit run app/streamlit_app.py
```

Full CLI options: `python -m src.train --help` (same pattern for `src.eda` / `src.evaluate`).

## Running with Docker

```bash
docker build -t sncf-waiting-time .
docker run -p 8501:8501 sncf-waiting-time
```

Then open http://localhost:8501.

## Running tests, lint & formatting

```bash
pip install flake8 black pytest
flake8 src app tests
black --check src app tests
pytest tests/
```
