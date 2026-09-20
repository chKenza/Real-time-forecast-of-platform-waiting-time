# Real time forecast of platform waiting time

We present our solution to the SNCF-Transilien 
["Platform Waiting-Time Forecast"](https://challengedata.ens.fr/participants/challenges/166/)
ENS Data Challenge. We predict, for a train two stations upstream,
the difference (in minutes) between the theoretical and the actual waiting time
displayed on the platform.

We train and tune the following models: Liner Regression, Random Forest,
Extra Trees, Histogram Gradient Boosting, LightGBM. We also include unit
tests, a CI pipeline, a Streamlit app, and Docker containerization.

CI: [![CI](https://github.com/chKenza/Real-time-forecast-of-platform-waiting-time/actions/workflows/ci.yml/badge.svg)](https://github.com/chKenza/Real-time-forecast-of-platform-waiting-time/actions/workflows/ci.yml)

Coverage: [![codecov](https://codecov.io/github/chkenza/real-time-forecast-of-platform-waiting-time/graph/badge.svg)](https://app.codecov.io/github/chkenza/real-time-forecast-of-platform-waiting-time)

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS: LightGBM needs the OpenMP runtime, not installed by default:

```bash
brew install libomp
```

`data/` is gitignored. Download `x_train`, `y_train`, `x_test`, `y_sample` from the
[challenge page](https://challengedata.ens.fr/participants/challenges/166/) and place
them in `data/` under their original file names.

## Training

To train and tune a model, run:

```bash
python -m src.train --model lightgbm --tune --output models/lightgbm.joblib --predict-test
```

`--model` selects any entry from the registry in `src/model.py`
(`random_forest`, `extra_trees`, `gradient_boosting`, `lightgbm`, `linear_regression`);
`--tune` runs a grid search scored on a held-out chronological validation split;
`--predict-test` refits on all available labelled data and writes a submission CSV.

Run `python -m src.train --help` for the full list of options.

To generate validation diagnostics
(predicted-vs-actual, residuals, error by weekday, daily trend) for one or more
trained model:

```bash
python -m src.evaluate --model-path models/lightgbm.joblib --labels lightgbm
```

## Results

Validation Mean Absolute Error (MAE):

| Model                           | Validation MAE   |
| ------------------------------- | ---------------- |
| Naive baseline (predict 0)      | 0.873            |
| Linear regression (Ridge)       | 0.816            |
| Random forest                   | 0.751            |
| Hist Gradient boosting               | 0.761            |
| Extra trees                     | 0.744            |
| **LightGBM (final choice)**     | **0.740**        |

LightGBM and extra trees were essentially tied. LightGBM was chosen since its saved
model is much smaller (1MB vs. ~400MB), which matters for the Docker image and app
startup time.

The final model is included in this repository under `models/`. The model is used by the app and refit refit on all
locally available labelled data with the tuned hyperparameters.


## Application

A Streamlit app (`app/streamlit_app.py`) serves the trained model for single or
batch predictions.

### Run it from source
You can run the Streamlit app using:

```bash
streamlit run app/streamlit_app.py
```

or containerized:

```bash
docker build -t sncf-waiting-time .
docker run -p 8501:8501 sncf-waiting-time
```

Then open http://localhost:8501.


### Run it from the pre-built Docker image

The image is published on [Docker Hub](https://hub.docker.com/r/chkenza/sncf-waiting-time)
with the trained model.
You can get the working app without cloning the repository using:

```bash
docker pull chkenza/sncf-waiting-time
docker run -p 8501:8501 chkenza/sncf-waiting-time
```

Then open http://localhost:8501.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and how to run the project.
