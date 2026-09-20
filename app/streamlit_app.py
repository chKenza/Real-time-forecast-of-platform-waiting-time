"""
Streamlit demo for the Transilien SNCF waiting-time forecasting model.

Run locally with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import RAW_FEATURE_COLUMNS, TARGET_COLUMN, preprocess_features  # noqa: E402
from src.utils import load_metadata, load_model  # noqa: E402

DEFAULT_MODEL_PATH = Path("models/lightgbm_production.joblib")
FALLBACK_MODEL_PATH = Path("models/lightgbm.joblib")
METADATA_PATH = Path("models/lightgbm.json")
SAMPLE_X_TEST_PATH = Path(__file__).resolve().parent / "sample_data" / "x_test_sample.csv"

st.set_page_config(page_title="SNCF waiting-time forecast", page_icon="🚆", layout="wide")


@st.cache_resource
def get_model_path() -> Path:
    """
    Resolve which saved pipeline to serve.

    :return: MODEL_PATH env var if set, else the production model if it
        exists, else the honest validation-split model
    """
    configured = os.environ.get("MODEL_PATH")
    if configured:
        return Path(configured)
    return DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH


@st.cache_resource
def load_pipeline(model_path: str):
    """
    Load the fitted pipeline at ``model_path``.

    :param model_path: path to a joblib-saved pipeline
    :return: the loaded pipeline
    """
    return load_model(model_path)


@st.cache_resource
def load_station_options(_pipeline) -> list[str]:
    """
    List the station codes the pipeline was trained on.

    :param _pipeline: fitted pipeline (leading underscore so Streamlit skips
        hashing it for the cache key)
    :return: sorted station codes
    """
    encoder = _pipeline.named_steps["preprocessor"].named_transformers_["gare"]
    return sorted(encoder.categories_[0].tolist())


@st.cache_data
def load_run_metadata(path: str) -> dict | None:
    """
    Load a training run's metadata JSON, if present.

    :param path: path to the metadata file written by :mod:`src.train`
    :return: the metadata dict, or ``None`` if the file doesn't exist
    """
    if not Path(path).exists():
        return None
    return load_metadata(path)


def build_raw_row(
    gare: str, arret: int, date: pd.Timestamp, past_values: dict[str, float]
) -> pd.DataFrame:
    """
    Assemble a single raw stop row in the same shape as the challenge's x_* files.

    :param gare: station code
    :param arret: stop number
    :param date: date of the stop
    :param past_values: the six past-delay features, keyed by column name
    :return: one-row dataframe with the raw feature columns
    """
    row = {"train": "N/A", "gare": gare, "date": date.strftime("%Y-%m-%d"), "arret": arret}
    row.update(past_values)
    return pd.DataFrame([row], columns=RAW_FEATURE_COLUMNS)


model_path = get_model_path()
if not model_path.exists():
    st.error(
        f"No trained model found at '{model_path}'. Train one first, e.g.:\n\n"
        "`python -m src.train --model lightgbm --tune "
        "--output models/lightgbm.joblib --predict-test`"
    )
    st.stop()

pipeline = load_pipeline(str(model_path))
station_options = load_station_options(pipeline)
metadata = load_run_metadata(str(METADATA_PATH))

st.title("🚆 Transilien SNCF: Platform Waiting-Time Forecast")
st.markdown(
    "Transilien SNCF Voyageurs displays an estimated waiting time on its platform "
    "screens. This project uses a **LightGBM** model, trained on real stop-level data "
    "from the [ENS Data Challenge](https://challengedata.ens.fr/participants/challenges/166/), "
    "to forecast how accurate that displayed time will turn out to be for a train two "
    "stations upstream. This app lets you try the model two ways: a **single "
    "prediction** for one train/station/date, or a **batch prediction** on a whole "
    "CSV file of stops."
)

with st.expander("About this project", expanded=False):
    st.markdown("""
Transilien SNCF Voyageurs operates over 6,200 trains a day for 3.4 million passengers
across Île-de-France, and displays an estimated waiting time at each station. This
project explores whether that estimate can be improved: given a train **k** at station
**s** on day **d**, predict **p0q0**, the difference (in minutes) between the theoretical
and the observed waiting time, for the train two stations *upstream* of where it
currently is, i.e. a short-term forecast, not a description of what already happened.

Full challenge description and data:
[ENS Data Challenge](https://challengedata.ens.fr/participants/challenges/166/).
        """)

with st.expander("About this model", expanded=False):
    if metadata is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Validation MAE", f"{metadata['validation_mae']:.3f} min")
        col2.metric("Naive baseline MAE", f"{metadata['baseline_mae']:.3f} min")
        col3.metric("Model", metadata["model_name"])
        st.caption(
            f"Trained on {metadata['train_rows']:,} rows, "
            f"validated on {metadata['validation_rows']:,} rows"
        )
    else:
        st.caption("No metadata file found — run src.train to generate one.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with tab_single:
    st.markdown(
        "Fill in one train's stop and its recent waiting-time history below, and the "
        "model will predict how far off the displayed waiting time will be for that "
        "stop, in minutes."
    )
    st.subheader("Stop context")
    col1, col2, col3 = st.columns(3)
    gare = col1.selectbox("Station (gare)", station_options)
    arret = col2.number_input("Stop number (arret)", min_value=1, max_value=100, value=10)
    date = col3.date_input("Date", value=pd.Timestamp.today())

    st.subheader("Recent waiting-time differences (minutes)")
    st.markdown(
        "For each value below: **positive** means that stop's actual wait ended up "
        "**shorter** than what was displayed (train arrived a bit early); **negative** "
        "means it was **longer** (train arrived a bit late). For example, `-2` means "
        "passengers waited 2 minutes longer than the screen said they would."
    )
    col_station, col_train = st.columns(2)
    with col_station:
        st.markdown("**Previous trains that stopped at this station**")
        p2q0 = st.number_input(
            "This station's waiting-time difference, 2 trains ago",
            value=0.0,
            step=1.0,
            help="How far off the displayed wait time was, for the 2nd train before this "
            "one that stopped at the same station. Positive = arrived earlier than "
            "displayed, negative = arrived later.",
        )
        p3q0 = st.number_input(
            "This station's waiting-time difference, 3 trains ago",
            value=0.0,
            step=1.0,
            help="Same as above, but for the 3rd-previous train at this station.",
        )
        p4q0 = st.number_input(
            "This station's waiting-time difference, 4 trains ago",
            value=0.0,
            step=1.0,
            help="Same as above, but for the 4th-previous train at this station.",
        )
    with col_train:
        st.markdown("**This same train, at its earlier stops**")
        p0q2 = st.number_input(
            "This train's waiting-time difference, 2 stations ago",
            value=0.0,
            step=1.0,
            help="How far off the displayed wait time was for this exact train, 2 stations "
            "before its current one. Positive = arrived earlier than displayed, "
            "negative = arrived later.",
        )
        p0q3 = st.number_input(
            "This train's waiting-time difference, 3 stations ago",
            value=0.0,
            step=1.0,
            help="Same as above, but 3 stations before its current one.",
        )
        p0q4 = st.number_input(
            "This train's waiting-time difference, 4 stations ago",
            value=0.0,
            step=1.0,
            help="Same as above, but 4 stations before its current one.",
        )

    if st.button("Predict waiting-time difference", type="primary"):
        past_values = {
            "p2q0": p2q0,
            "p3q0": p3q0,
            "p4q0": p4q0,
            "p0q2": p0q2,
            "p0q3": p0q3,
            "p0q4": p0q4,
        }
        raw_row = build_raw_row(gare, int(arret), pd.Timestamp(date), past_values)
        features = preprocess_features(raw_row)
        prediction = float(pipeline.predict(features)[0])
        st.metric("Predicted difference vs. the displayed waiting time", f"{prediction:+.2f} min")
        if prediction < 0:
            st.info(
                f"➡️ The model expects this train to arrive about **{abs(prediction):.2f} minutes "
                "later** than the time currently shown on the platform display — passengers "
                "should expect to wait a bit longer than announced."
            )
        else:
            st.info(
                f"➡️ The model expects this train to arrive about **{prediction:.2f} minutes "
                "earlier** than the time currently shown on the platform display — passengers "
                "should expect to wait a bit less than announced."
            )

with tab_batch:
    st.markdown(
        "Upload a CSV of many stops at once and "
        "the model will predict the waiting-time difference for every row, which you "
        "can then preview and download."
    )
    st.subheader("Upload a CSV in the challenge's x_test format")
    st.caption(
        "Required columns: " + ", ".join(RAW_FEATURE_COLUMNS) + ". Extra index columns are ignored."
    )
    if SAMPLE_X_TEST_PATH.exists():
        st.markdown("You can try the ENS Data Challenge's own `x_test.csv`:")
        st.download_button(
            "Download sample x_test.csv",
            data=SAMPLE_X_TEST_PATH.read_bytes(),
            file_name="x_test_sample.csv",
            mime="text/csv",
        )
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        unnamed_columns = [c for c in raw_df.columns if c.startswith("Unnamed")]
        raw_df = raw_df.drop(columns=unnamed_columns)
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw_df.columns]
        if missing:
            st.error(f"Missing required column(s): {missing}")
        else:
            features = preprocess_features(raw_df)
            predictions = pipeline.predict(features)
            result = raw_df.copy()
            result[TARGET_COLUMN] = predictions
            st.dataframe(result.head(100))
            st.download_button(
                "Download predictions as CSV",
                data=result[[TARGET_COLUMN]].to_csv().encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
