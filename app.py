"""Streamlit demo for historical and latest-observation LGA predictions."""

import joblib
import pandas as pd
import requests
import streamlit as st

from weather import (
    CATEGORY_LABELS, HOURS, OUTPUTS, PROCESSED, STATIONS, TZ,
    five_probabilities, local_cutoff, make_features, observation_table, previous_day_maxima,
)

METAR_URL = "https://aviationweather.gov/api/data/metar"
METAR_FIELDS = ["icaoId", "obsTime", "temp", "dewp", "wspd"]


def fetch_metar():
    response = requests.get(
        METAR_URL,
        params={"ids": ",".join(f"K{station}" for station in STATIONS), "format": "json", "hours": 48},
        timeout=(10, 30),
    )
    response.raise_for_status()
    if response.status_code == 204:
        raise ValueError("Aviation Weather Center returned no METAR observations.")
    records = response.json()
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of METAR observations.")
    frame = pd.DataFrame(records)
    if not {"icaoId", "obsTime"}.issubset(frame.columns):
        raise ValueError("The METAR response is missing station IDs or observation times.")
    frame = frame.reindex(columns=METAR_FIELDS)
    observations = {}
    for station in STATIONS:
        rows = frame.loc[frame["icaoId"] == f"K{station}"]
        observations[station] = observation_table(
            pd.to_datetime(pd.to_numeric(rows["obsTime"], errors="coerce"), unit="s", errors="coerce", utc=True),
            pd.to_numeric(rows["temp"], errors="coerce"),
            pd.to_numeric(rows["dewp"], errors="coerce"),
            pd.to_numeric(rows["wspd"], errors="coerce") * 0.514444,
        )
    return observations


def get_artifact(hour):
    return joblib.load(OUTPUTS / f"model_{hour:02d}.joblib")


def show_probabilities(artifact, features):
    columns = artifact["feature_columns"]
    probabilities = five_probabilities(artifact["model"], features[columns])[0]
    table = pd.DataFrame({"Category": CATEGORY_LABELS, "Probability": probabilities})
    st.success(f"Most likely category: {CATEGORY_LABELS[probabilities.argmax()]}")
    st.bar_chart(table, x="Category", y="Probability", sort=False)
    st.dataframe(
        table,
        hide_index=True,
        column_config={"Probability": st.column_config.NumberColumn(format="percent")},
    )


def format_hour(hour):
    return f"{hour:02d}:00"


def historical_replay():
    hour = st.selectbox("Prediction cutoff (New York time)", HOURS, format_func=format_hour)
    samples = pd.read_csv(PROCESSED / f"samples_{hour:02d}.csv", parse_dates=["date"])
    test = samples.loc[samples["date"].dt.year == 2025].set_index("date")
    day = st.selectbox("Eligible 2025 test date", list(test.index), format_func=lambda date: date.strftime("%Y-%m-%d"))
    row = test.loc[[day]]
    artifact = get_artifact(hour)
    show_probabilities(artifact, row)
    st.metric("Actual LGA daily hourly-observation maximum", f"{row['target_max'].iloc[0]:.1f}°F")
    st.caption(f"Model stations: {', '.join(artifact['stations'])}")


def latest_observations():
    now = pd.Timestamp.now(tz=TZ)
    day = now.date()
    reached = [hour for hour in HOURS if local_cutoff(day, hour) <= now]
    if not reached:
        st.info("Today's first prediction is available at 09:00 New York time.")
        return

    hour = reached[-1]
    st.write(f"Today's latest scheduled cutoff is **{format_hour(hour)}** New York time.")
    if not st.button("Fetch and predict", type="primary"):
        return

    try:
        with st.spinner("Fetching recent METAR observations…"):
            observations = fetch_metar()
    except (requests.RequestException, ValueError) as error:
        st.error(f"METAR request failed: {error}")
        return

    artifact = get_artifact(hour)
    stations = artifact["stations"]
    maxima = {station: previous_day_maxima(observations[station]) for station in stations}
    features, timestamps = make_features(observations, maxima, stations, day, hour)
    if timestamps["LGA"] is None:
        st.warning("LGA has missing or stale data. The cutoff requires an observation within the preceding hour.")
        return

    show_probabilities(artifact, pd.DataFrame([features]))
    st.dataframe(pd.DataFrame([
        {"Station": station, "Latest input observation (New York time)": (
            timestamp.strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp is not None else "Missing or stale"
        )}
        for station, timestamp in timestamps.items()
    ]), hide_index=True)
    st.caption("Live inputs come from Aviation Weather Center METAR. The models were trained on NOAA Global Hourly.")


def main():
    st.set_page_config(page_title="LGA daily high forecast", layout="centered")
    st.title("LGA daily high forecast")
    st.write("Estimate the five-category probability distribution for today's final hourly-observation maximum.")
    mode = st.radio("Mode", ("Historical replay", "Latest observations"), horizontal=True)
    try:
        if mode == "Historical replay":
            historical_replay()
        else:
            latest_observations()
    except FileNotFoundError:
        st.error("Run python weather.py to generate the local model and sample files.")


if __name__ == "__main__":
    main()
