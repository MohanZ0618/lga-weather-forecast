"""Streamlit demo for historical and latest-observation LGA predictions."""

import joblib
import pandas as pd
import json
import sys
from pathlib import Path
import streamlit as st

from weather import CATEGORY_LABELS, HOURS, OUTPUTS, PROCESSED, five_probabilities
from market_replay import market_replay

# The training and standalone replay entry points use these same module names.
sys.path.insert(0, str(Path(__file__).resolve().parent / "nowcast"))
from runtime import ARTIFACT, CACHE, latest_slot, run
from comparison_view import forecast_comparison, saved_temperature
from temperature import FINE_LABELS, MODEL


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


def fusion_backtest():
    path = ARTIFACT.parent.parent / "results" / "predictions.parquet"
    if not path.exists():
        st.info("Run python nowcast/study.py to generate the evaluated forecasts.")
        return
    rows = pd.read_parquet(path, filters=[("level","==","deployed"),("delay","==",0)])
    day = st.selectbox("Evaluated New York date", sorted(rows.day.unique(), reverse=True))
    selected = rows.loc[rows.day.eq(day)].sort_values("cutoff")
    stamp = st.selectbox("Forecast time", list(selected.cutoff),
                         format_func=lambda s: pd.Timestamp(s).strftime("%H:%M %Z"))
    row = selected.loc[selected.cutoff.eq(stamp)].iloc[0]
    probs = pd.Series([row[f"p{i}"] for i in range(5)],index=CATEGORY_LABELS,name="Probability")
    st.bar_chart(probs,horizontal=True)
    st.write(f"Predicted band: {probs.idxmax()}. Observed hourly maximum: {row['actual']:.2f}°F.")
    st.caption(f"Selected model input set: {row['route']}.")


@st.fragment(run_every="60s")
def latest_observations():
    at = latest_slot()
    st.caption("Every 15 minutes from 06:00 through 20:00 New York time while this page is open.")
    if not ARTIFACT.exists():
        st.error("Train the quarter-hour models with python nowcast/study.py.")
        return
    with st.spinner("Reading the latest published weather data…"):
        result = run(at)
    if result["status"] != "ok":
        st.warning(f"Forecast unavailable for {at:%H:%M %Z}. " + result.get("reason", "The LGA observation is missing or stale."))
        result = result.get("last_success")
        if not result:
            return
    stamp = pd.Timestamp(result["cutoff"])
    labels = {"station":"Airport observations", "latest":"Airports + HRRR",
              "local_cloud":"Airports + HRRR + local cloud", "upwind_cloud":"Airports + HRRR + upwind cloud"}
    st.subheader(stamp.strftime("%b %d, %Y · %H:%M %Z"))
    folder = CACHE/"live"/result["version"]
    input_path = folder / stamp.strftime("%Y%m%d-%H%M.inputs.json")
    fine = False
    if MODEL.exists() and input_path.exists():
        fine = st.radio("Range width",["5°F bands","Original five bands"],horizontal=True) == "5°F bands"
    bands = FINE_LABELS if fine else CATEGORY_LABELS
    values = saved_temperature(input_path,result)[1] if fine else result["probabilities"]
    probs = pd.Series(values, index=bands, name="Probability")
    st.metric("Most likely daily high", probs.idxmax(), help="Maximum of retained hourly observations at LaGuardia.")
    st.caption(f"{labels[result['tier']]} · model {result['version']}")
    st.bar_chart(probs, horizontal=True)
    st.dataframe(probs.to_frame(), column_config={"Probability":st.column_config.NumberColumn(format="percent")})
    history = []
    for file in sorted(folder.glob(stamp.strftime("%Y%m%d")+"-????.json")):
        saved = json.loads(file.read_text())
        if fine:
            inputs = file.with_suffix(".inputs.json")
            if not inputs.exists():
                continue
            _, values = saved_temperature(inputs,saved)
        else:
            values = saved["probabilities"]
        if values is not None:
            history.append({"Time":pd.Timestamp(saved["cutoff"]).strftime("%H:%M"), **dict(zip(bands,values))})
    if history:
        trends = pd.DataFrame(history).set_index("Time")
        if fine:
            trends = trends[trends.mean().nlargest(3).index.tolist()]
        st.line_chart(trends, x_label="Forecast time (New York)", y_label="Probability")
    with st.expander("Compare available input sets · original five bands"):
        comparisons = pd.DataFrame(result["available_predictions"],index=CATEGORY_LABELS).rename(columns=labels)
        st.dataframe(comparisons.style.format("{:.1%}"))
    with st.expander("Data timestamps and source status"):
        st.write("New York station observation and receipt times")
        st.dataframe(pd.DataFrame({"Observed":result["station_times"],"Received":result["station_receipts"]}))
        st.write({"HRRR run":result["hrrr_run"],"HRRR available":result["hrrr_available"],
                  "GOES scan end":result["satellite_scan"],"GOES available":result["satellite_available"]})
        inputs = result["inputs"]
        if inputs["sat10"] is not None:
            st.write(f"Cloud cover within 10 km: {inputs['sat10']:.0%}.")
        if inputs["nwp_high"] is not None:
            st.write(f"HRRR remaining hourly maximum: {inputs['nwp_high']:.1f}°F.")
        if result["source_errors"]:
            st.write(result["source_errors"])


def main():
    st.set_page_config(page_title="LaGuardia High Temperature Forecast", layout="wide")
    mode = st.sidebar.radio("Project view", ("Polymarket replay", "Forecast comparison", "Latest observations",
                                    "Fusion backtest", "Historical replay"))
    if mode == "Polymarket replay":
        market_replay()
        return
    st.title("LGA daily high forecast")
    try:
        if mode == "Forecast comparison":
            forecast_comparison(CATEGORY_LABELS)
        elif mode == "Historical replay":
            historical_replay()
        elif mode == "Fusion backtest":
            fusion_backtest()
        else:
            latest_observations()
    except FileNotFoundError:
        st.error("Run python weather.py to generate the local model and sample files.")


if __name__ == "__main__":
    main()
