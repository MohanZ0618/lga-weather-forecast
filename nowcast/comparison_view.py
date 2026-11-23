"""Streamlit comparison of day-ahead, intraday and observed LGA temperatures."""
from datetime import timedelta
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from features import TZ
from runtime import ARTIFACT, CACHE, latest_slot, run
from sources import HERE
from temperature import (FINE_COLUMNS, FINE_LABELS, MODEL, fine_probabilities, issue_time, live_day,
                         model, observations, recent_metars, rolling_point)

OUT = HERE / "results"
DAY, SAME, OBS, FINAL = "Day-ahead high", "Same-day high", "Observed temperature", "Final observed high"
COLORS = {DAY:"#ae6400", SAME:"#155bbe", OBS:"#00806e", FINAL:"#413557"}
PROBS = [f"p{i}" for i in range(5)]


@st.cache_data
def historic_tables(replay_stamp, obs_stamp):
    return pd.read_parquet(OUT/"temperature_replay.parquet"), pd.read_parquet(OUT/"temperature_observations.parquet")


@st.cache_resource
def reference(version):
    return joblib.load(ARTIFACT)["reference"]


@st.cache_data
def archived_point(path, modified, version, point_version):
    payload = json.loads(path.read_text())
    return rolling_point(pd.Timestamp(payload["cutoff"]).tz_convert(TZ),payload,reference(version))


def detailed(point, floor, probabilities, tier):
    return fine_probabilities([point],[floor],np.asarray([probabilities]),model()["tiers"][tier]["residuals"])[0]


def saved_temperature(path, saved):
    point = archived_point(path,path.stat().st_mtime_ns,saved["version"],MODEL.stat().st_mtime_ns)
    if point is None:
        return None, None
    return point, detailed(point,saved["high_so_far_f"],saved["probabilities"],"rolling")


@st.cache_data(ttl=900)
def observation_feed():
    records = recent_metars()
    return records, pd.Timestamp.now(tz=TZ)


def live_inputs(day, now):
    slot = latest_slot(now)
    latest = run(slot)
    if latest and latest["status"] != "ok":
        st.warning(f"Unable to update the {slot:%H:%M} forecast. Showing available saved observations.")
    version = json.loads((OUT/"selection.json").read_text())["version"]
    folder = CACHE/"live"/version
    files = sorted(folder.glob("*.inputs.json"))
    rows = []
    for path in files:
        if not path.name.startswith(day.strftime("%Y%m%d-")):
            continue
        saved_path = path.with_suffix("").with_suffix(".json")
        if not saved_path.exists():
            continue
        saved = json.loads(saved_path.read_text())
        value, fine = saved_temperature(path,saved)
        if value is not None:
            rows.append({"time":pd.Timestamp(saved["cutoff"]).tz_convert(TZ), "high":value, **dict(zip(FINE_COLUMNS,fine)),
                         **dict(zip(PROBS,saved["probabilities"]))})
    records = []
    fetched = None
    # A later 48-hour feed can verify a previous day after it closes.
    for path in reversed(files[-2:]):
        payload = json.loads(path.read_text())
        stamp = pd.Timestamp(payload["fetched"]).tz_convert(TZ)
        if stamp <= now and (now - stamp) < pd.Timedelta(hours=48):
            records, fetched = payload["metars"], stamp
            break
    if fetched is None or now - fetched > pd.Timedelta(minutes=15):
        try:
            records, fetched = observation_feed()
        except requests.RequestException:
            st.caption("The station feed is unavailable. Using the latest saved readings.")
    if not records:
        st.caption("No recent saved station feed is available for the observed curve.")
    try:
        use_records = records if fetched is not None and fetched >= issue_time(day) and records else None
        if fetched is not None and fetched - issue_time(day) > pd.Timedelta(hours=32):
            # Prefer an earlier 48-hour feed that still covers the issue-day morning.
            since = issue_time(day).strftime("%Y%m%d-%H%M")
            until = (issue_time(day) + pd.Timedelta(hours=24)).strftime("%Y%m%d-%H%M")
            path = next((p for p in files if since <= p.name[:13] <= until),None)
            if path:
                use_records = json.loads(path.read_text())["metars"]
        ahead = live_day(day, now, use_records)
    except (requests.RequestException, ValueError) as exc:
        st.warning(f"Day-ahead source unavailable. {exc}")
        ahead = None
    obs, actual = observations(records,day,fetched if fetched is not None else now)
    if ahead:
        ahead["fine"] = detailed(ahead["high"],-np.inf,ahead["probabilities"],ahead["tier"])
    return ahead, pd.DataFrame(rows,columns=["time","high",*PROBS,*FINE_COLUMNS]), obs, actual


def hour(t):
    return t.hour + t.minute/60 + t.second/3600


def trend_plot(data, ylabel, errors=False):
    domain = list(dict.fromkeys(data.series))
    color = alt.Color("series:N",title=None,sort=domain,scale=alt.Scale(domain=domain,range=[COLORS[s] for s in domain]),
                      legend=alt.Legend(orient="bottom",columns=2))
    base = alt.Chart(data).encode(
        x=alt.X("clock:Q",title="Target day · New York local time",scale=alt.Scale(domain=[0,24]),
                axis=alt.Axis(values=[0,3,6,9,12,15,18,21,24],labelExpr="format(datum.value, '02d') + ':00'")),
        y=alt.Y("value:Q",title=ylabel,scale=alt.Scale(zero=errors)),
        color=color,detail="series:N",
        tooltip=[alt.Tooltip("series:N",title="Series"),alt.Tooltip("when:N",title="Local time"),
                 alt.Tooltip("value:Q",title=ylabel,format=".1f")])
    line = base.mark_line(strokeWidth=2.6).encode(strokeDash=alt.StrokeDash("series:N",
        scale=alt.Scale(domain=[DAY,SAME,OBS,FINAL],range=[[7,4],[1,0],[1,0],[2,4]]),legend=None))
    chart = line + base.transform_filter("datum.marker").mark_point(filled=True,size=48)
    if errors:
        chart = alt.Chart(pd.DataFrame({"zero":[0]})).mark_rule(color="#888").encode(y="zero:Q") + chart
    st.altair_chart(chart.properties(height=215 if errors else 310), width="stretch")


def comparison(day, ahead, rolling, obs, actual, labels):
    completed = actual is not None and np.isfinite(actual)
    last = rolling.iloc[-1] if len(rolling) else None
    a,b,c = st.columns(3)
    a.metric("Previous noon forecast", f"{ahead['high']:.1f} °F" if ahead else "—")
    b.metric("Latest same-day forecast", f"{last.high:.1f} °F" if last is not None else "—")
    c.metric("Final observed high" if completed else "Latest measured temperature",
             f"{actual:.1f} °F" if completed else f"{obs.iloc[-1]:.1f} °F" if len(obs) else "—")
    if ahead:
        cycle = "airports + 12 UTC HRRR" if ahead["tier"]=="day_hrrr" else "airports"
        st.caption(f"Day-ahead cutoff {issue_time(day):%b %d, %H:%M %Z} · {cycle}.")
    points = []
    def add(clock, value, series, when, marker=True):
        points.append(dict(clock=clock,value=float(value),series=series,when=when,marker=marker))
    def horizontal(value, series):
        for x in (0,24):
            add(x,value,series,"Full target day",False)
    if ahead:
        horizontal(ahead["high"],DAY)
    if completed:
        horizontal(actual,FINAL)
    for t,value in obs.items():
        add(hour(t),value,OBS,t.strftime("%H:%M %Z"))
    for r in rolling.itertuples():
        add(hour(r.time),r.high,SAME,r.time.strftime("%H:%M %Z"))
    if points:
        trend_plot(pd.DataFrame(points),"Temperature (°F)")
    if completed:
        diffs = [dict(p,value=p["value"]-actual) for p in points if p["series"] in (DAY,SAME)]
        st.subheader("Forecast error")
        st.caption("Forecast high minus final observed high (°F). Positive values indicate an overestimate.")
        if diffs:
            trend_plot(pd.DataFrame(diffs),"Error (°F)",True)
        if ahead and last is not None:
            st.write(f"Day ahead {ahead['high']-actual:+.1f} °F  ·  Latest same day {last.high-actual:+.1f} °F")
    elif len(obs) and last is not None:
        st.write(f"Forecast high minus latest observation: **{last.high-obs.iloc[-1]:+.1f} °F** "
                 f"(observed at {obs.index[-1]:%H:%M %Z}). Final error is pending.")
        gaps = []
        for t,value in obs.items():
            for series,prediction in [(DAY,ahead["high"] if ahead else None),
                                      (SAME,rolling.loc[rolling.time <= t,"high"].iloc[-1]
                                       if (rolling.time <= t).any() else None)]:
                if prediction is not None:
                    gaps.append(dict(clock=hour(t),value=prediction-value,series=series,
                                     when=t.strftime("%H:%M %Z"),marker=True))
        if gaps:
            trend_plot(pd.DataFrame(gaps),"Forecast minus observation (°F)",True)
    if ahead or last is not None:
        st.subheader("Temperature range probabilities")
        fine = st.radio("Range width",["5°F bands","Original five bands"],horizontal=True) == "5°F bands"
        names, columns = (FINE_LABELS,FINE_COLUMNS) if fine else (labels,PROBS)
        table = {}
        if ahead:
            table[DAY] = ahead["fine" if fine else "probabilities"]
        if last is not None:
            table[SAME] = last[columns].astype(float).tolist()
        top = [f"{name}: {names[np.argmax(values)]} ({max(values):.1%})" for name, values in table.items()]
        st.write("  ·  ".join(top))
        probs = pd.DataFrame(table,index=pd.Index(names,name="Band")).reset_index().melt(
            id_vars="Band",var_name="Forecast",value_name="Probability")
        bars = alt.Chart(probs).mark_bar().encode(y=alt.Y("Band:N",sort=list(names),title="Daily high (°F)"),
            x=alt.X("Probability:Q",axis=alt.Axis(format="%"),scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Forecast:N",scale=alt.Scale(domain=list(table),range=[COLORS[key] for key in table]),
                            legend=alt.Legend(orient="bottom")),yOffset="Forecast:N",
            tooltip=["Forecast:N","Band:N",alt.Tooltip("Probability:Q",format=".2%")])
        st.altair_chart(bars.properties(height=20*len(names)+36),width="stretch")
        if len(rolling)>1:
            with st.expander("Probability changes during the day"):
                history = rolling.set_index("time")[columns].astype(float)
                history.columns = names
                selected = st.multiselect("Ranges to plot (°F)",list(names),
                                          default=history.mean().nlargest(3).index.tolist())
                history.index = history.index.strftime("%H:%M")
                if selected:
                    st.line_chart(history[selected],x_label="Forecast time (New York)",y_label="Probability")


@st.fragment(run_every="60s")
def forecast_comparison(labels):
    if not MODEL.exists():
        st.info("Run python nowcast/train_temperature.py to build the temperature models.")
        return
    now = pd.Timestamp.now(tz=TZ)
    today = now.date()
    mode = st.radio("Target day", ["Today","Tomorrow","Yesterday","Historical comparison"], horizontal=True)
    if mode == "Historical comparison":
        replay, history = historic_tables((OUT/"temperature_replay.parquet").stat().st_mtime_ns,
                                          (OUT/"temperature_observations.parquet").stat().st_mtime_ns)
        day = st.selectbox("Evaluated date",sorted(replay.day.unique(),reverse=True))
        rows = replay.loc[replay.day.eq(day)]
        prev = rows.loc[rows.horizon.eq("Day ahead")].iloc[0]
        ahead = {"high":prev.forecast,"tier":prev.tier,"probabilities":prev[PROBS].tolist(),
                 "fine":prev[FINE_COLUMNS].tolist()}
        rolling = rows.loc[rows.horizon.eq("Same day")].rename(columns={"forecast":"high"}).copy()
        rolling["time"] = pd.to_datetime(rolling.cutoff,utc=True).dt.tz_convert(TZ)
        comparison(pd.Timestamp(day).date(),ahead,rolling.sort_values("time"),history.loc[day,"temp"].dropna(),
                   prev.actual,labels)
    else:
        day = today + timedelta(days={"Today":0,"Tomorrow":1,"Yesterday":-1}[mode])
        st.subheader(day.strftime("%A, %B %d"))
        if issue_time(day) > now:
            st.info(f"The first forecast for this day is scheduled for {issue_time(day):%b %d at %H:%M %Z}.")
        else:
            ahead, rolling, obs, actual = live_inputs(day,now)
            if ahead is None:
                st.caption("The previous-noon inputs for this target are unavailable.")
            comparison(day,ahead,rolling,obs,actual,labels)
    with st.expander("Accuracy by forecast horizon"):
        metrics = pd.read_csv(OUT/"temperature_metrics.csv")
        st.caption("September 1 – November 6, 2026. Same target dates in both horizons. "
                   "Each date has equal weight. Lower MAE, RMSE and log loss are better.")
        cols = ["horizon","days","rows","accuracy","fine_accuracy","mae","rmse","bias","log_loss","fine_log_loss"]
        st.dataframe(metrics[cols].rename(columns=dict(zip(cols,["Horizon","Days","Forecasts","5-band hit rate",
            "5°F hit rate","MAE (°F)","RMSE (°F)","Bias (°F)","5-band log loss","5°F log loss"])))
            .style.format({"5-band hit rate":"{:.1%}","5°F hit rate":"{:.1%}","MAE (°F)":"{:.2f}",
                           "RMSE (°F)":"{:.2f}","Bias (°F)":"{:+.2f}","5-band log loss":"{:.3f}",
                           "5°F log loss":"{:.3f}"}),hide_index=True)
