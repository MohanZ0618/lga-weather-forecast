"""Read-only, as-of replay of the audited Polymarket temperature forecasts."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parent / "final_project"
MARKET = PROJECT / "results" / "rolling" / "polymarket"
BRACKET = PROJECT / "results" / "bracket_study"
ZONE = "America/New_York"
RULES = {"wunderground": "Weather Underground · KLGA", "noaa_timeseries": "NOAA · KLGA",
         "routine_metar": "Aviation Weather Center · KLGA routine METAR"}

STYLE = """
<style>
.replay-lead {display:flex;align-items:center;justify-content:space-between;gap:1rem;
 background:#123a41;color:#fff;border-radius:14px;padding:1.25rem 1.5rem;margin:.4rem 0 1rem}
.replay-lead small {color:#c1e5e2;font-size:.88rem}
.replay-lead strong {display:block;font-size:clamp(1.7rem,4vw,2.7rem);font-weight:650;line-height:1.25}
.replay-lead .chance {text-align:right;font-variant-numeric:tabular-nums}
.replay-bars {display:grid;gap:.52rem;margin:.3rem 0 1rem}
.replay-row {display:grid;grid-template-columns:minmax(6.2rem,8.7rem) 1fr 4.5rem;
 align-items:center;gap:.7rem;font-size:.94rem}
.replay-track {height:.75rem;border-radius:3px;background:rgba(120,140,150,.19)}
.replay-fill {height:100%;border-radius:3px;background:#819ba9}
.replay-top {background:#148175}
.replay-pct {text-align:right;font-variant-numeric:tabular-nums}
@media(max-width:600px){.replay-row{grid-template-columns:6.2rem 1fr 4rem;gap:.4rem}}
</style>"""


@st.cache_data(show_spinner=False)
def read_csv(path, stamp):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_degrees(path, stamp):
    return pd.read_parquet(path, columns=[f"p_{d}F" for d in range(-110, 151)])


@st.cache_data(show_spinner=False)
def read_weather(path, stamp):
    return pd.read_parquet(path, columns=["date", "hour", "latest_high", "cycle_available",
                                          "cycle_run", "model_cloud", "cloud_available", "sat10", "goes_available"])


def csv(name):
    path = MARKET / name
    return read_csv(path, path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def selected_bins(main_path, tail_path, metadata_path, stamps):
    main = pd.read_csv(main_path).query("delay == 0")[["date","hour","market_id","p"]]
    tail = pd.read_csv(tail_path).assign(hour=9)
    probs = pd.concat([main,tail],ignore_index=True)
    meta = pd.read_parquet(metadata_path,columns=["date","market_id","slug","bracket","regime","low","high","winner"])
    meta["date"] = pd.to_datetime(meta.date).dt.strftime("%Y-%m-%d")
    meta["market_id"] = meta.market_id.astype(str)
    probs["market_id"] = probs.market_id.astype(str)
    bins = probs.merge(meta.rename(columns={"slug":"event"}),on=["date","market_id"],how="left",validate="many_to_one")
    if bins.bracket.isna().any() or bins.duplicated(["date","hour","market_id"]).any():
        raise ValueError("Inconsistent archived forecast and market IDs")
    return bins


def overview(bins):
    picks = bins.loc[bins.groupby(["date","hour"]).p.idxmax()]
    common = picks.pivot(index="date",columns="hour",values="winner").reindex(columns=[9,12,15]).dropna()
    if common.empty:
        return
    hits = int(common.to_numpy(bool).sum())
    st.subheader("How often did the top bracket win?")
    st.write(f"Across {len(common)} days with forecasts at all three times, the top pick matched the official settlement in "
             f"**{hits:,}/{common.size:,} ({hits/common.size:.1%})** forecasts.")
    for col,hour in zip(st.columns(3),common.columns):
        col.metric(f"{hour:02}:00 forecast",f"{common[hour].mean():.1%}")
    st.caption("All three forecasts predict the same daily high. For a morning-only decision, use the 09:00 score.")


def local_stamp(value):
    return pd.Timestamp(value).tz_convert(ZONE).strftime("%m/%d %H:%M %Z")


def degree_terms(row, bracket):
    """Select all integer degrees in a listed bracket, including its open tail."""
    probs = row.rename(index=lambda col: int(col[2:-1]))
    left = probs.index.min() if pd.isna(bracket.low) else bracket.low
    right = probs.index.max() if pd.isna(bracket.high) else bracket.high
    return probs.loc[(probs.index >= left) & (probs.index <= right)]


def percent(value):
    return "<0.1%" if 0 < value < .001 else f"{value:.1%}"


def bars(frame, top):
    rows = []
    for row in frame.itertuples():
        mark = " replay-top" if row.bracket == top else ""
        rows.append(f'<div class="replay-row"><span>{escape(row.bracket)}</span>'
                    f'<div class="replay-track"><div class="replay-fill{mark}" '
                    f'style="width:{row.p * 100:.5f}%"></div></div>'
                    f'<span class="replay-pct">{percent(row.p)}</span></div>')
    st.html('<div class="replay-bars">' + ''.join(rows) + '</div>')


def market_replay():
    st.html(STYLE)
    st.title("LaGuardia daily high replay")
    st.write("Choose a New York date and forecast time to see which listed temperature bracket was most likely to win.")
    st.caption("Historical backtest · Training data through August 31, 2025 · Settlement hidden until revealed")
    selected_files = (BRACKET / "bracket_probabilities.csv.gz",BRACKET / "morning" / "reference_tail.csv.gz",
                      MARKET / "contracts.parquet")
    versions = []
    if all(p.exists() for p in selected_files):
        versions.append("Selected models by forecast time")
    if (MARKET / "contract_probabilities.csv.gz").exists():
        versions.append("Earlier model archive")
    if not versions:
        st.info("Market replay data is missing. See final_project/bracket_study/README.md for reproduction steps.")
        return
    version = st.radio("Model version",versions,horizontal=True)
    selected = version=="Selected models by forecast time"
    all_bins = (selected_bins(*selected_files,tuple(p.stat().st_mtime_ns for p in selected_files))
                if selected else csv("contract_probabilities.csv.gz"))
    overview(all_bins)
    dates = sorted(all_bins.date.unique(), reverse=True)
    date_col, hour_col = st.columns([1, 1.5])
    full_dates = all_bins.groupby("date").hour.nunique()
    complete = full_dates.index[full_dates.eq(3)]
    index = dates.index(complete.max()) if selected and len(complete) else 0
    day = date_col.selectbox("Market date", dates,index=index,key=f"date_{version}")
    today = all_bins.loc[all_bins.date.eq(day)]
    hours = sorted(today.hour.unique().tolist())
    hour = hour_col.radio("Forecast time (New York)", hours, format_func=lambda h:f"{h:02}:00",
                          horizontal=True)
    view = today.loc[today.hour.eq(hour)].sort_values("low", na_position="first")
    pick = view.loc[view.p.idxmax()]
    st.html(f'<div class="replay-lead"><div><small>{day} · {hour:02}:00 top bracket</small>'
            f'<strong>{escape(pick.bracket)}</strong></div><div class="chance"><small>Model probability</small>'
            f'<strong>{pick.p:.1%}</strong></div></div>')
    reveal = st.toggle("Reveal official settlement", key=f"settlement_{day}", value=False)
    if reveal:
        winner = view.loc[view.winner].iloc[0]
        settled = csv("settlement_check.csv")
        closed = settled.loc[settled.date.eq(day), "closed"].iloc[0]
        verdict = "correct" if pick.bracket == winner.bracket else "incorrect"
        st.info(f"The winning bracket was **{winner.bracket}**. The {hour:02}:00 top pick was {verdict}. "
                f"The model gave the winner **{winner.p:.1%}**. The winning contract closed at {local_stamp(closed)}.")
        st.markdown(f"[View the contract and settlement rules](https://polymarket.com/event/{pick.event})")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Probability by bracket")
        bars(view, pick.bracket)
        st.caption(f"The market listed {len(view)} brackets and settled using {RULES.get(pick.regime, pick.regime)}.")
        if selected:
            st.caption("The model estimates the probability of exceeding each temperature threshold. The difference between adjacent thresholds gives the probability of their bracket.")
        else:
            degree_details(day,hour,view,pick)

    with right:
        weather_and_history(day,hour,today)

    with st.expander("How were the forecasts reconstructed?"):
        st.write("Training used data through August 31, 2025. The replay reconstructs each forecast from data available by its issue time. "
                 "The airport archive uses observation times as a proxy for receipt times. Weather model inputs use archived publication times.")
        if selected:
            st.write("The 09:00 forecast averages threshold logistic regression and a two-stage warming model. Noon and 15:00 each use a separately trained monotonic XGBoost model. "
                     "They use four airports, HRRR and NBM forecasts. When a weather product is missing, logistic regression uses the available inputs. "
                     "The November 7–21, 2026 extension covers 09:00 only.")
        else:
            st.write("The earlier archive uses gradient-boosted trees at 09:00 and ridge regression at noon and 15:00. "
                     "Historical errors give a probability for each whole degree Fahrenheit, which is summed within each listed bracket.")
    if not selected:
        legacy_metrics()


def degree_details(day,hour,view,pick):
    with st.expander("How is a bracket probability calculated?"):
        names = view.bracket.tolist()
        name = st.selectbox("Show probabilities by whole degree", names, index=names.index(pick.bracket),
                            key=f"bin_{day}_{hour}")
        bracket = view.loc[view.bracket.eq(name)].iloc[0]
        path = MARKET / f"one_degree_{hour:02}.parquet"
        terms = degree_terms(read_degrees(path, path.stat().st_mtime_ns).loc[pd.Timestamp(day)], bracket)
        if len(terms) <= 4:
            st.write(" + ".join(f"{d}°F ({v:.2%})" for d,v in terms.items()) + f" = **{terms.sum():.2%}**")
        else:
            st.write(f"The probabilities of all whole degrees in {name} add up to **{terms.sum():.2%}**.")
            st.dataframe(pd.DataFrame({"Whole degree (°F)":terms.index,"Probability":[f"{v:.4%}" for v in terms]}),
                         hide_index=True, width="stretch")


def weather_and_history(day,hour,today):
    st.subheader("Weather available at forecast time")
    path = PROJECT / "data" / "rolling" / "external.parquet"
    if path.exists():
        weather = read_weather(path, path.stat().st_mtime_ns)
        snapshot = weather.loc[weather.date.eq(day) & weather.hour.eq(hour)]
        if len(snapshot):
            row = snapshot.iloc[0]
            if pd.notna(row.latest_high):
                st.metric("HRRR forecast high for the rest of the day", f"{row.latest_high:.1f} °F")
                st.caption(f"Model run {local_stamp(row.cycle_run)} · Published {local_stamp(row.cycle_available)}")
            else:
                st.caption("No recent HRRR run was available at this time.")
            if pd.notna(row.sat10):
                st.caption(f"Cloud cover was {row.sat10:.0%} within 10 km of the airport · Published {local_stamp(row.goes_available)}")
    st.markdown("**Forecasts available by this time**")
    history = today.loc[today.hour.le(hour)]
    best = history.loc[history.groupby("hour").p.idxmax()].sort_values("hour")
    st.dataframe(pd.DataFrame({"New York time":best.hour.map(lambda h:f"{h:02}:00"),"Top bracket":best.bracket,
                               "Probability":best.p.map(percent)}),hide_index=True,width="stretch")


def legacy_metrics():
    with st.expander("How did the earlier models perform at each forecast time?"):
        stats = csv("market_metrics.csv")
        stats = stats.loc[stats.period.eq("all") & stats.model.eq("selected")].sort_values("hour")
        st.dataframe(pd.DataFrame({"Forecast time":stats.hour.map(lambda h:f"{h:02}:00"),"Market days":stats.days,
                                   "Correct picks":stats.correct,"Accuracy":stats.accuracy.map(percent),
                                   "Log loss":stats.log_loss.round(3)}),hide_index=True,width="stretch")
