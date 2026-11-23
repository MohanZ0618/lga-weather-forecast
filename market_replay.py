"""Read-only, as-of replay of the audited Polymarket temperature forecasts."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parent / "final_project"
MARKET = PROJECT / "results" / "rolling" / "polymarket"
ZONE = "America/New_York"
RULES = {"wunderground": "Weather Underground · KLGA", "noaa_timeseries": "NOAA · KLGA",
         "routine_metar": "Aviation Weather Center · KLGA 常规 METAR"}

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
    st.title("LaGuardia 最高温回放")
    st.write("用纽约当天 09:00、12:00 或 15:00 之前的气象资料，回看模型怎样预测当天最高温。")
    st.caption("历史回测 · 训练数据截至 2025 年 8 月 31 日 · 结算结果默认隐藏")
    if not (MARKET / "contract_probabilities.csv.gz").exists():
        st.info("本地缺少市场回放结果。生成步骤见 final_project/rolling/README.md。")
        return
    all_bins = csv("contract_probabilities.csv.gz")
    dates = sorted(all_bins.date.unique(), reverse=True)
    date_col, hour_col = st.columns([1, 1.5])
    day = date_col.selectbox("历史交易日", dates)
    today = all_bins.loc[all_bins.date.eq(day)]
    hours = sorted(today.hour.unique().tolist())
    hour = hour_col.radio("预测时点 · 纽约时间", hours, format_func=lambda h:f"{h:02}:00",
                          horizontal=True)
    view = today.loc[today.hour.eq(hour)].sort_values("low", na_position="first")
    pick = view.loc[view.p.idxmax()]
    st.html(f'<div class="replay-lead"><div><small>{day} · {hour:02}:00 预测的首选温区</small>'
            f'<strong>{escape(pick.bracket)}</strong></div><div class="chance"><small>模型概率</small>'
            f'<strong>{pick.p:.1%}</strong></div></div>')
    reveal = st.toggle("揭示当日官方结算", key=f"settlement_{day}", value=False)
    if reveal:
        winner = view.loc[view.winner].iloc[0]
        settled = csv("settlement_check.csv")
        closed = settled.loc[settled.date.eq(day), "closed"].iloc[0]
        verdict = "命中" if pick.bracket == winner.bracket else "未命中"
        st.info(f"官方结算 **{winner.bracket}**　·　{hour:02}:00 首选{verdict}　·　"
                f"该时点给结算档 **{winner.p:.1%}**。赢家合约关闭于 {local_stamp(closed)}。")
        st.markdown(f"[查看当天合约和结算规则](https://polymarket.com/event/{pick.event})")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("各温区概率")
        bars(view, pick.bracket)
        st.caption(f"使用当日挂牌的 {len(view)} 个温区。结算来源为 {RULES.get(pick.regime, pick.regime)}。")
        with st.expander("一个温区的概率怎么算？"):
            names = view.bracket.tolist()
            name = st.selectbox("拆开查看 1°F 概率", names, index=names.index(pick.bracket),
                                key=f"bin_{day}_{hour}")
            bracket = view.loc[view.bracket.eq(name)].iloc[0]
            path = MARKET / f"one_degree_{hour:02}.parquet"
            terms = degree_terms(read_degrees(path, path.stat().st_mtime_ns).loc[pd.Timestamp(day)], bracket)
            if len(terms) <= 4:
                st.write(" + ".join(f"{d}°F（{v:.2%}）" for d,v in terms.items()) + f" = **{terms.sum():.2%}**")
            else:
                st.write(f"将 {name} 内每个整数温度的概率相加，总计 **{terms.sum():.2%}**。")
                st.dataframe(pd.DataFrame({"整数 °F":terms.index,"概率":[f"{v:.4%}" for v in terms]}),
                             hide_index=True, width="stretch")

    with right:
        st.subheader("当时有哪些信息？")
        path = PROJECT / "data" / "rolling" / "external.parquet"
        if path.exists():
            weather = read_weather(path, path.stat().st_mtime_ns)
            snapshot = weather.loc[weather.date.eq(day) & weather.hour.eq(hour)]
            if len(snapshot):
                row = snapshot.iloc[0]
                if pd.notna(row.latest_high):
                    st.metric("HRRR 预报的当天剩余最高温", f"{row.latest_high:.1f} °F")
                    st.caption(f"模式起报 {local_stamp(row.cycle_run)} · 档案可用 {local_stamp(row.cycle_available)}")
                else:
                    st.caption("该时点无可用的最新 HRRR 周期。模型可使用固定周期或机场观测。")
                if pd.notna(row.sat10):
                    st.caption(f"机场周围 10 公里卫星云量 {row.sat10:.0%} · 档案可用 {local_stamp(row.goes_available)}")
        st.markdown("**截至这个时点的预测**")
        history = today.loc[today.hour.le(hour)]
        best = history.loc[history.groupby("hour").p.idxmax()].sort_values("hour")
        st.dataframe(pd.DataFrame({"纽约时间":best.hour.map(lambda h:f"{h:02}:00"),"首选温区":best.bracket,
                                   "概率":best.p.map(percent)}),hide_index=True,width="stretch")

    with st.expander("这是提前预测吗？模型如何得到这些数字？"):
        st.write("这是同日历史回测。模型使用截至 2025 年 8 月 31 日的数据训练，再按每个交易日的时点重建预测。"
                 "没有当日实时发布凭证。机场历史记录用观测时间近似接收时间；HRRR 和卫星使用归档可用时间。")
        st.write("模型先看已观测最高温与最新可用天气预报，再用机场温度、湿度、风、升温趋势和云量估计偏差。"
                 "09:00 使用梯度提升树，12:00 和 15:00 使用 Ridge 回归。"
                 "根据历史预测误差为每个整数 °F 分配概率，再按当天挂牌区间相加。最终赢家用于事后评分。")
    with st.expander("各时点在历史市场上表现如何？"):
        stats = csv("market_metrics.csv")
        stats = stats.loc[stats.period.eq("all") & stats.model.eq("selected")].sort_values("hour")
        st.dataframe(pd.DataFrame({"时点":stats.hour.map(lambda h:f"{h:02}:00"),"交易日":stats.days,
                                   "命中天数":stats.correct,"命中率":stats.accuracy.map(percent),
                                   "Log loss":stats.log_loss.round(3)}),hide_index=True,width="stretch")
