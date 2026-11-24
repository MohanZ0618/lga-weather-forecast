# LGA 真实挂牌温度区间实验

[English](README.md) · [简体中文](README.zh-CN.md)

从 `weather-project` 目录在 Windows PowerShell 运行。建议使用 Python 3.13。首次安装依赖后，训练和评估从随附的紧凑快照读取，无需网络。

```powershell
.\.venv\Scripts\python.exe -m pip install -r final_project/bracket_study/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s final_project -p 'test*.py' -v
.\.venv\Scripts\python.exe final_project/bracket_study/train.py screen
.\.venv\Scripts\python.exe final_project/bracket_study/train.py tune
.\.venv\Scripts\python.exe final_project/bracket_study/train.py freeze
.\.venv\Scripts\python.exe final_project/bracket_study/replay.py
.\.venv\Scripts\python.exe final_project/bracket_study/report.py
```

`screen` 只读 2025 年 5 月以前的验证结算，分别锁定 09、12、15 点的模型和气象源。历史气象观测全部早于对应验证折，可以用于拟合整数温度。每个阶段只使用该折结束前已结算的合约。`tune` 只搜索锁定方案的超参数。`freeze` 重训到 2025-08-31。`train.py all` 依次完成这三步。单时点可使用 `--hour 9`、`--hour 12` 或 `--hour 15`。阶段之间用 `family.json` 与 `parameters.json` 记录选择。实验写入独立目录 `final_project/results/bracket_study`。

[中文结果报告](../results/bracket_study/REPORT.zh-CN.md) · [逐日概率](../results/bracket_study/bracket_probabilities.csv.gz) · [同日配对差值](../results/bracket_study/paired.csv)

## 上午 09:00 的首选命中率

单独针对首选档的实验以更早年份的天气训练模型，在 2025 年 1–4 月按实际合约命中率选择模型，再以 5–8 月确定参数。回放同时输出完全冻结和逐月更新两种概率。后者在每个月初留出七天时间间隔。评估脚本最后才将预测与实际赢家关联。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s final_project -p test_morning.py -v
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py select
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py predict
.\.venv\Scripts\python.exe final_project/bracket_study/morning.py score
```

选模与预测读取随附的 `final_project/data/morning_09.parquet`。从已有本机 NOAA、IEM 和 HRRR 原始缓存重建此表可运行 `.\.venv\Scripts\python.exe final_project/bracket_study/morning_data.py`。后续 15 天的旧模型概率保存在 `reference_tail.csv.gz`，可离线评分。 [上午预测实验报告](../results/bracket_study/morning/REPORT.zh-CN.md)

## 整体命中率与早晨模型选择

三个发报时点预测同一个交易日的最高温结算档。选定的分时模型在 413 个共同日期的 1,239 次预测中命中 746 次（60.2%）。每天只在 09:00 选一次时，命中 202/413（48.9%）。首页默认显示这一组模型。从项目根目录执行下列命令可复现对照和回放。

```powershell
.\.venv\Scripts\python.exe final_project/bracket_study/old09.py select
.\.venv\Scripts\python.exe final_project/bracket_study/old09.py evaluate
.\.venv\Scripts\python.exe final_project/bracket_study/old09.py forecast --date 2026-06-01
.\.venv\Scripts\python.exe final_project/bracket_study/time_study.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

早晨实验先在 2026-03-01 之前选定调整，再用其后的日期逐档复算并与原模型存档核对。共享模型实验只用 2025-09-01 以前的数据训练。其早中晚三个输入面板按各自的发报时点截断。[整体表现、新旧模型与调参结果](../results/bracket_study/old09/REPORT.zh-CN.md)

## 数据与复现

- NOAA LCD 四机场记录、按发布时间还原的 HRRR 输入、真实合约与官方结算来自已有滚动实验快照。市场按当天实际挂牌的全部区间评分。整数温度用于训练，历史 Weather Underground / NOAA 规则使用全部报告温度代理，routine METAR 规则使用其专用代理。市场结算结果直接决定赢家。每个预测时点只使用当时已挂牌且赢家尚未关闭的市场，档案缺少历史合约文字的版本记录。详见 [原始数据说明](../rolling/README.md)。
- 新增 NBM 为 NOAA 小时级 NBH 站点预报。优先尝试发报 UTC 时刻之前 1 小时的周期，再尝试之前 2、3 小时的周期。归档对象 Last-Modified 必须早于或等于发报时刻，且 KLGA 预报覆盖当地当天最后一个整点。北京时间与机器时区不参与计算。每个可用时点保存站点卡摘要、发布时间、预报周期、产品版本和 URL。重新抓取使用 `.\.venv\Scripts\python.exe final_project/bracket_study/nbm_archive.py`。该操作需要网络并复用本机 `final_project/build/nbm_cards` 缓存。归档修订可能改变未来重建的快照。[NOAA NBM 下载说明](https://vlab.noaa.gov/web/mdl/nbm-download)
- 输出概率 CSV 每行对应一个日期、发报时刻、观测延迟假设和市场合约，包含区间、市场 ID、预测概率和实际赢家。按日期、发报时刻和延迟分组，每组概率和为 1。`daily.csv.gz` 提供所有可比系统的命中、log loss 和 Brier。`course.csv` 单独记录课程原五档分数。`inputs.json` 记录紧凑输入的 SHA-256。
- 三个发报时点各自独立。每个时点评估范围截至 2026-11-06，这是四机场主快照最后一个完整有效日。后续旧研究中的 IEM 补充日期来自另一种观测归档。周区间按相同市场日期对齐，以七个日历日为块重采样 5,000 次。不同季节、结算来源和日期段的结果均为历史回放。
