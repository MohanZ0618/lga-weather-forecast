# 拉瓜迪亚机场当日最高温预测

[English](README.md) · [简体中文](README.zh-CN.md)

CS4260 项目，作者为 Mohan Zhang、Mingyeung Wu 和 Maxwell Liss。

默认页面 **Polymarket replay** 回放纽约时间 09:00、12:00 和 15:00 的预测。三次都回答同一个问题：拉瓜迪亚机场（LGA）当天的最高温会落入哪个实际挂牌的温度区间？选择交易日和预测时点，可以查看各档概率，再揭示官方结算。回放使用截至该时点可用的历史气象资料。

## 概率最高的一档有多准？

在 2025 年 9 月 3 日至 2026 年 11 月 6 日之间，有 413 个交易日同时具备三次预测。每次只选概率最高的一档，与官方结算赢家一致就记作命中。

| 纽约当地预测时点 | 命中 / 预测次数 | 命中率 |
| --- | ---: | ---: |
| 09:00 | 202 / 413 | 48.9% |
| 12:00 | 241 / 413 | 58.4% |
| 15:00 | 303 / 413 | 73.4% |
| 三次等权合计 | 746 / 1,239 | 60.2% |

如果每天只在早晨做一次决定，参考 48.9%。60.2% 包含中午和下午利用新观测发布的两次更新。早晨保留阈值逻辑回归和两阶段升温模型的等权组合，中午与下午各自训练单调 XGBoost。选定模型使用截至 2025 年 8 月 31 日的天气标签拟合。这里报告的是历史回放成绩。[模型选择、超参数和对照结果](final_project/results/bracket_study/old09/REPORT.zh-CN.md) · [复现实验](final_project/bracket_study/README.zh-CN.md)

## 其他预测页面

- **Forecast comparison** 展示同一目标日期的前日中午预测、当日更新、实际温度和误差，可切换 5°F 区间或课程原来的五档。
- **Latest observations** 在纽约当地 06:00 至 20:00 以 15 分钟为间隔，结合四座机场的 METAR、HRRR 数值预报和 GOES-East 云量数据。页面打开时会刷新，并显示观测时间、产品发布时间和使用的输入。
- **Fusion backtest** 回放上述系统的样本外预测。**Historical replay** 展示课程原三时点模型的概率分布和观测最高温。[短时预报运行指南（英文）](nowcast/README.md)

## 本地运行

使用 Python 3.13，在仓库根目录的 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS 或 Linux 可用 `source .venv/bin/activate` 激活环境。如果 PowerShell 不允许激活，可用 `.\.venv\Scripts\python.exe` 替代 `python`。历史回放直接读取仓库附带的概率表和市场档案。运行 `python -m unittest discover -s tests -v` 可检查应用。

## 原课程五档实验

最初的提案研究次日最高温，课程实现预测 LGA **当天逐小时观测的最高温**，按华氏度分成低于 40、40 至低于 55、55 至低于 70、70 至低于 85、85 及以上五档。逐小时最高温与正式日最高温或合约结算值可能不同。

- 数据来自 NOAA Global Hourly 的 LGA、JFK、EWR 和 TEB 站。程序筛选 FM-15、FM-16 报告并检查质量标记，每个 UTC 整点小时保留最后一条有效报告。[NOAA 数据说明](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) · [ISD 格式说明](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf)
- 目标日期至少有 20 个有效整点小时，并覆盖当地四个六小时段。原实验使用 2023 年训练、2024 年验证、2025 年可用日期测试。输入包含月份、各站最近的温湿度和风、已观测最高温、约三小时温差和前一日最高温。
- 在单站和四站输入上比较季节基线、逻辑回归、随机森林和 XGBoost。预处理和模型仅拟合 2023 年数据，各预测时点按 2024 年验证集的 log loss 选择模型。
- 依次运行 `python weather.py` 和 `python reporting.py` 可重建原课程实验。首次训练下载 12 个 NOAA 文件，后续复用缓存。`python weather.py --refresh` 可刷新原始数据，然后重新运行报告脚本。`data/raw/manifest.json` 记录源地址及 SHA-256。

## 文件索引

| 路径 | 内容 |
| --- | --- |
| `app.py` 与 `market_replay.py` | Streamlit 导航、市场回放和结算展示 |
| `final_project/bracket_study/` 与 `final_project/results/bracket_study/` | 挂牌区间模型、训练代码、已拟合模型、概率、评分和报告 |
| `weather.py` 与 `reporting.py` | 原课程实验的数据清洗、训练、评估和报告生成 |
| `tests/` | 输入时间截断、夏令时、标签覆盖、概率及回放交互测试 |
| `data/raw/` 与 `data/processed/` | 原始下载清单和本地缓存、逐小时数据、有效标签和分时样本 |
| `outputs/metrics.csv`、`outputs/selected_models.json` 与 `outputs/model_*.joblib` | 原实验 48 行验证与测试指标、模型选择和权重 |
| `outputs/selected_test_predictions.csv`、`outputs/prediction_errors.csv` 与 `outputs/noon_lga_results.csv` | 逐次概率、误判和 LGA 中午结果 |
| `outputs/*.png` 与 `outputs/Progress1_Report.*` | 类别分布、测试损失图以及 Markdown、Word 阶段报告 |
