"""Build the Chinese study report directly from frozen per-date scores."""
import json
from importlib.metadata import version

import pandas as pd

from panels import OUT,TABLE,contracts

FAMILY = {"threshold":"共享阈值逻辑回归","monotone_xgb":"单调 XGBoost","gaussian_cat":"CatBoost 温度分布回归",
          "hurdle":"两阶段升温模型","blend":"阈值逻辑回归＋两阶段升温（各 50%）"}
PERIOD = {"historical":"主回放","autumn":"秋季回放"}


def table(columns,rows):
    lines = ["| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
    lines.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    return "\n".join(lines)


def main():
    summary = pd.read_csv(OUT/"market_metrics.csv").set_index(["hour","period","delay","model"])
    daily = pd.read_csv(OUT/"daily.csv.gz",parse_dates=["date"])
    paired = pd.read_csv(OUT/"paired.csv").set_index(["hour","period","delay","comparison"])
    course = pd.read_csv(OUT/"course.csv").set_index(["hour","period","delay","model","subset"])
    bins = pd.read_csv(OUT/"bracket_probabilities.csv.gz",parse_dates=["date"])
    detail = pd.read_csv(OUT/"subgroups.csv")
    nbm = pd.read_parquet(TABLE)
    parts = ["# LGA 当日最高温能预测到哪个挂牌区间？",
             "纽约当地 09:00、12:00、15:00 分别预测当天 Polymarket 已挂牌的各个温度区间。每天用真实结算的唯一赢家计算分数。课程五档仍有助于衡量大范围气温变化，真实合约则检验模型能否区分相邻几度。"]
    parts.append("## 历史结果")
    parts.append("旧模型是项目上一轮按真实合约选出的冠军。09:00 用 15 叶梯度提升回归修正天气预报，12:00 和 15:00 用岭回归修正，再把校准误差分布加总到挂牌区间。新旧模型的最终拟合都截至 2025-08-31。")
    rows = []
    for period in PERIOD:
        for h in (9,12,15):
            key = (h,period,0)
            new = summary.loc[(*key,"new")]
            old = summary.loc[(*key,"incumbent")]
            ci = paired.loc[(*key,"overall")]
            rows.append([PERIOD[period],f"{h:02}:00",int(new.days),f"{old.accuracy:.1%} → {new.accuracy:.1%}",
                         f"{old.log_loss:.3f} → {new.log_loss:.3f}",
                         f"{ci.delta_loss:+.3f} [{ci.low:+.3f}, {ci.high:+.3f}]",
                         f"{old.brier:.3f} → {new.brier:.3f}"])
    parts.append(table(["区间","发报","共同日期","旧 → 新命中率","旧 → 新 log loss","新减旧（95% 周区间）","旧 → 新 Brier"],rows))
    parts.append("按预先冻结的配置，三个时点在两段回放中的平均命中率、log loss 和 Brier 都优于既有冠军。上午的配对损失区间在两段回放中都低于零。中午在秋季回放中达到这一条件。下午的两段区间都跨零，需更多独立日期来判断损失优势能否复现。")
    parts.append("主回放为 2025-09-01 至 2026-08-31，秋季为 2026-09-01 至 2026-11-06。两段分开配对。log loss 衡量模型给实际赢家的概率，数值越小越好。给错区间过高置信度会被重罚。Brier 同时衡量全部区间的概率偏差。周区间来自 5,000 次配对七日分块重采样，差值区间跨零表示改善方向尚不确定。这些日期在此前的项目研究中已经使用过。")
    parts.append("![实际挂牌区间 log loss 对比](market_log_loss.png)")
    parts.append("## 检验季节是否已经足以猜对")
    rows = []
    for h in (9,12,15):
        key = (h,"historical",0)
        def cell(model):
            r = summary.loc[(*key,model)]
            return f"{r.accuracy:.1%} / {r.log_loss:.3f}"
        rows.append([f"{h:02}:00",f"{course.loc[(*key,'season_only','same_market_dates')].accuracy:.1%}",
                     cell("season_only"),cell("listings_only"),cell("raw_HRRR"),cell("new")])
    parts.append(table(["发报","季节猜课程五档：命中率","季节猜实际档：命中率 / 损失",
        "仅知挂牌：命中率 / 损失","HRRR 基准：命中率 / 损失","新模型：命中率 / 损失"],rows))
    parts.append("季节基准只用训练期历史温度，按日历日的周期距离加权，再将整度概率加到当天挂牌区间。挂牌基准只学习过去已结算赢家在挂牌列表中的相对位置。HRRR 基准使用已有数值预报、观测下界及历史残差校正。三者都在新模型的相同市场日期评分。")
    rows = []
    for h in (9,12,15):
        g = detail.loc[(detail.hour==h)&detail.subset.eq("season")&detail.model.eq("new")].set_index("value")
        rows.append([f"{h:02}:00"]+[f"{g.loc[s].accuracy:.1%} / {g.loc[s].log_loss:.3f}（{int(g.loc[s].days)} 天）"
                                    for s in ("winter","spring","summer","autumn")])
    parts.append(table(["季节分组","冬季","春季","夏季","秋季"],rows))
    parts.append("上表合并两段回放，按当地气象季节分组。各季样本量和挂牌难度不同。以下例子取主回放中冬季和夏季各自第一个有 09:00 合约的日期。")
    main = bins.loc[bins.delay.eq(0)&bins.hour.eq(9)&bins.date.between("2025-09-01","2026-08-31")]
    for months in ([12,1,2],[6,7,8]):
        subset = main.loc[main.date.dt.month.isin(months)]
        date = subset.date.min()
        case = subset.loc[subset.date.eq(date)]
        top = "、".join(f"**{r.bracket}** {r.p:.1%}" for r in case.nlargest(3,"p").itertuples())
        win = case.loc[case.winner].iloc[0]
        parts.append(f"- {date:%Y-%m-%d} 共 {len(case)} 档，两端为 {case.iloc[0].bracket} 和 {case.iloc[-1].bracket}。前三档为 {top}。最终结算 **{win.bracket}**，模型给它 {win.p:.1%}。")
    parts.append("## 为什么选这些模型，参数怎样决定")
    parts.append("项目存档中的实际合约始于 2025-01-22。将 2025 年 1–2 月、3–4 月设为两个前向初选折，比较 5 种固定模型 × 3 套输入，再锁定每个时点的一套方案。随后在 5–6 月、7–8 月两个前向折，只对入选模型搜索参数。每折仅用更早的天气标签拟合，仅按折内已结算的真实合约评价，每折平均损失等权。模型初选差距不超过 0.001 时优先选择来源更少、结构更简单的方案。")
    parts.append("候选包括共享阈值逻辑回归、对相邻温度阈值施加单调约束的 XGBoost、学习预报误差均值和不确定性的 CatBoost、分别学习继续升温概率与正升温幅度的两阶段模型，以及阈值回归和两阶段模型的等权组合。每种都比较四机场、四机场＋HRRR、四机场＋HRRR＋NBM 三套输入。")
    rows = []
    for h in (9,12,15):
        folder = OUT/f"{h:02}"
        locked = json.loads((folder/"parameters.json").read_text())
        screen = pd.read_csv(folder/"screen.csv")
        rank = screen.groupby(["family","tier"]).log_loss.mean().sort_values()
        family,tier = locked["family"],locked["tier"]
        rival = rank.drop((family,tier)).index[0]
        tune = pd.read_csv(folder/"tune.csv")
        params = locked["params"]
        def desc():
            terms=[]
            if family in ("blend","threshold","hurdle"):
                terms.append(f"C={params['C']:.4g}")
            if family!="threshold":
                terms += [f"深度 {params['depth']}",f"学习率 {params['rate']:.4g}",
                          f"{params['trees']} 棵树",f"L2={params['l2']:.4g}"]
            return "，".join(terms)
        rows.append([f"{h:02}:00",FAMILY[family]+" / "+tier.upper(),f"{rank.loc[(family,tier)]:.4f}",
                     FAMILY[rival[0]]+" / "+rival[1].upper()+f"（{rank.loc[rival]:.4f}）",
                     desc(),f"{tune.loc[tune.params.eq(tune.params.iloc[0])].log_loss.mean():.4f} → {locked['validation_log_loss']:.4f}"])
    parts.append(table(["发报","选中模型 / 数据","初选损失","次优固定方案（损失）","最终参数","调参折默认 → 最优损失"],rows))
    parts.append("上午的组合同时估计最高温超过指定阈值的概率，以及从目前高点继续升温的幅度。两种预测等权平均后赢得初选，领先单独阈值回归的幅度很小。中午的单调树在固定参数下对次优方案的领先更明显。下午单调树与温度分布回归的初选成绩接近，按事先确定的损失规则选中了前者。")
    parts.append("共享阈值回归把不同分界线上的样本合在一起，C 越小，系数收缩越强。两阶段模型先用逻辑回归估计还会继续升温的概率，再由 CatBoost 拟合正升温量对数的条件分布。组合中两个逻辑组件共用 C，两个成员各占 50%。单调 XGBoost 利用温度相对阈值的距离，同时学习风、云和预报误差的非线性关系。树深控制交互复杂度，学习率控制每轮更新幅度，树数控制总迭代，L2 抑制叶子过拟合。")
    parts.append("单调约束保证在其余天气输入相同时，跨过更高温度门槛的概率只能下降或持平。CatBoost 的分布回归同时输出位置和尺度参数，允许同样的预期升温量具有不同的不确定性。[XGBoost 单调约束文档](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html) · [CatBoost 回归目标文档](https://catboost.ai/docs/en/concepts/loss-functions-regression)")
    parts.append("树模型的每个入选方案保留初选参数，再以种子 4260 抽取 64 组。深度 2–6、学习率 0.015–0.12、迭代数 150–1,000、L2 1–30。涉及逻辑回归时 C 为 0.01–100。深度和树数按整数抽样，连续参数按对数尺度抽样。调参联合改变多个参数，因此只能把收益归给整组参数。初选及调参逐折结果、拟合日期、覆盖和运行秒数保存在各时点的 CSV 中。")
    parts.append("## 改进来自数据、结构，还是调参")
    rows = []
    for period in PERIOD:
        for h in (9,12,15):
            def d(name):
                a=paired.loc[(h,period,0,name)]
                return f"{a.delta_loss:+.3f}"
            rows.append([PERIOD[period],f"{h:02}:00",d("add_nbm_routed"),d("add_nbm_matched"),
                         d("change_family"),d("tuning")])
    parts.append(table(["回放","发报","阈值回归加 NBM：全日期","加 NBM：相同训练与评估日期",
                        "相同输入改模型结构","同结构调参"],rows))
    parts.append("最一致的收益来自第二套气象预报。固定阈值回归并匹配训练和评估日期后，主回放中三个时点加入 NBM 的损失差值区间都低于零。深搜改善了三个时点的调参折评分，但主回放中上午和中午使用调后参数的损失更高。下午的微弱调参收益也没有在秋季重现。报告保留预先冻结的配置，以免看过回放后再挑参数。")
    parts.append("每格为后一种方法减前一种方法的 log loss，负数表示改善。NBM 消融固定阈值回归。结构消融固定入选输入与默认参数。调参消融固定结构与输入。三项对照各自回答不同问题。完整样本量和配对区间见 [paired.csv](paired.csv)。")
    rows=[]
    for h in (9,12,15):
        key=(h,"historical",0)
        new = course.loc[(*key,"new","all")]
        near = course.loc[(*key,"new","near_boundary")]
        old = course.loc[(*key,"incumbent","all")]
        rows.append([f"{h:02}:00",f"{old.accuracy:.1%} → {new.accuracy:.1%}",f"{old.log_loss:.3f} → {new.log_loss:.3f}",
                     f"{near.log_loss:.3f}（{int(near.n)} 天）",f"{new.macro_f1:.3f}",
                     " / ".join(f"{summary.loc[(h,'historical',delay,'new')].log_loss:.3f}" for delay in (0,15,30))])
    parts.append(table(["发报","课程五档旧 → 新命中率","课程五档旧 → 新损失","距课程边界 ≤2°F 损失",
                        "课程五档 Macro F1","市场损失：延迟 0 / 15 / 30 分钟"],rows))
    rise = detail.loc[(detail.hour==15)&detail.model.eq("new")&detail.subset.eq("rise")].set_index("value")
    stopped,warmer = rise.loc["at_floor"],rise.loc["rose"]
    parts.append(f"下午的主要难点是后续升温。两段回放合计，15:00 的已观测整度下界达到最终温度代理时，命中率为 {stopped.accuracy:.1%}（{int(stopped.days)} 天）。当最高温此后继续上升时降至 {warmer.accuracy:.1%}（{int(warmer.days)} 天）。观测延迟试验也显示下午的概率对最新读数尤其敏感。该分组用事后温度解释误差，预测时并不知道最高温是否已经出现。")
    parts.append("课程五档另以相同选中结构重训，目标为保留小时观测的当日最高温。课程档位的硬下界仅由已结束小时确定。市场评分则根据当日合约的测温规则选择整数温度训练目标，区间端点按最近整数华氏度处理，并在挂牌区间间均分 0.1% 的概率，以容纳温度代理与结算来源的差别。挂牌数及定义随日期变化，市场主指标为对真实赢家的 log loss、Brier 和命中率。")
    coverage=nbm.groupby("hour").status.agg(lambda x:f"{x.eq('ok').sum()}/{len(x)}")
    parts.append("## 数据覆盖与复现")
    parts.append(f"NOAA NBM 档案 {nbm.date.min()} 至 {nbm.date.max()}。三个发报时点可用数依次为 {', '.join(coverage)}。在截止前已发布且能覆盖当地当日最后一个整点的 KLGA 小时预报中，提取剩余最高温、峰值时距、峰值温度标准差和平均云量，并计算 NBM 与 HRRR 的分歧。站点卡包含不同 NBM 版本。富数据缺失时按 NBM、HRRR、仅机场顺序回退。较低层使用固定阈值逻辑回归。主评估限于四机场数据齐全的研究时间范围，复用既有模型的相同市场日期。")
    parts.append("历史站点报告时间代理实际接收时间。15/30 分钟敏感性试验将站点输入延后，延迟情况下 routine METAR 下界用早一个已结束小时的 LCD 常规报告重建。历史市场档案给出挂牌、关盘和最终规则，缺少逐时改版记录。Weather Underground 等结算来源的逐度天气标签是代理，真实挂牌评分始终以最终结算为准。这个实验评估天气区间概率。")
    libs = ", ".join(f"{k} {version(k)}" for k in ["numpy","pandas","scikit-learn","xgboost","catboost","scipy"])
    parts.append("训练使用 "+libs+"。 [命令与数据源说明](../../bracket_study/README.zh-CN.md) · [逐日合约概率](bracket_probabilities.csv.gz) · [分季节、结算规则和升温状态](subgroups.csv) · [输入哈希](inputs.json)")
    (OUT/"REPORT.zh-CN.md").write_text("\n\n".join(parts)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
