"""读取本次固定结果生成中文验收报告；不联网、不调参。"""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def pct(value):
    return 'NA' if value is None or pd.isna(value) else f'{value:.2%}'


def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    anchors=json.loads((OUT/'backtest/anchors.json').read_text())
    costs=json.loads((OUT/'backtest/summary.json').read_text())
    benchmarks=json.loads((OUT/'backtest/benchmarks.json').read_text())
    events=pd.read_csv(OUT/'backtest/price_only_events_NOT_STRATEGY.csv')
    daily=json.loads((OUT/'daily/audit.json').read_text())
    notes={('CBRS','2026-06-29'):'量比达标，但IPO暖机不足、洗盘/支撑分支未通过；当天开盘买到的17.48%无法用当天收盘信号事先取得。',('CBRS','2026-09-03'):'量比仅0.62，不是1.5倍放量；90日波动暖机不足。',('CRWV','2026-09-04'):'量比0.77、实体阳线已连续3根；规则明确禁止。',('CRWV','2026-06-10'):'无首根放量确认、周波动门槛未过，拒绝第一刀。',('CRWV','2026-09-08'):'实体连续4阳、收涨/板块领涨连击、回撤已不足35%；禁止追高。',('OKLO','2026-09-08'):'回撤超过65%，连续收涨/板块领涨达到禁入值，身份与基本面也未完成。',('AVGO','2026-09-04'):'核仓负对照，明确排除；不混入本模块。'}
    lines=['# Babybus 洗盘反抽 V1：实现与验收报告','',
      '独立研究模块已接入，参数、因子、信号、风险预算、模拟撮合、回测框架和四张日报CSV均已生成。**投资有效性尚未验证，不能据此实盘启用。**',
      '',f'运行冻结时间：{manifest["as_of"]}；最近完整交易日：{manifest["complete_session"]}。21个行情序列获取成功，14只候选/负对照。参数哈希：`{manifest["config_sha256"]}`。',
      '', '## 1. 真正改进了什么','',
      '- 收盘决策与成交彻底分开：下一交易日执行，节假日按交易所日历处理。未完成的日/周bar不能进入信号，旧盘中快照在未来重跑时也不能伪装成完整收盘。',
      '- 高开6%–8%只接受信号收盘+2%的限价回踩，高开至少8%放弃；未成交委托占用当天预算，不根据当天低点是否成交去追认另一个开盘买单。',
      '- 硬止损、事件、信号中位失效、分级止盈和第三交易日退出进入统一账本；同bar止损优先，不能凭模糊日线路径获取理想止盈。',
      '- 限制总弹性仓15%、单票5%、最多3票、同主题2票、与既有持仓相关性0.75；0.75% NAV是风险上限，不是单票必须用满的风险。',
      '- 新增事件T−3禁入、跨策略冲突检查、低位支撑/缺口/POC代理、RS连续领涨禁追、三态缺失处理与来源时间审计。',
      '- 解决了容易夸大利润的资金时序错误：下午止盈或收盘卖出不能释放上午的现金、仓位槽位。',
      '', '## 2. 三个正锚点没有被硬拟合','',
      '三个正锚点均未触发严格入场；不是验收为“已识别”，而是发现了任务书规则与案例之间的不兼容。下面收益为信号后下一交易日开盘到收盘的毛收益，仅用于事件描述；不表示本策略参与或赚到。所有锚点还缺历史完整故事/市值/事件证据。','',
      '| 标的/信号日 | 量比 | 实体连续阳线 | 下一开盘→收盘 | 解释 |','|---|---:|---:|---:|---|']
    for a in anchors:
        s=a['signal'];f=a['forward']
        lines.append(f'| {a["ticker"]} {a["date"]} | {s["vol_ratio_20"]:.2f} | {s["green_streak"]} | {pct(f["return_1d"])} | {notes[(a["ticker"],a["date"])]} |')
    lines += ['', '任务书所述CBRS 9/4收盘相对前收盘约+10.30%、CRWV 9/8约+11.72%可由本次日线复算。但后者次日开盘后只剩约+7.40%的当日涨幅，不能把隔夜跳空也算成次日开盘策略收益。CBRS 6/29日内约+17.48%属于当天开盘到收盘，和收盘确认后的交易不是同一个实验。',
      '', '## 3. 回测、成本和消融','',
      '覆盖区间为2025年3月至2026年9月16日，CRWV从上市日开始、CBRS从2026年5月14日开始。完整策略各版本均为 `not_evaluable_missing_pit_evidence`。**没有足够证据评估不等于策略收益为0、胜率为0，也不等于已证明能避开所有历史亏损。**','',
      '| 版本 | 单边滑点bp | 严格成交数 | 结论 |','|---|---:|---:|---|']
    for r in costs:
        lines.append(f'| {r["variant"]} | {r["slippage_bps"]:g} | {r["closed_trades"]} | 点时证据不足，不报告策略绩效 |')
    lines += ['', '胜率、盈亏比、净收益、最大回撤、平均持有期、换手和稀释暴露次数保持NA。事件列表、成交明细、费用、退出原因及每日NAV的输出接口均已实现；当前现金轨迹只是缺失闸门下的程序状态，不是已验证的零回撤策略。消融只移除相应软过滤，保留核心排除、事件未知、硬否决和第三根禁追规则。',
      '',f'另有 **{len(events)}个纯放量阳线事件**，仅作价格形态描述，存于 `price_only_events_NOT_STRATEGY.csv`。它们没有通过完整基本面/宏观/事件风控，存在样本重叠和当前候选池幸存者偏差，不是新增交易记录。','',
      '| 前瞻窗口 | 有完整价格的事件数 | 毛收益均值 | P10 | 中位数 | P90 |','|---|---:|---:|---:|---:|---:|']
    for n in [1,3,5]:
        values=events[f'return_{n}d'].dropna()
        lines.append(f'| {n}个交易日 | {len(values)} | {pct(values.mean())} | {pct(values.quantile(.1))} | {pct(values.median())} | {pct(values.quantile(.9))} |')
    counts=events.groupby('age_bucket').size().to_dict()
    lines += ['',f'上市交易日分层：0–30日 {counts.get("0-30",0)} 个，31–90日 {counts.get("31-90",0)} 个，91日以上 {counts.get("91+",0)} 个。采用固定阈值，无训练/参数搜索；这只是时间年龄分层，不是已经完成的滚动样本外验证。历史长寿股票只有截取窗口，因此早期年龄需完整证券主表进一步核验。',
      '', '## 4. 对照基准（不等于策略超额收益）','',
      '单股买持从各自区间首日收盘开始，起点不同，不能直接横向断言谁更优。SOX是价格指数；SMH为ETF复权代理。当前AI候选篮子每日等权、无费用且包含事后选池偏差。','',
      '| 对照 | 起止日 | 区间收益 | 最大回撤 |','|---|---|---:|---:|']
    for b in benchmarks:
        if b['ticker'] in ['CRWV','CBRS','NBIS','^SOX','SMH','AI_EQUAL_WEIGHT_SURVIVOR_PROXY']:
            lines.append(f'| {b["ticker"]} | {b["start"]}—{b["end"]} | {pct(b["return"])} | {pct(b["max_drawdown"])} |')
    lines += ['', '## 5. 最近完整交易日样本','',
      '9月16日：正式观察名单0只、严格新开仓0只、拒绝/待补证14只。部分主题宏观闸门可通过，不代表个股触发通过；行情未见本版全局硬否决。未取得当前账户快照，所以 `exits.csv` 只有表头，不代表账户实际空仓。','',
      '| 标的 | 收盘USD | 距一年/上市以来高点回撤 | 量比 | 当前说明 |','|---|---:|---:|---:|---|']
    for symbol in ['CRWV','CBRS','NBIS','CRDO']:
        r=next(x for x in daily if x['ticker']==symbol)
        explanation={'CRWV':'无放量阳线确认，故事/市值/事件证据未齐','CBRS':'90日RV未暖机，三连阳禁追，量能不足','NBIS':'回撤未到35%，无放量确认','CRDO':'量比未到1.5，身份及基本面尚待完整核验'}[symbol]
        lines.append(f'| {symbol} | {r["close"]:.2f} | {pct(r["dd_from_1y_high"])} | {r["vol_ratio_20"]:.2f} | {explanation} |')
    lines += ['', '9月17日新增且独立于昨日收盘的风险：CRWV拟发行30亿美元可转债，已写入事件警报。只从本次核验时间起生效，没有倒填到9月16日或假定能在盘前公告第一秒退出。[SEC公告](https://www.sec.gov/Archives/edgar/data/1769628/000176962826000429/ex9911.htm)',
      '', '## 6. 尚缺的数据与可用边界','',
      '- 缺点时市值/完整估值序列、历史指引相对共识、客户流失/会计风险完整覆盖、过去每日解锁/融资事件日历。已核验的部分季度数字单独保存，未据此自动认定故事仍在。',
      '- 当前是14只固定研究候选，不是全美股与ADR扫描；没有退市/失败者全量主表。9只候选身份保守保留未完成核验标记，不能进入严格信号。',
      '- 缺分钟数据、LULD暂停与盘口深度，不交付收盘前30分钟成交对照，也不声称具备盘中瞬时事件清仓能力。',
      '- 尚无可供本模块复用的共享持仓/下单/调度服务。本次提供输入接口和可定时执行的CLI，没有自动下单或创建常驻任务。',
      '- 半导体背景复用现有共享缓存，网站日期/抓取时间及超过48小时警告保留。memory、policy_events存在陈旧项，不进入今日新增交易证据；MU附带Memory Cycle Read-through。',
      '', '## 7. 工程验证与交付索引','',
      '29项确定性测试通过，包含数值断言与完整模拟账户回放。验证PIT未来数据排除、节假日/半日市、完整周、历史前缀不变、旧盘中快照不转成收盘、缺失闸门、核心/周期排除、连续上涨禁追、事件T−3、下一bar、高开限价、跳空止损、止盈幂等、第三天退出、停牌延期、手续费、分红应收、现金/主题/相关性/冲突、盘中退出与次日执行。测试验证功能，不验证盈利。',
      '', '- 参数与完整规则：[模块说明](../../strategies/bb_washout_bounce/README.md)、[参数YAML](../../config/bb_washout_bounce.yaml)。',
      '- 当前可读结果：[日报](results/daily/日报.md)、[待补证/拒绝表](results/daily/rejects.csv)、[观察表](results/daily/watchlist.csv)、[信号表](results/daily/signals.csv)、[退出表](results/daily/exits.csv)。',
      '- 研究输出：[锚点验收](results/backtest/anchors.csv)、[成本与消融](results/backtest/cost_ablation.csv)、[描述性价格事件](results/backtest/price_only_events_NOT_STRATEGY.csv)。',
      '- 原始行情及来源哈希：`raw-live/manifest.json`；重现参数/输入：`results/reproducibility/`；测试日志：`tests.log`。',
      '- 失效条件与变更清单见模块说明。原Babybus研究规格、原策略参数与历史输出未改动。',
      '', '建议的下一步是补齐可审计的点时输入和失败者证券样本，然后固定本版参数做独立时期验证；不能以“让三个正锚点都通过”为下一轮优化目标。',
      '', '## 核验来源','',
      '- [CRWV Q2官方财报](https://investors.coreweave.com/news/news-details/2026/CoreWeave-Reports-Strong-Second-Quarter-2026-Results/default.aspx)：收入与合同积压。',
      '- [CBRS Q2官方财报](https://investors.cerebras.ai/news-releases/news-release-details/cerebras-systems-fast-inference-cloud-business-nearly-quadruples)：GAAP增速与RPO；[IPO身份/价格](https://investors.cerebras.ai/shareholder-services/investor-faqs)。',
      '- [NBIS SEC财报](https://www.sec.gov/Archives/edgar/data/1513845/000110465926094568/tm2622968d1_ex99-1.htm)：合并收入同比。',
      '- [9月16日FOMC官方声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm)：加息决议于美东14:00公布，实施日9月17日；策略事件按决议日标注。',
      '- Yahoo chart日线接口具体URL、抓取UTC时间、报价时间和哈希逐标的保存在manifest；历史日线可用性使用收盘后15分钟假设，并非真实历史采集档案。', '']
    (HERE/'改进与回测报告.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
