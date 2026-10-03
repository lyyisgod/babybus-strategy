from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
rank=pd.read_csv(P/'ranking-enriched.csv').set_index('symbol');qs=json.loads((P/'final-snapshot/quotes.json').read_text());quotes={q['symbol']:q for q in qs};oldquotes={q['symbol']:q for q in json.loads((P.parent/'final-snapshot/quotes.json').read_text())}
order=['ALAB','NBIS','CRDO','COHR','AAOI'];oldplans={q['symbol']:q for q in json.loads((P.parent/'five-stock-plans.json').read_text())};plans=[]
for i,s in enumerate(order,1):
 r=rank.loc[s];q=quotes[s];o=oldplans[s]
 plans.append(dict(rank=i,symbol=s,last=q['last'],quote_at=q['last_at'],fetched_at=q['retrieved_at'],source=q['source'],after_hours_pct=q['post_pct'],previous_report_last=oldquotes[s]['last'],entry_ceiling=o['entry_ceiling'],entry_ceiling_policy='keep previous session cap; do not raise cap',invalidation_prior_low=float(r.low),first_resistance=float(r.high),p_open_to_high5=float(r.p_hit5),p_open_to_high10=float(r.p_hit10),p_open_to_low_minus3=float(r.p_down3),actual_night_entry_probability=None,model_output_certified=False,bid_ask=None,actual_purchase='unknown',status='conditional_watch_only_credit_OFF',net5_price_example=q['last']*1.052,net10_price_example=q['last']*1.102))
(P/'five-stock-plans.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2))
# Context is shared and fresh; no independent per-stock refresh.
from src.analysis.semiconductor_context import enrich_stock_analysis
from src.data.semiconductor_sentiment import get_semiconductor_market_context
context=get_semiconductor_market_context();sector=P/'semiconductor-final';sector.mkdir(exist_ok=True)
for s in ['ALAB','CRDO','COHR','AAOI','MU']:
 out=enrich_stock_analysis(s,{'symbol':s},context=context);(sector/(s+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False));(sector/(s+'.md')).write_text(out.get('semiconductor_industry_context_markdown',''))
now=datetime.now(timezone.utc).isoformat()
events=[dict(symbol='BA',known_by=now,source='https://boeing.mediaroom.com/2026-09-29-U-S-Navy-Selects-Boeing-for-F-A-XX-Program',announcement_date='2026-09-29',primary_verified=True,claim='Boeing selected for multibillion-dollar Navy F/A-XX fighter program',not_claimed=['current EPS upgrade','contract margin','tomorrow surge probability'],status='new-event-watch'),dict(symbol='MARKET',source='https://www.bea.gov/news/schedule',announcement_at='2026-09-30T08:30:00-04:00',claim='August personal income/outlays including PCE; Q2 GDP third estimate'),dict(symbol='NBIS',source='https://nebius.com/prices',effective_at='2026-10-01',claim='on-demand GPU price increases, not whole company earnings upgrade',first_announced_tonight=False)]
(P/'events-verified.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
lines=['''# 9/29 20:33 EDT 夜盘重新筛选

**相对首选仍为 ALAB 普通股；综合观察顺序 ALAB > NBIS > CRDO > COHR > AAOI。严格 Babybus 不新增，尚无一只达到“明天大概率暴涨”的买入证据门槛。** 如果只按五股的旧量价上行尾部输出，AAOI 第一；这与财务证据及风险筛选后的综合第一不同。综合排序不是拟合出的新概率。

决策时点已进入美东夜盘，目标是今晚实际买价至 9/30 常规盘中高点 +5%/+10%。20:25—20:26 重新抓取 121 股票、21 代理、日线和 5 分钟共 284 份响应；20:31:59 再核验 12 个标的的分钟末笔，并于 20:32 刷新 FRED。**公开行情只更新到 19:57—19:59 附近，无真正夜盘 ATS 双边盘口。下列价格不能当作 20:33 的可成交价。**

## 五股结果与当前价格

32 因子固定双模型重新推断，121 股全部可计算，117 股通过原价格/流动性过滤。没有新成熟标签，所以保留前次训练/校准参数，只刷新可知输入和事件；重复拟合同一数据不能增加信息或置信度。新日线历史变化对模型输出仅有浮点级影响，排序依据没有被新训练强行改写。

下列输出是下一常规开盘→最高/最低的有偏概率代理，**不是今晚实际买价起算的暴涨概率，也不是置信水平**。高点与低点事件可以同日发生。

|综合顺序|股票|末笔盘后价 / 涨跌|行情时间 EDT|旧 +5% 代理|旧 +10% 代理|旧 −3% 下探代理|原限价上限|
|---|---|---|---|---|---|---|---|
''']
for p in plans:lines.append(f"|{p['rank']}|{p['symbol']}|${p['last']:.2f} / {p['after_hours_pct']:+.2f}%|{p['quote_at'][11:19]}|{p['p_open_to_high5']:.1%}|{p['p_open_to_high10']:.1%}|{p['p_open_to_low_minus3']:.1%}|${p['entry_ceiling']:.2f}|\n")
lines.append('''
限价保留前次研究上限，仅至9/30有效，不因价格上涨抬高；不是公平价值或已回测最优买价。今晚本身没有确认买入信号。

- **ALAB**：从上一报告 361.39 回落至 359.58（约 −0.50%），已低于当日分钟 VWAP 代理 360.59；价格便宜一些，但延续确认变弱。20 日跑赢 QQQ 17.43 个百分点，RSI14 61.34，量比 0.65。8/4 Q2收入同比+104%，Q3收入指引5.4亿—5.6亿美元，是财务背景，不是今晚新催化。[公司财报](https://ir.asteralabs.com/news-releases/news-release-details/astera-labs-reports-second-quarter-2026-financial-results)。
- **NBIS**：240.12，接近旧上限240.6，仍略低于分钟 VWAP 240.52。10/1按需 GPU 涨价是临近的已知事件，H200 4.50→5.40，B300 7.85→9.50美元/小时；不能推为全公司利润上涨。量比0.86，回撤约20.8%不足严格Babybus的35%。[官方价格表](https://nebius.com/prices)、[Q2 SEC披露](https://www.sec.gov/Archives/edgar/data/1513845/000110465926094568/tm2622968d1_ex99-1.htm)。
- **CRDO**：193.88，仍低于VWAP194.21。回撤37.7%、在完成周中轨下方，符合部分反弹背景，但量比0.66、当日实体阴线，没有放量反转。9/1 Q1收入同比+114.7%、GAAP净利润1.294亿美元支持业务兑现。[公司财报](https://investors.credosemi.com/news-events/news/news-details/2026/Credo-Technology-Group-Holding-Ltd-Reports-First-Quarter-of-Fiscal-Year-2027-Financial-Results/default.aspx)。
- **COHR**：293.80，低于VWAP294.69；量比0.86、回撤33.6%，严格买点未通过。9/21 PhotonLink收入预计第四季度爬坡，不能当成明日必有新订单。[公司公告](https://www.coherent.com/news/press-releases/launches-photonlink-integrated-optics-platform-ai-infrastructure)。
- **AAOI**：101.45，略高于VWAP101.31；五股量价尾部最大，下探代理也最高。量比1.08不足1.5。今天100万台1.8GHz放大器累计出货是CATV产品里程碑，不是新增100万台AI订单；Q2 GAAP仍亏损。[9/29公司公告](https://investors.ao-inc.com/news-releases/news-release-details/aoi-marks-1-millionth-18ghz-amplifier-shipped-unveils-new)。

## 今晚新增：BA军机合同

**BA是这次核验中有实质新增财务事件的观察候选。** 公司9/29官方确认获得F/A-XX多十亿美元合同；生产开发、利润率和现金流兑现跨年，不能直接换算明日EPS。末笔191.81（19:59:31 EDT），较187.68常规收盘+2.20%，已高于当日高点190.905，部分利好已经被价格反映。[波音官方公告](https://boeing.mediaroom.com/2026-09-29-U-S-Navy-Selects-Boeing-for-F-A-XX-Program)。

其旧量价模型 +5% 代理3.24%、+10% 0.55%、下探−3%12.42%，**这些输出没有纳入今晚新合同，不能当合同后的概率**。没有经过点时历史事件样本校准，不人为加20个百分点，也不凭新闻大小宣布第一。BA的事件质量值得看，但它不满足严格Babybus；涨幅目标5%还需从实际入场另算，不能把已涨的2.2%重复记作未来收益。列为单独事件观察，不挤掉原五股来伪造模型改善。

事件延续观察：明日8:30宏观数据后，若190.90附近守住、9:35后站稳开盘VWAP且量能延续，可重新评估；若回落190.90下方且不能收复，事件突破失效。此为未回测的观察条件，没有当下夜盘下单授权。

## Babybus与宏观开关

信用开关继续 **OFF**：JNK20日总回报−2.74%、5日−1.91%；9/25完成周93.61 <此前8周低点94.55×99.5%=94.0773。完成月线未走成空头，不抵消周破位。原14股严格模块重新运行：0观察、0信号、14拒绝；输入覆盖不足与量价未通过均保留，不把缺字段说成公司不可靠。五股都未达1.5放量标准，完成周带价格触发也未成立。

20:32重新抓取的FRED最新观测仍是9/28：10年实际利率2.90%，相对8/28+48bp；高收益OAS3.02%，+42bp。[DFII10](https://fred.stlouisfed.org/series/DFII10)、[HY OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)。

QQQ最后740.30（19:59:58、较收盘+0.32%）、SMH610.18（19:59:46、+0.54%）；半导体相对较强，但信用走弱并未转向。布伦特连续合约日线跳变可能受换月污染，不将−8.84%解释成今晚油价暴跌；股票模型仅使用滞后一日收益，旧敏感性结果仍作为局限保存。

**9/30 08:30 EDT：8月个人收入/支出含PCE、二季度GDP第三次估计。** 这是今晚到明早的关键共同风险，发布前没有结果，未获得可追溯一致预期就不编造惊喜值。[BEA日程](https://www.bea.gov/news/schedule)。MU的财报电话会9/30 16:30，为明日正常盘后，不能先用于盘中判断。

## 可执行边界

严格策略结论为不新增。若用户主动选择偏离严格规则的隔夜投机，ALAB仍是条件首选；先核对券商真正夜盘ask与时间，ask超过362.4或价差超过0.3%不追。最后359.58不是当前可成交卖价。明日盘初重新收复360.59并守住开盘VWAP才增加延续证据；这改变入场时点，不再是现在夜盘买入。

反弹失效观察使用9/29低点：ALAB349.21、NBIS233.61、CRDO189.01、COHR285.85、AAOI97.43。若主动投机且最大价格亏损预算为3%，独立风险触发价为max(上述日低, 实际买价P×0.97)；AAOI以101.45为例约98.41，比日低更紧。失效触发不保证成交损失上限，隔夜跳空可能穿越。触发退出后不自动等待反弹。

第一阻力依次366.70、246.975、199.38、302.72、103.83；**触及这些阻力不等于从当前报价赚5%**。实际P、总成本0.2个百分点的示例净+5%需要P×1.052，净+10%需要P×1.102；ALAB以359.58示例分别378.28/396.28。实际夜盘成本未知，需重算。最长持有到9/30常规收盘，不自动转长期，不同时把五股当独立低相关仓位。没有账户风险预算，不编股数。

## 置信度、覆盖与验证

模型仍用2016—2022训练76293行、2023—2024校准37742行；旧2025—2026评估45328行、435日期。全池有幸存者及事后选择偏差，旧评估已经使用，没有新未碰审计段、真实历史夜盘入场和盘口，不能给经验证当前暴涨概率。

旧评估每日原始第一名 +5%触及43.68%，日期块95%历史区间38.85%—48.28%；+10%16.09%；下探−3%61.61%。保守+5%止盈/−3%止损，同日双触先止损，扣0.2%成本后平均−0.543%，区间−0.858%至−0.234%。因此**95%只是历史统计区间，并不表示95%把握明天上涨**。

原量价靠前的IOVA/KOD未因高分强推：两者+5%旧输出约67%/66%，下探−3%也约85%/73%；超买和跳涨已发生。CBRS末笔192.95、盘后−1.03%，低于当天低点194.21，继续剔除。完整候选及失败保留在ranking-enriched.csv。LRCX趋势超卖原型仍触发INTC/MSTR/COIN/TEAM/HOOD/RNG/QCOM/NOW/CRM，其执行为次日收盘、最长20天，不替换这次次日盘中目标；WDAY原始认可不编参数、不替换成WDC。

统一机会分覆盖率仍20%，仅宏观1维，**机会分留空**；缺全池点时EPS修订、FCF、PB杠杆、恒定期限IV/偏斜、固定AI/非AI池、完整事件/资金流/资本开支、借券及真正夜盘深度。当前财务事件审阅不能填补历史模型缺失。独立风险警报：信用OFF、PCE/GDP、低量反弹、行业集中、缺可执行报价。

半导体背景共享18小时缓存：来源Semiconductor Fear & Greed，网站模块数据9/29约02:01—02:35 UTC，实际抓取9/29 23:59:11 UTC；AI情绪72 Greed、广义61、周期83 Peak conditions，仅作背景。政策模块9/7陈旧超过48小时，不计新增。MU Memory Cycle Read-through为Bullish规则背景，MU自身库存未知；评级变化不等于EPS上修。网站分组和本地业务暴露在各股文件分开记录。[行业来源](https://semiconductor.feargreedchart.com)。

数据质量：20:25后股票及部分代理当前日OHLC为空，使用19:50左右此前保存完整的同日记录在内存补齐，原新响应保持不动；4条期货日线出现纽约日期重复，用此前完整日线且只滞后布伦特入模。共138条回退记录（含期货），不是138次成功实时更新。回退文件时间、原始SHA及原因可追溯。

验证：284份新响应哈希、12份末笔行情哈希、FRED哈希一致；冻结系数哈希一致；121股特征时间到9/29、无计算缺失；六股/事件候选前缀因果检查、概率有限与嵌套检查通过；新旧五股模型输出复现。功能验证通过不等于投资有效。严格模块复用原已完整收盘快照、20:33重新运行，没有宣称回退日线是20:33新收盘。

未下单，用户是否实际买入未知，9/30结果尚未发生。新BA事件作为本轮可知证据独立记录，不回填进前次模型或模拟收益。
''')
(P/'研究结论.md').write_text(''.join(lines))
ledger=ROOT/'量化参考/优先策略/案例登记.json';d=json.loads(ledger.read_text());rid='2026-09-29-2033-overnight-refresh';assert not any(r.get('run_id')==rid for r in d['research_runs'])
d['research_runs'].append(dict(run_id=rid,decision_at=now,relative_first='ALAB',five_ranked=order,absolute_gate='failed_credit_OFF_strict_signals_0_of_14',strategy='same frozen 32-factor model with refreshed public inputs and independent event review',universe_count=121,liquid_count=117,source=str((P/'研究结论.md').relative_to(ROOT)),all_candidates=str(P/'ranking-enriched.csv'),plans=plans,new_event_watch='BA',new_event_source=events[0]['source'],quote_limit='last available after-hours observations, no overnight executable bid/ask',raw_field_fallback_count=138,actual_bought='unknown',later_evidence=[],outcome='not_yet_known',probability_certified=False,orders_placed=False));d['saved_at']=now;ledger.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
verified=[]
for r in json.loads((P/'manifest.json').read_text()):
 f=P/'raw'/f"{r['symbol']}-{r['interval']}.json";assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];verified.append(str(f))
for q in qs:assert hashlib.sha256((P/'final-snapshot'/(q['symbol']+'.json')).read_bytes()).hexdigest()==q['sha256']
for q in json.loads((P/'macro-manifest.json').read_text()):assert hashlib.sha256((P/(q['series']+'.csv')).read_bytes()).hexdigest()==q['sha256']
assert len(verified)==284 and len(plans)==5 and all(p['entry_ceiling']==oldplans[p['symbol']]['entry_ceiling'] for p in plans)
assert rank.hit5_change_pp.abs().max()<1e-6
(P/'delivery-verification.json').write_text(json.dumps(dict(status='pass',raw_hashes=284,minute_quote_hashes=12,macro_hashes=2,frozen_scores_match=True,ceilings_preserved=True,ledger_run_id=rid,orders_placed=False),indent=2))
print('saved refresh report, plans, industry context, ledger and delivery verification')
