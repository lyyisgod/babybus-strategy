# 独立选择层

`macro_regime/` 全部 Python 文件保持 v1.1 原样，冻结摘要见`docs/frozen_v1_1.sha256`。本包只增加接口桥接和选择功能，不调整作者价位、63 个 NYSE 日有效期、200 日均线、MOVE 80 分位、252 根最少历史、两天确认/解除或跌幅档位。旧宏观 CLI 的必填 `--regime` 保持兼容；新入口只读取核验过的官方证据，不接受手填 HIKING 覆盖 UNKNOWN。

## 运行与来源

```sh
.venv-macro/bin/python -m macro_selection.snapshot \
  --official-evidence examples/official_evidence.example.json \
  --calendars examples/calendars.example.json \
  --output data/result.json
```

运行日期来自带时区的系统时钟，t 为已收盘 NYSE 日，每次强制刷新 Yahoo/FRED 并存 parquet、日期、来源和摘要。缺失必需序列返回 `data_valid=false`、缺失因子 `null`、空名单，退出码 2；输出是失败仪表，不能冒充完整宏观快照。MOVE 依旧是唯一可代理序列。政策来源与行情齐全是两个独立条件。

官方证据输入是最近两次 FOMC 原文的日期、动作、区间、短原文、链接及核验标记，以及下一场会议 CME FedWatch 的 cut/hold/hike 概率、市场日期、带时区页面数据时间。必须保存原始核验记录；不以网页抓取时钟冒充行情时间。缺失/矛盾/旧概率为 UNKNOWN，DFF 只能作为标签。仅凭两次声明不能证明降息轮数时保持 UNKNOWN，不用价格补政策历史。

`bridge.py` 调用 v1.1 数值因子和压力滞回，首次完整 pair 的旧 WATCH 只在新接口标为 PENDING；单维 WATCH 不封锁 margin。OAS 由 `--oas-publications` 提供经过核验的**实际公布日期**、数值、来源与 `publication_dates_verified=true`。先在原公布坐标 shift(1)，然后计算五次公布变化并 backward asof；未经核验的 FRED observation_date 不能冒充公布日。DGS10 与 MOVE 的日历仍由 v1.1 处理。

当前官方证据/日历文件只供对应日期复现。概率或成分的日期不匹配时，不能在以后运行中沿用。当前 Yahoo/FRED 最新修订值不是历史 point-in-time 数据，不用于收益有效性论证。设计期间行情未用于参数选择或盈利证明。

## 选择输入与闸门

宏观有效后，才读取可选 `--candidates` JSON 列表。候选是有出处的事实，不是推荐名单；`selection.py` 不导入宏观代码，只消费 JSON。价格输入应使用 `prepare.price_facts`：相同复权口径，t 前截断，至少 200 根已收盘价格，63 根高点窗口的首日作为 EPS 基准日期。不得用 60 天分析师趋势代替 63 个交易日的同财年一致预期。

每条记录含 `symbol/market/role/asof/window_start/facts/consensus/membership`。每个 facts 字段含 `value/date/series/source/closed`；基本字段为 `ret_1d/ret_5d/drawdown_63d/below_ma50/below_ma200/rsi14/new_low20`。卫星额外需要 `hike_beta/liquid` 或 `dividend_yield/mature_company`；A 股需要 `roe/revenue_growth/high_valuation/financing_sensitive`，不得用新闻填财务缺项。成分证据必须为 t 日的 SMH、XLU 或当地最后收盘日的 CSI300，包含来源。外部流动股必须提供 `smh_membership={member:false,date:t,source:...}`，不能用未知成分状态冒充 SMH 以外。

consensus 包含 `baseline_eps/current_eps/baseline_date/date/series/source/is_consensus/fiscal_period/baseline_fiscal_period/baseline_available_at/available_at`，同财年、准确端点且当时已知；基准 EPS 不为正时百分比下修不可可靠解释，保持 fundamental_gate=null。当前未接入能提供这组 63 日历史一致预期的商业服务，不伪造采集结果；缺项逐股删除。SLV/SILJ 虽在范围内，也没有免除两种估值模板，无法通过时不会自动给出接飞刀。

错杀分为高点回撤减 EPS 下修幅度，EPS 上修不额外加故事分。下修超过 5% 删除；核心还要满足 SMH 范围及低于 MA50 或回撤至少 10%。A 股范围为 CSI300，ROE、营收增速均正、回撤至少 15%；符合高估值/融资敏感且 HIKING、收益率五次公布上升时扣 5 个百分点。主分相同采用有出处的 secondary_score；仍无法打破并列则无核心，不用熟名字填空。每个市场最多一个核心。

卫星成长模板按错杀分、成熟高股息模板按股息率减 DGS10 排序，两组不合成总分。股息溢价必须正；两组按固定交替顺序选取，总数最多三个，该顺序只是可复现的容量安排。全部需要 RSI<30 或 20 根新低。yield_stable=true 时不发卫星接飞刀，核心依然必须通过完整一致预期与错杀分检查，单纯 RSI 超卖不构成资格，不额外添加“EPS 必须上修”的阈值；null 不进入收益率稳定分支。

UNKNOWN 或 PENDING 加关键空值不新开仓；FIRST_CUT 为 CORE_EXIT_WINDOW；EASING 为 HOLD。TRIGGER/CONFIRMED_SELLOFF 只允许已有核心收跌后按原档位加，禁止新核心与新卫星，margin=false、gross 上限 1。gross>1 优先 DELEVER_TO_1X；gross 未知时压力加仓保持观察。HIKING/PAUSE 无压力才可选新核心，PAUSE 不加 margin。t 收涨或平盘等待阴线，五日涨幅>8% 删除，跌幅没达到原档位则观察。休市只给观察，不发订单，输出下次开市日。

## 数量与验证

核心 `size_unit` 是 0/.25/.5/1 **序数**，通过桥接层的 v1.1 `dip_units` 输出，不是净值比例。HOLD/退出/减杠杆的序数为 0。卫星 .02 是建议单名名义上限，不计算持仓；卫星 size_unit=0 表示核心序数不适用。无用户资金预算时 nominal_amount=null，不能据此生成股数。`可执行`仅表示研究规则通过，仍受 gross 上限和实际预算约束。

`pytest` 检验政策证据缺失/冲突、优先级、压力两天进入/解除、持有与加仓序数、并列处理、EPS 缺失/下修、股息模板、A 股休市、实际运行日、OAS 发布滞后及追加未来行不改变历史。冻结摘要测试确认未改动原 v1.1。合成测试通过不代表策略收益有效；新选择层纸面记录 N=0，与旧记录分开，不合并为旧模型样本。
