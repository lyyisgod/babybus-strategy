# Babybus Strategy

## 因子定义

纯研究日频策略：先计算冻结的 `macro_regime v1.1`，再由独立 `macro_selection` 消费宏观输出与有出处的收盘价格、一致预期、指数/基金成分事实。只输出研究状态、名单和约束；不下单、不接券商、不读推文、不输出暴涨概率。

宏观组合采用完整作者组合 OR 完整统计组合。IXG 绝对收盘与 MOVE 指数点的两个 125 单位不同；长期压力使用 IXG/SPY 的 200 日均线与 MOVE 五年 80 分位，最低 252 根。两次收盘确认，两次解除；选择接口把首次完整组合标为 PENDING，不交叉拼接两种定义。MOVE 是唯一允许代理的序列。

DGS10 在公布日计算变化、10 次公布波动、252 次公布中位数及连续 5 次公布确认；股票日只 backward asof 映射已算完的结果，不补收益率、不写假零。OAS 需提供实际公布日，至少滞后一次公布。JNK 下跌先区分利率与信用，再看股票确认。

核心范围为当期 SMH 持仓，错杀分为 63 根高点回撤减同财年一致预期 EPS 下修幅度；下修超过 5% 删除，缺一致预期不能升为核心。卫星成长与股息溢价模板分别处理。A 股先验证交易所日历及 CSI300 成分、财务；休市只给观察。完整定义见 [宏观说明](macro_regime/README.md) 与 [选择层说明](macro_selection/README.md)。

安装与离线测试（Python 3.10+）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

手动传入政策/敞口的原始宏观入口：

```sh
.venv/bin/python -m macro_regime.snapshot \
  --date YYYY-MM-DD --regime HIKING --gross-exposure 0.8
```

官方政策与选择入口自动使用运行时最新已收盘 NYSE 日期：

```sh
.venv/bin/python -m macro_selection.snapshot \
  --official-evidence PATH_TO_VERIFIED_OFFICIAL_EVIDENCE.json \
  --calendars PATH_TO_VERIFIED_CALENDARS.json \
  --oas-publications PATH_TO_VERIFIED_OAS_PUBLICATIONS.json \
  --candidates PATH_TO_VERIFIED_CANDIDATES.json \
  --gross-exposure 0.8 --output data/result.json
```

`examples/` 仅展示格式，核验标记关闭，不提供实时买入信号。官方证据、会议概率、基金成分与 63 日一致预期需由调用方提供并核验；当前没有自动采集完整历史一致预期的商业数据接入。必需宏观序列缺失时停止选择，输出失败仪表并退出码 2；不把最新修订历史当点时回测。下载的缓存保存在被 Git 忽略的 data 目录。

## 动作优先级

政策优先：UNKNOWN 不新开仓；FIRST_CUT 为核心退出窗口；EASING 保持 HOLD。压力确认禁止 margin，总 gross 上限 1，禁止新卫星及新核心；已有核心收跌且达到原跌幅档位才能 CORE_ADD_NO_MARGIN，gross>1 先减杠杆。压力触发不等于清仓。WATCH 只是标签，不新增压力限制。

无压力的 HIKING/PAUSE，每个市场最多一个核心、美股最多三个卫星；yield_stable=true 时不发卫星接飞刀。核心 t 日收涨等阴线，五日涨幅>8% 删除。加仓 size_unit=0/.25/.5/1 是序数，不能当净值比例；卫星 .02 只是建议单名名义上限，不计算持仓或股数。原始宏观动作与选择层闸门的区别分别在两个模块 README 中注明。

`tests/` 包含离线合成表驱动、公布日、防泄露、缺失源与冻结摘要检验；不需要 GitHub 凭据、券商或行情网络即可验证规则。测试通过证明实现符合规则，不能证明投资效果。选择层纸面 N 从 0 开始，不合并旧研究收益。

## 绝对点位会过期

作者点位独立存于 `macro_regime/config.py`，as_of=2026-09-30，之后 63 个 NYSE 交易日有效，第 64 日作者组合永久关闭。过期不影响统计腿；MOVE 代理时作者组合关闭。AVGO/SLV/BWXT/APP 加仓参考仅展示，不进入状态机。

原 v1.1 Python 文件逐字保留，SHA256 清单在 [docs/frozen_v1_1.sha256](docs/frozen_v1_1.sha256)。本仓库包含策略源码、规则说明、格式示例、合成测试，以及[历史短线案例参考](references/short-term-cases/README.md)。参考资料保留原建议、失败与事后结果，个人账户信息已脱敏；这些旧研究不计入 v1.1 的纸面 N。

## A股因子映射

新增[A股主升浪因子映射 v1.1](docs/a-share-factor-mapping.md)及[机器可读数据字典](docs/a-share-factor-mapping.json)：12项美股因子的中国对应口径、6项A股补充因子、数据来源、缺失处理与验证边界。机构总杠杆无可靠公开等价项，与融资拥挤度合并，筹码供给独立核验。该研究规格尚未接入运行入口，权重与暴涨概率未校准；现有冻结 v1.1 保持原定义。

## macro_engine

新规则的唯一入口为 `macro_engine.decide(book, facts, policy_statements, asof)`。返回不可变的 `Decision`，用 `to_dict()` / `to_json()` 导出 `regime、stress、constraints、action、size_unit、cores、satellites、rejects、trace、surge`；每天只有一个账户动作。冻结 v1.1 与原 `macro_selection` 的独立入口保持原行为。

账本指定全局唯一 CORE，所有 nominal、nav 和 budget 必须使用同一计价货币。gross 由持仓名义金额绝对值之和 / nav 计算；budget 是当日可用于增加名义金额的资金。缺 nav、budget 或任一持仓 nominal 时 `trace.execution_status="观察"`，不能标为可执行；若 gross 仍能算出且压力确认超过 1 倍，唯一动作提示仍为 DELEVER_TO_1X，不产生可执行加仓行。`opened_on` 是实际开仓日，watchlist 允许留空。引擎不按错杀分每天换 CORE。

```python
from macro_engine import decide

book = {
    "nav": 100_000, "budget": 10_000,
    "positions": [
        {"symbol": "NVDA", "market": "US", "role": "CORE",
         "industry": "SEMICONDUCTOR", "nominal": 60_000,
         "opened_on": "2026-09-30"}
    ],
    "watchlist": [
        {"symbol": "SLV", "market": "US", "role": "SATELLITE",
         "industry": "PRECIOUS_METALS"}
    ],
    "max_names": 16,
}
# facts 与 policy_statements 是已核验的当时可用原始证据。
decision = decide(book, facts, policy_statements, asof="2026-10-01")
print(decision.to_json())
```

输入及重放约定：

- `policy_statements` 是完整官方声明序列，每条含 `date、action(HIKE/HOLD/CUT)、target_range([下限,上限])、source、verified`。可另附带时区的 `available_at`；未来才可用的声明会被过滤。不提供它时，声明日期是其已公开日期。来源必须是官方 Federal Reserve HTTPS 页面。加息后的任意长度暂停再第一次降息为 FIRST_CUT；后续降息保留 `cut_index`。矛盾或无法证明的路径为 UNKNOWN，FedWatch 仅作标签。
- `facts.prices[symbol]` 为复权收盘原序列：`date、adj_close、available_at、source、verified、closed`。CORE 还需同口径 `adj_open` 验证阴线，缺开盘价只观察。价格必须在本交易日收盘时已经可用，且按已核验日历连续；引擎计算一日/五日收益、RSI、20 日新低及前 252 日最高价，不使用调用方的 `ret_5d`、错杀分或未收盘行情。
- `facts.memberships.SMH / CSI300` 是成分变更序列：`symbol、date(生效日)、member、available_at、source、verified`。`facts.consensus[symbol]` 为一致预期版本：`date、eps、fiscal_period、is_consensus、available_at、source、verified`。CORE 必须是当日 SMH 成分；与 63 根窗口起点当时已知的同财年基准比较，下修超过 5% 或缺一致预期不能加仓。
- `facts.macro_prices` 提供 IXG、SPY、JNK、QQQ、SMH、TLT 的原始 `close、adj_close`，另含上述日期、可用时间及核验字段。`facts.publications` 提供 OAS、DGS10、MOVE 的实际公布日 `date、value、available_at、source、verified`。先按公布日 `shift(1)`，再计算五次公布差分，最后 backward asof 到股票日；不读取已铺到股票日的差分。MOVE 缺失使用完整 TLT 波动代理历史，并关闭作者点位腿；国债休市按实际国债日历处理。缺少必需证据时观察；已独立确认的压力仍可优先提示去杠杆。
- `facts.expected_hikes_remaining` 是 `{value: 正整数, source, verified, available_at}`。HIKING/PAUSE 时只有已核验正整数才打开金银逢低桶；第一刀是股票 CORE_EXIT_WINDOW，金银仍持有；第二刀金银停止加仓并退出；第三刀及以后只观察。`facts.earnings[symbol]` 有财报日期时，日期前 CORE 只持有不加；没有该字段不增加限制。
- 卫星限于 SLV、SILJ、TLT，以及账本 UTILITY / SOE_DIVIDEND 压舱，必须有来源、收盘事实及 RSI<30 或 20 日新低。新卫星达到前 252 日高点被拒；该限制不用于既有 CORE。全局硬顶 16 名，可用 `max_names` 下调；同业最多 5 名。卫星建议单名名义上限为 NAV 的 0.02，已持有金额占用额度，不计算股数，不为填名额买入。yield_stable=true 只禁止新开卫星。
- 压力仅收紧 margin 和 gross：TRIGGER / CONFIRMED_SELLOFF 下 `allow_margin=false、max_gross=1`；gross>1 当天唯一动作 DELEVER_TO_1X。gross≤1 时指定 CORE 收跌、收阴且五日涨幅≤8%，仍按冻结跌幅序数加仓；`size_unit=0/.25/.5/1` 是序数，须另行做资金分配，不是净值比例。
- 卫星升 CORE 必须在账本显式提供 `core_promotion={symbol, previous_core, issued_on}`，旧 CORE 必须已退出或降为其他 role；晋升目标标 `previous_role`。A 股只接收 `market="CN"、role="BALLAST"、industry="SOE_DIVIDEND"` 且当前 CSI300 成分；`facts.calendars.CN={sessions:[交易日期], source, verified}` 必须已核验。休市输出观察与 `next_open`。不使用 A 股因子映射权重。

`trace.coverage` 即使名单为空也列出输入数、输出数及被成分、一致预期、五日涨幅、未收阴、产业上限、创新高、压力去杠杆和金银时钟拦下的计数。十个连续合成 XNYS 交易日的原始输入与手写预期保存在 `tests/fixtures/golden/`；未来行及未来才可用的回填行不得改变历史 Decision JSON。

`macro_engine.surge` 独立计算历史频率，不在 `decide` 中运行，也不能修改动作、序数或约束。目标固定为下一 XNYS 交易日复权收盘相对当日复权收盘上涨≥5%，不用日内高点。金银、TLT 和 A 股压舱不计算该股票频率。

```python
from macro_engine.surge import estimate

metrics = decision.trace["computed_price_facts"]["NVDA"]
bucket = {"regime": decision.regime, "stress": decision.stress["state"],
          "dip_unit": metrics["dip_unit"],
          "yield_stable": decision.trace["yield_stable"],
          "at_252_high": metrics["at_252_high"]}
frequency = estimate("NVDA", "2026-10-01", bucket, panel=pit_panel)
display = decision.with_surge([frequency])  # 只读注释，不重新裁决
```

PIT 面板格式是 `{pit_verified: true, rows: [...], prices: {symbol: [...]}}`。每个 rows 记录含 `symbol、date、regime、stress、dip_unit、yield_stable、at_252_high、available_at、source、verified`；prices 为有来源且已核验的原始复权收盘记录。状态和价格须在各自交易日收盘时已知；标签由下一实际交易日收盘自算，且须在 asof 前一交易日收盘前已经成熟。同桶股票合并计数，桶只使用上述五个离散状态，dip_unit 只允许冻结四档；股票代码、股价、市值、推文、错杀分及手填标签不进入桶。缺 PIT 面板全部股票 `p=null、abstain=true、reason=no_pit_panel`；n<30 为 insufficient_sample。区间使用 Wilson 95%，不会用 0.5 补缺失。

输出始终包含 `symbol、asof、target、p、n、ci_low、ci_high、method、calibrated、beats_baseline、abstain、reason`。这是历史频率，不是暴涨预测；n、区间、abstain 必须一起展示。未校准前 calibrated=false，尚未验证基准优势时 beats_baseline=null。改动展示用 p（包括 0.99 或 null）不改变 action、size_unit、constraints。

测试通过不等于有收益，纸面 N 仍从 0 开始。
