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
