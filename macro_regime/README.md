# 因子定义

纯研究用日频状态机 **v1.1**，输出可对账的规则状态与约束；不下单、不接券商、不读推文、不优化阈值。v1.1只修正收益率相关计算的日历，10/252/5窗口和阈值保持不变；10个交易日持有期仍按XNYS计数。它是已有机会分和短线模型旁边的独立风险层，不修改原分数，不生成上涨概率。规则正确不等于策略有收益；FIRST_CUT 的“最后逃命反弹”只是待验证的作者假设。

因子口径变更后，旧版纸面观察不能累计到v1.1的N，v1.1应从N=0开始。当前项目未发现macro_regime纸面持仓/收益观察记录；缓存、合成测试和验证快照不计入N，也不改写旧验证记录。

**设计说明 1：两个125单位不同。** IXG为美元ETF价格，MOVE为债券波动指数点；绝对价格会随市场和拆股尺度漂移。长期统计使用复权 `IXG/SPY` 的200日均线和MOVE近1260个交易日的经验分位，不能拿125当长期阈值。

**设计说明 3：JNK下跌没有唯一动作。** 利率上涨、HY OAS不升时标记利率压低债价；利差走阔与股票确认分别观察，不把JNK跌幅直接变成清仓信号。

输入约定：无时区且严格递增、无重复的交易日期索引。`IXG/JNK/QQQ/SMH/SPY/SLV/TLT`及调用方核心标的存原始Close，同时提供对应`SYMBOL_adj`总回报调整收盘；绝对点位只比Close，收益/RS/RSI/回归用复权序列。`MOVE/VIX/TNX`为指数原值，`move_proxy`逐日为布尔值；`DGS10/DGS2/DFF/BAMLH0A0HYM2`单位为百分数，差值为百分点（乘100才为bp）。所有函数只看传入的已对齐数据；调用方回测时必须按可用时间截断。

| 因子 | 固定定义 |
|---|---|
| policy_regime | 调用方手填 HIKING / PAUSE / FIRST_CUT / EASING，不从价格或DFF反推 |
| nfp_change | 可选人数标签，无固定160000门槛；不推断政策，不触发有色卖出 |
| author_combo | IXG收盘<125且MOVE收盘>125；两点位均生效、未过期，且非MOVE代理 |
| stat_combo | 复权IXG/SPY<自身200日均线，且MOVE五年经验分位>80；作者组合与统计组合为或关系，禁止跨组合拼接 |
| stress | pair=author_combo或stat_combo，连续2个完整收盘为真进入TRIGGER；连续2天为假解除，第一天继续TRIGGER。其余有任一有效弱势/高波动维度或尚待配对确认为WATCH，否则NONE |
| MOVE_pct5y | 含当期最多1260收盘：100×(低于当期数量+0.5×等于当期数量)/有效观测数。以最多1260根真实观测计算，至少252根才算分位；不足时为缺失（JSON为null）、stat_combo=false，不记为0；同时输出观测数和80分位数值用于对账 |
| balance_sheet_stress | TRIGGER且5日ΔDGS10>0且5日ΔHY OAS>0，描述资产负债表压力，不称“经济太好” |
| RATE_NOT_CREDIT | JNK五日<-1.5%，五日ΔHY OAS≤0，五日ΔDGS10>0 |
| RISK_ON_DIVERGENCE | JNK五日>0且QQQ五日<0 |
| UNCONFIRMED_CREDIT | (JNK五日<-1.5%或连跌≥5天)且SMH五日≥-1% |
| CONFIRMED_SELLOFF | JNK五日<-1.5%且QQQ五日<-1.5%且TRIGGER；分类重叠时优先确认抛售，其次利率归因、未确认信用、风险偏好背离 |
| yield_stable | DGS10先dropna，再算本次公布减上次公布；最近10次公布变化的样本标准差（min_periods=10），低于此波动列最近252次公布的中位数（min_periods=200），连续5个DGS10公布日成立。MOVE单独在自身公布日上算5次公布变化≤0，连续5个MOVE公布日成立。两条腿均已知且成立才为true；任一腿未知则为null |
| hike_beta | 过去60个股票交易日内，日收益小数对当日公布ΔDGS10百分点的带截距OLS，正常国债休市日不参加配对回归、不写入零变化；SMH≥0为CORE_ELIGIBLE，负值为RATE_SUPPRESSED。零方差/不足数据报错，绝不当成0 |
| oversold | Wilder RSI14<30或收盘低于此前20日最低；只作卫星候选，不能赋予核心资格 |
| size_unit | 仅针对调用方core_symbol（默认SMH）：日收益≥-1%为0；[-3%,-1%)为0.25；[-6%,-3%)为0.5；<-6%为1。这里只是序数0/0.25/0.5/1，不是NAV比例、建议持仓或股数 |
| chase / LEAP | 核心五日>8%则size_unit=0；QQQ单日≤-2%可加LEAP_2Y_0_6D标签，触发后第1至20个XNYS交易日禁止重发，第21日才能再次触发；按传入完整日频历史回放标签，不计算期权价格 |
| vix_spike | VIX>20日均线×1.25或单日增>3点；只解释限价等待，不产生卖出 |

`data.py`用yfinance拉IXG、JNK、QQQ、SMH、SPY、SLV、^VIX、^TNX、^MOVE、TLT，展示点位额外尝试AVGO/BWXT/APP；IXG longName必须含Global Financials，下载和缓存都验证身份，失败退出。FRED拉DGS10、DGS2、DFF、BAMLH0A0HYM2和PAYEMS；PAYEMS按月单独缓存，单位千人，不填充为日线、不从最新修订值生成历史非农事件。非农标签由调用方提供。

parquet和JSON元数据保存来源、序列名、抓取时间、观测范围、单位、复权口径与SHA256。缓存按序列和请求日期范围隔离；默认读准确匹配的缓存，`--refresh`重新拉取；只有MOVE允许代理；其它必需序列下载失败或被标记为代理立即报错。`--allow-stale-cache`只适用于MOVE匹配缓存，标记`proxy=true/stale_cache=true/fetch_error`，仍关闭作者组合；不能用于IXG、其它ETF或FRED。错误证券身份或格式不允许用缓存遮掩。

^MOVE缺当日、已有观测区间中有缺口或代理缓存回退时，整段改用TLT复权收盘对数收益20日样本标准差×√252×100，`move_proxy=true`且`proxy=true`；以同一五年经验分位定义统计腿，不与125比较，不拼接不同单位历史。源代理身份变化后重新累计252根；只有历史短且当日真实MOVE存在时，保留原序列，不能仅因未满五年就换代理。

股票日期按XNYS对齐，尚未收盘、周末、缺其它必需序列或关键因子窗口时报错。DGS10及MOVE另外保留原公布日序列：先计算各自的波动/变化、连续计数和布尔值，再用`merge_asof(direction="backward")`将publication_date≤股票日期的最后结果及asof日期映射回来；不映射或填充收益率水平，不插值、不在国债假期写入Δ=0。原频率还保留NYSE不开市但实际有公布的记录。API可通过`dgs10_publications`、`move_publications`传入原频率序列；未提供时从frame的非空原观测计算。

仅放行NYSE开市而国债全天休市的哥伦布日、退伍军人节缺口，标记`treasury_closed`，收益率腿的asof与计数保持上次公布结果；早收市有公布则正常使用。其它NYSE开市日期缺少DGS10，或最新DGS10公布距目标日超过5个XNYS交易日，整份快照报错；MOVE不参加DGS10的完整性检查。假期参考[SIFMA固定收益休市表](https://www.sifma.org/resources/guides-playbooks/holiday-schedule)。年度中位数窗口中不足200个有效波动值时，yield_stable=null，原因记录`insufficient_yield_history`，不会触发EARNINGS_SLEEVE；不再要求252个股票交易日逐日存在波动。输出`yield_calendar`记录两条腿的asof、状态、连续公布次数及DGS10当日状态。FRED OAS可能只有三年历史，不要求它有五年数据，MOVE压力分位仍使用五年最大窗口和252根最低观测要求。

Yahoo/FRED当前下载为供应商复权/最新修订观察，不证明历史点时可得性；元数据`pit_verified=false`、FRED历史`available_at=null`，抓取时间不冒充发布日期。不能把该快照入口当点时回测认证。[yfinance日线接口](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)、[IXG身份](https://www.ishares.com/us/products/239742/ishares-global-financials-etf)、[FRED HY OAS历史限制](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)、[PAYEMS月频](https://fred.stlouisfed.org/series/PAYEMS)。

# 动作优先级

**设计说明 2：压力触发不等于清仓。** 加息下联合触发设置allow_margin=false、max_gross=1.0、停止新卫星仓；保留半导体核心，跌幅加仓单位仍可展示，但调用方必须遵守总名义敞口上限。

**设计说明 4：政策路径优先于价格。** FIRST_CUT是作者定义的核心退出窗口；IXG破位或超卖均不能替代、取消这个政策条件。它是策略假设，不是关于首次降息必跌的已验证判断。

每天仅一个`action`，按下面顺序先命中返回；`satellite_candidates`和LEAP仅为研究标签，不是第二条账户指令。

1. FIRST_CUT → CORE_EXIT_WINDOW，size_unit=0，包括已有压力和gross>1的情况。
2. 任意非FIRST_CUT政策下，TRIGGER或CONFIRMED_SELLOFF且gross_exposure>1 → DELEVER_TO_1X，size_unit=0。
3. HIKING且CONFIRMED_SELLOFF → DELEVER_TO_1X，允许保留核心、禁margin/新卫星仓；gross≤1时该名称表示风险约束而非必须卖出。
4. HIKING且TRIGGER，跌幅表单位>0 → CORE_ADD_NO_MARGIN，size_unit使用跌幅表；否则CORE_HOLD_NO_MARGIN且size_unit=0。两者均allow_margin=false、max_gross=1，不改成清仓；追高封锁后单位为0也只持有。
5. HIKING且UNCONFIRMED_CREDIT或RATE_NOT_CREDIT → BUY_CORE_SEMI；负hike_beta且超卖的SLV/JNK/TLT仅给SATELLITE_DIP候选，单名0.02为建议名义/NAV上限，本模块不计算持仓、不检查剩余额度，也不将size_unit转为净值比例。传入核心beta<0时保留动作解释，但新核心size_unit=0，已有核心可持有。
6. HIKING且RISK_ON_DIVERGENCE → INDEX_DIP_BUY。
7. HIKING且yield_stable，未命中以上压力/背离动作 → EARNINGS_SLEEVE，size_unit=0，不发卫星接飞刀，只保留已有核心。
8. 其余HIKING → DIP_ADD，不跌不加；PAUSE → CORE_HOLD，size_unit=0，不新增。
9. EASING且VIX>自身252日80分位 → REENTRY_WATCH，否则HOLD；均无自动买回，size_unit=0。

WATCH只是解释标签；它与NONE得到相同动作和约束，不关闭margin，也不因gross>1强制减仓。只有TRIGGER或CONFIRMED_SELLOFF把allow_margin=false、max_gross=1；其它状态allow_margin=true、max_gross=null。allow_margin在本模块只表示压力约束，PAUSE/EASING的持有和FIRST_CUT的退出动作仍独立生效，不能把true读成自动融资或新增指令。输出包含regime、stress各维度/组合/确认计数、divergence、yield_stable、balance_sheet_stress、hike_beta、constraints、action、size_unit、数值reason、author_levels及数据元数据。

DELEVER_TO_1X、CORE_EXIT_WINDOW、HOLD、EARNINGS_SLEEVE的size_unit恒为0；CORE_HOLD_NO_MARGIN也为0。只有CORE_ADD_NO_MARGIN表达压力下的核心加仓单位。表驱动入口`decide_factors`的`pair_prev`指昨日原始组合是否成立，`trigger_on`指昨日是否已确认TRIGGER；当天输出的`pair`和`trigger_on`可作为下一日输入，保留两天确认、两天解除。完整日线入口仍按历史逐收盘回放，不只看当天布尔。

从仓库根目录运行，政策和敞口必须明确传入（只给日期会报错，不猜账户或政策）：

```sh
python3 -m venv --system-site-packages .venv-macro
.venv-macro/bin/python -m pip install -r macro_regime/requirements.txt
.venv-macro/bin/python -m macro_regime.snapshot --date 2026-09-30 --regime HIKING --gross-exposure 0.8
.venv-macro/bin/python -m pytest -q
```

`--regime`必填且仅接受HIKING/PAUSE/FIRST_CUT/EASING，没有默认值，policy-file也不能替代。也可用`--policy-file FILE.json`提供`{"date":"2026-09-30","gross_exposure":0.8}`，可加nfp_change；日期须严格匹配。如保留policy_regime字段，须与--regime一致。`--core-symbol`切换调用方指定核心，宏观确认仍使用SMH/QQQ；`--nfp-change`仅标签。API入口为`macro_regime.state.evaluate(frame, policy_regime, gross_exposure, ...)`。与旧模型结合时由调用方读取风险约束，不自动修改旧机会分、仓位或历史报告。

# 绝对点位会过期

`config.py`的AUTHOR_LEVELS与STATISTICS分开存：IXG125美元、MOVE125指数点；AVGO334、SLV53.3、BWXT131.8、APP277美元仅展示。每条as_of=2026-09-30。按用户指定有效期：as_of之后63个XNYS交易日内expired=false（不含as_of当天，第1至63日均有效），第64个交易日起expired=true，作者组合永久关闭；假期和周末不推进，历史数据行数也不代替交易日历。valid_sessions=63，输出age_sessions供对账；不是通过回测优化的参数。

过期或未到as_of时作者组合恒false，过期只发告警、不删除展示点位、不影响统计组合。MOVE代理时整个作者组合恒false；MOVE绝对距离为null，不能拿代理波动率减125。其他点位输出Close、distance=Close-level、distance_pct=Close/level-1、expired和as_of；展示价未知留null并告警，不伪造距离。加仓参考点位没有接入状态机。
