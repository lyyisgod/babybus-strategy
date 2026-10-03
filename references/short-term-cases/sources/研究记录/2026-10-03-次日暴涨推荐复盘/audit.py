"""离线复核已发布候选；不重新训练、不生成买单或当前上涨概率。"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NY = ZoneInfo('America/New_York')
AS_OF = datetime.fromisoformat('2026-10-03T08:10:08-04:00')
DAILY = ROOT / '研究记录/2026-10-02-Babybus今日至下个交易日/daily-raw'
AUDIT_ID = '2026-10-03-published-next-session-picks-audit'


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def calc(reference, bar):
    if reference is None:
        return None
    assert reference > 0
    high, close = bar['high'], bar['close']
    result = {'high_return_pct': 100 * (high / reference - 1),
              'close_return_pct': 100 * (close / reference - 1)}
    for threshold in (5, 10):
        # 直接比较价格，避免先四舍五入收益率造成边界误判。
        result[f'touch{threshold}'] = high >= reference * (1 + threshold / 100) - 1e-10
        result[f'close{threshold}'] = close >= reference * (1 + threshold / 100) - 1e-10
    return result


def summarize(rows):
    n = len(rows)
    result = {'n': n}
    for field in ('touch5', 'touch10', 'close5', 'close10'):
        hits = sum(r['reference_outcome'][field] for r in rows)
        result[field] = {'hits': hits, 'misses': n - hits,
                         'rate_pct': 100 * hits / n if n else None}
    return result


def main():
    import exchange_calendars as xc
    inputs = read_json(HERE / '案例输入.json')
    web = read_json(HERE / '已收盘行情核验.json')
    web_index = {(r['symbol'], r['session']): r for r in web}
    calendar = xc.get_calendar('XNYS')
    session_dates = calendar.sessions_in_range('2026-09-01', '2026-10-06').strftime('%Y-%m-%d').tolist()
    manifests, used_paths = [], set()

    def use(path):
        if path not in used_paths:
            manifests.append(fingerprint(path))
            used_paths.add(path)

    def daily_bar(symbol, session):
        override = web_index.get((symbol, session))
        if override:
            use(HERE / '已收盘行情核验.json')
            return dict(override)
        path = DAILY / f'{symbol}.json'
        if not path.exists():
            return None
        data = read_json(path)['chart']['result'][0]
        quote = data['indicators']['quote'][0]
        for i, timestamp in enumerate(data['timestamp']):
            if datetime.fromtimestamp(timestamp, NY).date().isoformat() != session:
                continue
            # 本地抓取发生在10/2盘中，因此它的10/2日线禁止作完整结果。
            if session >= '2026-10-02':
                return None
            bar = {field: round(quote[field][i], 2) for field in ('open', 'high', 'low', 'close')}
            bar.update(symbol=symbol, session=session, complete=True,
                       source_url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
                       source_file=str(path.relative_to(ROOT)), provider='Yahoo Chart archive',
                       archive_market_time=data['meta'].get('regularMarketTime'))
            use(path)
            return bar
        return None

    results, primary_keys = [], set()
    for original in inputs:
        r = dict(original)
        source_paths = [r['source_report']] + [v['source_report'] for v in r.get('versions', [])]
        for source in source_paths:
            path = ROOT / source
            assert path.exists(), source
            use(path)
        target = r['target_session']
        r['target_session_complete'] = False
        r['reference_outcome'] = None
        if target is None:
            r['audit_status'] = 'original_not_located'
            results.append(r)
            continue
        assert target in session_dates, target
        # 各报告的目标已逐条人工确认；凌晨报告指即将开盘的同日常规盘。
        decision = r['decision_date']
        if decision:
            expected = next(d for d in session_dates if d >= decision) if target == decision else next(d for d in session_dates if d > decision)
            assert target == expected, (decision, target, expected)
        if datetime.fromisoformat(target + 'T16:00:00-04:00') > AS_OF:
            r['audit_status'] = 'pending'
            results.append(r)
            continue
        bar = daily_bar(r['symbol'], target)
        if bar is None:
            r['audit_status'] = 'missing_outcome_price'
            results.append(r)
            continue
        assert bar['low'] <= min(bar['open'], bar['close']) <= max(bar['open'], bar['close']) <= bar['high']
        r['target_session_complete'] = True
        r['outcome_bar'] = bar
        r['reference_outcome'] = calc(r['reference_price'], bar)
        previous = session_dates[session_dates.index(target) - 1]
        prev_bar = daily_bar(r['symbol'], previous)
        if prev_bar:
            r['previous_session_close'] = prev_bar['close']
            r['target_day_outcome_vs_previous_close'] = calc(prev_bar['close'], bar)
        r['entry_cap_sensitivity'] = calc(r.get('entry_cap'), bar)
        r['version_sensitivities'] = [dict(v, outcome=calc(v['reference_price'], bar))
                                      for v in r.get('versions', [])]
        role = r['selection_role']
        if role == 'primary':
            key = (r['symbol'], target)
            assert key not in primary_keys, key
            primary_keys.add(key)
            r['audit_status'] = 'matured_primary_selection'
        else:
            r['audit_status'] = role
        results.append(r)

    primary = [r for r in results if r['audit_status'] == 'matured_primary_selection']
    secondary = [r for r in results if r['selection_role'] == 'secondary' and r['reference_outcome']]
    assert len(primary) == 16, len(primary)
    assert len([r for r in results if r['audit_status'] == 'pending']) == 2
    assert calc(100, {'high': 110, 'close': 105})['touch5']
    assert not calc(100, {'high': 104.999, 'close': 100})['touch5']
    assert calc(None, {'high': 105, 'close': 100}) is None
    # 独立按阈值价重数，与汇总器结果对照。
    summary = summarize(primary)
    for threshold in (5, 10):
        assert summary[f'touch{threshold}']['hits'] == sum(
            r['outcome_bar']['high'] >= r['reference_price'] * (1 + threshold / 100) for r in primary)
    assert all(r['outcome_bar']['complete'] for r in primary)
    assert len(secondary) == 15
    use(HERE / '案例输入.json')
    use(HERE / 'audit.py')
    payload = {
        'audit_id': AUDIT_ID, 'as_of': AS_OF.isoformat(),
        'scope': '本地已保存的美股隔夜/下一常规盘相对首选，非全部未保存对话；备选、不同持有期和缺证案例分列。',
        'protocol': {
            'unit': '同一股票、同一目标常规交易日合并；不同股票/目标交易日各计一次',
            'selection': '原文显式综合/单一/量价首选；包括未过绝对买入门槛的相对首选，不据结果删除失败',
            'reference': '该股票成为该目标盘首选时的第一份已定位报告报价；保留后续报价及假设限价敏感度',
            'IREN_reference': '下午40.18为形态首选、夜盘40.81为综合首选；主表采用后者，前者另列',
            'thresholds_pct': [5, 10], 'formula': '目标常规盘high或close / 原推荐报价 - 1',
            'excluded_from_main_rate': ['仅备选观察篮子', '旧收盘/日期不明且未取得当晚报价', '10–20日等不同期限', '当时主要劝等待的错失机会', '目标盘未到期', '原始推荐未定位'],
            'execution': '不推定已成交、止盈先于止损、已扣费用或实际赚到；本统计不验证策略未来胜率',
            'market_data': '10/1及以前用保存Yahoo完整日表；10/2必须用本次浏览已收盘日表，不用昨天盘中日K',
        },
        'summary_primary': summary, 'summary_secondary': summarize(secondary),
        'summary_primary_plus_secondary': summarize(primary + secondary),
        'coverage': {'discovery_shortlist_documents': len(read_json(HERE / '检索文档索引.json')),
                     'recorded_cases': len(results), 'matured_primary': len(primary),
                     'pending_primary': 2, 'secondary': len(secondary),
                     'missing_entry_quote': sum(r['selection_role'] == 'missing_entry_quote' for r in results),
                     'original_not_located': sum(r['selection_role'] == 'original_not_located' for r in results),
                     'limits': '检索本地资料不能保证涵盖所有历史对话；没有逐笔买卖成交记录。'},
        'cases': results, 'input_manifest': manifests,
        'validation': {'duplicate_primary_keys': False, 'calendar_dates_verified': True,
                       'all_primary_bars_complete': True, 'threshold_boundary_checks': True,
                       'independent_hit_count_agrees': True, 'unknown_execution_not_imputed': True},
    }
    (HERE / '命中登记.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (HERE / '输入SHA256.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + '\n')

    lines = ['次日暴涨推荐复盘｜2026-10-03',
             '统计截止：2026-10-02常规盘收盘。范围：本地已保存、可定位原文的美股推荐。',
             '主表统计相对首选的事后价格表现，包括明确“不立即买”的相对首选；不是实际交易胜率。',
             '同股同目标交易日的重复推荐合并；从首次首选报价算，不挑后来的更低报价。',
             f"已成熟首选{summary['n']}次：盘中达到+5% {summary['touch5']['hits']}次（{summary['touch5']['rate_pct']:.2f}%）；达到+10% {summary['touch10']['hits']}次（{summary['touch10']['rate_pct']:.2f}%）。",
             f"次日收盘相对原报价仍≥+5%：{summary['close5']['hits']}/{summary['n']}（{summary['close5']['rate_pct']:.2f}%）；仍≥+10%：{summary['close10']['hits']}/{summary['n']}。",
             '计算：涨幅=目标常规盘最高价或收盘价÷原推荐参考价−1；毛价格变动，未扣费用。',
             '原推荐与结果逐条如下（美元）。', '',
             '推荐日 → 目标盘 | 股票 | 原报价 | 次日最高 | 高点涨幅 | 次日收盘 | 收盘涨幅 | +5%/+10%']
    for r in primary:
        q, b = r['reference_outcome'], r['outcome_bar']
        lines.append(f"{r['decision_date'][5:]}→{r['target_session'][5:]} | {r['symbol']} | {r['reference_price']:.4f} | {b['high']:.2f} | {q['high_return_pct']:+.2f}% | {b['close']:.2f} | {q['close_return_pct']:+.2f}% | {'中' if q['touch5'] else '未中'}/{'中' if q['touch10'] else '未中'}")
    lines.extend(['', '重要口径与原建议：',
        '1. 上次75%=3/4，仅覆盖四条Babybus首选；这次扩大到16条可复核首选，不能继续把75%当整体命中率。',
        '2. VICR 9/21盘前228.56→9/22高268.65为+17.54%；当晚252→同一高点仅+6.61%，不是晚间买入也涨17.54%。用户明确没买VICR。',
        '3. ALAB 9/21首选340.32→高369.15为+8.47%；9/22盘前刷新334.75→高为+10.28%，但同一目标盘不重复计数，也不把更低刷新价替换原报价。',
        '4. APLD 10/1报价24.235→10/2高26.85为+10.79%；若按原假设成交24.50，只有+9.59%，不满足+10%。收盘25.38相对原报价+4.72%，未保持+5%。',
        '5. IREN夜盘40.81→高43.91为+7.60%；下午形态参考40.18→高为+9.28%。收盘41.76相对夜盘报价+2.33%，未保持+5%。',
        '6. 9/17与9/28凌晨报告说的“下一常规盘”是当日尚未开盘的美股常规盘，不能错算成下一日。周末、劳动节均按交易日历。',
        '7. 日线高点触及不保证卖到，也不证明入场确认满足、止盈早于止损。推荐是否说中和用户实际赚钱分开保存。',
        '8. 原文只列备选的WDC、ALAB、VICR等，即使涨到目标也不偷换成原综合首选；未涨到的AAOI等同样保留。', '',
        '未到期：10/2首选APLD、AAOI，目标10/5；不计成功/失败分母。APLD周五早盘旧计划当日已失效，这项与周一价格结果分别记录。',
        '缺证：9/7 AMD仅9/4收盘参考；MU缺当晚可靠买价；WDAY原始推荐未定位。三项均不编造胜负。',
        '不同持有期：LRCX 9/17原方案最长20日，虽9/18价格上升，本统计另存价格旁证，不算隔夜首选命中。',
        '错失机会：AAOI 9/30在97.781附近的原建议以等待为主，10/1大涨应记错失，不能归为我的成功推荐。', '',
        f"备选/观察篮子另有{len(secondary)}条成熟记录，+5%触及{payload['summary_secondary']['touch5']['hits']}条；与首选合并的宽口径为{payload['summary_primary_plus_secondary']['touch5']['hits']}/{payload['summary_primary_plus_secondary']['n']}，仅作敏感度，不能替代首选指标。",
        '其逐项价格、条件、各次刷新、来源、原报告路径和SHA256均保存在命中登记.json。', '',
        '主表每条的原建议与来源：'])
    for r in primary:
        lines.extend([f"{r['case_id']}：{r['original_advice']}", '原报告：' + r['source_report']])
    lines.extend(['', '补充记录：'])
    for r in results:
        if r in primary:
            continue
        q = r['reference_outcome']
        perf = f"；目标盘高点相对参考{q['high_return_pct']:+.2f}%，收盘{q['close_return_pct']:+.2f}%" if q else ''
        lines.append(f"{r['case_id']} [{r['audit_status']}] {r['original_advice']}{perf}")
    lines.extend(['', '行情来源：',
        'Yahoo Chart原始日表：研究记录/2026-10-02-Babybus今日至下个交易日/daily-raw/；仅采用10/1及以前完整日。',
        '10/2完整结果及SCHL由2026-10-03本次浏览StockAnalysis历史日表核验，网站标注S&P Global来源。',
        *[r['source_url'] for r in web],
        '验证：目标交易日、去重、完整日排除、阈值边界及独立重数通过；输入原文/行情哈希已保存。',
        '限制：只覆盖本地可定位的已保存记录，不能声称覆盖所有历史聊天；样本少且相互相关，结果不代表未来命中概率。'])
    (HERE / '复盘结果.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'summary': summary, 'secondary': payload['summary_secondary'],
                      'recorded_cases': len(results), 'sources': len(manifests)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
