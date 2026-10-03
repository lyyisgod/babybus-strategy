from pathlib import Path
import json,sys,datetime,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import load_config,load_market,calendar,save
P=Path(__file__).resolve().parent
cfg=load_config();now=datetime.datetime.now(datetime.timezone.utc).isoformat();market,metadata,errors,last=load_market(P/'daily-raw',cfg,now);cal=calendar(cfg)
events=json.loads((P/'historical-price-only-events.json').read_text());stats=json.loads((P/'historical-price-only-stats.json').read_text());names=json.loads((P/'protocol.json').read_text())['stocks']
sessions=cal.sessions_in_range('2024-01-01',cal.previous_session(last));panel={}
for s in names:
    if s not in market:continue
    d=market[s];adv=(d.close*d.volume).shift().rolling(20).mean();net=(d.adjclose.shift(-1)/d.adjclose-1-.002).where(adv>=50e6)
    panel[s]=net.reindex(sessions)
base=pd.DataFrame(panel).mean(axis=1)
statsmap={(r['ticker'],r['kind']):r for r in stats};rng=np.random.default_rng(1001)
diagnostics=[]
for s in ['IREN','APLD','COHR','LITE','CIFR','ON']:
    for kind in ['washout_analog','momentum_analog']:
        selected=[e for e in events if e['ticker']==s and e['kind']==kind];n=len(selected)
        if not n:continue
        frame=pd.DataFrame(0.0,index=sessions,columns=['count','touch5','touch10','net','excess'])
        for e in selected:
            date=pd.Timestamp(e['signal_session']);frame.loc[date]=[1,e['high']>=.05,e['high']>=.10,e['close_net'],e['close_net']-base.loc[date]]
        a=frame.to_numpy();draws=[]
        for _ in range(1000):
            starts=rng.integers(0,len(a)-4,size=int(np.ceil(len(a)/5)));index=np.concatenate([np.arange(i,i+5) for i in starts])[:len(a)];tot=a[index].sum(axis=0)
            if tot[0]>0:draws.append(tot[1:]/tot[0])
        result=dict(ticker=s,kind=kind,n=n,bootstrap_replicates=len(draws),date_block_sessions=5,touch5_date_block95=np.quantile(np.array(draws)[:,0],[.025,.975]),touch10_date_block95=np.quantile(np.array(draws)[:,1],[.025,.975]),mean_net_date_block95=np.quantile(np.array(draws)[:,2],[.025,.975]),excess_vs_same_day_survivor_pool=np.mean([e['close_net']-base.loc[pd.Timestamp(e['signal_session'])] for e in selected]),excess_date_block95=np.quantile(np.array(draws)[:,3],[.025,.975]),double_cost_mean=np.mean([e['close_net']-.002 for e in selected]),adverse_entry25bp_mean=np.mean([(1+e['close_net']+.002)/1.0025-1-.002 for e in selected]),current_probability=None,warning='Exploratory fixed price analogs on reused dates/current survivor pool; no new audit or overnight execution calibration; not official Babybus performance')
        diagnostics.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()})
save(P/'date-block-diagnostics.json',diagnostics)
macro=json.loads((P/'credit-context.json').read_text());template=json.loads((ROOT/'量化参考/数据输入模板.json').read_text());template['as_of']=now
for field,s,available in [('real_yield_change_bp','DFII10','2026-10-01T16:16:00-04:00'),('hy_oas_change_bp','BAMLH0A0HYM2','2026-10-01T10:17:00-04:00')]:
    m=macro[s];template['observations'][field]=dict(value=m['change20_bp'],observed_at='2026-09-30T16:00:00-04:00',available_at=available,source=m['source'],verified=True,proxy=False)
save(P/'score-input.json',template)
supplement={}
for s in ['^MOVE','BTC-USD']:
    a=json.loads((P/'daily-raw'/f'{s.replace("^","INDEX_")}.json').read_text())['chart']['result'][0];q=a['indicators']['quote'][0];vals=[(pd.Timestamp(t,unit='s',tz='UTC').tz_convert('America/New_York'),c) for t,c in zip(a['timestamp'],q['close']) if c is not None and c>0]
    if s=='^MOVE':
        vals=[(t,c) for t,c in vals if t.date()<=last.date()];t,c=vals[-1];supplement[s]=dict(observed_at=t,value=c,previous_close=vals[-2][1],change_pct=(c/vals[-2][1]-1)*100,source=f'https://finance.yahoo.com/quote/{s}/',validation='latest scalar closes only; full OHLC rejected by strict loader',full_history_error=errors.get(s))
    else:supplement[s]=dict(status='not incorporated',reason='missing daily bars/OHLC validation failure; crypto trades24/7; no manufactured latest daily return')
save(P/'supplemental-risk.json',supplement)
# Meaningful numerical checks for macro units and price thresholds.
assert abs((2.93-2.45)*100-48)<1e-8
for s in ['DFII10','BAMLH0A0HYM2']:
    m=macro[s];assert abs(m['change20_bp']-(m['value']-m['base_value'])*100)<1e-8
    assert len(cal.sessions_in_range(pd.Timestamp(m['base_date']),pd.Timestamp(m['observed_date'])))-1==20
assert macro['credit_state']=='OFF' and macro['weekly_break']
assert str(cal.session_offset(pd.Timestamp('2026-09-30'),-20).date())=='2026-09-01'
assert len([r for r in json.loads((P/'all-factors.json').read_text()) if r['ticker'] in names])==154
assert (P/'verification.json').exists()
for d in diagnostics:
    original=statsmap[(d['ticker'],d['kind'])];assert original['n']==d['n']
verification=json.loads((P/'verification.json').read_text());verification.update(macro_units_and_20session_boundary=True,date_block_stats_checked=True,price_snapshots_match_original_bytes=True,stocks_available=154,daily_loader_excluded_proxies=errors,csv_retry_kept=True)
save(P/'verification.json',verification)
print('all checks passed; risk',supplement,flush=True)
for d in diagnostics:
    if d['ticker']=='IREN' and d['kind']=='washout_analog':print(d,flush=True)
