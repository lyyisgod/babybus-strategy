from pathlib import Path
import importlib.util, json, datetime, hashlib, sys
import numpy as np, pandas as pd

P=Path(__file__).resolve().parent; ROOT=P.parents[1]
spec=importlib.util.spec_from_file_location('prior',ROOT/'研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from strategies.bb_washout_bounce.data import clean
proto=json.loads((P/'protocol.json').read_text()); now=pd.Timestamp.now(tz='UTC')
market,meta,errors,last=m.load_market(P/'daily-raw',m.CFG,now.isoformat());cal=m.calendar(m.CFG)
assert str(last.date())=='2026-10-01'
save=m.save
live_manifest={r['symbol']:r for r in json.loads((P/'live-raw/manifest.json').read_text())}

def intraday(s):
    raw=json.loads((P/'live-raw'/f'{s.replace("^","INDEX_").replace("=","_")}.json').read_text())['chart']['result'][0]
    mt=raw['meta']; d=pd.DataFrame(raw['indicators']['quote'][0],index=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York'))
    snapshot_at=min(now,pd.Timestamp(live_manifest[s]['fetched_at']))
    d=d.dropna(subset=['close']); d=d[d.index+pd.Timedelta(minutes=5)<=snapshot_at]
    reg=d[(d.index.hour*60+d.index.minute>=570)&(d.index.hour*60+d.index.minute<960)]
    cur=reg[reg.index.strftime('%Y-%m-%d')==proto['today']]
    assert len(cur)>0,'no completed current regular bars'
    qt=pd.Timestamp(mt['regularMarketTime'],unit='s',tz='UTC');assert qt<=now
    prev=float(market[s].close.iloc[-1]);price=float(mt['regularMarketPrice'])
    v=cur.volume.fillna(0);vwap=float((((cur.high+cur.low+cur.close)/3)*v).sum()/v.sum()) if v.sum()>0 else None
    return dict(price=price,quote_at=qt.tz_convert('America/New_York'),fetched_at=now,prior_close=prev,change_pct=(price/prev-1)*100,open=float(cur.open.iloc[0]),gap_pct=(cur.open.iloc[0]/prev-1)*100,completed_5m_vwap=vwap,above_vwap=price>vwap if vwap else None,return_from_open_pct=(price/cur.open.iloc[0]-1)*100,high_completed=float(cur.high.max()),low_completed=float(cur.low.min()),completed_bar_end=cur.index[-1]+pd.Timedelta(minutes=5),volume_completed=float(v.sum()),bid_ask=None,rvol20_same_clock=None,source=f'https://finance.yahoo.com/quote/{s}/',intraday_not_fitted=True)

rows=[];factors={};ferrs={};liveproxies={}
for s in proto['stocks']:
    try:
        d=market[s];assert d.index[-1]==last
        f=m.light(d,market['SMH'],cal);factors[s]=f;r=f.iloc[-1].to_dict()
        r.update(ticker=s,session=last,price_gates=m.gates(r),prior_high=float(d.high.iloc[-1]),prior_low=float(d.low.iloc[-1]),prior_open=float(d.open.iloc[-1]))
        r['gate_count']=sum(r['price_gates'].values());r['live']=intraday(s);r['daily_signal_complete']=True;rows.append(r)
    except Exception as e:ferrs[s]=str(e)
for s in proto['proxies']+['IXG','^MOVE','NQ=F']:
    try:liveproxies[s]=intraday(s)
    except Exception as e:liveproxies[s]=dict(error=str(e))
save(P/'all-factors.json',rows);save(P/'factor-errors.json',ferrs);save(P/'live-proxies.json',liveproxies)
eligible=[r for r in rows if r['adv20']>=50e6]
wash=sorted([r for r in eligible if r['dd_high']>=.35],key=lambda r:(r['gate_count'],bool(r['live']['above_vwap']),r['vol_ratio20']),reverse=True)
trend=sorted([r for r in eligible if r['close']>r['ma200'] and r['live']['above_vwap']],key=lambda r:(r['vol_ratio20'],r['relative20']),reverse=True)
save(P/'washout-ranking.json',wash);save(P/'momentum-ranking.json',trend);save(P/'lrcx-branch.json',[r for r in eligible if r['lrcx_branch']])
securities=json.loads((ROOT/'config/bb_universe.json').read_text())['securities'];inputs={k:m.read_records(ROOT/'data/bb_washout_bounce'/(k+'.jsonl')) for k in ['fundamentals','macro','events','event_coverage','supports','conflicts']}
e=m.Engine(market,securities,inputs,m.CFG);save(P/'strict-summary.json',m.daily_report(e,last,P/'strict-original',now=now.isoformat()))
j=market['JNK'].adjclose;mon=j.resample('ME').last();mon=mon[mon.index<=last];w=j.resample('W-FRI').last();w=w[w.index<=last]
credit=dict(as_of=now,session=last,jnk_ret20=j.iloc[-1]/j.iloc[-21]-1,jnk_ret5=j.iloc[-1]/j.iloc[-6]-1,completed_month=mon.index[-1],bear_month=bool(mon.tail(3).mean()<mon.tail(10).mean()),completed_week=w.index[-1],weekly_break=bool(w.iloc[-1]<w.iloc[-9:-1].min()*.995),credit_state='UNKNOWN',nfp_protection_through='2026-10-02 close',current_nfp=29000,nfp_release='2026-10-02T08:30:00-04:00',nfp_source='https://www.bls.gov/news.release/empsit.nr0.htm')
for s in ['BAMLH0A0HYM2','DFII10']:
    path=P/f'{s}.csv'
    try:
        a=pd.read_csv(path);a.index=pd.to_datetime(a.iloc[:,0]);v=pd.to_numeric(a[s],errors='coerce').dropna().loc[:last];date=v.index[-1];base=cal.session_offset(cal.date_to_session(date,direction='previous'),-20);prior=v.loc[:base]
        credit[s]=dict(observed_date=date,value=v.iloc[-1],base_date=prior.index[-1],base_value=prior.iloc[-1],change20_bp=(v.iloc[-1]-prior.iloc[-1])*100,source=f'https://fred.stlouisfed.org/series/{s}',fetched_at=now,current_download_not_vintage=True)
    except Exception as ex:credit[s]=dict(error=str(ex))
if credit['weekly_break'] or credit['bear_month'] or credit.get('BAMLH0A0HYM2',{}).get('change20_bp',-999)>=50:credit['credit_state']='OFF'
save(P/'credit-context.json',credit)
probes={}
for s in ['IREN','APLD','CIFR','COHR']:
    d=market[s];cut=d.index[-25];tr=m.light(d.loc[:cut],market['SMH'].loc[:cut],cal)
    cols=['dd_high','dd10','rv90','weekly_mid','weekly_vol','ma200','rsi2','atr14','vol_ratio20'];probes[s]=bool(np.allclose(tr.loc[cut,cols].astype(float),factors[s].loc[cut,cols].astype(float),equal_nan=True))
hashchecks={}
for folder in ['daily-raw','live-raw']:
    mf=json.loads((P/folder/'manifest.json').read_text());hashchecks[folder]=all(hashlib.sha256((P/folder/a['file']).read_bytes()).hexdigest()==a['sha256'] for a in mf if 'file' in a)
assert all(probes.values()) and all(hashchecks.values());assert cal.next_session(pd.Timestamp('2026-10-02'))==pd.Timestamp('2026-10-05')
save(P/'verification.json',dict(as_of=now,stocks=len(rows),requested=len(proto['stocks']),ferrs=ferrs,daily_loader_errors=errors,last_completed=last,next_session='2026-10-05',truncation_checks=probes,raw_hashes=hashchecks,live_partial_bars_excluded=True,probability_validation=False))
for label,rs in [('washout',wash),('momentum',trend),('lrcx',[r for r in rows if r['lrcx_branch']])]:
    print(label)
    for r in rs[:15]:print(json.dumps(clean(dict(ticker=r['ticker'],gates=r['gate_count'],dd=r['dd_high'],dd10=r['dd10'],rv=r['rv90'],wv=r['weekly_vol'],rsi2=r['rsi2'],vol=r['vol_ratio20'],close=r['close'],atr=r['atr14'],live=r['live'])),ensure_ascii=False))
print('CREDIT',json.dumps(clean(credit),ensure_ascii=False))
