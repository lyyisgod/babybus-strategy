"""Refresh the previous 38-name protocol without changing acceptance thresholds."""
import importlib.util,json,sys,math
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
import urllib.request,ssl
import certifi
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'2026-09-17-高胜率操作'
sp=importlib.util.spec_from_file_location('old_screen',OLD/'model.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.ROOT=HERE
m.P=json.loads((OLD/'protocol.json').read_text());m.P['date']='2026-09-21'
m.P['scope_limitations']+=' This is a repeated use of previously inspected historical data, not a new untouched test. No claim of independent certification.'
m.TODAY=pd.Timestamp(m.P['date'],tz=m.NY)
OFFLINE='--offline' in sys.argv
(HERE/'raw').mkdir(exist_ok=True)
if not OFFLINE:
    m.save('protocol.json',m.P)
    m.fetch()
m.checks()
if (HERE/'locked-shortlist.json').exists() and (HERE/'development.json').exists():
    previous=json.loads((HERE/'development.json').read_text())
    if not previous['rows'] and previous['errors']:
        (HERE/'locked-shortlist.json').rename(HERE/'failed-no-data-lock.json')
if not (HERE/'locked-shortlist.json').exists():m.discover()
lock_meta=json.loads((HERE/'locked-shortlist.json').read_text())
lock_meta['test_outcomes_not_read']=False
lock_meta['historical_outcomes_previously_inspected']=True
lock_meta['note']='Original historical outcomes were inspected in prior research. This refresh is not an untouched holdout test.'
m.save('locked-shortlist.json',lock_meta)

rows=[]
for s in m.P['universe']:
    try:
        d,meta,ratio=m.load(s);f=m.factors(d);z=f.iloc[-1];p=float(meta['regularMarketPrice']);scale=float(ratio.iloc[-1])
        last=float(d.raw_close.iloc[-1]);atr=float(z.atr/scale);cap=last+.5*atr
        rows.append(dict(symbol=s,price=p,time=datetime.fromtimestamp(meta['regularMarketTime'],m.NY).isoformat(),daily_cutoff=str(d.index[-1].date()),change_pct=100*(p/last-1),cap=math.floor(cap*100)/100,atr=atr,rsi2=float(z.rsi2),sma200=float(z.sma200/scale),ema20=float(z.ema20/scale),ema50=float(z.ema50/scale),r3_pct=float(z.r3*100),r20_pct=float((d.close.iloc[-1]/d.close.iloc[-21]-1)*100),r63_pct=float(z.r63*100),active=[r for r in m.RULES if z[r]],price_ok=p<=cap,entry_band_low=round(last-.5*atr,2),entry_band_note='Lower edge descriptive only; historical rule imposes an upper ceiling only, not this lower edge'))
    except Exception as e:rows.append(dict(symbol=s,error=str(e)))
m.save('current-factors.json',rows)
pd.DataFrame(rows).to_csv(HERE/'current-factors.csv',index=False)

def recent(d,f,rule):
    bars=d[['raw_open','raw_high','raw_low','raw_close']].to_numpy();last=len(d)-1;busy=-1;rows=[];pending=[]
    for i in np.flatnonzero(d.index>=pd.Timestamp('2024-01-01',tz=m.NY)):
        if i<=busy or i+1>last or not f[rule].iloc[i]:continue
        e=i+1;entry=bars[e,3];atr=f.atr.iloc[i]/(d.close.iloc[i]/d.raw_close.iloc[i])
        if entry>bars[i,3]+.5*atr:continue
        path=bars[e+1:min(e+20,last)+1]
        if len(path):j,px,reason=m.exit_trade(path,entry,atr)
        else:j,px,reason=-1,entry,'time'
        row=dict(signal=str(d.index[i].date()),entry_date=str(d.index[e].date()),entry=entry,atr=atr,hold=j+1)
        if reason=='time' and len(path)<20:
            row.update(marked_return=bars[last,3]/entry-1-m.COST,stop=entry-2*atr,target=entry+2*atr);pending.append(row);busy=last
        else:
            ex=e+1+j;row.update(exit_date=str(d.index[ex].date()),exit=px,ret=px/entry-1-m.COST,reason=reason);rows.append(row);busy=ex
    return pd.DataFrame(rows),pending

lock=json.loads((HERE/'locked-shortlist.json').read_text());results=[]
for pick in lock['shortlist']:
    s,rule=pick['symbol'],pick['rule'];d,meta,ratio=m.load(s);f=m.factors(d)
    t,pending=recent(d,f,rule);st=m.stats(t)
    if len(t):
        st['family_adjusted_win_lower']=m.wilson(st['wins'],st['n'],NormalDist().inv_cdf(1-.05/max(1,lock['selection_count'])))[0]
        st['mean_return_block_ci95']=m.block_mean_ci(t.ret.to_numpy())
        st['accepted']=bool(st['n']>=20 and st['win']>=.65 and st['family_adjusted_win_lower']>.5 and st['mean']>0 and st['pf'] is not None and st['pf']>=1.3 and st['win_loss_ratio'] is not None and st['win_loss_ratio']>=.8 and st['mean_return_block_ci95'][0]>0)
        assert (pd.to_datetime(t.entry_date)>pd.to_datetime(t.signal)).all()
        assert (pd.to_datetime(t.exit_date)<=pd.Timestamp('2026-09-18')).all()
        assert (t.entry_date.iloc[1:].to_numpy()>t.exit_date.iloc[:-1].to_numpy()).all()
    else:st['accepted']=False
    st['open_trade_count']=len(pending)
    t.to_csv(HERE/f'completed-{s}-{rule}.csv',index=False)
    base,bpending=recent(d,f,'baseline')
    results.append(dict(symbol=s,rule=rule,test=st,pending=pending,baseline=m.stats(base),baseline_pending=bpending))
m.save('test-results.json',dict(asof=datetime.now(timezone.utc).isoformat(),results=results,decision='BUY' if any(r['test']['accepted'] for r in results) else 'NO_BUY',new_test_independent=False))
print(pd.DataFrame(rows).reindex(columns=['symbol','price','change_pct','rsi2','r3_pct','active','cap','price_ok','error']).to_string(index=False))
print(json.dumps(m.clean(results),ensure_ascii=False,indent=2))
