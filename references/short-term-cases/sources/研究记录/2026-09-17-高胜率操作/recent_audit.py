"""Chronological endpoint audit: retain early resolved recent trades and censor open ones.
No new symbols, parameters, or replacement shortlist. Original mature-window test retained.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
from statistics import NormalDist
p=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('core',p/'model.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
lock=json.loads((p/'locked-shortlist.json').read_text());out=[]
for pick in lock['shortlist']:
    s=pick['symbol'];rule=pick['rule'];d,meta,ratio=m.load(s);f=m.factors(d)
    bars=d[['raw_open','raw_high','raw_low','raw_close']].to_numpy();last=len(d)-1;busy=-1;rows=[];pending=[]
    for i in np.flatnonzero(d.index>=pd.Timestamp('2024-01-01',tz=m.NY)):
        if i<=busy or i+1>last or not f[rule].iloc[i]:continue
        e=i+1;entry=bars[e,3];atr=f.atr.iloc[i]/ratio.iloc[i]
        if entry>bars[i,3]+.5*atr:continue
        path=bars[e+1:min(e+20,last)+1]
        if len(path):j,px,reason=m.exit_trade(path,entry,atr)
        else:j,px,reason=-1,entry,'time'
        row=dict(symbol=s,signal=str(d.index[i].date()),entry_date=str(d.index[e].date()),entry=entry,atr=atr,hold=j+1)
        if reason=='time' and len(path)<20:
            row.update(asof=str(d.index[last].date()),marked_return=bars[last,3]/entry-1-m.COST,stop=entry-2*atr,target=entry+2*atr)
            pending.append(row);busy=last
        else:
            ex=e+1+j;row.update(exit_date=str(d.index[ex].date()),exit=px,ret=px/entry-1-m.COST,reason=reason)
            rows.append(row);busy=ex
    t=pd.DataFrame(rows);st=m.stats(t);z=NormalDist().inv_cdf(1-.05/max(1,lock['selection_count']))
    st['family_adjusted_win_lower']=m.wilson(st['wins'],st['n'],z)[0]
    st['mean_return_block_ci95']=m.block_mean_ci(t.ret.to_numpy())
    st['open_trade_count']=len(pending)
    st['win_fraction_if_open_all_lose']=st['wins']/(st['n']+len(pending))
    st['accepted']=bool(st['n']>=20 and st['win']>=.65 and st['family_adjusted_win_lower']>.5 and st['mean']>0 and st['pf']>=1.3 and st['win_loss_ratio']>=.8 and st['mean_return_block_ci95'][0]>0)
    st['last5']=m.stats(t.tail(5))
    t.to_csv(p/f'recent-completed-{s}.csv',index=False)
    out.append(dict(symbol=s,completed=st,pending=pending,recent_closed=rows[-5:]))
m.save('recent-audit.json',dict(purpose='Endpoint censoring audit; not re-optimization. Open trades not counted as successes.',results=out))
print(json.dumps(m.clean(out),ensure_ascii=False,indent=2))
