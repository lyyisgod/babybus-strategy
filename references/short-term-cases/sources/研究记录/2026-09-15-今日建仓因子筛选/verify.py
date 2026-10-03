"""Small numerical and input-boundary checks; no claim of investment efficacy."""
import ast,json,math
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
tree=ast.parse((R/'model.py').read_text())
ns={'np':np,'pd':pd,'math':math}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['rsi','enrich','logistic','predict','auc']],type_ignores=[]),'numerical_functions','exec'),ns)
assert ns['rsi'](np.arange(1,31))[-1]==100
assert ns['rsi'](np.arange(30,0,-1))[-1]==0
assert ns['rsi'](np.ones(30))[-1]==50
# Published-style Wilder example: initial averages imply RSI=70.464135...
prices=[44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,45.89,46.03,45.61,46.28,46.28]
assert abs(ns['rsi'](prices)[-1]-70.4641350211)<1e-8
assert ns['auc'](np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))==1
assert ns['auc'](np.array([0,0,1,1]),np.array([.5,.5,.5,.5]))==.5
X=np.linspace(-2,2,200)[:,None];y=(X[:,0]>0).astype(float)
model=ns['logistic'](X,y);pr=ns['predict'](model,np.array([[-2],[0],[2]]))
assert pr[0]<.1 and abs(pr[1]-.5)<1e-6 and pr[2]>.9
assert np.all(np.isfinite(ns['predict'](ns['logistic'](np.ones((100,1)),np.tile([0,1],50)),np.ones((2,1)))))
# Do not mistake nontrading timestamps or a partial current day for completed signals.
results=json.loads((R/'model-results.json').read_text())
assert all(x['date']=='2026-09-14' for x in results['predictions'])
assert all(v['validation']['calibrated_current_probability'] is False for v in results['models'].values())
snap=json.loads((R/'snapshot.json').read_text());assert len(snap)==117
bad=[];large=[];counts={}
for p in (R/'raw').glob('*-1d.json'):
 d=json.loads(p.read_text())['chart']['result'][0];q=pd.DataFrame(d['indicators']['quote'][0]);a=np.asarray(d['indicators'].get('adjclose',[{}])[0].get('adjclose',q.close),float)
 q=q.dropna();s=d['meta']['symbol'];counts[s]=len(q)
 if ((q.high+0.02<q[['open','close','low']].max(axis=1))|(q.low-0.02>q[['open','close','high']].min(axis=1))).any():bad.append(s)
 assert min(d['timestamp'])<=max(d['timestamp'])
 assert pd.to_datetime(max(d['timestamp']),unit='s',utc=True).strftime('%Y-%m-%d')<='2026-09-15'
 jumps=a[1:]/a[:-1]-1
 if np.nanmax(abs(jumps))>.6:large.append({'symbol':s,'max_abs_daily_adjusted_return':float(np.nanmax(abs(jumps)))})
assert bad==['OXY'],bad
assert 'OXY' not in [x['symbol'] for x in snap]
print(json.dumps({'checks':'passed','numeric_checks':['RSI increasing/decreasing/flat/reference example','AUC perfect/tied','logistic monotonic synthetic/constant feature'],'snapshot_count':len(snap),'quarantined_ohlc_invalid':bad,'historical_large_returns_for_disclosure':large,'rows_min':min(counts.values()),'rows_max':max(counts.values()),'probability_interpretation':'unapproved for current probability; chronological backtest reported separately'},indent=2))
