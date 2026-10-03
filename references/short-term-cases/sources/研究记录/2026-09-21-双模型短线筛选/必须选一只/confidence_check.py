"""Conditional model uncertainty; fixed previous features/regularization, no retuning."""
from pathlib import Path
import importlib.util,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sp=importlib.util.spec_from_file_location('dual',BASE/'dual_model.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
NBOOT=120;SEED=9212026
protocol={'asof':datetime.now(timezone.utc).isoformat(),'objective':'Forced one-name comparison of frozen next-day +5% target, not reversal of original no-buy decision','bootstrap_replicates':NBOOT,'seed':SEED,'resampling':'Calendar-month blocks, all stocks jointly, training <=2022 only; same fixed train feature scaling and ridge 0.01','comparison':'All 49 current stocks. No changes to model labels, weights or shortlist based on bootstrap output.','limitations':'Conditional fitting uncertainty only: excludes calibration error, new information, model misspecification and survivor universe. First-place frequency is not trade success probability.'}
(ROOT/'confidence-protocol.json').write_text(json.dumps(protocol,indent=2))
coef=json.loads((BASE/'model-coefficients.json').read_text());cols=coef['features'];mu=np.array(coef['train_mean']);sd=np.array(coef['train_sd'])
ds={s:m.m.load(s)[0] for s in m.SYMBOLS};pan=[];cur=[]
for s in m.STOCKS:
 d=ds[s];f=m.factors(s,d,ds);f['r2']=d.raw_close.shift(-2)/d.raw_open.shift(-1)-1-.002;f['exit3']=pd.Series(d.index,index=d.index).shift(-3);f['date']=d.index;f['symbol']=s
 cur.append(f.iloc[-1]);pan.append(f.dropna(subset=cols+['r2','exit3']))
p=pd.concat(pan);tr=p[(p.date.dt.year<=2022)&(p.exit3.dt.year<=2022)];cc=pd.DataFrame(cur)
def xmat(d):return np.c_[np.ones(len(d)),np.clip((d[cols].to_numpy()-mu)/sd,-8,8)]
x=xmat(tr);cx=xmat(cc);y=(tr.r2>=.05).to_numpy(float);initial=np.array(coef['logistic'][0]);pred=lambda b,xx:1/(1+np.exp(-np.clip(xx@b,-30,30)))
rank=pd.read_csv(BASE/'ranking.csv').set_index('symbol');original=pred(initial,cx)
assert np.max(np.abs(original-np.array([rank.loc[s,'p_next5'] for s in cc.symbol])))<1e-6
codes,months=pd.factorize(tr.date.dt.strftime('%Y-%m'));g=len(months);rng=np.random.default_rng(SEED)
pen=np.eye(x.shape[1])*.01;pen[0,0]=0
out=[]
for rep in range(NBOOT):
 counts=np.bincount(rng.integers(g,size=g),minlength=g);wt=counts[codes].astype(float);norm=wt.sum();b=initial.copy()
 for step in range(15):
  pr=pred(b,x);grad=x.T@(wt*(pr-y))/norm+pen@b;h=(x.T*(wt*pr*(1-pr)))@x/norm+pen
  delta=np.linalg.solve(h,grad);b-=delta
  if np.max(np.abs(delta))<1e-7:break
 assert np.max(np.abs(delta))<1e-5
 out.append(pred(b,cx))
 if (rep+1)%30==0:print('bootstrap',rep+1,flush=True)
out=np.array(out);win=np.argmax(out,axis=1);rows=[];leader=int(np.argmax(original));runner=int(np.argsort(original)[-2])
for i,s in enumerate(cc.symbol):
 rows.append({'symbol':s,'point_output':original[i],'conditional_bootstrap95':np.quantile(out[:,i],[.025,.975]).tolist(),'first_place_fraction':float((win==i).mean())})
rows.sort(key=lambda a:a['point_output'],reverse=True)
diff=out[:,leader]-out[:,runner]
res={'protocol':protocol,'train_rows':len(tr),'month_blocks':g,'rows':rows,'leader_vs_runner_difference95':np.quantile(diff,[.025,.975]).tolist(),'leader_vs_runner_fraction':float((diff>0).mean()),'checks':'original point outputs reproduced; all 120 weighted fits converged; no current outcomes used'}
(ROOT/'confidence-results.json').write_text(json.dumps(res,indent=2));print(json.dumps({**{k:v for k,v in res.items() if k not in ['rows','protocol']},'top_rows':rows[:6]},indent=2))
