"""Numerical checks and forward-information audit; not profit validation."""
import importlib.util,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('model',R/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
X=np.linspace(-2,2,200)[:,None];y=(X[:,0]>0).astype(int);p=m.pred(m.fit(X,y),np.array([[-2],[0],[2]]))
assert p[0]<p[1]<p[2] and abs(p[1]-.5)<1e-6
assert m.auc(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))==1
assert m.auc(np.array([0,0,1,1]),np.ones(4))==.5
assert abs(m.wilson([0,1])['frequency']-.5)<1e-12
assert m.wilson([0]*20)['wilson95'][0]==0
assert len([d for d in m.CAL if '2021-01-01'<=d<'2026-09-15'])==45
quality={}
for s in ['AVGO','MU']:
    f=m.F[s];bad=(f.high+.02<f[['open','close','low']].max(axis=1))|(f.low-.02>f[['open','close','high']].min(axis=1))
    assert not bad.any(),s
    a=m.features(s);assert a.loc[a.index>='2026-09-14','next_return'].isna().all()
    assert a.loc[a.index>='2026-09-14','break_low'].isna().all()
    # Future price perturbation cannot change earlier feature values.
    target='2025-09-15';saved=m.F[s].copy();m.F[s].loc[m.F[s].index>target,'adj']*=10
    b=m.features(s);m.F[s]=saved
    assert np.allclose(a.loc[target,m.FULL].to_numpy(dtype=float),b.loc[target,m.FULL].to_numpy(dtype=float))
    quality[s]={'rows':len(f),'earliest':f.index.min(),'latest':f.index.max(),'invalid_ohlc':int(bad.sum())}
result=json.loads((R/'model-results.json').read_text())
for s,x in result['results'].items():
    for k,v in x['models'].items():
        assert 0<=v['raw_probability']<=1
        assert v['validation']['through']<'2026-09-14'
        assert v['oos_fomc']['n']==21
out={'status':'passed','checks':['logistic monotonic synthetic / constant midpoint','AUC known/tied','Wilson bounds','45 official scheduled FOMC meetings before current date','AVGO/MU OHLC integrity','unfinished outcome excluded','future-price perturbation leaves earlier features unchanged','OOS timestamp boundaries','21 OOS FOMC observations per stock'],'quality':quality,'investment_effectiveness':'direction and down2 models fail stable OOS edge; break-low models pass broad-history discrimination but event-day calibration unproven'}
(R/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
