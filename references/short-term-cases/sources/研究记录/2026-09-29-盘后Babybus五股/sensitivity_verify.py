from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('model',P/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=json.loads((P/'coefficients.json').read_text());cols=c['features'];mu=np.array(c['train_mean']);sd=np.array(c['train_sd']);ds={s:m.load(s)[0] for s in m.STOCKS+m.PROXIES}
a=pd.read_csv(P/'ranking.csv');a=a[a.eligible];rows=[]
for s in a.symbol:
 assert ds[s].index[-1]==m.DAY
 x=m.features(ds[s],ds);pd.testing.assert_frame_equal(x.iloc[:-1],m.features(ds[s].iloc[:-1],{k:v.loc[:ds[s].index[-2]] for k,v in ds.items()}))
 z=np.clip((x.iloc[-1].to_numpy()-mu)/sd,-8,8);rec={'symbol':s}
 for version in ['original','oil_neutralized']:
  zz=z.copy()
  if version=='oil_neutralized':zz[cols.index('BZ=F_r5')]=0
  probs=[]
  for kind in ['a','b']:
   xx=np.r_[1,zz] if kind=='a' else np.r_[1,zz,zz*zz,*[zz[cols.index(l)]*zz[cols.index(r)] for l,r in [('ret1','volume_ratio'),('range_position','volume_ratio'),('ret1','QQQ_r1'),('atr_pct','VIX_level'),('ret20','rs20')]]]
   model=c['models'][kind+'_hit5'];logit=xx@np.array(model['coefficients']);cb=model['calibration'];probs.append(float(m.sigmoid(cb[0]+cb[1]*logit)))
  rec[version]=np.mean(probs)
 assert abs(rec['original']-a[a.symbol==s].p_hit5.iloc[0])<1e-10
 rows.append(rec)
m.save('oil-sensitivity.json',rows)
# Raw-source provenance validation; allows only declared original current OHLC envelope corrections.
manifest=json.loads((P/'manifest.json').read_text())
assert all(hashlib.sha256((P/'raw'/f'{r["symbol"]}-{r["interval"]}.json').read_bytes()).hexdigest()==r['sha256'] for r in manifest)
sel=pd.read_csv(P/'evaluation-top1.csv');ret=sel.toy_trade_return.to_numpy();m.save('extended-validation.json',dict(status='pass',symbols_prefix_checked=len(a),source_hashes_checked=len(manifest),original_inference_reproduced=True,oil_neutralization='diagnostic only; no re-selection or retraining',oil_roll_caution='Vendor BZ continuous futures front contract discontinuity; current last day excluded from model using lag, older rolls still potential contamination',cost_base20bps_mean=ret.mean(),cost_double40bps_mean=(ret-.002).mean(),cost_double40bps_dateblock95=m.block_ci(ret-.002),probability_certified=False))
print(pd.DataFrame(rows).query('symbol in ["ALAB","NBIS","CRDO","COHR","AAOI"]').round(4).to_string(index=False))
print('Verified',len(a),'symbols')
