"""Exploratory next-day filtered historical simulation; public saved daily data only."""
import json,math,statistics
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
def quantile(a,p):
 a=sorted(a);v=(len(a)-1)*p;i=int(v);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(v-i)
def wilson(k,n):
 z=1.96;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [center-w,center+w]
def proportion(x,th):return sum(v>=th for v in x)/len(x)
def mean(a):return statistics.mean(a)
results={}
for s in ['AVAV','KTOS','RCAT','ONDS']:
 d=json.loads((R/'raw'/f'{s}-1d.json').read_text())['chart']['result'][0];q=d['indicators']['quote'][0];adj=d['indicators']['adjclose'][0]['adjclose'];valid=[i for i,c in enumerate(q['close']) if c and adj[i]];c=[q['close'][i] for i in valid];a=[adj[i] for i in valid];h=[q['high'][i] for i in valid];v=[q['volume'][i] for i in valid];lr=[None]+[math.log(a[i]/a[i-1]) for i in range(1,len(a))];n=len(a)
 def sigma(t,w):return statistics.stdev(lr[t-w+1:t+1])
 def samples(t,w):
  # shock for day j standardized using only data through j-1; no look-ahead.
  z=[lr[j]/sigma(j-1,w) for j in range(max(w+1,t-251),t+1)]
  avg=mean(z)
  return [math.expm1(sigma(t,w)*(x-avg)) for x in z]
 models={}
 for w in [20,60]:
  sam=samples(n-1,w)
  models[f'fhs{w}']=dict(n=len(sam),p_up=proportion(sam,0),p5=proportion(sam,.05),p10=proportion(sam,.1),p_down5=mean([x<=-.05 for x in sam]),quantile_10_50_90=[quantile(sam,p) for p in [.1,.5,.9]])
 # same-stock analogs: recent trend, daily move, volume. Return standardized by prior volatility.
 def features(t):
  sg=sigma(t,20)
  return [lr[t]/sg,math.log(a[t]/a[t-5])/(sg*math.sqrt(5)),math.log(a[t]/a[t-20])/(sg*math.sqrt(20)),math.log(v[t]/mean(v[t-20:t]))]
 target=features(n-1);candidates=list(range(60,n-1));fs=[features(t) for t in candidates];scales=[max(statistics.stdev([x[k] for x in fs]),.01) for k in range(4)]
 dist=[(sum(((f[k]-target[k])/scales[k])**2 for k in range(4)),t) for f,t in zip(fs,candidates)];chosen=[t for _,t in sorted(dist)[:50]]
 analog=[math.expm1(lr[t+1]/sigma(t,20)*sigma(n-1,20)) for t in chosen]
 models['analog50']=dict(n=50,p_up=proportion(analog,0),p5=proportion(analog,.05),p10=proportion(analog,.1),p_down5=mean([x<=-.05 for x in analog]),wilson_p5=wilson(sum(x>=.05 for x in analog),50),dates=[datetime.fromtimestamp(d['timestamp'][valid[t]],NY).date().isoformat() for t in chosen])
 raw=[math.expm1(x) for x in lr[-252:]];highs=[h[i]/c[i-1]-1 for i in range(n-252,n)]
 # walk-forward check of each volatility model, not analog selection: same-stock available history.
 validation={}
 for w in [20,60]:
  preds=[];ys=[]
  for t in range(312,n-1):
   sam=samples(t,w);preds.append(proportion(sam,.05));ys.append(int(math.expm1(lr[t+1])>=.05))
  baseline=[mean([int(math.expm1(lr[j])>=.05) for j in range(max(1,t-251),t+1)]) for t in range(312,n-1)]
  validation[str(w)]=dict(n=len(ys),events=sum(ys),mean_prediction=mean(preds),actual_frequency=mean(ys),brier=mean([(p-y)**2 for p,y in zip(preds,ys)]),baseline_brier=mean([(p-y)**2 for p,y in zip(baseline,ys)]))
 results[s]=dict(close=c[-1],threshold5=c[-1]*1.05,threshold10=c[-1]*1.1,rv20_daily=sigma(n-1,20),rv60_daily=sigma(n-1,60),models=models,raw252=dict(n=252,p5=proportion(raw,.05),p10=proportion(raw,.1),p_down5=mean([x<=-.05 for x in raw]),p_high5=proportion(highs,.05),high5_closebelow5=sum(x>=.05 and y<.05 for x,y in zip(highs,raw))/252),validation=validation)
print(json.dumps(results,indent=2))
(R/'next-day-results.json').write_text(json.dumps(results,indent=2))
assert abs(quantile([0,1,2],.5)-1)<1e-12
assert proportion([-.1,0,.05,.1],.05)==.5
assert wilson(0,50)[0]>=-1e-12
