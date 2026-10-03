import json,math,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
m=json.loads((R/'model-results.json').read_text())
def payoff(s,k,c):return max(s-k,0)/c-1
assert payoff(100,100,5)==-1
assert abs(payoff(105,100,5))<1e-12
assert payoff(110,100,5)==1
N=lambda x:(1+math.erf(x/math.sqrt(2)))/2
def bs(s,k,t,v,r=.04):
 if t<=0:return max(s-k,0)
 d1=(math.log(s/k)+(r+v*v/2)*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
 return s*N(d1)-k*math.exp(-r*t)*N(d2)
assert abs(bs(100,100,1,.2,.05)-10.450583572185565)<1e-8
rows=[]
for s,d in m.items():
 k={'AVAV':160,'KTOS':50,'RCAT':8,'ONDS':8}[s]
 o=next(x for x in d['options'] if x['expiry']=='2026-10-16' and x['strike']==k)
 row=dict(symbol=s,option=o['id'],ask=o['ask'],dollar_cost=100*o['ask'],expiry_returns={str(r):round(100*payoff(d['close']*(1+r),k,o['ask']),2) for r in [-.1,0,.05,.1,.2]},seven_calendar_days_later={})
 # Sensitivity only: calendar ACT/365, 32 days at capture to Oct16 -> 25 days.
 for r in [0,.05,.1]:
  row['seven_calendar_days_later'][str(r)]={str(dv):round(100*(bs(d['close']*(1+r),k,25/365,max(o['iv']+dv,.01))/o['ask']-1),2) for dv in [0,-.2]}
 rows.append(row)
(R/'payoff-results.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
# Preserve missingness under the original framework. Cboe IV30 is an explicitly named proxy;
# ATM methodology and annual percentile have not been independently replicated.
spec=importlib.util.spec_from_file_location('framework',R.parents[1]/'量化参考/计算评分.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
f={}
for s,d in m.items():
 data=json.loads((R.parents[1]/'量化参考/数据输入模板.json').read_text());data['as_of']='2026-09-14T22:20:31-04:00'
 data['observations']['iv_rv_ratio']=dict(value=d['iv_rv'],observed_at='2026-09-14T16:00:00-04:00',available_at='2026-09-14T22:20:30-04:00',source=f'https://cdn.cboe.com/api/global/delayed_quotes/options/{s}.json',verified=True,proxy=True)
 (R/f'{s}-框架输入.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));f[s]=module.calculate(data)
(R/'框架结果.json').write_text(json.dumps(f,ensure_ascii=False,indent=2))
print('Validation: payoff at OTM/breakeven/double; Black-Scholes reference passed. Original score unavailable, coverage 0% (proxy excluded).')
