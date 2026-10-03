"""Auditable descriptive factors and research plans, separate from regression."""
import json, importlib.util
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent; P=R.parents[1]
D={x['symbol']:x for x in json.loads((R/'snapshot.json').read_text())}
I={x['symbol']:x for x in json.loads((R/'intraday.json').read_text()) if 'error' not in x}
now=datetime.now(timezone.utc).isoformat()
macro={'retrieved_at':now,'verification':'FRED web tables read in current task; equity trading-day lag excludes Labor Day 2026-09-07',
 'DFII10':{'date':'2026-09-11','value':2.60,'lag20_date':'2026-08-13','lag20_value':2.39,'change_bp':21,'available_at':'2026-09-14T16:16:00-04:00','source':'https://fred.stlouisfed.org/data/DFII10'},
 'HY_OAS':{'date':'2026-09-14','value':2.71,'lag20_date':'2026-08-14','lag20_value':2.67,'change_bp':4,'available_at':'2026-09-15T10:44:00-04:00','source':'https://fred.stlouisfed.org/data/BAMLH0A0HYM2'},
 'core_cpi':{'2026-04':335.423,'2026-05':336.121,'2026-06':336.065,'2026-07':336.789,'2026-08':337.765,'source':'https://fred.stlouisfed.org/series/CPILFESL','available_at':'2026-09-11T09:37:00-04:00'}}
macro['core_cpi_3m_annualized_pct']=100*((337.765/336.121)**4-1)
macro['core_cpi_3m_acceleration_pp']=macro['core_cpi_3m_annualized_pct']-100*((336.789/335.423)**4-1)
macro['brent_futures_proxy']={k:D['BZ=F'][k] for k in ['price','day_pct','quote_time','ret20']}
macro['recession_rate_cut_trigger']=False
macro['energy_inflation_joint_trigger']=bool(D['BZ=F']['ret20']>=.1 and macro['core_cpi_3m_acceleration_pp']>=.5)
(R/'macro.json').write_text(json.dumps(macro,ensure_ascii=False,indent=2))
inp=json.loads((P/'量化参考/数据输入模板.json').read_text());inp['as_of']=now
for k,m in [('real_yield_change_bp',macro['DFII10']),('hy_oas_change_bp',macro['HY_OAS'])]:
    inp['observations'][k]={'value':m['change_bp'],'observed_at':m['date']+'T16:00:00-04:00','available_at':m['available_at'],'source':m['source'],'verified':True,'proxy':False}
inp['observations']['verified_catalyst_count']={'value':1,'observed_at':now,'available_at':now,'source':'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm','verified':True,'proxy':False}
inp['notes']='FOMC September 15-16 counts once as a material macro event, not a bullish catalyst. Current technical screen is separate from fixed AI/non-AI framework. EPS revisions, FCF-margin breadth, PB leverage, ATM-IV history and capex revisions unavailable.'
(R/'框架输入.json').write_text(json.dumps(inp,ensure_ascii=False,indent=2))
spec=importlib.util.spec_from_file_location('framework',P/'量化参考/计算评分.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
(R/'框架结果.json').write_text(json.dumps(mod.calculate(inp),ensure_ascii=False,indent=2))
factors=[]
for s,x in D.items():
    if x['group']=='benchmarks':continue
    z=I.get(s,{})
    factors.append({'symbol':s,'group':x['group'],'price':z.get('price',x['price']),'quote_at':z.get('quote_at',x['quote_time']),'day_pct':z.get('day_pct',x['day_pct']),
      'completed_signal_date':x['last_complete_date'],'prior_ret20_pct':100*x['prior_ret20'],'prior_rs20_spy_pp':100*(x['prior_ret20']-D['SPY']['prior_ret20']),
      'prior_rsi14':x['prior_rsi14'],'atr_pct':100*x['atr14']/x['price'],'dist_ma20_pct':100*(z.get('price',x['price'])/x['ma20']-1),
      'dollar_volume20':x['dollar_volume20'],'above_prior_ma50':z.get('price',x['price'])>x['ma50'],
      'vwap_proxy':z.get('vwap_proxy'),'last3_above_vwap':z.get('above_vwap_last3'),'rvol_same_time_4session':z.get('rvol_same_time_4session')})
pd.DataFrame(factors).to_csv(R/'因子比较.csv',index=False)
groups={}
for g in sorted(set(x['group'] for x in factors)):
    a=[x for x in factors if x['group']==g]
    groups[g]={'n':len(a),'positive_today':sum(x['day_pct']>0 for x in a),'equal_weight_day_pct':float(np.mean([x['day_pct'] for x in a]))}
(R/'group-breadth.json').write_text(json.dumps(groups,indent=2))

def history(s):
    x=json.loads((R/'raw'/f'{s}-1d.json').read_text())['chart']['result'][0]
    f=pd.DataFrame(x['indicators']['quote'][0],index=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d'))
    f['adj']=x['indicators']['adjclose'][0]['adjclose'];return f.dropna()
f=history('TSLA');f=f.loc[f.index>='2026-03-15'];ratio=f.adj/f.close
for c in ['high','low','open','close']:f[c]=f[c]*ratio/ratio.iloc[-1]
def remaining_gaps(frame,price):
    gaps=[]
    for n in range(1,len(frame)):
        p,b=frame.iloc[n-1],frame.iloc[n]
        if b.low>p.high:
            lo,hi=p.high,frame.iloc[n:].low.min();kind='up'
        elif b.high<p.low:
            lo,hi=frame.iloc[n:].high.max(),p.low;kind='down'
        else:continue
        if hi>lo:gaps.append({'formed':frame.index[n],'type':kind,'remaining_low':float(lo),'remaining_high':float(hi),'relative_to_current':'above' if lo>price else 'below' if hi<price else 'overlap'})
    return gaps
gaps=remaining_gaps(f,I['TSLA']['price'])
fixture=pd.DataFrame({'low':[8,12,11],'high':[10,14,13]})
assert [(x['remaining_low'],x['remaining_high']) for x in remaining_gaps(fixture,12)]==[(10,11)]
fixture.loc[2,'low']=10
assert remaining_gaps(fixture,12)==[]
fixture=pd.DataFrame({'low':[12,8,9],'high':[14,10,11]})
assert [(x['remaining_low'],x['remaining_high']) for x in remaining_gaps(fixture,10)]==[(11,12)]
tg={'asof':D['TSLA']['quote_time'],'window':'2026-03-15 through latest partial September 15 daily bar','definition':'Opening gap=open/previous close-1. Remaining structural gaps use full high/low interval, adjusted to current share basis; only positive-width untraded portions. Current daily bar is provisional.','open':D['TSLA']['day_open'],'previous_close':D['TSLA']['prev_close'],'opening_gap_pct':D['TSLA']['gap_pct'],'remaining':gaps,'TSLL':I['TSLL'],'TSLL_distance_to_10_pct':100*(I['TSLL']['price']/10-1)}
(R/'TSLA-gaps.json').write_text(json.dumps(tg,indent=2))
plans=[
 {'symbol':'XOM','entry_low':167.8,'entry_high':168.5,'stop':164.5,'targets':[175.5,179.0],'condition':'回踩167.8–168.5后，三根完整5分钟K线收回更新后的VWAP；不追169.6上方；放量收盘突破169.65后再复核加仓','interpretation':'priority conditional pullback; oil reversal risk'},
 {'symbol':'CVX','entry_low':215.0,'entry_high':216.0,'stop':211.8,'targets':[223.0,228.0],'condition':'XOM替代：回踩215–216后重回VWAP；217.65为上方近期阻力。与XOM二选一','interpretation':'same energy exposure, not diversification'},
 {'symbol':'CRWD','entry_low':238.5,'entry_high':240.0,'stop':231.0,'targets':[256.0,265.0],'condition':'回踩前高239.37和VWAP约238.86并止跌；三根完整5分钟K线重回更新VWAP；跌破239.37后不能收回则撤单','interpretation':'higher volatility, stretched momentum, no chase at 243'},
 {'symbol':'TSLA','entry_low':355.0,'entry_high':358.0,'stop':348.0,'targets':[375.44,384.04],'condition':'用户左侧框架：TSLL低于10且TSLA停止创新低，355–358试计划份额1/3；收回359.3附近VWAP且回踩守住再1/3；FOMC后站稳362.4再1/3。跌破354.63先暂停加仓，跌破348退出。','interpretation':'user-specified unvalidated left-side framework; stop 348 is risk-budget line, not proven support'}]
for p in plans:
    mid=(p['entry_low']+p['entry_high'])/2
    p['reference_entry']=mid;p['risk_pct']=100*(mid-p['stop'])/mid;p['target_R']=[(t-mid)/(mid-p['stop']) for t in p['targets']]
    p['target_note']='XOM/CVX/CRWD levels are planned reward scenarios, not model price forecasts. TSLA targets are historical highs.'
(R/'交易计划.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2))
assert len(factors)==88
assert all(x['completed_signal_date']=='2026-09-14' for x in factors)
assert len(I)==22 and all(z['last_completed_bar']['time'][11:16]<z['quote_at'][11:16] for z in I.values())
assert abs(macro['DFII10']['change_bp']-100*(2.60-2.39))<1e-9
assert abs(macro['HY_OAS']['change_bp']-100*(2.71-2.67))<1e-9
assert all(p['stop']<p['entry_low']<=p['entry_high']<p['targets'][0] for p in plans)
print(json.dumps({'stock_count':len(factors),'groups':groups,'macro':macro,'TSLA_gaps':tg,'plans':plans,'checks':'passed'},ensure_ascii=False,indent=2))
