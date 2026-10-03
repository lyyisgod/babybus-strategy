import json, datetime, math, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York'); now=datetime.datetime.now(NY); today=str(now.date())
def read(symbol,kind):
    obj=json.loads((ROOT/f'{symbol}_{kind}.json').read_text()); d=obj['data']
    f=pd.DataFrame(d['indicators']['quote'][0],index=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert(NY))
    if 'adjclose' in d['indicators']:f['adjclose']=d['indicators']['adjclose'][0]['adjclose']
    return obj,f.dropna(subset=['open','high','low','close'])
def summary(f):
    v=f.volume.sum(); p=f.close.iloc[-1];h=f.high.max();l=f.low.min()
    return dict(open=float(f.open.iloc[0]),high=float(h),low=float(l),price=float(p),volume=float(v),vwap=float((((f.high+f.low+f.close)/3)*f.volume).sum()/v) if v else None,location=float((p-l)/(h-l)) if h>l else .5)
daily={}; intr={}; meta={}; quotes={}
for s in ['MU','SMH','QQQ','SNDK','WDC','STX','NVDA','VIX']:
    ob,d=read(s,'daily');daily[s]=d[d.index.date<now.date()]
    ob,f=read(s,'intraday');meta[s]=ob['data']['meta']; intr[s]=f[(f.index.date==now.date())&(f.index.hour*60+f.index.minute>=570)&(f.index.hour*60+f.index.minute<960)]
    prev=float(daily[s].close.iloc[-1]);m=meta[s]
    quotes[s]={'price':m['regularMarketPrice'],'time_et':datetime.datetime.fromtimestamp(m['regularMarketTime'],NY).isoformat(),'previous_close':prev,'change_pct':100*(m['regularMarketPrice']/prev-1),'source':ob['source'],'retrieved_at':ob['retrieved_at']}
cut=min(f.index[-1].floor('min') for s,f in intr.items() if s!='VIX')
snap={s:summary(f[f.index<cut]) for s,f in intr.items()}
for s,a in snap.items():
    prev=quotes[s]['previous_close'];a['change_pct']=100*(a['price']/prev-1);a['gap_pct']=100*(a['open']/prev-1)
mu=daily['MU'];c=mu.close;delta=c.diff().dropna();g=delta.clip(lower=0);l=(-delta).clip(lower=0)
ag=g.iloc[:14].mean();al=l.iloc[:14].mean()
for x,y in zip(g.iloc[14:],l.iloc[14:]):ag=(ag*13+x)/14;al=(al*13+y)/14
rsi=100-100/(1+ag/al) if al else (100 if ag else 50)
tr=pd.concat([mu.high-mu.low,(mu.high-c.shift()).abs(),(mu.low-c.shift()).abs()],axis=1).max(axis=1).iloc[1:]
atr=tr.iloc[:14].mean()
for x in tr.iloc[14:]:atr=(13*atr+x)/14
returns=pd.DataFrame({s:pd.Series(np.log(d.adjclose).diff().values,index=d.index.normalize()) for s,d in daily.items()})
xy=returns[['MU','SMH']].dropna().tail(60);beta=float(xy.cov().loc['MU','SMH']/xy.SMH.var());alpha=float(xy.MU.mean()-beta*xy.SMH.mean())
res=xy.MU-alpha-beta*xy.SMH
reg={'n':len(xy),'start':str(xy.index[0].date()),'end':str(xy.index[-1].date()),'smh_beta':beta,'alpha_daily_log':alpha,'r_squared':float(1-res.var()/xy.MU.var()),'residual_sigma_daily_pct':float(res.std()*100),'current_residual_pp':100*(math.log(snap['MU']['price']/quotes['MU']['previous_close'])-alpha-beta*math.log(snap['SMH']['price']/quotes['SMH']['previous_close'])),'note':'60日复权日收益同期解释回归，当前残差为截至共同分钟的日内近似；无预测能力验证。'}
tech={'as_of':str(mu.index[-1].date()),'rsi14':rsi,'atr14':atr,'rv20_annualized_pct':float(np.log(mu.adjclose).diff().tail(20).std()*math.sqrt(252)*100),'sma':{str(n):float(c.tail(n).mean()) for n in [5,10,20,50,200]},'returns20_pct':{s:float((d.adjclose.iloc[-1]/d.adjclose.iloc[-21]-1)*100) for s,d in daily.items()},'previous_5_sessions':[dict(date=str(i.date()),**{k:float(row[k]) for k in ['open','high','low','close','volume']}) for i,row in mu.tail(5).iterrows()]}
# Same completed 5-minute cutoff for current and historical sessions, no final-day features.
hist={s:read(s,'5m')[1] for s in ['MU','SMH','QQQ']}
last_complete=min(now-pd.Timedelta(minutes=5),cut-pd.Timedelta(minutes=5))
cutminute=(last_complete.hour*60+last_complete.minute)//5*5
sessions={s:{str(k):v for k,v in f[(f.index.hour*60+f.index.minute>=570)&(f.index.hour*60+f.index.minute<960)].groupby(f[(f.index.hour*60+f.index.minute>=570)&(f.index.hour*60+f.index.minute<960)].index.date)} for s,f in hist.items()}
records=[]
for day,f in sessions['MU'].items():
    selected=f[f.index.hour*60+f.index.minute<=cutminute]; after=f[f.index.hour*60+f.index.minute>cutminute]
    if len(selected)<(cutminute-570)//5+1:continue
    dt=datetime.date.fromisoformat(day);prev=mu[mu.index.date<dt]
    if len(prev)==0:continue
    a=summary(selected);a['gap_pct']=100*(a['open']/prev.close.iloc[-1]-1);a['change_pct']=100*(a['price']/prev.close.iloc[-1]-1);a['date']=day
    a['vwap_distance_pct']=100*(a['price']/a['vwap']-1)
    if day<today and len(after)>0 and f.index[-1].hour*60+f.index[-1].minute>=955:
        a.update(close_return_pct=100*(f.close.iloc[-1]/a['price']-1),remaining_low_return_pct=100*(after.low.min()/a['price']-1),new_low=bool(after.low.min()<a['low']),remaining_high_return_pct=100*(after.high.max()/a['price']-1))
        records.append(a)
    elif day==today:current=a
def stats(rows):
    if not rows:return {'n':0}
    x=pd.DataFrame(rows)
    return {'n':len(x),'dates':x.date.tolist(),'close_up_count':int((x.close_return_pct>0).sum()),'new_low_count':int(x.new_low.sum()),'median_close_return_pct':float(x.close_return_pct.median()),'close_quantiles_pct':{str(q):float(x.close_return_pct.quantile(q)) for q in [.1,.25,.5,.75,.9]},'median_remaining_low_pct':float(x.remaining_low_return_pct.median())}
strict=[r for r in records if r['gap_pct']<=-1 and r['location']<=.25 and r['vwap_distance_pct']<0]
loose=[r for r in records if r['location']<=.25 and r['vwap_distance_pct']<0]
volume_history=[r['volume'] for r in records[-20:]]
study={'cutoff_bar_start_et':f'{cutminute//60:02d}:{cutminute%60:02d}','current':current,'same_time_volume_ratio20':float(current['volume']/np.mean(volume_history)),'baseline':stats(records),'near_low_below_vwap':stats(loose),'gap_down_near_low_below_vwap':stats(strict),'note':'阈值预设：低开至少1%、区间位置≤25%、低于VWAP。60日分钟数据条件频数，非样本外胜率，未扣成本；未使用当日收盘进行特征筛选。'}
macro={}
for s in ['DFII10','BAMLH0A0HYM2','DCOILBRENTEU','CPILFESL']:
    try:
        df=pd.read_csv(ROOT/(s+'.csv'),index_col=0,parse_dates=True);a=pd.to_numeric(df.iloc[:,0],errors='coerce').dropna();a=a[a.index.date<=now.date()]
        info={'latest':float(a.iloc[-1]),'observed_at':str(a.index[-1].date()),'source':f'https://fred.stlouisfed.org/series/{s}'}
        if s!='CPILFESL':
            dates=[i.date() for i in mu.index if i.date()<=a.index[-1].date()];ref=pd.Timestamp(dates[-21]);old=a.loc[:ref].iloc[-1]
            info.update(reference_date=str(ref.date()),reference_value=float(old),change20_bp=float((a.iloc[-1]-old)*100),change20_pct=float((a.iloc[-1]/old-1)*100))
        else:info.update(annualized3m_pct=float(((a.iloc[-1]/a.iloc[-4])**4-1)*100),previous_annualized3m_pct=float(((a.iloc[-2]/a.iloc[-5])**4-1)*100))
        macro[s]=info
    except Exception as e:macro[s]={'error':str(e)}
inp=json.loads((ROOT.parents[1]/'量化参考/数据输入模板.json').read_text());inp['as_of']=now.isoformat()
for factor,s in [('real_yield_change_bp','DFII10'),('hy_oas_change_bp','BAMLH0A0HYM2')]:
    if 'change20_bp' in macro[s]:
        a=macro[s];inp['observations'][factor].update(value=a['change20_bp'],observed_at=a['observed_at']+'T00:00:00-04:00',available_at=now.isoformat(),source=a['source'],verified=True)
(ROOT/'量化输入.json').write_text(json.dumps(inp,ensure_ascii=False,indent=2))
sys.path.insert(0,str(ROOT.parents[1]/'量化参考'))
from 计算评分 import calculate
coverage=calculate(inp)
(ROOT/'量化覆盖率.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))
out={'generated_at':now.isoformat(),'common_intraday_cutoff_exclusive':str(cut),'quotes':quotes,'synchronized_intraday':snap,'technical':tech,'regression':reg,'intraday_study':study,'macro':macro,'framework':coverage}
(ROOT/'模型结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
