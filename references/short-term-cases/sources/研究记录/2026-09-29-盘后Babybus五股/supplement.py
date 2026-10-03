from pathlib import Path
import importlib.util,json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('m',P/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ds={s:m.load(s)[0] for s in m.STOCKS+m.PROXIES}
rows=[]
for s in m.STOCKS:
 d=ds[s];c=d.close
 delta=c.diff().dropna();g=m.wilder(delta.clip(lower=0),2);l=m.wilder(-delta.clip(upper=0),2);rsi=100*g/(g+l)
 rsi[(g==0)&(l==0)]=50
 w=c.resample('W-FRI').last();w=w[w.index<=m.DAY];mid=w.tail(20).mean();sd=w.tail(20).std(ddof=0);lower=mid-2*sd;pos=(c.iloc[-1]-mid)/(2*sd)
 z=(c.iloc[-1]-c.tail(20).mean())/c.tail(20).std(ddof=0)
 tr=bool((c.iloc[-2]<=lower and c.iloc[-1]>lower) or (lower<c.iloc[-1]<mid and c.iloc[-1]>d.high.iloc[-2]))
 rows.append(dict(symbol=s,weekly_bb_pos=pos,weekly_bb_mid=mid/(c.iloc[-1]/d.raw_close.iloc[-1]),daily_z20=z,baby_price_trigger=tr,baby_no_chase=bool(pos<.8 and z<2),rsi2=rsi.iloc[-1],lrcx_signal=bool(c.iloc[-1]>c.rolling(200).mean().iloc[-1] and rsi.iloc[-1]<=20 and c.pct_change(3).iloc[-1]<0),drawdown_252=1-c.iloc[-1]/d.high.tail(252).max()))
m.save('strategy-crosscheck.json',rows)
j=ds['JNK'].close;w=j.resample('W-FRI').last();w=w[w.index<=m.DAY];months=j.resample('ME').last();months=months[months.index<m.DAY.replace(day=1)];bear=months.tail(3).mean()<months.tail(10).mean();breakdown=w.iloc[-1]<w.iloc[-9:-1].min()*.995
macro={}
for s in ['DFII10','BAMLH0A0HYM2']:
 a=pd.read_csv(P/(s+'.csv'),index_col=0,parse_dates=True).apply(pd.to_numeric,errors='coerce').dropna();a=a[a.index<=m.DAY.tz_localize(None)];last=a.index[-1];idx=ds['QQQ'].index.tz_localize(None);prior_date=idx[idx<=last][-21];prev=a.loc[:prior_date].iloc[-1,0]
 macro[s]=dict(observed_date=str(last.date()),value=a.iloc[-1,0],prior_date=str(prior_date.date()),prior_value=prev,change20_bp=100*(a.iloc[-1,0]-prev))
state='OFF' if bear or breakdown or macro['BAMLH0A0HYM2']['change20_bp']>=50 else 'NOT_ON' if j.iloc[-1]/j.iloc[-21]<=1 or macro['BAMLH0A0HYM2']['change20_bp']>0 else 'ON_UNCONFIRMED'
m.save('babybus-credit.json',dict(state=state,jnk_ret20=100*(j.iloc[-1]/j.iloc[-21]-1),jnk_ret5=100*(j.iloc[-1]/j.iloc[-6]-1),bear_month=bear,month_fast3=months.tail(3).mean(),month_slow10=months.tail(10).mean(),weekly_breakdown=breakdown,latest_completed_week=str(w.index[-1]),latest_week_value=w.iloc[-1],prior8_min=w.iloc[-9:-1].min(),weekly_threshold=w.iloc[-9:-1].min()*.995,macro=macro,new_buy=False))
market=[]
for s in m.PROXIES:
 d=ds[s];r=json.loads((P/'raw'/f'{s}-5m.json').read_text())['chart']['result'][0];q=r['indicators']['quote'][0];idx=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(m.NY);bars=pd.Series(q['close'],index=idx).dropna();bars=bars[bars.index<=pd.Timestamp.now(tz=m.NY)]
 market.append(dict(symbol=s,daily_date=str(d.index[-1]),close=d.raw_close.iloc[-1],daily_pct=100*(d.raw_close.iloc[-1]/d.raw_close.iloc[-2]-1),ret20_pct=100*(d.close.iloc[-1]/d.close.iloc[-21]-1),last=bars.iloc[-1],last_at=str(bars.index[-1]),vs_close_pct=100*(bars.iloc[-1]/d.raw_close.iloc[-1]-1)))
m.save('market-summary.json',market)
a=pd.read_csv(P/'ranking.csv').merge(pd.DataFrame(rows),on='symbol');a.to_csv(P/'ranking-enriched.csv',index=False)
print('CREDIT',state,macro);print('LRCX',a[a.lrcx_signal].symbol.tolist());print(a[a.symbol.isin(['ALAB','NBIS','AAOI','CRDO','CBRS','COHR','BA','IOVA','KOD','VICR','MU','CRWV'])][['symbol','last','high','low','atr','volume_ratio','weekly_bb_pos','baby_price_trigger','baby_no_chase','drawdown_252','p_hit5','p_down3']].round(3).to_string(index=False));print(pd.DataFrame(market)[['symbol','close','daily_pct','last','vs_close_pct']].round(3).to_string(index=False))
# Raw same-scale prefix verification is already in model.py; weekly uses only finished Fridays.
assert w.index[-1]<m.DAY and months.index[-1].month==8
base=json.loads((Path('量化参考')/'数据输入模板.json').read_text());base['as_of']=pd.Timestamp.now(tz=m.NY).isoformat()
for key,s in [('real_yield_change_bp','DFII10'),('hy_oas_change_bp','BAMLH0A0HYM2')]:
 v=macro[s];base['observations'][key]=dict(value=v['change20_bp'],observed_at=v['observed_date']+'T16:00:00-04:00',available_at='2026-09-29T19:51:34-04:00',source='https://fred.stlouisfed.org/series/'+s,verified=True,proxy=False)
m.save('framework-input.json',base)
