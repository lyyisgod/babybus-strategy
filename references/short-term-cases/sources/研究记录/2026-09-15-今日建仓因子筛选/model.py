"""Exploratory technical model; no probability is accepted without chronological validation."""
import json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
U=json.loads((R/'universe.json').read_text())
TODAY='2026-09-15'
def read(s):
    if s=='OXY': raise ValueError('Quarantined: current daily low exceeds open; raw retained')
    d=json.loads((R/'raw'/f'{s}-1d.json').read_text())['chart']['result'][0]
    f=pd.DataFrame(d['indicators']['quote'][0],index=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d'))
    f['adj']=d['indicators'].get('adjclose',[{}])[0].get('adjclose',f['close'])
    f=f.dropna(subset=['open','high','low','close','volume','adj'])
    f=f[~f.index.duplicated(keep='last')];f=f.loc[f.index<=TODAY]
    ratio=f['adj']/f['close']
    for col in ['open','high','low']: f['a'+col]=f[col]*ratio
    return f,d['meta']
def rsi(c):
    a=np.asarray(c,float);delta=np.diff(a);out=np.full(len(a),np.nan)
    if len(a)<=14:return out
    gain=np.maximum(delta,0);loss=np.maximum(-delta,0);g=gain[:14].mean();l=loss[:14].mean()
    val=lambda g,l:50. if g==0 and l==0 else 100. if l==0 else 100-100/(1+g/l)
    out[14]=val(g,l)
    for i in range(15,len(a)):
        g=(13*g+gain[i-1])/14;l=(13*l+loss[i-1])/14;out[i]=val(g,l)
    return out
def enrich(f):
    f=f.copy();c=f.adj;lr=np.log(c/c.shift())
    for h in [5,10,20,60]: f[f'ret{h}']=c/c.shift(h)-1
    for h in [20,50,200]: f[f'ma{h}']=c.rolling(h).mean()
    tr=pd.concat([f.ahigh-f.alow,(f.ahigh-c.shift()).abs(),(f.alow-c.shift()).abs()],axis=1).max(axis=1)
    f['atr14']=tr.rolling(14).mean() # SMA ATR disclosed; RSI is Wilder
    f['rv20']=lr.rolling(20).std()*np.sqrt(252)
    f['rsi14']=rsi(c);f['rvol']=f.volume/f.volume.shift().rolling(20).mean()
    f['h20']=f.ahigh.shift().rolling(20).max();f['l20']=f.alow.shift().rolling(20).min()
    f['h5']=f.ahigh.shift().rolling(5).max();f['l5']=f.alow.shift().rolling(5).min()
    f['dist20']=c/f.ma20-1;f['dist50']=c/f.ma50-1;f['dist200']=c/f.ma200-1
    f['dist_high20']=c/f.h20-1
    f['day']=c/c.shift()-1;f['gap']=f.aopen/c.shift()-1
    f['closepos']=(c-f.alow)/(f.ahigh-f.alow).replace(0,np.nan)
    f['dollar_volume20']=(f.close*f.volume).shift().rolling(20).mean()
    return f
FRAMES={};META={}
for symbols in U['groups'].values():
    for s in symbols:
        try:f,m=read(s);FRAMES[s]=enrich(f);META[s]=m
        except Exception as e:print('PARSE ERROR',s,e)
SPY=FRAMES['SPY'];QQQ=FRAMES['QQQ']
FEATURES=['ret5','ret20','ret60','dist50','dist200','rv20','rsi14','log_rvol','rs20','market20','market_dist200','market_rv20']
rows=[];train=[];latest=[]
for group,syms in U['groups'].items():
 for s in syms:
    if s not in FRAMES:continue
    f=FRAMES[s];m=META[s];complete=f.loc[f.index<TODAY]
    if len(complete)<25:continue
    last=f.iloc[-1];prev=complete.iloc[-1];adjratio=last.adj/last.close
    r={'symbol':s,'group':group,'quote_time':datetime.fromtimestamp(m['regularMarketTime'],timezone.utc).astimezone().isoformat(),'date':f.index[-1],'price':m.get('regularMarketPrice',last.close),'prev_close':prev.close,'day_pct':last.day*100,'gap_pct':last.gap*100,'last_complete_date':complete.index[-1]}
    for col in ['ret5','ret20','ret60','rsi14','rv20','rvol','dollar_volume20','closepos']:
        r[col]=float(last[col]);r['prior_'+col]=float(prev[col])
    for col in ['ma20','ma50','ma200','atr14','h5','l5','h20','l20']:
        r[col]=float(prev[col]/adjratio) if col.startswith('ma') or col=='atr14' else float(last[col]/adjratio)
    r.update(day_high=float(last.high),day_low=float(last.low),day_open=float(last.open),prior_high=float(prev.high),prior_low=float(prev.low),rs20=float(last.ret20-SPY.ret20.reindex(f.index).iloc[-1]))
    rows.append(r)
    if group=='benchmarks':continue
    ff=f.loc[f.index<TODAY].copy()
    ff['log_rvol']=np.log(ff.rvol.clip(lower=.05));ff['rs20']=ff.ret20-SPY.ret20
    ff['market20']=SPY.ret20;ff['market_dist200']=SPY.dist200;ff['market_rv20']=SPY.rv20
    ff['symbol']=s;ff['date']=ff.index
    for h in [5,10,20]:
        # Signal at close t, trade at next adjusted open, exit h trading bars later.
        ff[f'y{h}']=ff.adj.shift(-h)/ff.aopen.shift(-1)-1-.002
        dates=pd.Series(ff.index,index=ff.index);ff[f'end{h}']=dates.shift(-h)
    valid=ff.dropna(subset=FEATURES)
    if len(valid):latest.append(valid.iloc[-1][['symbol','date']+FEATURES].to_dict())
    train.append(valid)
pd.DataFrame(rows).to_csv(R/'市场与个股指标.csv',index=False)
for row in rows:
    for key,value in list(row.items()):
        if isinstance(value,float) and not math.isfinite(value):row[key]=None
(R/'snapshot.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
print('SNAPSHOT',len(rows))
for r in sorted([x for x in rows if x['group']!='benchmarks'],key=lambda x:x['day_pct'],reverse=True):
 print(r['symbol'],round(r['price'],2),'day',round(r['day_pct'],2),'5d',round(100*r['prior_ret5'],1),'20d',round(100*r['prior_ret20'],1),'RSI',round(r['prior_rsi14'],1),'MA20',round(r['ma20'],2),'ATR%',round(r['atr14']/r['price']*100,1))

def logistic(X,y,lam=.02):
    mean=X.mean(0);std=X.std(0);std[std<1e-10]=1
    Z=np.column_stack([np.ones(len(X)),np.clip((X-mean)/std,-5,5)])
    b=np.zeros(Z.shape[1]);reg=np.eye(len(b))*lam;reg[0,0]=0
    for _ in range(45):
        p=1/(1+np.exp(-np.clip(Z@b,-30,30)))
        grad=Z.T@(p-y)/len(y)+reg@b
        H=(Z.T*(p*(1-p)))@Z/len(y)+reg+np.eye(len(b))*1e-7
        step=np.linalg.solve(H,grad);b-=step
        if np.max(np.abs(step))<1e-6:break
    return mean,std,b
def predict(model,X):
    mean,std,b=model;z=np.column_stack([np.ones(len(X)),np.clip((X-mean)/std,-5,5)])
    return 1/(1+np.exp(-np.clip(z@b,-30,30)))
def auc(y,p):
    n1=y.sum();n0=len(y)-n1
    if not n1 or not n0:return None
    return float((pd.Series(p).rank().to_numpy()[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def stats(df):
    y=df.y.to_numpy();p=df.p.to_numpy();base=df.base.to_numpy()
    return {'n':len(df),'dates':df.date.nunique(),'event_rate':float(y.mean()),'brier':float(np.mean((p-y)**2)),'baseline_brier':float(np.mean((base-y)**2)),'auc':auc(y,p)}
if __name__=='__main__':
    data=pd.concat(train,ignore_index=True);live=pd.DataFrame(latest)
    bounds=['2024-01-01','2024-07-01','2025-01-01','2025-07-01','2026-01-01','2026-07-01',TODAY]
    output={'feature_names':FEATURES,'assumptions':{'cost_roundtrip':.002,'regularization':.02,'current_signal_date':'2026-09-14','current_universe_survivorship_bias':True,'fundamentals_and_current_news':'not in regression; independent current evidence overlay','models':'expanding chronological logistic, training labels strictly mature before test fold; no tuning','today_intraday':'excluded from regression; used only for separate entry screen'},'models':{},'predictions':live[['symbol','date']].to_dict('records')}
    for h,target in [(5,0),(10,0),(20,0),(20,.2),(20,-.15)]:
        key=f'h{h}_{"lt" if target<0 else "gt"}{target}';d=data.dropna(subset=[f'y{h}',f'end{h}']).copy()
        d['label']=((d[f'y{h}']<target) if target<0 else (d[f'y{h}']>target)).astype(int)
        tests=[];folds=[]
        for start,end in zip(bounds,bounds[1:]):
            tr=d[d[f'end{h}']<start];te=d[(d.date>=start)&(d.date<end)]
            if len(tr)<1000 or len(te)==0:continue
            model=logistic(tr[FEATURES].to_numpy(),tr.label.to_numpy())
            test=pd.DataFrame({'date':te.date,'symbol':te.symbol,'y':te.label,'p':predict(model,te[FEATURES].to_numpy()),'base':tr.label.mean()})
            vm=logistic(tr[['rv20']].to_numpy(),tr.label.to_numpy())
            test['vol_only']=predict(vm,te[['rv20']].to_numpy())
            tests.append(test);folds.append({'start':start,'end':end,**stats(test)})
        oos=pd.concat(tests);oos.to_csv(R/f'validation-{key}.csv',index=False)
        full=logistic(d[FEATURES].to_numpy(),d.label.to_numpy());p=predict(full,live[FEATURES].to_numpy())
        summary=stats(oos)
        summary['vol_only_brier']=float(np.mean((oos.vol_only-oos.y)**2))
        summary['vol_only_auc']=auc(oos.y.to_numpy(),oos.vol_only.to_numpy())
        # Date blocks preserve cross-stock dependence and overlapping horizon locally.
        daily=oos.assign(gain=(oos.base-oos.y)**2-(oos.p-oos.y)**2).groupby('date').gain.mean().to_numpy()
        rng=np.random.default_rng(20260914);boot=[];block=20
        for _ in range(1000):
            ids=np.concatenate([(j+np.arange(block))%len(daily) for j in rng.integers(len(daily),size=math.ceil(len(daily)/block))])[:len(daily)]
            boot.append(float(daily[ids].mean()))
        summary['date_block_brier_gain_ci95']=np.quantile(boot,[.025,.975]).tolist()
        summary['historical_discrimination_gate']=bool(summary['date_block_brier_gain_ci95'][0]>0 and summary['auc']>.55)
        summary['calibrated_current_probability']=False
        output['models'][key]={'validation':summary,'folds':folds,'coefficients':dict(zip(['intercept']+FEATURES,full[2].tolist()))}
        for i,v in enumerate(p):output['predictions'][i][key]=float(v)
        print(key,summary)
    (R/'model-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
