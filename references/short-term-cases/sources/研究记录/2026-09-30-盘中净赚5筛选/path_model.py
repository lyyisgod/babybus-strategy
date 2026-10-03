"""Conservative historical +5.2% target / -3% stop paths from 09:20.

Current candidate selection and the historical universe are biased; all model
outputs are exploratory and must not be represented as certified win rates.
"""
import json
from datetime import datetime,time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
TODAY='2026-09-30'
TARGET=.052
STOP=.03
COST=.002


def load(symbol,mode):
    raw=json.loads((ROOT/'raw'/f'{symbol}-{mode}.json').read_text())
    z=raw['data']['chart']['result'][0];q=z['indicators']['quote'][0];out={}
    for i,t in enumerate(z['timestamp']):
        dt=datetime.fromtimestamp(t,NY)
        if q['close'][i] is None:continue
        out.setdefault(dt.date().isoformat(),[]).append((dt,{k:q[k][i] for k in ['open','high','low','close','volume']}))
    return out,raw['fetched_at']


def snapshot(day):
    snap=[q for dt,q in day if dt.time()==time(9,15)]
    return float(snap[0]['close']) if snap else None


def trade_path(day,entry):
    high=entry*(1+TARGET);low=entry*(1-STOP)
    for dt,q in day:
        if dt.time()<time(9,20) or dt.time()>=time(16):continue
        o=float(q['open'] or q['close']);h=float(q['high'] or q['close']);l=float(q['low'] or q['close'])
        # Stops apply from entry; target realization is limited to regular hours.
        if o<=low:return {'hit_target':False,'exit':'stop_gap','net':o/entry-1-COST}
        target_allowed=dt.time()>=time(9,30)
        if target_allowed and o>=high:return {'hit_target':True,'exit':'target_gap','net':TARGET-COST}
        if l<=low:return {'hit_target':False,'exit':'stop','net':-STOP-COST}
        if target_allowed and h>=high:return {'hit_target':True,'exit':'target','net':TARGET-COST}
    regular=[q for dt,q in day if time(9,30)<=dt.time()<time(16)]
    return {'hit_target':False,'exit':'close','net':float(regular[-1]['close'])/entry-1-COST}


def wilson(h,n):
    p=h/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-width),float(center+width)]


def fit_predict(train,current):
    x=np.array([r['features'] for r in train]);y=np.array([r['hit_target'] for r in train],int)
    mean=x.mean(axis=0);std=x.std(axis=0);std[std<1e-12]=1
    a=np.column_stack([np.ones(len(x)),(x-mean)/std])
    beta=np.zeros(a.shape[1]);penalty=np.diag([0]+[2]*(a.shape[1]-1))
    # Sum log loss + (1/(2*C))*slope^2; C=.5, intercept unpenalized.
    for _ in range(100):
        p=1/(1+np.exp(-np.clip(a@beta,-35,35)));w=np.maximum(p*(1-p),1e-12)
        grad=a.T@(p-y)+penalty@beta
        hess=a.T@(w[:,None]*a)+penalty+np.eye(a.shape[1])*1e-10
        step=np.linalg.solve(hess,grad);beta-=step
        if np.max(abs(step))<1e-9:break
    else:raise RuntimeError('Logistic fit did not converge')
    v=np.column_stack([np.ones(len(current)),(np.array([r['features'] for r in current])-mean)/std])
    return 1/(1+np.exp(-np.clip(v@beta,-35,35)))


def main():
    # Additional strategy/catalyst contrasts retained separately from rank-selected pool.
    syms=json.loads((ROOT/'selected-history-symbols.json').read_text())+['HOOD','MSTR','NVTS','JBL','AVGO']
    qqq,_=load('QQQ','history');hist=[];current=[]
    for s in syms:
        data,fetched=load(s,'history');daily,_=load(s,'daily')
        closes={d:float(v[-1][1]['close']) for d,v in daily.items() if d<TODAY}
        for d in sorted(data):
            if d>TODAY or d not in qqq:continue
            a=snapshot(data[d]);qq=snapshot(qqq[d]);prior=[z for z in closes if z<d];qprior=[z for z in qqq if z<d]
            if a is None or qq is None or len(prior)<21 or not qprior:continue
            prior=sorted(prior);pd=prior[-1];ps=max(qprior)
            if pd!=ps:continue
            qr=[q for dt,q in qqq[ps] if time(9,30)<=dt.time()<time(16)]
            if not qr:continue
            ret=a/closes[pd]-1;qret=qq/float(qr[-1]['close'])-1
            old=[daily[z][-1][1] for z in prior[-21:]]
            cc=np.array([z['close'] for z in old]);hh=np.array([z['high'] for z in old]);ll=np.array([z['low'] for z in old]);tr=np.maximum(hh[1:]-ll[1:],np.maximum(abs(hh[1:]-cc[:-1]),abs(ll[1:]-cc[:-1])))
            feat=[ret,ret-qret,float(tr[-14:].mean()/cc[-1]),float(cc[-1]/cc[-4]-1)]
            row={'symbol':s,'date':d,'anchor':a,'premarket_return':ret,'features':feat}
            if d==TODAY:
                if datetime.fromisoformat(fetched).astimezone(NY).time()>=time(9,20):current.append(row)
                continue
            regular=[(dt,q) for dt,q in data[d] if time(9,30)<=dt.time()<time(16)]
            if len(regular)<70 or regular[-1][0].time()!=time(15,55):continue
            path=trade_path(data[d],a);row.update(path)
            row['touch_5p2_ignoring_stop']=max(float(q['high'] or q['close']) for dt,q in regular)>=a*(1+TARGET)
            hist.append(row)
    hist=sorted(hist,key=lambda r:(r['date'],r['symbol']));dates=sorted({r['date'] for r in hist});oos=[]
    for i in range(25,len(dates)):
        train=[r for r in hist if r['date']<dates[i]];test=[dict(r) for r in hist if r['date']==dates[i]]
        preds=fit_predict(train,test);base=float(np.mean([r['hit_target'] for r in train]))
        for r,p in zip(test,preds):r.update(model_output=float(p),base_output=base,train_last_date=dates[i-1]);oos.append(r)
    scored=[]
    if current:
        pred=fit_predict(hist,current)
        for r,p in zip(current,pred):
            selected=[z for z in hist if z['symbol']==r['symbol']];n=len(selected);h=sum(z['hit_target'] for z in selected)
            if n==0:
                r.update(model_output_unverified=None,history_n=0,reason='No valid complete historical paths; excluded from model interpretation')
                scored.append(r)
                continue
            matching=[z for z in selected if abs(z['premarket_return']-r['premarket_return'])<=.01]
            r.update(model_output_unverified=float(p),history_n=n,hit_before_stop_count=h,hit_before_stop_fraction=h/n,
                     hit_wilson95_descriptive=wilson(h,n),touch_ignoring_stop_fraction=float(np.mean([z['touch_5p2_ignoring_stop'] for z in selected])),
                     plan_mean_net_return=float(np.mean([z['net'] for z in selected])),plan_profitable_fraction=float(np.mean([z['net']>0 for z in selected])),
                     matching_premarket_n=len(matching),matching_target_before_stop_count=sum(z['hit_target'] for z in matching),
                     gross_target_price=r['anchor']*(1+TARGET),stop_price=r['anchor']*(1-STOP))
            scored.append(r)
    scored.sort(key=lambda r:r['model_output_unverified'] if r['model_output_unverified'] is not None else -1,reverse=True)
    actual=np.array([r['hit_target'] for r in oos],float);pred=np.array([r['model_output'] for r in oos]);base=np.array([r['base_output'] for r in oos])
    top=[max([r for r in oos if r['date']==d],key=lambda r:r['model_output']) for d in sorted({r['date'] for r in oos})]
    results={'generated_at':datetime.now(NY).isoformat(),'features':['premarket return','premarket relative vs QQQ','prior14 simple ATR fraction','prior3day return'],
             'fixed_logistic_C':.5,'hist_stock_days':len(hist),'hist_dates':len(dates),'oos_stock_days':len(oos),'oos_dates':len(top),
             'brier_model':float(np.mean((actual-pred)**2)),'brier_expanding_rate_baseline':float(np.mean((actual-base)**2)),
             'daily_top1_target_hit_fraction':float(np.mean([r['hit_target'] for r in top])),
             'daily_top1_mean_plan_net':float(np.mean([r['net'] for r in top])),
             'same_date_equal_pool_mean_plan_net':float(np.mean([np.mean([r['net'] for r in oos if r['date']==d]) for d in sorted({r['date'] for r in oos})])),
             'current_candidates':scored,'certified_win_rate':None,'actual_execution_bid_ask_verified':False,
             'limitations':['chosen current survivors, selected high-volatility candidates: selection bias','33 or fewer unseen-by-fit market dates not independent across stocks','cost .2% assumed; actual spread unknown','five-minute stop/target order conservative but not tick reconstruction','current 09:20 anchor is not executable ask','no untouched prospective audit; no reliable high-win certification']}
    (ROOT/'path-model-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'historical-paths.json').write_text(json.dumps(hist,indent=2)+'\n');(ROOT/'path-oos.json').write_text(json.dumps(oos,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
