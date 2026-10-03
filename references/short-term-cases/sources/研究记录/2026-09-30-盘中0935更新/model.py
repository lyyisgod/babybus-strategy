"""Exploratory, timestamped opening-five-minute screen; no certified win rate."""
import json, importlib.util
from pathlib import Path
from datetime import datetime,time
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'2026-09-30-盘中净赚5筛选'
NY=ZoneInfo('America/New_York'); TODAY='2026-09-30'
sp=importlib.util.spec_from_file_location('previous_model',OLD/'path_model.py')
pm=importlib.util.module_from_spec(sp);sp.loader.exec_module(pm)

def load(s,mode):
    path=ROOT/'raw'/f'{s}-{mode}.json'
    if not path.exists():path=OLD/'raw'/f'{s}-{mode}.json'
    raw=json.loads(path.read_text()); z=raw['data']['chart']['result'][0]
    q=z['indicators']['quote'][0]; days={}
    for i,t in enumerate(z['timestamp']):
        if q['close'][i] is None:continue
        dt=datetime.fromtimestamp(t,NY)
        days.setdefault(dt.date().isoformat(),[]).append((dt,{k:q[k][i] for k in ['open','high','low','close','volume']}))
    return days,raw['fetched_at']

def first(day,live=False):
    a=[(dt,q) for dt,q in day if time(9,30)<=dt.time()<time(9,35)]
    if not a or a[0][0].time()!=time(9,30):return None
    if live and (len(a)<5 or a[-1][0].time()!=time(9,34)):return None
    o=float(a[0][1]['open']);c=float(a[-1][1]['close'])
    return dict(open=o,close=c,ret=c/o-1,volume=sum(q['volume'] or 0 for _,q in a),high=max(q['high'] or q['close'] for _,q in a),low=min(q['low'] or q['close'] for _,q in a))

def path(day,entry):
    for dt,q in day:
        if not time(9,35)<=dt.time()<time(16):continue
        o=q['open'] or q['close'];h=q['high'] or q['close'];l=q['low'] or q['close']
        if o<=entry*.97:return dict(hit_target=False,net=o/entry-1-.002,exit='stop_gap')
        if o>=entry*1.052:return dict(hit_target=True,net=.05,exit='target_gap')
        if l<=entry*.97:return dict(hit_target=False,net=-.032,exit='stop')
        if h>=entry*1.052:return dict(hit_target=True,net=.05,exit='target')
    r=[q for dt,q in day if time(9,35)<=dt.time()<time(16)]
    return dict(hit_target=False,net=r[-1]['close']/entry-1-.002,exit='close')

def screen():
    old=json.loads((OLD/'screen-results.json').read_text())['candidates']; rows=[]
    qqq,_=load('QQQ','live');qf=first(qqq[TODAY],True)
    for x in old:
        if not x['eligible']:continue
        data,fetched=load(x['symbol'],'live');day=data.get(TODAY,[]);f=first(day,True)
        if not f or datetime.fromisoformat(fetched).astimezone(NY).time()<time(9,35):continue
        r=[(dt,q) for dt,q in day if time(9,30)<=dt.time()<time(16)]
        volumes=[(q['close'],q['volume'] or 0) for _,q in r];vol=sum(v for _,v in volumes)
        vwap=sum(c*v for c,v in volumes)/vol if vol>0 else None
        row=dict(x);row.update(first5=f,price=r[-1][1]['close'],observed_at=r[-1][0].isoformat(),fetched_at=fetched,
            return_pct=(r[-1][1]['close']/x['previous_close']-1)*100,first5_relative_pp=(f['ret']-qf['ret'])*100,
            reg_volume=vol,vwap_close_approximation=vwap,first5_valid_volume=f['volume']>0,
            opening_score_descriptive=x['atr14_pct']+.5*max(-5,min(5,(f['ret']-qf['ret'])*100)))
        rows.append(row)
    rows.sort(key=lambda x:x['opening_score_descriptive'],reverse=True)
    (ROOT/'opening-screen.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([{k:x[k] for k in ['symbol','price','observed_at','return_pct','first5','first5_relative_pp','reg_volume','vwap_close_approximation','opening_score_descriptive']} for x in rows[:20]],indent=2))

def model():
    syms=json.loads((ROOT/'selected-history-symbols.json').read_text());qqq,_=load('QQQ','history');liveqqq,_=load('QQQ','live');qnow=first(liveqqq[TODAY],True)
    hist=[];curr=[];missing=[]
    for s in syms:
        intr,_=load(s,'history');daily,_=load(s,'daily');live,fetched=load(s,'live')
        priorfirst={d:first(v) for d,v in intr.items() if d<TODAY}
        for d in sorted(set(intr)|{TODAY}):
            if d>TODAY:continue
            isnow=d==TODAY;f=first(live[TODAY],True) if isnow else first(intr[d]);qf=qnow if isnow else first(qqq.get(d,[]))
            priordays=sorted(z for z in daily if z<d)
            volumes=[priorfirst[z]['volume'] for z in sorted(priorfirst) if z<d and priorfirst[z] and priorfirst[z]['volume']>0][-20:]
            if not f or not qf or f['volume']<=0 or len(volumes)<20 or len(priordays)<21:continue
            ds=[daily[z][-1][1] for z in priordays[-21:]];cc=np.array([v['close'] for v in ds]);hh=np.array([v['high'] for v in ds]);ll=np.array([v['low'] for v in ds])
            tr=np.maximum(hh[1:]-ll[1:],np.maximum(abs(hh[1:]-cc[:-1]),abs(ll[1:]-cc[:-1])))
            feat=[f['ret'],f['ret']-qf['ret'],float(tr[-14:].mean()/cc[-1]),f['volume']/float(np.mean(volumes))]
            row=dict(symbol=s,date=d,anchor=f['close'],features=feat,first5=f)
            if isnow:
                if datetime.fromisoformat(fetched).astimezone(NY).time()>=time(9,35):curr.append(row)
            else:
                reg=[(dt,q) for dt,q in intr[d] if time(9,30)<=dt.time()<time(16)]
                if len(reg)<70 or reg[-1][0].time()!=time(15,55):continue
                row.update(path(intr[d],f['close']));hist.append(row)
        if not any(r['symbol']==s for r in curr):missing.append(s)
    hist.sort(key=lambda r:(r['date'],r['symbol']));dates=sorted({r['date'] for r in hist});oos=[]
    for i in range(25,len(dates)):
        train=[r for r in hist if r['date']<dates[i]];test=[dict(r) for r in hist if r['date']==dates[i]]
        pred=pm.fit_predict(train,test);base=float(np.mean([r['hit_target'] for r in train]))
        for r,p in zip(test,pred):r.update(p=float(p),base=base,train_last_date=dates[i-1]);oos.append(r)
    top=[max([r for r in oos if r['date']==d],key=lambda r:r['p']) for d in sorted({r['date'] for r in oos})]
    if hist and curr:
        for r,p in zip(curr,pm.fit_predict(hist,curr)):
            own=[z for z in hist if z['symbol']==r['symbol']];n=len(own);h=sum(z['hit_target'] for z in own)
            r.update(model_output_unverified=float(p),history_n=n,history_target_before_stop_fraction=h/n if n else None,
                history_mean_net=float(np.mean([z['net'] for z in own])) if n else None,
                first5_relative_volume=r['features'][3],momentum_gate=r['features'][0]>0 and r['features'][1]>0)
    curr.sort(key=lambda r:r.get('model_output_unverified',-1),reverse=True)
    result=dict(generated_at=datetime.now(NY).isoformat(),hist_dates=len(dates),hist_stockdays=len(hist),oos_stockdays=len(oos),oos_dates=len(top),
       oos_top1_target_hits=sum(r['hit_target'] for r in top),oos_top1_target_fraction=float(np.mean([r['hit_target'] for r in top])) if top else None,
       oos_top1_mean_net=float(np.mean([r['net'] for r in top])) if top else None,
       brier_model=float(np.mean([(r['p']-r['hit_target'])**2 for r in oos])) if oos else None,
       brier_baseline=float(np.mean([(r['base']-r['hit_target'])**2 for r in oos])) if oos else None,
       current=curr,missing=missing,certified_win_rate=None,
       limitations=['Current selected survivors; selection bias','20 historical opening bars for volume leaves a shorter evaluation period','No untouched prospective audit','First5 model anchor is 09:35; later executable price is different','Actual bid ask unavailable; 0.2% cost assumption','Same 5min bar stop first; no tick reconstruction'])
    (ROOT/'model-results.json').write_text(json.dumps(result,indent=2));(ROOT/'oos.json').write_text(json.dumps(oos,indent=2));(ROOT/'historical-paths.json').write_text(json.dumps(hist,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    screen() if sys.argv[1]=='screen' else model()
