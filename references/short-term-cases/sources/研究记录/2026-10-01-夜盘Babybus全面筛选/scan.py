from pathlib import Path
import sys,json,datetime,hashlib,urllib.request,ssl,argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,certifi
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import load_config,fetch_market,load_market,read_records,save,calendar
from strategies.bb_washout_bounce.factors import rsi,streak
from strategies.bb_washout_bounce.backtest import Engine
from strategies.bb_washout_bounce.daily import daily_report
P=Path(__file__).resolve().parent;PROTO=json.loads((P/'protocol.json').read_text());CFG=load_config()
def getraw(s):
    url=f'https://query2.finance.yahoo.com/v8/finance/chart/{s}?range=5d&interval=5m&includePrePost=true&events=div%2Csplits'
    a=dict(symbol=s,url=url,fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r:raw=r.read()
        out=P/'live-raw'/f'{s.replace("^","INDEX_").replace("=","_")}.json';out.write_bytes(raw)
        a.update(file=out.name,sha256=hashlib.sha256(raw).hexdigest())
        assert json.loads(raw)['chart']['result'][0]
    except Exception as e:a['error']=str(e)
    return a
def fetch():
    cfg=dict(CFG);cfg['data']=dict(CFG['data'],workers=8)
    symbols=PROTO['stocks']+PROTO['proxies']+['IXG','^MOVE','BTC-USD','NQ=F']
    res=fetch_market(symbols,P/'daily-raw',cfg);print('daily requested/success',len(res),sum('error' not in r for r in res),flush=True)
    (P/'live-raw').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=8) as pool:res=list(pool.map(getraw,symbols))
    save(P/'live-raw/manifest.json',res);print('live requested/success',len(res),sum('error' not in r for r in res),flush=True)
def macrofetch():
    result=[]
    for s in ['BAMLH0A0HYM2','DFII10']:
        url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}'
        row=dict(series=s,source=url,fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Research/1.0'})
            with urllib.request.urlopen(req,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r:raw=r.read()
            assert raw.startswith(b'DATE,') or raw.startswith(b'observation_date,'),'FRED returned non-CSV'
            (P/f'{s}.csv').write_bytes(raw);row.update(file=f'{s}.csv',sha256=hashlib.sha256(raw).hexdigest())
        except Exception as e:row['error']=str(e)
        result.append(row)
    save(P/'macro-manifest.json',result);print(result,flush=True)
def gates(f):
    return dict(dd35_65=bool(.35<=f['dd_high']<=.65),rv90=bool(f['rv90']>=.6),weekly_vol=bool(f['weekly_vol']>=.12),washout_or_low20=bool(f['dd10']>=.15 or 0<=f['dist_low20']<=.08),bb_or_low60=bool(f['close']<=f['weekly_mid'] or 0<=f['dist_low60']<=.12),no_chase=bool(max(f['green_streak'],f['up_streak'],f['rs_streak'])<3),volume_green=bool(f['close']>f['open'] and f['vol_ratio20']>=1.5 and f['close_range']>=.5))
def light(d,sector,cal):
    a=d[['open','high','low','close']].mul(d.factor,axis=0);c=a.close
    f=pd.DataFrame(index=d.index);f['open']=d.open;f['close']=d.close;f['day_return']=c.pct_change(fill_method=None)
    f['dd_high']=1-c/a.high.rolling(252,min_periods=20).max();f['dd10']=1-c/a.high.rolling(10).max()
    for n in [20,60]:
        low=a.low.shift().rolling(n).min();f[f'low{n}']=low/d.factor;f[f'dist_low{n}']=c/low-1
    f['rv90']=np.log(c/c.shift()).rolling(90).std(ddof=1)*np.sqrt(252)
    f['vol_ratio20']=d.volume/d.volume.shift().rolling(20).mean();f['adv20']=(d.close*d.volume).shift().rolling(20).mean()
    f['green_streak']=streak(a.close>a.open);f['up_streak']=streak(c>c.shift())
    f['rs_streak']=streak(c.pct_change(fill_method=None)>sector.adjclose.pct_change(fill_method=None).reindex(c.index))
    f['close_range']=(a.close-a.low)/(a.high-a.low).replace(0,np.nan)
    f['ma200']=c.rolling(200).mean()/d.factor;f['rsi2']=rsi(c,2);f['rsi14']=rsi(c,14)
    tr=pd.concat([a.high-a.low,(a.high-c.shift()).abs(),(a.low-c.shift()).abs()],axis=1).max(axis=1)
    # Wilder ATR with explicit initial seed.
    atr=np.full(len(tr),np.nan)
    if len(tr)>=14:
        atr[13]=tr.iloc[:14].mean()
        for i in range(14,len(tr)):atr[i]=(atr[i-1]*13+tr.iloc[i])/14
    f['atr14']=atr/d.factor
    sessions=cal.sessions_in_range(d.index[0],cal.next_session(d.index[-1]));ends=pd.Series(sessions,index=sessions).groupby(sessions.to_period('W-FRI')).max()
    ends=pd.DatetimeIndex([x for x in ends if cal.next_session(x).to_period('W-FRI')!=x.to_period('W-FRI')]);w=c.reindex(ends).dropna()
    mid=w.rolling(20).mean();sd=w.rolling(20).std(ddof=0)
    f['weekly_mid']=mid.reindex(f.index,method='ffill')/d.factor;f['weekly_lower']=(mid-2*sd).reindex(f.index,method='ffill')/d.factor
    f['weekly_vol']=w.pct_change(fill_method=None).rolling(12).std(ddof=1).reindex(f.index,method='ffill')
    f['relative20']=c.pct_change(20,fill_method=None)-sector.adjclose.pct_change(20,fill_method=None).reindex(c.index)
    f['next_high']=a.high.shift(-1)/c-1;f['next_low']=a.low.shift(-1)/c-1;f['next_close_net']=c.shift(-1)/c-1-PROTO['cost_roundtrip']
    f['washout_analog']=(f.dd_high.between(.35,.65))&((f.dd10>=.15)|f.dist_low20.between(0,.08))&(f.rv90>=.6)
    f['momentum_analog']=(f.day_return>=.03)&(f.vol_ratio20>=1.2)&(f.close>f.open)&(f.close>f.ma200)
    f['lrcx_branch']=(f.close>f.ma200)&(f.rsi2<=20)&(c.pct_change(3,fill_method=None)<0)
    return f
def live(s):
    raw=json.loads((P/'live-raw'/f'{s.replace("^","INDEX_").replace("=","_")}.json').read_text())['chart']['result'][0];m=raw['meta']
    d=pd.DataFrame(raw['indicators']['quote'][0],index=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York')).dropna(subset=['close'])
    d=d[d.index.strftime('%Y-%m-%d')==PROTO['today']];reg=d[(d.index.hour*60+d.index.minute>=570)&(d.index.hour*60+d.index.minute<960)]
    post=d[(d.index.hour*60+d.index.minute>=960)&(d.index.hour*60+d.index.minute<1200)];last=post.iloc[-1] if len(post) else None
    vwap=(((reg.high+reg.low+reg.close)/3)*reg.volume).sum()/reg.volume.sum() if len(reg) and reg.volume.sum()>0 else np.nan
    return dict(last_regular_meta=m.get('regularMarketPrice'),last_regular_quote_at=pd.Timestamp(m['regularMarketTime'],unit='s',tz='UTC'),last_extended_price=float(last.close) if last is not None else None,last_extended_bar_at=post.index[-1] if last is not None else None,last_extended_bar_volume=int(last.volume) if last is not None else None,post_volume=float(post.volume.sum()) if len(post) else 0,post_volume_warning='zero-volume is provider missing trade-size; preserved price proxy, cannot confirm afterhours volume',vwap5m_proxy=vwap,bid_ask=None,overnight_quote=None,source='Yahoo5m extendedbars; bar timestamps not executable quote')
def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;mid=(p+z*z/(2*n))/(1+z*z/n);rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [mid-rad,mid+rad]
def analyze():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();market,meta,errors,last=load_market(P/'daily-raw',CFG,now);market={s:d.loc['2021-01-01':] for s,d in market.items()};cal=calendar(CFG)
    save(P/'loader-audit.json',dict(as_of=now,last=last,errors=errors))
    manifests=json.loads((P/'live-raw/manifest.json').read_text());factors={};rows=[];errs={}
    for s in PROTO['stocks']:
        try:
            d=market[s];assert d.index[-1]==last,'daily stale';f=light(d,market['SMH'],cal);factors[s]=f;r=f.iloc[-1].to_dict();r.update(ticker=s,session=last,gates=gates(r),high20=float(d.high.iloc[-21:-1].max()),low_today=float(d.low.iloc[-1]),high_today=float(d.high.iloc[-1]));r['gate_count']=sum(r['gates'].values());r.update(live(s));p=r['last_extended_price'];r['extended_ret_vs_close']=p/r['close']-1 if p else None;rows.append(r)
        except Exception as e:errs[s]=str(e)
    save(P/'all-factors.json',rows);save(P/'factor-errors.json',errs)
    pd.DataFrame(rows).drop(columns=['gates']).to_csv(P/'all-factors.csv',index=False)
    low=sorted([r for r in rows if r['adv20']>=50e6 and r['day_return']<=.03],key=lambda r:(r['gate_count'],r['close']>r['vwap5m_proxy'],r['vol_ratio20']),reverse=True);save(P/'washout-ranking.json',low)
    trend=sorted([r for r in rows if r['adv20']>=50e6 and r['close']>r['ma200'] and r['close']>r['vwap5m_proxy'] and r['day_return']>.03],key=lambda r:(r['vol_ratio20'],r['relative20']),reverse=True);save(P/'momentum-ranking.json',trend)
    save(P/'lrcx-branch.json',[r for r in rows if r['lrcx_branch']]);print('factors',len(rows),'errors',errs,flush=True)
    for label,rank in [('WASHOUT',low),('MOMENTUM',trend)]:
        print(label,flush=True)
        for r in rank[:14]:print({k:r[k] for k in ['ticker','close','day_return','last_extended_price','dd_high','dd10','vol_ratio20','gate_count','weekly_vol','rsi2','atr14','vwap5m_proxy']},flush=True)
    # Historical price-only diagnostics; all dates are reused research, not untouched audits.
    records=[];stats=[]
    for s,f in factors.items():
        for kind in ['washout_analog','momentum_analog']:
            selected=f.loc['2024-01-01':].iloc[:-1];selected=selected[selected[kind]&selected.next_close_net.notna()&(selected.adv20>=50e6)];prev=None;events=[]
            for day,row in selected.iterrows():
                if prev is not None and day<=cal.next_session(prev):continue
                prev=day;events.append(dict(ticker=s,kind=kind,signal_session=day,exit_session=cal.next_session(day),high=row.next_high,low=row.next_low,close_net=row.next_close_net))
            records+=events;n=len(events);k5=sum(x['high']>=.05 for x in events);k10=sum(x['high']>=.10 for x in events)
            stats.append(dict(ticker=s,kind=kind,n=n,touch5=k5,touch10=k10,close_net5=sum(x['close_net']>=.05 for x in events),down3=sum(x['low']<=-.03 for x in events),touch5_wilson95_descriptive=wilson(k5,n),touch10_wilson95_descriptive=wilson(k10,n),mean_close_net=np.mean([x['close_net'] for x in events]) if n else None,current_probability=None,warning='prior regularclose proxy; not overnight fill; priceonly; reused period; current survivor universe; Wilson assumes independent trials and not calibration'))
    save(P/'historical-price-only-events.json',records);save(P/'historical-price-only-stats.json',stats)
    securities=json.loads((ROOT/'config/bb_universe.json').read_text())['securities'];inputs={k:read_records(ROOT/'data/bb_washout_bounce'/(k+'.jsonl')) for k in ['fundamentals','macro','events','event_coverage','supports','conflicts']}
    e=Engine(market,securities,inputs,CFG);strict=daily_report(e,last,P/'strict-original',now=now);save(P/'strict-summary.json',strict);print('STRICT',strict,flush=True)
    j=market['JNK'].adjclose;mon=j.resample('ME').last();mon=mon[mon.index<=last];completed=j.resample('W-FRI').last();completed=completed[completed.index<=last]
    credit=dict(as_of=now,session=last,jnk_ret20=j.iloc[-1]/j.iloc[-21]-1,jnk_ret5=j.iloc[-1]/j.iloc[-6]-1,completed_month=mon.index[-1],bear_month=mon.tail(3).mean()<mon.tail(10).mean(),completed_week=completed.index[-1],weekly_break=completed.iloc[-1]<completed.iloc[-9:-1].min()*.995,oas20_missing=True,credit_state='UNKNOWN',nfp_protection=True)
    for s in ['BAMLH0A0HYM2','DFII10']:
        if (P/f'{s}.csv').exists():
            try:
                mf=pd.read_csv(P/f'{s}.csv');mf.index=pd.to_datetime(mf.iloc[:,0]);vals=pd.to_numeric(mf[s],errors='coerce').dropna();vals=vals.loc[:last];date=vals.index[-1];base=cal.session_offset(cal.date_to_session(date,direction='previous'),-20);v=vals.loc[:base].iloc[-1]
                credit[s]=dict(observed_date=date,value=vals.iloc[-1],base_date=vals.loc[:base].index[-1],base_value=v,change20_bp=(vals.iloc[-1]-v)*100,source=f'https://fred.stlouisfed.org/series/{s}',current_download_not_historical_vintage=True)
                if s=='BAMLH0A0HYM2':credit['oas20_missing']=False
            except Exception as e:credit[s]=dict(error=str(e))
    if credit['weekly_break'] or credit['bear_month'] or credit.get('BAMLH0A0HYM2',{}).get('change20_bp',-999)>=50:credit['credit_state']='OFF'
    for s in ['QQQ','SMH','IXG','^MOVE','^VIX','^TNX','BZ=F','NQ=F','BTC-USD']:
        if s in market:
            d=market[s];credit[s]=dict(session=d.index[-1],close=d.close.iloc[-1],dayreturn=d.adjclose.iloc[-1]/d.adjclose.iloc[-2]-1,return20=d.adjclose.iloc[-1]/d.adjclose.iloc[-21]-1)
    save(P/'credit-context.json',credit);print('CREDIT',credit,flush=True)
    # Verify current factor path is unchanged by removing future rows.
    probes={}
    for s in ['IREN','APLD','COHR','ON']:
        d=market[s];cut=d.index[-25];trunc=light(d.loc[:cut],market['SMH'].loc[:cut],cal);full=factors[s]
        cols=['dd_high','dd10','rv90','weekly_mid','weekly_vol','ma200','rsi2','atr14','vol_ratio20']
        probes[s]=bool(np.allclose(trunc.loc[cut,cols].astype(float),full.loc[cut,cols].astype(float),equal_nan=True))
    hashok=all(hashlib.sha256((P/'live-raw'/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifests if 'error' not in x)
    save(P/'verification.json',dict(prefix_factor_checks=probes,live_hashes=hashok,daily_hashes_checked_by_loader=True,last_completed=last,full_babybus_investment_validation=False,overnight_execution_validation=False));assert all(probes.values()) and hashok
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='fetch':fetch()
    elif mode=='macro':macrofetch()
    elif mode=='analyze':analyze()
