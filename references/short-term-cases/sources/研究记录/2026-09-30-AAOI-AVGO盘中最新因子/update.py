import datetime as dt
import importlib.util
import json
import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
P=json.loads((BASE/'protocol.json').read_text())
source=BASE.parent/'2026-09-30-AAOI十交易日走势/model.py'
spec=importlib.util.spec_from_file_location('frozen10',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def daily(symbol):
    raw=json.loads((BASE/'raw'/f'{symbol}-daily.json').read_text())['chart']['result'][0]
    d=pd.DataFrame(raw['indicators']['quote'][0])
    d.index=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York').date
    d=d.dropna(subset=['open','high','low','close'])
    d.iloc[-1,d.columns.get_loc('close')]=float(raw['meta']['regularMarketPrice'])
    return d


def minute_factors(symbol):
    raw=json.loads((BASE/'raw'/f'{symbol}-live.json').read_text())['chart']['result'][0]
    q=pd.DataFrame(raw['indicators']['quote'][0])
    q.index=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York')
    q=q.loc[(q.index.date==dt.date(2026,9,30)) & (q.index.hour*60+q.index.minute>=570) & (q.index.hour*60+q.index.minute<960)].dropna()
    q=q.loc[q.volume>0]
    vw=float((((q.high+q.low+q.close)/3)*q.volume).sum()/q.volume.sum())
    return {'bar_approx_vwap':vw,'volume':int(q.volume.sum()),'last_completed_nonzero_bar':q.index[-1].isoformat(),'bars':len(q)}


def forecast(symbol):
    raw=daily(symbol);df=raw.iloc[:-1].copy()
    f=m.prior.indicators(df)
    f['rv20']=np.log(df.close/df.close.shift()).rolling(20).std(ddof=1)
    risk=f.rv20*np.sqrt(10)
    logs=np.log(df.close.shift(-10)/df.close);y=logs/risk
    valid=f[m.NAMES].notna().all(axis=1)&y.notna()&(risk>0)
    ids=np.flatnonzero(valid.to_numpy()).tolist()
    start=next(i for i,d in enumerate(df.index) if str(d)>='2023-01-01')
    rows=[]
    for i in range(start,len(df)-10,11):
        preds,_=m.fit_predict(f,y,[j for j in ids if j+10<=i],f.iloc[i])
        rows.append({'signal_date':str(df.index[i]),'label_end':str(df.index[i+10]),'actual_log_return':float(logs.iloc[i]),'risk_scale':float(risk.iloc[i]),**{k:v*float(risk.iloc[i]) for k,v in preds.items()},'zero':0.})
    cf=m.prior.indicators(raw);cf['rv20']=np.log(raw.close/raw.close.shift()).rolling(20).std(ddof=1)
    query=cf.iloc[-1];preds,neighbors=m.fit_predict(f,y,ids,query)
    px=P['quotes'][symbol]['price'];rs=float(query.rv20*np.sqrt(10))
    pl={k:v*rs for k,v in preds.items()}
    residual=np.array([(x['actual_log_return']-x['ensemble'])/x['risk_scale'] for x in rows])
    bands=px*np.exp(pl['ensemble']+np.quantile(residual,[.1,.25,.5,.75,.9])*rs)
    forward=np.array([float(df.close.iloc[i+10]/df.close.iloc[i]-1) for i in neighbors])
    highs=np.array([float(df.high.iloc[i+1:i+11].max()/df.close.iloc[i]-1) for i in neighbors])
    lows=np.array([float(df.low.iloc[i+1:i+11].min()/df.close.iloc[i]-1) for i in neighbors])
    out={'prices':{k:float(px*np.exp(v)) for k,v in pl.items()},'center_return':float(np.expm1(pl['ensemble'])),
         'residual_reference_quantiles':dict(zip(['p10','p25','p50','p75','p90'],map(float,bands))),
         'evaluation':m.metrics(rows),'evaluation_by_year':{},'evaluation_rows':rows,
         'neighbors':[{'signal_date':str(df.index[i]),'terminal_date':str(df.index[i+10]),'index':i} for i in neighbors],
         'diagnostic_counts':{'n':len(neighbors),'terminal_up':int((forward>0).sum()),'terminal_plus5':int((forward>=.05).sum()),'terminal_plus10':int((forward>=.1).sum()),'high_plus5':int((highs>=.05).sum()),'low_minus5':int((lows<=-.05).sum())}}
    for year in sorted(set(x['signal_date'][:4] for x in rows)):
        out['evaluation_by_year'][year]=m.metrics([x for x in rows if x['signal_date'].startswith(year)])
    if symbol=='AAOI':
        paths=[m.stop_path(df,i,float(df.close.iloc[i]),px) for i in neighbors]
        out['user_93_110']={'path_counts':{k:sum(x['kind']==k for x in paths) for k in ['target','stop','timeout']},
            'mean_net':float(np.mean([x['net'] for x in paths])),'terminal_ge110':int((forward>=110/px-1).sum()),
            'high_ge110':int((highs>=110/px-1).sum()),'low_le93':int((lows<=93/px-1).sum()),
            'rr':(110-px)/(px-93),'risk_pct':1-93/px,'reward_pct':110/px-1}
    return out


def factors(symbol):
    d=daily(symbol);c=d.close;px=float(c.iloc[-1]);f=m.prior.indicators(d).iloc[-1]
    mf=minute_factors(symbol)
    out={'price':px,'quote_time':dt.datetime.fromtimestamp(P['quotes'][symbol]['epoch'],dt.timezone.utc).astimezone(ZoneInfo('America/New_York')).isoformat(),
         'day_change':px/c.iloc[-2]-1,'open':float(d.open.iloc[-1]),'high':float(d.high.iloc[-1]),'low':float(d.low.iloc[-1]),
         'opening_gap_vs_prior_close':float(d.open.iloc[-1]/c.iloc[-2]-1),
         'return3':float(px/c.iloc[-4]-1),'return5':float(f.return5),'return20':float(f.return20),
         'rsi2':float(f.rsi2),'rsi14':float(f.rsi14),'atr14_partial':float(f.atr14),
         'atr14_last_complete':float(m.prior.indicators(d.iloc[:-1]).atr14.iloc[-1]),
         'rv20_annual':float(np.log(c/c.shift()).tail(20).std(ddof=1)*np.sqrt(252)),
         'ma':{str(n):float(c.tail(n).mean()) for n in [5,10,20,30,50,60,200]},
         'ma20_slope5':float(c.rolling(20).mean().iloc[-1]/c.rolling(20).mean().iloc[-6]-1),
         'low20_complete':float(d.low.iloc[-21:-1].min()),'high20_complete':float(d.high.iloc[-21:-1].max()),
         'volume20_mean_complete':float(d.volume.iloc[-21:-1].mean()),'intraday':mf,
         'distance_vwap':px/mf['bar_approx_vwap']-1,'relative':{}}
    for bench in ['QQQ','SOXX']:
        b=daily(bench);bc=b.close
        pairs=pd.concat([np.log(c.iloc[:-1]/c.iloc[:-1].shift()),np.log(bc.iloc[:-1]/bc.iloc[:-1].shift())],axis=1).dropna().tail(60)
        beta=float(pairs.iloc[:,0].cov(pairs.iloc[:,1])/pairs.iloc[:,1].var(ddof=1))
        out['relative'][bench]={'day_excess_pp':float((px/c.iloc[-2]-bc.iloc[-1]/bc.iloc[-2])*100),
            'ret20_excess_pp':float((px/c.iloc[-21]-bc.iloc[-1]/bc.iloc[-21])*100),'beta60_completed':beta,
            'corr60_completed':float(pairs.iloc[:,0].corr(pairs.iloc[:,1]))}
    out['LRCX_strict_gate']={'above_ma200':bool(px>out['ma']['200']),'rsi2_le20':bool(f.rsi2<=20),'negative3day':bool(out['return3']<0),'passed':bool(px>out['ma']['200'] and f.rsi2<=20 and out['return3']<0)}
    return out


def options(symbol,rv):
    raw=json.loads((BASE/'raw'/f'{symbol}-options.json').read_text());data=raw['data']
    stamp=dt.datetime.fromisoformat(raw['timestamp']).replace(tzinfo=dt.timezone.utc)
    spot=float(data['current_price']);groups={}
    for x in data['options']:
        match=re.search(r'(\d{6})([CP])(\d{8})$',x['option'])
        if not match:continue
        date,side,k=match.groups();expiration=dt.datetime.strptime(date,'%y%m%d').replace(hour=16,tzinfo=ZoneInfo('America/New_York'))
        days=(expiration-stamp).total_seconds()/86400
        bid,ask,iv=map(float,[x['bid'],x['ask'],x['iv']])
        if not (7<=days<=90 and 0<bid<=ask and 0<iv<5):continue
        if (ask-bid)/((ask+bid)/2)>.5:continue
        strike=int(k)/1000
        groups.setdefault(date,{'days':days,'strikes':{},'sides':{'C':[],'P':[]}})
        groups[date]['strikes'].setdefault(strike,{})[side]=x
        groups[date]['sides'][side].append(x)
    terms=[]
    for date,g in groups.items():
        ks=[k for k,v in g['strikes'].items() if 'C' in v and 'P' in v and abs(k/spot-1)<.05]
        if not ks:continue
        k=min(ks,key=lambda k:abs(k-spot));pair=g['strikes'][k]
        iv=(pair['C']['iv']+pair['P']['iv'])/2
        sk=[]
        for side,delta in [('P',-.25),('C',.25)]:
            chain=g['sides'][side];z=min(chain,key=lambda x:abs(x['delta']-delta))
            sk.append(z['iv'] if abs(z['delta']-delta)<.08 else None)
        terms.append({'expiry':date,'days':g['days'],'strike':k,'atm_iv':iv,'straddle_mid':sum((pair[s]['bid']+pair[s]['ask'])/2 for s in ['C','P']),
            'straddle_bid':pair['C']['bid']+pair['P']['bid'],'straddle_ask':pair['C']['ask']+pair['P']['ask'],
            'put25_iv':sk[0],'call25_iv':sk[1],'skew_pp':(sk[0]-sk[1])*100 if all(x is not None for x in sk) else None})
    terms.sort(key=lambda x:x['days']);below=[x for x in terms if x['days']<=30];above=[x for x in terms if x['days']>=30]
    proxy=None;bracket=None
    if below and above:
        a,b=below[-1],above[0];w=(30-a['days'])/(b['days']-a['days']) if b['days']!=a['days'] else 0
        proxy=math.sqrt(((1-w)*a['atm_iv']**2*a['days']+w*b['atm_iv']**2*b['days'])/30);bracket=[a,b]
    out={'retrieved_from_manifest':json.loads((BASE/'fetch-manifest.json').read_text())[symbol+'-options']['retrieved_at'],
         'provider_timestamp_assumed_utc':stamp.isoformat(),'underlying_last_trade_local':data['last_trade_time'],
         'delayed_underlying':spot,'delayed_bid':data['bid'],'delayed_ask':data['ask'],
         'provider_iv30':float(data['iv30'])/100,'provider_iv30_rv20_ratio':float(data['iv30'])/100/rv,
         'own_atm30_proxy':proxy,'own_proxy_ivrv':proxy/rv if proxy else None,'own_proxy_not_original_score':True,
         'annual_iv_percentile':None,'bracket':bracket,'term_points':terms}
    out['provider_iv_14calendar_day_scale_pct']=float(data['iv30'])/100*math.sqrt(14/365)*100
    return out


def main():
    result={'protocol':P,'symbols':{},'benchmarks':{}}
    for s in ['AAOI','AVGO']:
        fac=factors(s)
        result['symbols'][s]={'factors':fac,'options':options(s,fac['rv20_annual']),'forecast':forecast(s)}
        (BASE/f'{s}-recent.csv').write_text(daily(s).tail(30).to_csv())
    for s in ['QQQ','SOXX','SMH','COHR','LITE']:
        d=daily(s);result['benchmarks'][s]={'price':float(d.close.iloc[-1]),'day_change':float(d.close.iloc[-1]/d.close.iloc[-2]-1)}
    macro={};calendar=daily('QQQ').index[:-1]
    for s in ['DFII10','BAMLH0A0HYM2','DCOILBRENTEU','CPILFESL']:
        p=BASE/'raw'/f'{s}.csv'
        if not p.exists():macro[s]={'available':False};continue
        x=pd.read_csv(p,index_col=0);x.index=pd.to_datetime(x.index).date;x[s]=pd.to_numeric(x[s],errors='coerce');x=x.dropna();last=x.iloc[-1,0];date=x.index[-1]
        record={'available':True,'date':str(date),'value':float(last),'source':'https://fred.stlouisfed.org/series/'+s}
        if s!='CPILFESL':
            prior_dates=[v for v in calendar if v<=date];ago=prior_dates[-21];xago=x.loc[[v<=ago for v in x.index]].iloc[-1,0]
            record.update(prior20_date=str(ago),prior20_value=float(xago),change20_bp=float((last-xago)*100) if s!='DCOILBRENTEU' else None,return20_pct=float((last/xago-1)*100) if s=='DCOILBRENTEU' else None)
        else:
            annual=((x.iloc[-1,0]/x.iloc[-4,0])**4-1)*100;oldannual=((x.iloc[-2,0]/x.iloc[-5,0])**4-1)*100
            record.update(core3m_annual_pct=float(annual),change_vs_prior_month_pp=float(annual-oldannual))
        macro[s]=record
    result['macro']=macro
    (BASE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for s,v in result['symbols'].items():
        print(s,'factors',json.dumps(v['factors'],ensure_ascii=False))
        print(s,'options',json.dumps({k:z for k,z in v['options'].items() if k!='term_points'},ensure_ascii=False))
        fc=v['forecast'];print(s,'forecast',json.dumps({k:z for k,z in fc.items() if k not in ['evaluation_rows','neighbors','evaluation_by_year']},ensure_ascii=False))
    print('Macro',json.dumps(macro,ensure_ascii=False));print('Benchmarks',json.dumps(result['benchmarks']))


if __name__=='__main__':main()
