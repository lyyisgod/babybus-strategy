"""Next-session models with chronological OOS validation. Today is provisional."""
import ast,json,re,math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent;TODAY='2026-09-15';NEXT='2026-09-16'

def load(s):
    d=json.loads((R/'raw'/f'{s}-1d.json').read_text())['chart']['result'][0]
    f=pd.DataFrame(d['indicators']['quote'][0],index=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d'))
    f['adj']=d['indicators'].get('adjclose',[{}])[0].get('adjclose',f.close)
    f=f.dropna(subset=['open','high','low','close','adj']).loc[lambda z:z.index<=TODAY]
    assert f.index.is_unique
    ratio=f.adj/f.close
    for k in ['open','high','low']:f['a'+k]=f[k]*ratio
    return f,d['meta']

def calendar():
    soup=BeautifulSoup((R/'raw/fomc.html').read_text(),'html.parser');out=[]
    for row in soup.select('div.fomc-meeting'):
        heading=row.find_previous('h4');year=re.search(r'(20\d\d)',heading.get_text() if heading else '')
        if not year:continue
        month=row.select_one('.fomc-meeting__month');days=row.select_one('.fomc-meeting__date')
        if month is None or days is None or '-' not in days.get_text():continue
        mo=month.get_text(strip=True).split('/')[-1];dd=re.findall(r'\d+',days.get_text())[-1]
        mo={'Apr':'April','Jun':'June','Jul':'July'}.get(mo,mo)
        try:date=datetime.strptime(f'{year[1]} {mo} {dd}','%Y %B %d')
        except ValueError:date=datetime.strptime(f'{year[1]} {mo} {dd}','%Y %b %d')
        out.append(date.strftime('%Y-%m-%d'))
    assert '2026-09-16' in out and '2025-09-17' in out
    return sorted(set(out))

def fit(X,y,lam=.05):
    mean=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1
    z=np.column_stack([np.ones(len(X)),np.clip((X-mean)/sd,-5,5)]);b=np.zeros(z.shape[1]);reg=np.eye(len(b))*lam;reg[0,0]=0
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(z@b,-30,30)));g=z.T@(p-y)/len(y)+reg@b
        h=(z.T*(p*(1-p)))@z/len(y)+reg+np.eye(len(b))*1e-8
        step=np.linalg.solve(h,g);b-=step
        if abs(step).max()<1e-7:break
    return mean,sd,b

def pred(m,X):
    mu,sd,b=m;z=np.column_stack([np.ones(len(X)),np.clip((X-mu)/sd,-5,5)])
    return 1/(1+np.exp(-np.clip(z@b,-30,30)))

def auc(y,p):
    a=y.sum();b=len(y)-a
    return float((pd.Series(p).rank().to_numpy()[y==1].sum()-a*(a+1)/2)/(a*b)) if a and b else None

def wilson(y):
    n=len(y)
    if not n:return None
    p=np.mean(y);z=1.96;den=1+z*z/n;c=(p+z*z/(2*n))/den;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'n':n,'events':int(sum(y)),'frequency':float(p),'wilson95':[max(0,c-w),min(1,c+w)]}

def stats(t):
    y=t.y.to_numpy();p=t.p.to_numpy();g=(t.base-y)**2-(p-y)**2;rng=np.random.default_rng(916);boot=[];n=len(t)
    for _ in range(1000):
        ids=np.concatenate([(j+np.arange(20))%n for j in rng.integers(n,size=math.ceil(n/20))])[:n];boot.append(float(g.iloc[ids].mean()))
    ci=np.quantile(boot,[.025,.975]).tolist()
    return {'n':n,'from':t.date.min(),'through':t.date.max(),'event_rate':float(y.mean()),'auc':auc(y,p),'brier':float(np.mean((p-y)**2)),'baseline_brier':float(np.mean((t.base-y)**2)),'block20_brier_improvement95':ci,'validated_edge':bool(ci[0]>0 and auc(y,p)>.55)}

SYMS=json.loads((R/'universe.json').read_text())['symbols'];F={};META={}
for s in SYMS:F[s],META[s]=load(s)
CAL=calendar();(R/'fomc-dates.json').write_text(json.dumps(CAL,indent=2))
BASE=['shock1','momentum5','momentum20','distance20','relative5','sigma20']
FULL=BASE+['qqq1','semis_excess1','yield1_bp','yield20_bp','vix_level','vix_change','oil5','dollar5','credit_proxy5','breadth','fomc_next']
PEERS='NVDA AMD TSM MRVL AMAT LRCX KLAC WDC STX'.split()

def features(s,close_mult=1.):
    f=F[s].copy()
    if close_mult!=1:
        f.iloc[-1,f.columns.get_loc('adj')]*=close_mult
        f.iloc[-1,f.columns.get_loc('close')]*=close_mult
    c=f.adj;lr=np.log(c/c.shift());sig=lr.rolling(20).std();out=pd.DataFrame(index=f.index)
    out['shock1']=lr/sig;out['momentum5']=np.log(c/c.shift(5))/(sig*np.sqrt(5));out['momentum20']=np.log(c/c.shift(20))/(sig*np.sqrt(20));out['distance20']=(c/c.rolling(20).mean()-1)/sig
    out['relative5']=(c/c.shift(5)-F['SMH'].adj/F['SMH'].adj.shift(5))/sig;out['sigma20']=sig
    ret=lambda x,h:F[x].adj.pct_change(h,fill_method=None).reindex(f.index)
    out['qqq1']=ret('QQQ',1);out['semis_excess1']=ret('SMH',1)-ret('QQQ',1)
    # All inputs are contemporary public price proxies, no macro revised-history backfill.
    rate=F['^TNX'].close.reindex(f.index).ffill(limit=3);out['yield1_bp']=rate.diff()*100;out['yield20_bp']=rate.diff(20)*100
    vix=F['^VIX'].close.reindex(f.index).ffill(limit=3);out['vix_level']=vix;out['vix_change']=vix.diff()
    out['oil5']=ret('BZ=F',5);out['dollar5']=ret('DX-Y.NYB',5);out['credit_proxy5']=ret('HYG',5)-ret('LQD',5)
    # Nine predeclared peers; each needs a valid return. No forward filling equity returns.
    peer=pd.DataFrame({x:ret(x,1) for x in PEERS});out['breadth']=peer.gt(0).sum(axis=1)/peer.notna().sum(axis=1);out.loc[peer.notna().sum(axis=1)<8,'breadth']=np.nan
    dates=pd.Series(f.index,index=f.index);out['end']=dates.shift(-1);out.loc[TODAY,'end']=NEXT
    out['fomc_next']=out.end.isin(CAL).astype(float)
    out['next_return']=c.shift(-1)/c-1
    out['break_low']=(f.alow.shift(-1)<f.alow).astype(float)
    out.loc[out.next_return.isna(),'break_low']=np.nan
    # Never label an earlier signal with today's unfinished outcome.
    out.loc[out.end>=TODAY,['next_return','break_low']]=np.nan
    out['symbol']=s;out['date']=out.index
    return out

def run():
    result={'asof':datetime.now().astimezone().isoformat(),'target':'Tomorrow adjusted close below today adjusted close; provisional today-close-equals-snapshot assumption. No scheduled ex-dividend for target session verified in reviewed announcements.','dates':{'today':TODAY,'tomorrow':NEXT},'method':'Two prespecified per-stock ridge logistic models, quarterly expanding OOS since 2024; fixed lambda .05, no tuning. Separate historical FOMC and analog frequencies.','features':{'simple':BASE,'full':FULL},'live_probability_not_calibrated_for_fomc':True,'results':{}}
    bounds=['2024-01-01','2024-04-01','2024-07-01','2024-10-01','2025-01-01','2025-04-01','2025-07-01','2025-10-01','2026-01-01','2026-04-01','2026-07-01',TODAY]
    for s in ['AVGO','MU']:
        allf=features(s);allf.to_csv(R/f'{s}-features.csv',index=False)
        live=allf.loc[[TODAY]];data=allf.dropna(subset=FULL+['next_return']).copy();data=data[data.date>='2021-01-01']
        out={'quote':{'price':META[s]['regularMarketPrice'],'timestamp':META[s]['regularMarketTime']},'models':{},'feature_snapshot':live[FULL].iloc[0].to_dict()}
        for label in ['down','down2','break_low']:
            data['y']=((data.next_return<(-.02 if label=='down2' else 0)).astype(int) if label!='break_low' else data.break_low.astype(int))
            out[label+'_base_252']=wilson(data.y.tail(252).tolist())
            out[label+'_fomc']=wilson(data.loc[data.fomc_next==1,'y'].tolist())
            out[label+'_fomc_after_down']=wilson(data.loc[(data.fomc_next==1)&(data.shock1<0),'y'].tolist())
            subset=data[(data.shock1<0)&(data.distance20<0)&(data.relative5<0)]
            out[label+'_weak_analog']=wilson(subset.y.tolist())
            for name,cols in [('simple',BASE),('full',FULL)]:
                pieces=[]
                for start,end in zip(bounds,bounds[1:]):
                    tr=data[data.end<start];te=data[(data.date>=start)&(data.date<end)]
                    if len(tr)<500 or te.empty:continue
                    assert tr.end.max()<te.date.min()
                    fitted=fit(tr[cols].to_numpy(),tr.y.to_numpy());p=pred(fitted,te[cols].to_numpy())
                    pieces.append(pd.DataFrame({'date':te.date,'end':te.end,'y':te.y,'p':p,'base':tr.y.mean(),'fomc':te.fomc_next}))
                test=pd.concat(pieces,ignore_index=True);key=label+'_'+name;test.to_csv(R/f'{s}-{key}-oos.csv',index=False)
                fitted=fit(data[cols].to_numpy(),data.y.to_numpy());prob=float(pred(fitted,live[cols].to_numpy())[0])
                sensitivity={str(mult):float(pred(fitted,features(s,mult).loc[[TODAY],cols].to_numpy())[0]) for mult in [.99,1.,1.01]}
                rng=np.random.default_rng(915);boot=[];n=len(data)
                for _ in range(200):
                    ids=np.concatenate([(j+np.arange(20))%n for j in rng.integers(n,size=math.ceil(n/20))])[:n]
                    fb=fit(data[cols].to_numpy()[ids],data.y.to_numpy()[ids]);boot.append(float(pred(fb,live[cols].to_numpy())[0]))
                oosstats=stats(test);fomctest=test[test.fomc==1]
                bins=[]
                for lo,hi in [(0,.4),(.4,.5),(.5,.6),(.6,1.0001)]:
                    b=test[(test.p>=lo)&(test.p<hi)]
                    bins.append({'range':[lo,hi],'n':len(b),'mean_probability':float(b.p.mean()) if len(b) else None,'observed':float(b.y.mean()) if len(b) else None})
                out['models'][key]={'raw_probability':prob,'conditional_parameter_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'today_close_plus_minus1pct_sensitivity':sensitivity,'validation':oosstats,'oos_fomc':{'n':len(fomctest),'frequency':float(fomctest.y.mean()),'brier':float(np.mean((fomctest.p-fomctest.y)**2)),'baseline_brier':float(np.mean((fomctest.base-fomctest.y)**2))},'calibration_bins':bins,'standardized_coefficients':dict(zip(['intercept']+cols,fitted[2].tolist()))}
        # Daily return correlation and volatility, not a causal attribution.
        ret=pd.DataFrame({x:F[x].adj.pct_change(fill_method=None) for x in [s,'QQQ','SMH']}).loc[lambda z:z.index<TODAY].dropna().tail(60)
        out['corr60']=ret.corr().to_dict();out['beta60_smh']=float(ret[s].cov(ret.SMH)/ret.SMH.var())
        result['results'][s]=out
        print(s,json.dumps({k:{'p':v['raw_probability'],'ci':v['conditional_parameter_bootstrap95'],'auc':v['validation']['auc'],'edge':v['validation']['validated_edge']} for k,v in out['models'].items()}),flush=True)
    (R/'model-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=='__main__':run()
