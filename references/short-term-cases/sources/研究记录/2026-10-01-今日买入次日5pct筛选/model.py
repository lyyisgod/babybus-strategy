"""Frozen exploratory same-clock next-session research. Public data; no orders."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json, sys, hashlib, importlib.util
import numpy as np
import pandas as pd
import exchange_calendars as xc

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.short_term_factors import wilder
NY = ZoneInfo('America/New_York')
CFG = json.loads((P/'protocol.json').read_text())
COLS = CFG['features']
spec = importlib.util.spec_from_file_location('stable_logistic', ROOT/'研究记录/2026-09-21-隔夜至次日爆发/model.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
cal = xc.get_calendar('XNYS', start='2026-01-01', end='2027-01-01')
SESSIONS = [x.date().isoformat() for x in cal.sessions_in_range('2026-07-01','2026-10-02')]

def save(name, value):
    (P/name).write_text(json.dumps(old.clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def payload(symbol, mode):
    return json.loads((P/'raw'/f'{symbol}-{mode}.json').read_text())['chart']['result'][0]

def daily(symbol):
    z = payload(symbol, 'daily')
    ix = pd.to_datetime(z['timestamp'], unit='s', utc=True).tz_convert(NY).normalize().tz_localize(None)
    d = pd.DataFrame(z['indicators']['quote'][0], index=ix).dropna(subset=['open','high','low','close','volume'])
    d = d.loc[:CFG['daily_cutoff']].copy()
    assert d.index.is_unique and d.index.is_monotonic_increasing
    assert str(d.index[-1].date()) == CFG['daily_cutoff'], symbol
    return d

def minutes(symbol):
    z = payload(symbol, 'history')
    d = pd.DataFrame(z['indicators']['quote'][0], index=pd.to_datetime(z['timestamp'],unit='s',utc=True).tz_convert(NY))
    live = P/'raw'/f'{symbol}-live.json'
    if live.exists():
        a = payload(symbol,'live')
        b = pd.DataFrame(a['indicators']['quote'][0], index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert(NY))
        fetched = next(r['fetched_at'] for r in json.loads((P/'manifest-live.json').read_text()) if r['symbol']==symbol and r['ok'])
        b = b[(b.index + pd.Timedelta(minutes=1)) <= pd.Timestamp(fetched).tz_convert(NY)]
        b = b.dropna(subset=['open','high','low','close'])
        b = b.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        # Only complete five-minute bars may enter this model's fixed-clock signal.
        b = b[(b.index + pd.Timedelta(minutes=5)) <= pd.Timestamp(fetched).tz_convert(NY)]
        d = pd.concat([d.loc[d.index.date < pd.Timestamp(CFG['date']).date()],b])
    d = d.dropna(subset=['open','high','low','close','volume'])
    d = d[(d.index.hour*60+d.index.minute >= 570)&(d.index.hour*60+d.index.minute < 960)]
    d = d[~d.index.duplicated(keep='last')].sort_index()
    assert d.index.is_unique
    return d

def daily_features(d, q):
    c = d.close
    tr = pd.concat([d.high-d.low, (d.high-c.shift()).abs(), (d.low-c.shift()).abs()],axis=1).max(axis=1)
    dc = c.diff().iloc[1:]
    gain = wilder(dc.clip(lower=0),2).reindex(c.index)
    loss = wilder((-dc).clip(lower=0),2).reindex(c.index)
    rsi = (100-100/(1+gain/loss)).where(loss!=0,100).where((gain+loss)!=0,50)
    return pd.DataFrame({'rv20':np.log(c/c.shift()).rolling(20).std(ddof=1),
        'rs20':c.pct_change(20,fill_method=None)-q.close.pct_change(20,fill_method=None).reindex(c.index),
        'ma20':c.rolling(20).mean(),'ma200':c.rolling(200).mean(),'ret3':c.pct_change(3,fill_method=None),
        'atr':wilder(tr,14),'rsi2':rsi,'adv20':(c*d.volume).rolling(20).mean(),'previous_close':c})

def signal(g, f, qg, prior_volumes):
    pre = g[g.index.hour*60+g.index.minute < 795]
    qp = qg[qg.index.hour*60+qg.index.minute < 795]
    if len(pre)<41 or len(qp)<41 or pre.index[0].strftime('%H:%M')!='09:30' or pre.index[-1].strftime('%H:%M')!='13:10':
        return None
    if len(prior_volumes)<20 or pre.volume.sum()<=0:
        return None
    price = float(pre.close.iloc[-1])
    vwap = float((((pre.high+pre.low+pre.close)/3)*pre.volume).sum()/pre.volume.sum())
    rvol = float(pre.volume.sum()/np.mean(prior_volumes[-20:]))
    return {'rv20':float(f.rv20),'rs20':float(f.rs20),'ma20_dist':price/float(f.ma20)-1,
        'intraday':price/float(pre.open.iloc[0])-1,
        'rel_intraday_qqq':price/float(pre.open.iloc[0])-float(qp.close.iloc[-1])/float(qp.open.iloc[0]),
        'log_rvol20':np.log(max(rvol,1e-8)),'vwap_dist':price/vwap-1,
        'signal_price':price,'vwap':vwap,'rvol':rvol,'pre_low':float(pre.low.min()),'pre_high':float(pre.high.max()),
        'eligible':bool(f.previous_close>=5 and f.adv20>=5e7)}

def simulate(path, entry):
    stop = entry*.97; target = entry*1.05
    tomorrow = str(path.index[-1].date())
    for stamp, r in path.iterrows():
        if r.open<=stop:
            return {'firstwin':0.,'net':float(r.open/entry-1-CFG['cost']),'exit_at':stamp,'cause':'stop_gap'}
        if str(stamp.date())==tomorrow and r.open>=target:
            return {'firstwin':1.,'net':.05-CFG['cost'],'exit_at':stamp,'cause':'target_gap_conservative_limit'}
        if r.low<=stop:
            return {'firstwin':0.,'net':-.03-CFG['cost'],'exit_at':stamp,'cause':'stop_or_samebar_both'}
        if str(stamp.date())==tomorrow and r.high>=target:
            return {'firstwin':1.,'net':.05-CFG['cost'],'exit_at':stamp,'cause':'target'}
    return {'firstwin':0.,'net':float(path.close.iloc[-1]/entry-1-CFG['cost']),
        'exit_at':path.index[-1]+pd.Timedelta(minutes=5),'cause':'next_close'}

def build(symbol, d, b, qd, qb):
    f = daily_features(d,qd)
    splits=payload(symbol,'daily').get('events',{}).get('splits',{})
    split_days={pd.Timestamp(v['date'],unit='s',tz='UTC').tz_convert(NY).date().isoformat() for v in splits.values()}
    groups = {str(day):g for day,g in b.groupby(b.index.date)}
    qgroups = {str(day):g for day,g in qb.groupby(qb.index.date)}
    rows=[]; volumes=[]; current=None
    for day,g in groups.items():
        previous = f.loc[f.index < pd.Timestamp(day)]
        if len(previous)==0 or day not in qgroups:continue
        vals = signal(g,previous.iloc[-1],qgroups[day],volumes)
        pre=g[g.index.hour*60+g.index.minute < 795]
        if len(pre)>=41 and pre.index[-1].strftime('%H:%M')=='13:10':volumes.append(float(pre.volume.sum()))
        if vals is None or not np.isfinite([vals[x] for x in COLS]).all():continue
        vals.update(symbol=symbol,signal_day=day,atr=float(previous.atr.iloc[-1]),ma200=float(previous.ma200.iloc[-1]),
            rsi2=float(previous.rsi2.iloc[-1]),ret3=float(previous.ret3.iloc[-1]),adv20=float(previous.adv20.iloc[-1]),
            previous_close=float(previous.previous_close.iloc[-1]))
        if day==CFG['date']:
            current=vals;continue
        if day not in SESSIONS or SESSIONS.index(day)+1>=len(SESSIONS):continue
        nxt=SESSIONS[SESSIONS.index(day)+1]
        if nxt>=CFG['date'] or nxt not in groups:continue
        if day in split_days or nxt in split_days:continue
        ng=groups[nxt]
        if len(ng)<70 or ng.index[0].strftime('%H:%M')!='09:30' or ng.index[-1].strftime('%H:%M')!='15:55':continue
        ent=g[g.index.hour*60+g.index.minute>=795]
        if len(ent)<30 or ent.index[0].strftime('%H:%M')!='13:15' or ent.index[-1].strftime('%H:%M')!='15:55':continue
        price=float(ent.open.iloc[0]);path=pd.concat([ent,ng]);sim=simulate(path,price)
        vals.update(entry=price,exit_day=nxt,high_return=float(ng.high.max()/price-1),
            close_return=float(ng.close.iloc[-1]/price-1-CFG['cost']),
            hit5=float(ng.high.max()>=price*1.05),close5=float(ng.close.iloc[-1]/price-1-CFG['cost']>=.05),
            down3=float(path.low.min()<=price*.97),firstwin=sim['firstwin'],net=sim['net'],cause=sim['cause'],
            adverse_slip_net=simulate(path,price*1.0025)['net'])
        rows.append(vals)
    return rows,current

def fitting(train, test, targets):
    # Equal weight per date; stocks sharing one macro shock are not independent dates.
    counts=train.groupby('signal_day').symbol.transform('count').to_numpy(float)
    weight=1/counts;weight/=weight.sum()
    x=train[COLS].to_numpy();mu=np.average(x,axis=0,weights=weight)
    sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=weight));sd=np.where(sd>1e-10,sd,1.)
    xx=np.c_[np.ones(len(x)),np.clip((x-mu)/sd,-8,8)]
    tx=np.c_[np.ones(len(test)),np.clip((test[COLS].to_numpy()-mu)/sd,-8,8)]
    out=test.copy();diagnostics={}
    for target in targets:
        y=train[target].to_numpy();b=np.zeros(xx.shape[1]);mean=np.average(y,weights=weight)
        if mean==0 or mean==1:
            out['rank_'+target+'_unverified']=mean;diagnostics[target]={'constant':True,'n':len(y)};continue
        b[0]=np.log(mean/(1-mean));pen=np.eye(xx.shape[1])*.1;pen[0,0]=0
        converged=False
        def loss(z):return np.sum(weight*(np.logaddexp(0,xx@z)-y*(xx@z)))+.5*z@pen@z
        for i in range(100):
            pp=old.sigmoid(xx@b);grad=xx.T@(weight*(pp-y))+pen@b
            if np.max(np.abs(grad))<1e-7:converged=True;break
            hs=(xx.T*(weight*pp*(1-pp)))@xx+pen+np.eye(xx.shape[1])*1e-9
            step=np.linalg.solve(hs,grad);rate=1.;before=loss(b)
            while rate>1e-10 and loss(b-rate*step)>before-1e-4*rate*(grad@step):rate*=.5
            assert rate>1e-10
            b-=rate*step
        assert converged,'logistic convergence'
        out['rank_'+target+'_unverified']=old.sigmoid(tx@b)
        diagnostics[target]={'constant':False,'gradient':float(np.max(np.abs(grad))),'n':len(y),'dates':train.signal_day.nunique()}
    return out,diagnostics

def run():
    qd=daily('QQQ');qb=minutes('QQQ');panels=[];current=[];errors={}
    for s in CFG['stocks']:
        try:
            d=daily(s);b=minutes(s);rows,cur=build(s,d,b,qd,qb)
            if rows:panels.extend(rows)
            if cur is not None:current.append(cur)
        except Exception as exc:errors[s]=str(exc)
    panel=pd.DataFrame(panels);panel=panel[panel.eligible].copy()
    panel.to_csv(P/'historical-panel.csv',index=False)
    targets=['firstwin','hit5','close5','down3'];dates=sorted(panel.signal_day.unique());evaluated=[];tops=[];last_exit=None;fits=[]
    for day in dates[8:]:
        if last_exit is not None and day<=last_exit:continue
        train=panel[panel.exit_day<day]
        if train.signal_day.nunique()<8:continue
        test=panel[panel.signal_day==day]
        pred,diag=fitting(train,test,targets);fits.append({'signal_day':day,'fit':diag,'last_train_exit':train.exit_day.max()})
        pred['train_last_exit']=train.exit_day.max();evaluated.append(pred)
        top=pred.sort_values(['rank_firstwin_unverified','symbol'],ascending=[False,True]).iloc[0]
        simple=pred.sort_values(['rv20','symbol'],ascending=[False,True]).iloc[0]
        row=top.to_dict();row.update(pool_hit5=float(pred.hit5.mean()),pool_firstwin=float(pred.firstwin.mean()),
            pool_net=float(pred.net.mean()),simple_symbol=simple.symbol,simple_hit5=float(simple.hit5),simple_net=float(simple.net),
            pool_brier=float(((pred.hit5-train.groupby('signal_day').hit5.mean().mean())**2).mean()),model_brier=float(((pred.hit5-pred.rank_hit5_unverified)**2).mean()))
        tops.append(row);last_exit=top.exit_day
    top=pd.DataFrame(tops);ev=pd.concat(evaluated);top.to_csv(P/'evaluation-top1.csv',index=False);ev.to_csv(P/'evaluation-all.csv',index=False)
    bins=[]
    for low in np.arange(0,1,.1):
        a=ev[(ev.rank_hit5_unverified>=low)&(ev.rank_hit5_unverified<low+.1)]
        if len(a):bins.append({'low':float(low),'rows':len(a),'independent_signal_dates':a.signal_day.nunique(),
            'mean_unverified_score':float(a.rank_hit5_unverified.mean()),'historical_event_fraction':float(a.hit5.mean())})
    save('calibration-bins-descriptive.json',bins)
    assert (ev.train_last_exit<ev.signal_day).all()
    assert all(top.signal_day.iloc[i]>top.exit_day.iloc[i-1] for i in range(1,len(top)))
    metrics={'independent_nonoverlap_signals':len(top),'all_signal_dates':len(dates),'training_rows':len(panel),
        'first_signal':dates[0],'last_mature_signal':dates[-1],'hit5_rate_descriptive':float(top.hit5.mean()),
        'firstwin_rate_descriptive':float(top.firstwin.mean()),'close5_rate_descriptive':float(top.close5.mean()),
        'down3_rate_descriptive':float(top.down3.mean()),'selected_net_mean':float(top.net.mean()),
        'selected_net_mean_ci95':old.block_ci(top.net),'selected_hit5_ci95':old.block_ci(top.hit5),
        'same_date_pool_hit5':float(top.pool_hit5.mean()),'same_date_pool_net':float(top.pool_net.mean()),
        'simple_highest_rv20_hit5':float(top.simple_hit5.mean()),'simple_highest_rv20_net':float(top.simple_net.mean()),
        'model_minus_simple_net_ci95':old.block_ci(top.net-top.simple_net),
        'model_minus_pool_net_ci95':old.block_ci(top.net-top.pool_net),
        'brier_daily_mean':float(top.model_brier.mean()),'baseline_brier_daily_mean':float(top.pool_brier.mean()),
        'double_cost_selected_net':float(top.net.mean()-.002),'adverse_entry25bp_selected_net':float(top.adverse_slip_net.mean()),
        'not_untouched_audit':True,'universe_survivorship_bias':True,'certification_pass':False,
        'probability_scope':'Historical13:15open proxy; no executable-spread/available_at event history. Few independent dates; numbers descriptive, not current-stock winrates.'}
    if current:
        cur=pd.DataFrame(current);cur,diag=fitting(panel[panel.exit_day<CFG['date']],cur,targets)
        cur=cur.sort_values(['rank_firstwin_unverified','symbol'],ascending=[False,True]);cur.to_csv(P/'ranking-all.csv',index=False)
        cur[cur.eligible].to_csv(P/'ranking-eligible.csv',index=False)
        lrcx=cur[(cur.previous_close>cur.ma200)&(cur.rsi2<=20)&(cur.ret3<0)]
        lrcx.to_csv(P/'lrcx-branch.csv',index=False)
        save('current-rank.json',cur.to_dict('records'));save('current-fit.json',diag)
        print(cur[cur.eligible][['symbol','signal_price','rank_firstwin_unverified','rank_hit5_unverified','rank_close5_unverified','rank_down3_unverified','rvol','intraday','vwap_dist','rv20']].head(25).to_string(index=False))
    save('metrics.json',metrics);save('errors.json',errors);save('fit-diagnostics.json',fits)
    # Meaningful target, gap, ambiguity, and causality checks.
    idx=pd.DatetimeIndex(['2026-09-28 13:15','2026-09-29 09:30'],tz=NY)
    toy=pd.DataFrame({'open':[100,100],'high':[107,106],'low':[99,98],'close':[106,105]},index=idx)
    assert simulate(toy,100)['firstwin']==1 and abs(simulate(toy,100)['net']-.048)<1e-12
    toy.loc[idx[-1],'low']=96
    assert simulate(toy,100)['cause']=='stop_or_samebar_both'
    toy.loc[idx[-1],'open']=94
    assert abs(simulate(toy,100)['net']+.062)<1e-12
    f=daily_features(qd,qd);cut=qd.index[-50]
    pd.testing.assert_frame_equal(f.loc[:cut],daily_features(qd.loc[:cut],qd.loc[:cut]))
    qgroups={str(day):g for day,g in qb.groupby(qb.index.date)}
    volumes=[];prefix_check=False
    for day,g in qgroups.items():
        pre=g[g.index.hour*60+g.index.minute<795]
        prior=f[f.index<pd.Timestamp(day)]
        if len(volumes)>=20 and len(prior):
            full=signal(g,prior.iloc[-1],g,volumes)
            prefix=signal(pre,prior.iloc[-1],pre,volumes)
            if full is not None:
                np.testing.assert_allclose([full[k] for k in COLS],[prefix[k] for k in COLS])
                assert signal(g,prior.iloc[-1],g,volumes[:19]) is None
                prefix_check=True;break
        if len(pre)>=41 and pre.index[-1].strftime('%H:%M')=='13:10':volumes.append(float(pre.volume.sum()))
    assert prefix_check
    test_train=pd.DataFrame({k:np.zeros(100) for k in COLS})
    test_train['signal_day']='2026-09-01';test_train['symbol']=[str(i) for i in range(100)]
    test_train['hit5']=np.r_[np.ones(20),np.zeros(80)]
    fitted,_=fitting(test_train,test_train.head(1),['hit5'])
    assert abs(float(fitted.rank_hit5_unverified.iloc[0])-.2)<1e-6
    hashes=[]
    for name in ['manifest-initial.json','manifest-retry.json','manifest-live.json']:
        if (P/name).exists():
            for r in json.loads((P/name).read_text()):
                if r['ok']:assert hashlib.sha256((P/'raw'/f"{r['symbol']}-{r['mode']}.json").read_bytes()).hexdigest()==r['sha256'];hashes.append(r['symbol'])
    save('verification.json',{'mature_labels_strict_before_signal':True,'nonoverlapping_evaluation':True,
        'today_target_does_not_exit':True,'dual_touch_stop_first':True,'negative_gap_slippage':True,
        'daily_feature_prefix':True,'raw_hashes_verified':len(hashes),'functional_validation_only':True,
        'intraday_feature_prefix':True,'RVOL20_requires20fullpriorclocks':True,'known20pct_intercept':True,
        'split_crossing_labels_excluded':True,
        'not_investment_validation':True})
    print(json.dumps(old.clean({'metrics':metrics,'errors':errors,'current_count':len(current)}),ensure_ascii=False,indent=2))

if __name__=='__main__':run()
