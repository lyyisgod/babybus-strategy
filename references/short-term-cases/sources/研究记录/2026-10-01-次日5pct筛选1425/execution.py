"""Current execution diagnostics separate from unverified model rank; no orders."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,importlib.util
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from src.analysis.short_term_factors import completed_regular_minutes
spec=importlib.util.spec_from_file_location('same_clock_model',P/'model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    current=pd.read_csv(P/'ranking-all.csv');manifest=json.loads((P/'manifest-live.json').read_text());out=[]
    for _,r in current.iterrows():
        s=r.symbol
        try:
            a=next(x for x in manifest if x['symbol']==s and x['ok'])
            raw=json.loads((P/'raw'/f'{s}-live.json').read_text());meta=raw['chart']['result'][0]['meta']
            q=completed_regular_minutes(raw,a['fetched_at']);px=float(meta['regularMarketPrice'])
            tm=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').tz_convert(m.NY)
            vv=float(q.volume.sum());vwap=float((((q.high+q.low+q.close)/3)*q.volume).sum()/vv)
            b=q.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            b=b[(b.index+pd.Timedelta(minutes=5))<=pd.Timestamp(a['fetched_at']).tz_convert(m.NY)]
            vw=(((b.high+b.low+b.close)/3)*b.volume).cumsum()/b.volume.cumsum()
            two=bool(len(b)>=2 and (b.close.tail(2)>vw.tail(2)).all())
            d=m.daily(s);delta=min(.005*px,.1*float(r.atr));cap=round(px+delta,2)
            age=(pd.Timestamp(a['fetched_at'])-tm).total_seconds()
            out.append({**r.to_dict(),'latest_price':px,'quote_at':tm.isoformat(),'fetched_at':a['fetched_at'],
                'change_pct':100*(px/float(r.previous_close)-1),'quote_age_seconds_at_fetch':age,
                'minute_vwap_approx':vwap,'above_vwap':bool(px>vwap),'two_completed5m_above_vwap':two,
                'reference_entry_cap':cap,'reference_stop3':round(px*.97,2),'reference_target5':round(px*1.05,2),
                'reference_target_net5_with_cost_proxy':round(px*1.052,2),'completed_minute_low':float(q.low.min()),
                'completed_minute_high':float(q.high.max()),'last60minute_low':float(q.low.tail(60).min()),
                'last60minute_high':float(q.high.tail(60).max()),'prior20day_high':float(d.high.tail(20).max()),
                'longest_entry_age_not_validated':True,'rightside_price_volume_gate':bool(r.eligible and px>vwap and two and r.rvol>=1.2 and age<=90),
                'bid_ask_available':False,'absolute_validated_profit_gate':False,
                'signal_price_to_actual_quote_change_pct':100*(px/float(r.signal_price)-1)})
        except Exception as exc:out.append({'symbol':s,'execution_error':str(exc)})
    m.save('execution-all.json',out)
    d=pd.DataFrame(out);d.to_csv(P/'execution-all.csv',index=False)
    d[d.eligible.fillna(False)].to_csv(P/'execution-eligible.csv',index=False)
    print(d[d.eligible.fillna(False)][['symbol','latest_price','quote_at','change_pct','rvol','above_vwap','two_completed5m_above_vwap','rightside_price_volume_gate','rank_firstwin_unverified']].head(25).to_string(index=False))
    print('Focused candidates')
    print(d[d.symbol.isin(['AAOI','COHR','VICR','CRDO','AEHR','NVTS','LITE','ACN','TSEM'])][['symbol','latest_price','change_pct','rvol','minute_vwap_approx','rightside_price_volume_gate','reference_entry_cap','reference_stop3','reference_target5','rv20','rank_firstwin_unverified']].to_string(index=False))

if __name__=='__main__':main()
