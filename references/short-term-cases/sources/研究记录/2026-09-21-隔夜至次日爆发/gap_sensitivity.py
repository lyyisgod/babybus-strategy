"""Descriptive gap analogue after an observed opening gap, not an overnight forecast."""
import model as m
import numpy as np
import pandas as pd

def main():
    rows=[];errors={}
    for s in m.STOCKS:
        try:d,_=m.load(s)
        except Exception as e:errors[s]=str(e);continue
        y=m.outcomes(d)
        eligible=(d.raw_close>=5)&((d.raw_close*d.volume).shift().rolling(20).mean()>=50e6)
        mask=eligible&y.next_gap.between(.08,.15)&(d.close.pct_change().abs()<=.05)&y.exit_date.notna()
        for date,r in y[mask].iterrows():rows.append(dict(symbol=s,date=date,**r.to_dict()))
    a=pd.DataFrame(rows);reports=[]
    semi=set(m.STOCKS[:34])
    for name,q in [('all_history',a),('since2025',a[a.date.dt.year>=2025]),('semi_optical_power_since2025',a[(a.date.dt.year>=2025)&a.symbol.isin(semi)]),('VICR_only',a[a.symbol=='VICR'])]:
        reports.append(dict(scope=name,rows=len(q),dates=q.date.nunique(),hit5=q.hit5.mean(),hit10=q.hit10.mean(),down3=q.down3.mean(),
            mean_net_close=q.close_return.mean(),median_net_close=q.close_return.median(),worst_net_close=q.close_return.min(),
            hit5_dateblock95=m.frequency_ci_by_date(q,'hit5') if len(q) else [None,None],
            stop3_target5_mean=q.toy_trade_return.mean()))
    a.to_csv(m.ROOT/'gap-analogue-events.csv',index=False)
    m.save('gap-sensitivity.json',dict(definition='Historical next open +8% to +15% versus prior close, prior daily absolute return <=5%, same liquidity filter. Conditional on the opening gap being observed; event type not classified. Chosen after discovering VICR announcement, no independent validation.',reports=reports,excluded=errors,
        warning='These include earnings, deals and other causes, and do not measure buying VICR in after-hours at the quoted price. May contain sector/date dependence and current survivor selection.'))
    print(reports)
if __name__=='__main__':main()
