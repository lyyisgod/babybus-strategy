"""Option cost, skew and risk-neutral spread proxies; never physical win rates."""
import json,re,math
from pathlib import Path
R=Path(__file__).resolve().parent

def parse(name):
    m=re.fullmatch(r'([A-Z]+)(\d{6})([CP])(\d{8})',name)
    return (m[2],m[3],int(m[4])/1000) if m else None

def analyze(s):
    raw=json.loads((R/'raw'/f'{s}-options.json').read_text());d=raw['data'];spot=d['current_price'];rows=[]
    for x in d['options']:
        p=parse(x['option'])
        if not p:continue
        exp,typ,k=p;b=x.get('bid');a=x.get('ask')
        if b is None or a is None or a<b or b<0:continue
        mid=(a+b)/2
        rows.append(dict(x,expiry=exp,type=typ,strike=k,mid=mid,spread_pct=(a-b)/mid if mid else None))
    out={'symbol':s,'source':f'https://cdn.cboe.com/api/global/delayed_quotes/options/{s}.json','vendor_timestamp_without_zone':raw.get('timestamp'),'underlying_timestamp_without_zone':d.get('last_trade_time'),'underlying_spot_at_chain':spot,'vendor_iv30_pct_proxy':d.get('iv30'),'vendor_iv30_change_points':d.get('iv30_change'),'iv30_sigma_one_trading_day_pct_proxy':d['iv30']/math.sqrt(252),'expiries':[],'notes':'Delayed quotes. IV30 is vendor aggregate, not verified constant-maturity ATM. US equity American options: put spread slope only a rough risk-neutral price proxy, not realized probability. No trade-side/hedge interpretation of volume or OI.'}
    for exp in ['260916','260918','260925','261002']:
        a=[x for x in rows if x['expiry']==exp];calls={x['strike']:x for x in a if x['type']=='C'};puts={x['strike']:x for x in a if x['type']=='P'}
        both=set(calls)&set(puts)
        if not both:continue
        k=min(both,key=lambda k:abs(k-spot));c,p=calls[k],puts[k]
        tight=[x for x in a if x['bid']>0 and x['spread_pct'] is not None and x['spread_pct']<=.5 and x.get('iv',0)>0]
        c25=min([x for x in tight if x['type']=='C'],key=lambda x:abs(x['delta']-.25),default=None)
        p25=min([x for x in tight if x['type']=='P'],key=lambda x:abs(x['delta']+.25),default=None)
        cv=sum(x.get('volume',0) or 0 for x in a if x['type']=='C');pv=sum(x.get('volume',0) or 0 for x in a if x['type']=='P')
        co=sum(x.get('open_interest',0) or 0 for x in a if x['type']=='C');po=sum(x.get('open_interest',0) or 0 for x in a if x['type']=='P')
        strike=sorted(k for k,x in puts.items() if x['bid']>0 and x['spread_pct'] is not None and x['spread_pct']<=.5)
        under=[k for k in strike if k<spot];over=[k for k in strike if k>spot];digital=None
        if under and over:
            k1=under[-1];k2=over[0];p1,p2=puts[k1],puts[k2];w=k2-k1
            digital={'strike_interval':[k1,k2],'mid_slope':(p2['mid']-p1['mid'])/w,'quote_bound':[max(0,(p2['bid']-p1['ask'])/w),min(1,(p2['ask']-p1['bid'])/w)],'interpretation':'Approximate probability around the strike interval under pricing measure; American exercise, bid-ask and nonsynchronous quotes limit validity; no risk-free adjustment over short tenor.'}
        out['expiries'].append({'expiry':exp,'contracts':len(a),'ATM_strike':k,'ATM_call':{j:c.get(j) for j in ['option','bid','ask','mid','iv','delta','volume','open_interest','last_trade_time']},'ATM_put':{j:p.get(j) for j in ['option','bid','ask','mid','iv','delta','volume','open_interest','last_trade_time']},'straddle_mid':c['mid']+p['mid'],'straddle_cost_pct':100*(c['mid']+p['mid'])/spot,'straddle_bid_ask_cost_pct':[100*(c['bid']+p['bid'])/spot,100*(c['ask']+p['ask'])/spot],'put_call_volume':pv/cv if cv else None,'put_call_OI':po/co if co else None,'iv_25delta_skew_pp':100*(p25['iv']-c25['iv']) if p25 and c25 else None,'skew_contracts':[p25['option'],c25['option']] if p25 and c25 else None,'put_spread_pricing_proxy':digital})
    return out

if __name__=='__main__':
    assert parse('AVGO260916C00340000')==('260916','C',340)
    assert parse('invalid') is None
    out=[analyze(s) for s in ['AVGO','MU']]
    (R/'options-analysis.json').write_text(json.dumps(out,indent=2))
    for x in out:
        print(x['symbol'],'IV30',x['vendor_iv30_pct_proxy'])
        for y in x['expiries']:print(y['expiry'],'straddle%',y['straddle_cost_pct'],'skew',y['iv_25delta_skew_pp'],'PC',y['put_call_volume'],'digital',y['put_spread_pricing_proxy'])
