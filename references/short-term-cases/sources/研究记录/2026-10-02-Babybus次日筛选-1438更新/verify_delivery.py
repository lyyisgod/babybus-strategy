from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
def read(f):return json.loads((P/f).read_text())
checks={}
for folder in ['daily-raw','live-raw','shortlist-raw','final-raw','tsla-daily-raw']:
    manifest=read(folder+'/manifest.json')
    checks[folder+'_hashes']=all('error' not in r and hashlib.sha256((P/folder/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in manifest)
events=read('path-events.json');checks['mature_only']=all(e['exit_session']<='2026-10-01' for e in events)
checks['entry_clock']=all(e['entry_clock']=='14:35' for e in events)
checks['double_cost']=all(abs(e['cost_double_net']-(e['net_return']-.002))<1e-12 for e in events)
checks['nonoverlap']=True
for s in {e['ticker'] for e in events}:
    es=sorted([e for e in events if e['ticker']==s],key=lambda e:e['entry_session'])
    checks['nonoverlap'] &= all(es[i]['entry_session']>es[i-1]['exit_session'] for i in range(1,len(es)))
# Independent replay of stored path trades checks price/time/cost and stop-first handling.
checks['all_path_replay']=True
for s in {e['ticker'] for e in events}:
    a=read('shortlist-raw/'+s+'.json')['chart']['result'][0]
    df=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert('America/New_York')).dropna(subset=['open','high','low','close','volume'])
    minutes=df.index.hour*60+df.index.minute;df=df[(minutes>=570)&(minutes<960)]
    for e in [e for e in events if e['ticker']==s]:
        en=df[df.index.strftime('%Y-%m-%d')==e['entry_session']];ex=df[df.index.strftime('%Y-%m-%d')==e['exit_session']]
        p=e['entry_price'];stop=p*.97;target=p*1.05
        assert np.isclose(float(en[en.index.hour*60+en.index.minute<875].close.iloc[-1]),p)
        ex=ex[ex.index.hour*60+ex.index.minute<955];path=pd.concat([en[en.index.hour*60+en.index.minute>=875],ex]);exitprice=float(ex.close.iloc[-1])
        for ts,b in path.iterrows():
            if b.open<=stop:exitprice=float(b.open);break
            if b.open>=target:exitprice=target;break
            if b.low<=stop:exitprice=stop;break
            if b.high>=target:exitprice=target;break
        checks['all_path_replay'] &= abs(exitprice/p-1-.002-e['net_return'])<1e-12
for r in read('final-live.json'):
    assert pd.Timestamp(r['quote_at'])<=pd.Timestamp(r['fetched_at'])
    assert pd.Timestamp(r['last_completed_end'])<=pd.Timestamp(r['fetched_at'])
checks['quote_and_bar_time']=True
checks['factor_prefix']=all(read('verification.json')['truncation_checks'].values()) and all(read('path-verification.json')['prefix'].values())
checks['all_factors']=len(read('all-factors.json'))==154
checks['original_early_report_preserved']=(P.parent/'2026-10-02-Babybus今日至下个交易日/decision.json').exists()
assert all(checks.values()),checks
(P/'delivery-verification.json').write_text(json.dumps({'checks':checks,'path_trades':len(events),'investment_effectiveness_verified':False,'current_probability_certified':False},ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False));print('trades',len(events))
