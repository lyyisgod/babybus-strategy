import json,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
P=Path(__file__).resolve().parent;Z=ZoneInfo('America/New_York');out={}
for s in ['AAPL','HOOD','APP','XOM','QQQ']:
 a=json.loads((P/(s+'_5m.json')).read_text())['chart']['result'][0];assert a['meta']['dataGranularity']=='5m'
 f=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert(Z));m=f.index.hour*60+f.index.minute
 # Common complete 5 minute bars 09:30 through 13:30 inclusive.
 f=f[(m>=570)&(m<815)].dropna(subset=['close']);d=f.groupby(f.index.date).volume.sum();today=datetime.date(2026,9,10);prior=d[d.index<today].tail(20)
 out[s]={'volume_before_1335':float(d.loc[today]),'prior20_same_clock_mean':float(prior.mean()),'same_clock_volume_ratio':float(d.loc[today]/prior.mean()),'history_n':len(prior),'latest_price':a['meta']['regularMarketPrice'],'quote_time':datetime.datetime.fromtimestamp(a['meta']['regularMarketTime'],Z).isoformat()}
for s in ['VIX','TNX','BZ=F']:
 a=json.loads((P/(s+'_live.json')).read_text())['chart']['result'][0]['meta'];out[s]={'price':a['regularMarketPrice'],'quote_time':datetime.datetime.fromtimestamp(a['regularMarketTime'],Z).isoformat()}
(P/'量能结果.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
