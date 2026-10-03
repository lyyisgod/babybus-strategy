"""Refresh only the locked finalists; preserve selection and test snapshots."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json,ssl,urllib.request
import certifi
p=Path(__file__).resolve().parent
out=[]
for s in ['LRCX','AMAT']:
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=1d&interval=1m'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    raw=urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=25).read()
    (p/f'{s}-latest.json').write_bytes(raw)
    m=json.loads(raw)['chart']['result'][0]['meta']
    row=dict(symbol=s,price=m['regularMarketPrice'],quote_time=datetime.fromtimestamp(m['regularMarketTime'],ZoneInfo('America/New_York')).isoformat(),fetched_at=datetime.now(timezone.utc).isoformat(),source=url)
    out.append(row)
(p/'latest-quotes.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
