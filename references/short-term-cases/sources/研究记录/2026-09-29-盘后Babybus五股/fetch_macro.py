from pathlib import Path
import urllib.request,ssl,certifi,json,hashlib
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
rows=[]
for s in ['DFII10','BAMLH0A0HYM2']:
 u=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=2026-06-01&coed=2026-09-29'
 try:
  raw=urllib.request.urlopen(u,context=ssl.create_default_context(cafile=certifi.where()),timeout=25).read();(P/(s+'.csv')).write_bytes(raw)
  rows.append(dict(series=s,url=u,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest()))
 except Exception as e:rows.append(dict(series=s,url=u,error=str(e)))
(P/'macro-manifest.json').write_text(json.dumps(rows,indent=2));print(rows)
