import json,ssl,urllib.request,concurrent.futures,math,statistics
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import certifi
R=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
ctx=ssl.create_default_context(cafile=certifi.where())
def get(task):
 s,kind=task
 url=(f'https://cdn.cboe.com/api/global/delayed_quotes/options/{s}.json' if kind=='options' else f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range={"5d" if kind=="5m" else "2y"}&interval={kind}&includePrePost={"true" if kind=="5m" else "false"}&events=div%2Csplits')
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),context=ctx,timeout=30) as f: data=json.load(f)
  (R/'raw'/f'{s}-{kind}.json').write_text(json.dumps(data))
  return dict(symbol=s,kind=kind,url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),status='ok')
 except Exception as e:return dict(symbol=s,kind=kind,url=url,error=str(e))
if __name__=='__main__':
 tasks=[(s,k) for s in ['AVAV','KTOS','RCAT','ONDS'] for k in ['1d','5m','options']]+[('SPY','1d')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: result=list(ex.map(get,tasks))
 (R/'manifest.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
