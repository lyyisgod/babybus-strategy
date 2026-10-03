"""Read-only public factor capture. No trading or private account access."""
import json, ssl, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import certifi
R=Path(__file__).resolve().parent
SYMS='AVGO MU NVDA AMD TSM MRVL AMAT LRCX KLAC WDC STX SNDK QQQ SPY SMH SOXX RSP HYG LQD TLT ^VIX ^VIX9D ^VIX3M ^TNX ^IRX DX-Y.NYB BZ=F CL=F'.split()
def fetch(task):
    name,url=task
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=22) as f: data=f.read()
        (R/'raw'/name).write_bytes(data)
        return dict(file=name,source=url,retrieved_at=datetime.now(timezone.utc).isoformat(),status='ok',bytes=len(data))
    except Exception as e:return dict(file=name,source=url,retrieved_at=datetime.now(timezone.utc).isoformat(),status='unavailable',error=str(e))
if __name__=='__main__':
    (R/'raw').mkdir(parents=True,exist_ok=True)
    (R/'universe.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'symbols':SYMS,'target':['AVGO','MU'],'horizon':'2026-09-16 close versus 2026-09-15 close; today still provisional','selection':'predeclared liquid semiconductor peers and macro proxies; not all US stocks'},indent=2))
    tasks=[(s+'-1d.json',f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?range=10y&interval=1d&events=div%2Csplits') for s in SYMS]
    tasks += [(s+'-options.json',f'https://cdn.cboe.com/api/global/delayed_quotes/options/{s}.json') for s in ['AVGO','MU']]
    tasks += [('fred-'+s+'.csv',f'https://fred.stlouisfed.org/graph/graph.csv?id={s}&cosd=2026-07-01&coed=2026-09-15') for s in ['DFII10','BAMLH0A0HYM2','T10YIE']]
    tasks += [('fomc.html','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm')]
    with ThreadPoolExecutor(max_workers=6) as ex:result=list(ex.map(fetch,tasks))
    (R/'capture.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
