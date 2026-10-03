"""Public read-only market capture. Snapshot universe saved before requests."""
import concurrent.futures, json, ssl, urllib.request, urllib.parse, certifi
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GROUPS={
 'ai_hardware':'NVDA AMD AVGO MU SNDK WDC STX MRVL ARM ALAB CRDO COHR LITE VRT DELL HPE SMCI ANET TSM AMAT LRCX KLAC',
 'cloud_software':'MSFT AMZN GOOGL META ORCL PLTR NET SNOW DDOG CRM NOW ADBE',
 'cybersecurity':'CRWD PANW ZS FTNT S OKTA',
 'high_beta':'CRWV NBIS IREN CORZ APLD WULF CIFR HUT CLSK ASTS RKLB IONQ OKLO SMR TSLA',
 'energy_power':'XOM CVX COP EOG OXY SLB VLO MPC LNG CEG VST NRG GEV ETN PWR',
 'other_sectors':'AAPL JPM GS BAC BRK-B LLY UNH ISRG ABBV WMT COST HD CAT GE RTX LMT NEM FCX UBER',
 'benchmarks':'SPY QQQ IWM RSP DIA SMH IGV CIBR XLE XLF XLV XLI XLU XLP XLY XLB XLRE XLC XLK TLT HYG LQD GLD TSLL ^VIX ^TNX DX-Y.NYB BZ=F CL=F'
}
def get(url):
    # Public read-only GETs with certificate validation.
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=35) as r:
        return r.read()
def capture(s,interval='1d',period='5y'):
    u=f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?range={period}&interval={interval}&events=div%2Csplits'
    raw=get(u); data=json.loads(raw)
    if not data['chart']['result']: raise ValueError(str(data['chart'].get('error')))
    p=ROOT/'raw'/f'{s}-{interval}.json';p.write_bytes(raw)
    return {'symbol':s,'url':u,'retrieved_at':datetime.now(timezone.utc).isoformat(),'file':p.name,'meta':data['chart']['result'][0]['meta']}
if __name__=='__main__':
    symbols=list(dict.fromkeys(' '.join(GROUPS.values()).split()))
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'groups':{g:v.split() for g,v in GROUPS.items()},'weighting':'equal within groups for descriptive breadth only','purpose':'exploratory current liquid thematic cross-sector universe, not point-in-time historical constituents','horizons':[5,10,20],'jump_definition':'20 trading-day terminal total return >= 20%','max_drawdown_preference':'10-15% per stock; not a guarantee'}
    (ROOT/'universe.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    out={'quotes':{},'errors':[]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures={ex.submit(capture,s):s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            s=futures[f]
            try: out['quotes'][s]=f.result()
            except Exception as e: out['errors'].append({'symbol':s,'error':str(e)})
    (ROOT/'capture.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('Captured',len(out['quotes']),'/',len(symbols),'errors',out['errors'])
