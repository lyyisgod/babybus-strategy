"""Read-only public quote acquisition, preserving timestamped raw responses."""
import concurrent.futures
import hashlib
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent
CTX = ssl.create_default_context(cafile=certifi.where())


def fetch(job):
    symbol, mode = job
    query = {'daily': 'range=2y&interval=1d',
             'live': 'range=1d&interval=1m&includePrePost=true',
             'history': 'range=60d&interval=5m&includePrePost=true'}[mode]
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol, safe='') + '?' + query
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=25, context=CTX) as response:
            raw = response.read()
        data = json.loads(raw)
        if not data.get('chart', {}).get('result'):
            raise ValueError(str(data.get('chart', {}).get('error')))
        output = {'source_url': url, 'fetched_at': datetime.now(timezone.utc).isoformat(),
                  'raw_sha256': hashlib.sha256(raw).hexdigest(), 'data': data}
        (ROOT / 'raw' / f'{symbol}-{mode}.json').write_text(json.dumps(output))
        return {'symbol': symbol, 'mode': mode, 'ok': True}
    except Exception as exc:
        return {'symbol': symbol, 'mode': mode, 'ok': False, 'error': str(exc)}


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'screen':
        symbols = json.loads((ROOT / 'protocol.json').read_text())['universe'] + ['SOXX', 'QQQ']
        jobs = [(s, m) for s in symbols for m in ['daily', 'live']]
    else:
        jobs = [(s, 'history') for s in sys.argv[2:]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(fetch, jobs))
    (ROOT / f'fetch-{mode}.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'requests': len(results), 'successes': sum(r['ok'] for r in results),
                      'errors': [r for r in results if not r['ok']]}))
