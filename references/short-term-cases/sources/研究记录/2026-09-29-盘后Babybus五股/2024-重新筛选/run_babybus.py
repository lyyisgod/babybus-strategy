from pathlib import Path
import sys,json,datetime
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import load_config,load_market,read_records,save
from strategies.bb_washout_bounce.backtest import Engine
from strategies.bb_washout_bounce.daily import daily_report
P=Path(__file__).resolve().parent;cfg=load_config();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
m,meta,errors,last=load_market(P/'bb-raw',cfg,now)
# Existing calendar starts in 2020; retain original snapshots, trim in-memory warmup only.
m={k:v.loc['2021-01-01':] for k,v in m.items()}
u=json.loads((ROOT/'config/bb_universe.json').read_text())
r={k:read_records(ROOT/'data/bb_washout_bounce'/(k+'.jsonl')) for k in ['fundamentals','macro','events','event_coverage','supports','conflicts']}
e=Engine(m,u['securities'],r,cfg)
save(P/'babybus-strict/run_manifest.json',dict(as_of=now,last=str(last),errors=errors,source_manifest='bb-raw/manifest.json',trim='2021 onward for existing calendar; no change to stored raw data',universe=u))
print(daily_report(e,last,P/'babybus-strict/daily',now=now))
