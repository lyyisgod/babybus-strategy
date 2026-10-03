"""Numerical and timestamp checks; no claim of investment validity."""
import ast,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from strategies.bb_washout_bounce.data import load_market,load_config,save,calendar
cfg=load_config();data,meta,errors,last=load_market(P/'raw',cfg)
res=json.loads((P/'weekend-results.json').read_text())
protocol=json.loads((P/'weekend_protocol.json').read_text());names=protocol['features']
tree=ast.parse((P/'weekend_model.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='nearest')
ns={'np':np,'pd':pd,'FEATURES':names}
exec(compile(ast.Module(body=[node],type_ignores=[]),'nearest_unit','exec'),ns)
x=pd.DataFrame({k:np.arange(100,dtype=float) for k in names});cur=pd.Series({k:50. for k in names})
got=ns['nearest'](x,cur)
assert len(got)==40 and got.index[0]==50
# Outcomes cannot affect neighbor selection.
x['monday_close']=np.arange(100)*999
assert ns['nearest'](x,cur).index.tolist()==got.index.tolist()
checks=0
for row in res['rows']:
 s=row['symbol'];d=data[s]
 assert str(d.index[-1].date())=='2026-09-17'
 if 'analog' not in row:continue
 a=pd.read_csv(P/f'{s}-analogs.csv')
 for _,r in a.iterrows():
  sig=pd.Timestamp(r.signal_date);entry=pd.Timestamp(r.entry_date);end=pd.Timestamp(r.exit_date)
  assert sig.weekday()==3 and entry.weekday()==4 and end.weekday()==0
  assert end<last and (end-entry).days==3
  for label,col in [('monday_close','close'),('monday_high','high'),('monday_low','low')]:
   assert np.isclose(r[label],d.loc[end,col]/d.loc[entry,'close']-1)
  checks+=1
 assert np.isclose((a.monday_close-.002).mean(),row['analog']['mean_net'])
 z=pd.read_csv(P/f'{s}-oos.csv')
 for label in ['up','close5']:
  assert np.isclose(((z['forecast_'+label]-z['actual_'+label].astype(float))**2).mean(),row['oos'][label]['brier'])
cal=calendar(cfg);assert cal.next_session(pd.Timestamp('2026-09-18'))==pd.Timestamp('2026-09-21')
assert data['NBIS'].index[0]>=pd.Timestamp('2024-10-21')
save(P/'validation.json',dict(status='passed',checked_analog_labels=checks,checks=['nearest exact match','outcomes do not influence neighbors','no incomplete current daily bar','Thu/Fri/Mon dates and holiday exclusions','raw price return labels','fee subtraction','Brier recomputation','next trading day Sep21','NBIS predecessor absent'],limitations='Does not establish causal edge, probability calibration or efficacy of live-Friday filters, entry band or exits.'))
print('PASS',checks,'historical analog labels and numerical/time checks')
