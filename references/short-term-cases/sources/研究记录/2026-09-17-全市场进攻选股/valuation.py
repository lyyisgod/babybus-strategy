"""CLS dilution and valuation scenarios, not probabilities or a fitted forecast."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
d=pd.read_csv(ROOT/'factors.csv').set_index('symbol')
price=float(d.loc['CLS','price'])
price_time=str(d.loc['CLS','time_et'])
if (ROOT/'latest/quotes.csv').exists():
    latest=pd.read_csv(ROOT/'latest/quotes.csv').set_index('symbol')
    price=float(latest.loc['CLS','price']);price_time=str(latest.loc['CLS','time_et'])
old_eps=11.30;old_weighted_shares_m=116.2
new_shares_m=(9677419+1451612)/1e6
full_year_base=old_eps*old_weighted_shares_m/(old_weighted_shares_m+new_shares_m)
calendar_year_base=old_eps*old_weighted_shares_m/(old_weighted_shares_m+new_shares_m*147/365)
rows=[]
for name,eps,pe in [('severe',10,20),('bear',13,20),('base',17,25),('bull',20,30)]:
    target=eps*pe
    rows.append({'scenario':name,'assumed_2027_adjusted_eps':eps,'assumed_pe':pe,'price_scenario':target,'return_pct':100*(target/price-1),'probability':None})
out={'snapshot_price':price,'price_time_et':price_time,'source_eps_announcement':'2026-07-27','old_guidance_eps':old_eps,'old_guidance_assumes_no_issuance':True,'q2_weighted_diluted_shares_m_proxy':old_weighted_shares_m,'new_shares_m_issued_2026_08_07':new_shares_m,'greenshoe_fully_exercised':True,'annual_weighted_dilution_sensitivity_eps':calendar_year_base,'full_year_new_share_run_rate_sensitivity_eps':full_year_base,'price_over_full_year_sensitivity_eps':price/full_year_base,'scenarios':rows,'planned_exit_reference':294.0,'planned_loss_pct':100*(294/price-1),'notes':['Scenario EPS and multiples are analyst assumptions, not company guidance, consensus or probabilities.','Full-year base assumes new shares existed for the entire year; excludes interest/investment returns of new cash.','3-6 month horizon assumes market values 2027 earnings before their actual realization.','Severe scenario is not a loss floor; stop prices do not guarantee execution.','Manufacturing customer concentration, working capital and low margins justify valuation discount to fabless companies.']}
assert abs(full_year_base-10.3121967)<.001
assert rows[2]['price_scenario']==425 and rows[3]['price_scenario']==600
(ROOT/'valuation.json').write_text(json.dumps(out,indent=2))
pd.DataFrame(rows).to_csv(ROOT/'valuation_scenarios.csv',index=False)
print(json.dumps(out,indent=2))
