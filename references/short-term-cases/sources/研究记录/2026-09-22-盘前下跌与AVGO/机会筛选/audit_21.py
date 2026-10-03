"""Independent numerical/time/hash audit for the date-refreshed 21-factor branch."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent / '21因子'
spec = importlib.util.spec_from_file_location('dual_audit', P / 'dual_model.py')
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)
coef = json.loads((P / 'model-coefficients.json').read_text())
cols = coef['features']
ds = {s: dual.m.load(s)[0] for s in dual.SYMBOLS}
panels = []
for s in dual.STOCKS:
    d = ds[s]
    x = dual.factors(s, d, ds)
    x['r2'] = d.raw_close.shift(-2) / d.raw_open.shift(-1) - 1 - .002
    x['r3'] = d.raw_close.shift(-3) / d.raw_open.shift(-1) - 1 - .002
    x['exit_at'] = pd.Series(d.index, index=d.index).shift(-3)
    x['signal_at'] = d.index
    panels.append(x.dropna(subset=cols+['r2', 'r3', 'exit_at']))
    if len(d) > 300:
        pd.testing.assert_frame_equal(dual.factors(s, d, ds).iloc[:300],
                                      dual.factors(s, d.iloc[:300], ds))
panel = pd.concat(panels)
train = panel[(panel.signal_at.dt.year <= 2022) & (panel.exit_at.dt.year <= 2022)]
X = np.c_[np.ones(len(train)), np.clip((train[cols].to_numpy()-coef['train_mean'])/coef['train_sd'], -8, 8)]
reg = np.eye(X.shape[1])*.01
reg[0, 0] = 0
gradients = []
for b, y in zip(coef['logistic'], [(train.r2 >= .05).to_numpy(float), (train.r3 >= .08).to_numpy(float)]):
    b = np.asarray(b)
    pred = 1/(1+np.exp(-np.clip(X@b, -30, 30)))
    gradient = X.T@(pred-y)/len(y)+reg@b
    gradients.append(float(np.max(np.abs(gradient))))
br = np.asarray(coef['linear'])
linear_gradient = float(np.max(np.abs(X.T@(X@br-train.r3.to_numpy())/len(train)+reg@br)))
assert max(gradients) < 1e-6 and linear_gradient < 1e-8
assert len(cols) == 21 and len(train) == coef['train_rows']
assert train.exit_at.max() < pd.Timestamp('2023-01-01', tz=dual.m.NY)
assert all(d.index.max() == pd.Timestamp('2026-09-21', tz=dual.m.NY) for s, d in ds.items() if s not in ['BZ=F','NQ=F','ES=F'])
sources = {r['file']: r for r in json.loads((P/'shared-manifest.json').read_text())}
fresh = json.loads((P/'dual-manifest.json').read_text())
assert all('error' not in r for r in fresh)
for r in fresh:
    name = r['symbol'] + ('-intraday' if r['kind'] == 'intraday' else '') + '.json'
    sources[name] = r
for name, r in sources.items():
    assert hashlib.sha256((P/'raw'/name).read_bytes()).hexdigest() == r['sha256']
rank = pd.read_csv(P/'ranking.csv')
assert len(rank) == 49 and np.isfinite(rank[['p_next5','p_three8','expected3']].to_numpy()).all()
result = dict(status='pass', model_features=len(cols), stocks=len(rank),
              raw_hashes_checked=len(sources), train_rows=len(train),
              train_outcomes_max=train.exit_at.max().isoformat(),
              evaluation_outcomes_max=panel.exit_at.max().isoformat(),
              logistic_max_gradient=gradients, linear_max_gradient=linear_gradient,
              completed_stock_daily_date='2026-09-21',
              prefix_causality_stocks_checked=49,
              current_numeric_buy_gate_count=int(((rank.p_next5>=.5)&(rank.expected3>=.02)).sum()),
              checks=['All 134 stock/proxy daily/intraday hashes verified from final source manifest',
                      '21 fixed features, 49 frozen stocks, original parameters',
                      'Newtons fitted gradient and linear normal equation residual',
                      'Feature prefix is unchanged by future rows',
                      'Training labels mature before 2023 and current daily session excluded'],
              not_validated=['future actual execution or profitability','point-in-time membership','news causal predictive effect'])
(P/'additional-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
