"""下载解析、缓存身份、明确代理和CLI失败边界；不依赖网络。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macro_regime.config import FRED_SYMBOLS, YAHOO_SYMBOLS
from macro_regime.data import (Dataset, assemble, load_series, normalize_index,
                               parse_fred_csv, require_closed_session, yahoo_fetch)
from macro_regime.factors import DataError
from macro_regime.snapshot import main
from test_macro_regime import frame


FIXTURE = Path(__file__).parent / "fixtures" / "macro_regime_fred.csv"


def test_fred_fixture_keeps_missing_observation():
    raw = parse_fred_csv(FIXTURE.read_text(), "DGS10")
    assert len(raw) == 4
    assert pd.isna(raw.loc["2026-09-28", "DGS10"])
    assert raw.loc["2026-09-30", "DGS10"] == 4.15
    with pytest.raises(DataError, match="错误序列"):
        parse_fred_csv(FIXTURE.read_text(), "DGS2")
    with pytest.raises(DataError, match="重复"):
        parse_fred_csv("observation_date,DGS10\n2026-09-30,4\n2026-09-30,5\n", "DGS10")


def fake_fetch(symbol, start, end):
    raw = pd.DataFrame({"Close": [124., 123.], "Adj Close": [124., 123.]},
                       index=pd.to_datetime(["2026-09-29", "2026-09-30"]))
    return raw, {"source_symbol": symbol, "long_name": "iShares Global Financials ETF", "source": "test_fixture"}


def test_parquet_roundtrip_checksum_and_cached_identity(tmp_path):
    a, ma = load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, fake_fetch)
    def never_fetch(*args):
        raise AssertionError("cache hit should not fetch")
    b, mb = load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, never_fetch)
    pd.testing.assert_frame_equal(a, b)
    assert not ma["proxy"] and mb["cached"]
    manifest = next(tmp_path.glob("*.json"))
    meta = json.loads(manifest.read_text())
    meta["source_symbol"] = "XLF"
    manifest.write_text(json.dumps(meta))
    with pytest.raises(DataError, match="身份不匹配"):
        load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, never_fetch)


def test_move_network_fallback_is_opt_in_and_marked_proxy(tmp_path):
    load_series("^MOVE", "2026-09-29", "2026-10-01", tmp_path, fake_fetch)
    def failing(*args):
        raise ConnectionError("offline")
    with pytest.raises(DataError):
        load_series("^MOVE", "2026-09-29", "2026-10-01", tmp_path, failing, refresh=True)
    _, meta = load_series("^MOVE", "2026-09-29", "2026-10-01", tmp_path, failing,
                          refresh=True, allow_stale_cache=True)
    assert meta["proxy"] and meta["stale_cache"]
    assert meta["fetch_error"] == "offline"


@pytest.mark.parametrize("symbol", ["IXG", "JNK", "QQQ", "SMH", "DGS10", "DGS2", "DFF", "PAYEMS", "BAMLH0A0HYM2"])
def test_non_move_download_failure_cannot_use_proxy_cache(tmp_path, symbol):
    load_series(symbol, "2026-09-29", "2026-10-01", tmp_path, fake_fetch)
    def failing(*args):
        raise ConnectionError("offline")
    with pytest.raises(DataError):
        load_series(symbol, "2026-09-29", "2026-10-01", tmp_path, failing,
                    refresh=True, allow_stale_cache=True)


def test_cache_integrity_error_is_not_network_failure(tmp_path):
    load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, fake_fetch)
    parquet = next(tmp_path.glob("*.parquet"))
    parquet.write_bytes(parquet.read_bytes() + b"corruption")
    with pytest.raises(DataError, match="摘要不匹配"):
        load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, fake_fetch)


@pytest.mark.parametrize("wrong", ["XLF", "SOX", None, "global financials"])
def test_wrong_ixg_name_raises_without_substitution(monkeypatch, wrong):
    fake = SimpleNamespace(get_info=lambda: {"longName": wrong})
    symbols = []
    def ticker(symbol):
        symbols.append(symbol)
        return fake
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker))
    with pytest.raises(DataError, match="身份校验"):
        yahoo_fetch("IXG", "2026-09-29", "2026-10-01")
    assert symbols == ["IXG"]


def test_yahoo_close_and_adjusted_basis_are_kept_separate(monkeypatch):
    calls = []
    def history(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"Close": [124.], "Adj Close": [120.]},
                            index=pd.DatetimeIndex(["2026-09-30"], tz="America/New_York"))
    fake = SimpleNamespace(get_info=lambda: {"longName": "iShares Global Financials ETF"}, history=history)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda symbol: fake))
    raw, meta = yahoo_fetch("IXG", "2026-09-29", "2026-10-01")
    assert raw.Close.iloc[-1] == 124 and raw["Adj Close"].iloc[-1] == 120
    assert calls[0]["auto_adjust"] is False
    assert calls[0]["repair"] is False
    assert raw.index.tz is None
    assert meta["source_symbol"] == "IXG"


def sources():
    source = frame()
    # 代理波动率不能用零波动TLT作为测试输入。
    t = np.arange(len(source))
    source["TLT"] = source["TLT_adj"] = 100 * np.exp(.01 * np.sin(t / 8))
    yahoo = {}
    for s in YAHOO_SYMBOLS:
        name = {"^VIX": "VIX", "^TNX": "TNX", "^MOVE": "MOVE"}.get(s, s)
        raw = pd.DataFrame({"Close": source[name]}, index=source.index)
        if not s.startswith("^"):
            raw["Adj Close"] = source[f"{s}_adj"]
        yahoo[s] = raw
    fred = {s: source[[s]].copy() for s in FRED_SYMBOLS if s != "PAYEMS"}
    fred["PAYEMS"] = pd.DataFrame({"PAYEMS": [159000., 159100.]},
                                   index=pd.to_datetime(["2026-08-01", "2026-09-01"]))
    return source, yahoo, fred


def test_move_missing_uses_tlt_with_explicit_units_and_monthly_payems():
    source, yahoo, fred = sources()
    del yahoo["^MOVE"]
    data = assemble(yahoo, fred, source.index[-1], source.index, {})
    assert data.frame.move_proxy.all()
    assert data.metadata["proxy"]
    assert data.metadata["series"]["MOVE"]["unit"] == "annualized_percent"
    expected = np.log(source.TLT_adj / source.TLT_adj.shift()).rolling(20).std(ddof=1) * np.sqrt(252) * 100
    pd.testing.assert_series_equal(data.frame.MOVE, expected, check_names=False)
    assert len(data.payems) == 2
    assert "PAYEMS" not in data.frame
    json.dumps(data.metadata, allow_nan=False)


def test_assembled_real_move_metadata_is_json_serializable():
    source, yahoo, fred = sources()
    data = assemble(yahoo, fred, source.index[-1], source.index, {})
    assert data.metadata["move_proxy"] is False
    json.dumps(data.metadata, allow_nan=False)


def test_short_real_move_history_is_not_replaced_with_tlt():
    source, yahoo, fred = sources()
    yahoo["^MOVE"] = yahoo["^MOVE"].tail(251)
    data = assemble(yahoo, fred, source.index[-1], source.index, {})
    assert not data.frame.move_proxy.any()
    assert data.frame.MOVE.iloc[-1] == source.MOVE.iloc[-1]
    assert data.frame.MOVE.iloc[:-251].isna().all()


@pytest.mark.parametrize("symbol", ["IXG", "JNK", "DGS10", "PAYEMS", "TLT"])
def test_non_move_proxy_metadata_is_rejected(symbol):
    source, yahoo, fred = sources()
    with pytest.raises(DataError, match="只有MOVE允许代理"):
        assemble(yahoo, fred, source.index[-1], source.index, {symbol: {"proxy": True}})


def test_fetcher_cannot_hide_non_move_proxy_by_overwriting_metadata(tmp_path):
    def proxy_fetch(*args):
        raw, meta = fake_fetch(*args)
        return raw, {**meta, "proxy": True}
    with pytest.raises(DataError, match="只有MOVE允许代理"):
        load_series("IXG", "2026-09-29", "2026-10-01", tmp_path, proxy_fetch)


def test_missing_one_move_date_never_splices_raw_move_into_proxy():
    source, yahoo, fred = sources()
    yahoo["^MOVE"].loc[source.index[-1], "Close"] = np.nan
    data = assemble(yahoo, fred, source.index[-1], source.index, {})
    assert data.frame.move_proxy.all()
    assert data.frame.MOVE.iloc[-2] < 10  # 明确为TLT RV，而非MOVE的100。


@pytest.mark.parametrize("missing", ["JNK", "DFF", "PAYEMS"])
def test_missing_required_source_fails(missing):
    source, yahoo, fred = sources()
    if missing in yahoo:
        del yahoo[missing]
    else:
        del fred[missing]
    with pytest.raises(DataError):
        assemble(yahoo, fred, source.index[-1], source.index, {})


def test_daily_fred_unexpected_gap_fails():
    source, yahoo, fred = sources()
    fred["DGS10"].loc[source.index[-20], "DGS10"] = np.nan
    with pytest.raises(DataError, match="非treasury_closed"):
        assemble(yahoo, fred, source.index[-1], source.index, {})


def test_daily_fred_treasury_holiday_is_not_forward_filled():
    source, yahoo, fred = sources()
    holiday = pd.Timestamp("2025-10-13")
    fred["DGS10"].loc[holiday, "DGS10"] = np.nan
    data = assemble(yahoo, fred, source.index[-1], source.index, {})
    assert pd.isna(data.frame.loc[holiday, "DGS10"])
    assert data.metadata["fred_daily_missing_sessions"]["DGS10"] == 1
    assert pd.isna(data.dgs10_publications.loc[holiday])


@pytest.mark.parametrize("on,now", [
    ("2026-10-03", "2026-10-04T00:00:00Z"),  # 周末
    ("2026-09-30", "2026-09-30T18:00:00Z"),  # 未收盘
    ("2026-11-27", "2026-11-27T17:59:00Z"),  # 半日市尚未收盘
])
def test_unclosed_or_non_session_dates_fail(on, now):
    with pytest.raises(DataError):
        require_closed_session(on, now)


def test_closed_half_day_session_is_accepted():
    assert require_closed_session("2026-11-27", "2026-11-27T18:01:00Z") == pd.Timestamp("2026-11-27")


def test_cli_missing_input_and_data_error_emit_no_json(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--date", "2026-09-30"])
    assert error.value.code == 2
    assert capsys.readouterr().out == ""
    def failing(*args):
        raise DataError("必需数据缺失")
    monkeypatch.setattr("macro_regime.snapshot.fetch_dataset", failing)
    assert main(["--date", "2026-09-30", "--regime", "HIKING", "--gross-exposure", "0.8"]) == 2
    streams = capsys.readouterr()
    assert streams.out == "" and "必需数据缺失" in streams.err


@pytest.mark.parametrize("gross", ["nan", "inf", "-1"])
def test_invalid_exposure_rejected_before_download(monkeypatch, capsys, gross):
    def never_fetch(*args):
        raise AssertionError("非法敞口不应触发下载")
    monkeypatch.setattr("macro_regime.snapshot.fetch_dataset", never_fetch)
    assert main(["--date", "2026-09-30", "--regime", "HIKING", "--gross-exposure", gross]) == 2
    assert capsys.readouterr().out == ""


def test_cli_prints_finite_json_with_explicit_caller_policy(monkeypatch, capsys, tmp_path):
    dataset = Dataset(frame(), pd.Series([159000.]), {"warnings": [], "proxy": False, "pit_verified": False})
    monkeypatch.setattr("macro_regime.snapshot.fetch_dataset", lambda *args: dataset)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"date": "2026-09-30", "policy_regime": "PAUSE", "gross_exposure": .8}))
    assert main(["--date", "2026-09-30", "--regime", "PAUSE", "--policy-file", str(policy)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["regime"] == "PAUSE" and result["action"] == "CORE_HOLD"
    assert result["constraints"]["allow_margin"] is True
    assert result["data"]["pit_verified"] is False
    assert main(["--date", "2026-09-29", "--regime", "PAUSE", "--policy-file", str(policy)]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", ["hiking", "UNKNOWN", ""])
def test_cli_regime_has_strict_choices(value, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--date", "2026-09-30", "--regime", value, "--gross-exposure", ".8"])
    assert error.value.code == 2
    assert capsys.readouterr().out == ""


def test_policy_file_cannot_supply_missing_regime(tmp_path, capsys):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"date": "2026-09-30", "policy_regime": "HIKING", "gross_exposure": .8}))
    with pytest.raises(SystemExit) as error:
        main(["--date", "2026-09-30", "--policy-file", str(policy)])
    assert error.value.code == 2
    assert capsys.readouterr().out == ""
