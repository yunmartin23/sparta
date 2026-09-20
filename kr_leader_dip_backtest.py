"""
기간별 대장주(주도주) TOP7 × '20% 하락 후 절반 회복' 매수 전략 백테스트

전략 출처: 유튜브 쇼츠 lw5e3wvG4xE — 트레이더 오스틴 실버(Austin Silver)
  1) 사고 싶었지만 너무 올라서 못 산 주식이 고점 대비 -20% 이상 빠질 때까지 '가만히 있기'
  2) 저점에서 하락폭의 절반(50%)을 회복하면 그때 매수
     ("바닥에서 사면 더 떨어질 수 있지만, 절반 회복은 다시 오르기 시작했다는 신호")
  * 매도 규칙은 영상에 없음 → 아래 EXIT_MODES로 여러 가정을 검증한다.

데이터: FinanceData/marcap (KRX 전종목 일별 시가총액·주가, 상장폐지 종목 포함)
  https://github.com/FinanceData/marcap
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".marcap_cache")
RAW = "https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-{}.parquet"

# 기간 구분 (5년 블록, 마지막은 현재까지)
PERIODS = [
    ("2000-01-01", "2004-12-31"),
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
    ("2025-01-01", "2026-12-31"),
]

EXCLUDE_NAME = re.compile(r"우[BC]?\)?$|우선|스팩|SPAC|리츠$")
TRADING_DAYS = 250


# ---------------------------------------------------------------- 데이터 로딩

def load_universe(start_year: int = 2000) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    merged = os.path.join(CACHE, "krx_all.parquet")
    if os.path.exists(merged):
        return pd.read_parquet(merged)

    cols = ["Code", "Name", "Close", "Marcap", "Stocks", "Market", "Date", "Amount"]
    frames = []
    for year in range(start_year, 2027):
        path = os.path.join(CACHE, f"marcap-{year}.parquet")
        if not os.path.exists(path):
            print(f"  다운로드 marcap-{year}.parquet …", file=sys.stderr)
            urllib.request.urlretrieve(RAW.format(year), path)
        frames.append(pd.read_parquet(path, columns=cols))

    df = pd.concat(frames, ignore_index=True)
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    df = df[~df["Name"].str.contains(EXCLUDE_NAME, na=False, regex=True)]
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    df.to_parquet(merged, index=False)
    return df


def adjusted_series(g: pd.DataFrame) -> pd.Series:
    """액면분할·병합을 보정한 수정주가 시계열. 배당은 반영하지 않는다(가격수익률)."""
    close = g["Close"].astype(float).to_numpy()
    stocks = g["Stocks"].astype(float).to_numpy()
    n = len(close)
    if n < 2:
        return pd.Series(close, index=pd.to_datetime(g["Date"]))

    s = np.ones(n)
    c = np.ones(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        s[1:] = np.where(stocks[:-1] > 0, stocks[1:] / stocks[:-1], 1.0)
        c[1:] = np.where(close[:-1] > 0, close[1:] / close[:-1], 1.0)
    s = np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
    c = np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=1.0)

    # 주식수가 크게 변했는데 시가총액은 거의 그대로 → 액면분할/병합
    is_split = (np.abs(s * c - 1.0) < 0.15) & ((s > 1.4) | (s < 0.72))
    factor = np.where(is_split, s, 1.0)
    ret = c * factor - 1.0
    ret[0] = 0.0
    adj = close[0] * np.cumprod(1.0 + ret)
    return pd.Series(adj, index=pd.to_datetime(g["Date"].to_numpy()))


FRED_KR3M = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
             "?id=IR3TIB01KRM156N&cosd=1999-01-01")


def load_cash_rate(mode: str) -> pd.Series | float:
    """현금 보유 시 적용할 연이율. mode='real'이면 한국 3개월 은행간금리(FRED) 월별 시계열."""
    if mode != "real":
        return float(mode)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "kr_3m_rate.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(FRED_KR3M, path)
    r = pd.read_csv(path, parse_dates=["observation_date"]).set_index("observation_date")
    return (r.iloc[:, 0].astype(float) / 100.0).dropna()


def daily_cash_factor(rate, index: pd.DatetimeIndex) -> np.ndarray:
    """일별 현금 증가율 (1 + r/250)."""
    if isinstance(rate, float):
        ann = np.full(len(index), rate)
    else:
        ann = rate.reindex(index.union(rate.index)).ffill().reindex(index).ffill().bfill().to_numpy()
    return 1.0 + ann / TRADING_DAYS


def load_kospi() -> pd.Series | None:
    try:
        import FinanceDataReader as fdr
        k = fdr.DataReader("KS11", "1999-12-01")["Close"]
        return pd.Series(k.to_numpy(dtype=float), index=pd.to_datetime(k.index))
    except Exception as exc:  # 네트워크 미가용 등
        print(f"  (KOSPI 벤치마크 생략: {exc})", file=sys.stderr)
        return None


def leaders_by_period(df: pd.DataFrame, top_n: int, mode: str) -> dict:
    """기간별 주도주 TOP N.

    mode='turnover' : 해당 기간 일평균 거래대금 상위(시총 100위 이내) = 그 시기 실제 주도주
    mode='exante'   : 기간 시작 직전 1년 거래대금 상위(시총 100위 이내) = 사전 선정 가능
    mode='marcap'   : 기간 시작일 시가총액 상위
    """
    out = {}
    all_dates = df["Date"]
    for a, b in PERIODS:
        if mode == "marcap":
            d0 = all_dates[all_dates >= a].min()
            snap = df[df["Date"] == d0].sort_values("Marcap", ascending=False)
            picks = snap.head(top_n)[["Code", "Name"]].to_numpy().tolist()
        else:
            if mode == "exante":
                lo = (pd.Timestamp(a) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
                win = df[(df["Date"] >= lo) & (df["Date"] < a)]
            else:
                win = df[(df["Date"] >= a) & (df["Date"] <= b)]
            if win.empty:
                continue
            ndays = win["Date"].nunique()
            g = win.groupby("Code").agg(
                Name=("Name", "last"), amt=("Amount", "mean"),
                mc=("Marcap", "mean"), n=("Date", "size"))
            g = g[g["n"] > ndays * 0.3]
            g = g[g["mc"].rank(ascending=False) <= 100]
            g = g.sort_values("amt", ascending=False).head(top_n)
            picks = [[code, row.Name] for code, row in g.iterrows()]
        out[(a, b)] = picks
    return out


# ---------------------------------------------------------------- 시그널 엔진

@dataclass
class Trade:
    code: str
    name: str
    entry_date: pd.Timestamp
    entry_px: float
    exit_date: pd.Timestamp | None = None
    exit_px: float | None = None
    reason: str = ""
    peak: float = 0.0
    trough: float = 0.0

    @property
    def ret(self) -> float:
        if self.exit_px is None:
            return float("nan")
        return self.exit_px / self.entry_px - 1.0

    @property
    def days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class Result:
    code: str
    name: str
    trades: list = field(default_factory=list)
    signals: list = field(default_factory=list)  # (date, idx) 매수 시그널 발생 시점
    equity: pd.Series | None = None
    bh: pd.Series | None = None
    exposure: float = 0.0
    in_mkt: np.ndarray | None = None


def run_signal(px: pd.Series, drop: float, rec: float, exit_mode: str,
               exit_param: float, cost_bps: float, code: str, name: str,
               cash_factor: np.ndarray | None = None) -> Result:
    """
    상태기계
      WATCH : 고점(peak) 갱신하며 대기. peak 대비 drop 이상 하락 → ARMED
      ARMED : 저점(trough) 갱신. 종가 >= trough + rec*(peak-trough) → 다음날 종가 매수
      HOLD  : exit_mode 규칙에 따라 청산 → 다시 ARMED(트레일링) 또는 WATCH
    모든 판단은 당일 종가까지의 정보만 사용하고, 체결은 다음 거래일 종가로 한다.
    """
    p = px.to_numpy(dtype=float)
    dates = px.index
    n = len(p)
    res = Result(code=code, name=name)
    cost = cost_bps / 10000.0

    state = "WATCH"
    peak = p[0]
    trough = np.inf
    pending = None          # (action, ) 다음 거래일 종가에 체결
    hold_peak = 0.0
    entry_i = -1
    cur: Trade | None = None

    in_mkt = np.zeros(n, dtype=bool)
    equity = np.ones(n)
    cash = 1.0
    shares = 0.0
    cf = np.ones(n) if cash_factor is None else cash_factor

    for i in range(n):
        if i > 0 and cash > 0:
            cash *= cf[i]
        price = p[i]
        if not np.isfinite(price) or price <= 0:
            equity[i] = cash + shares * (p[i - 1] if i else 0)
            continue

        # ---- 전일 시그널 체결 ----
        if pending == "BUY":
            shares = cash / (price * (1 + cost))
            cash = 0.0
            cur = Trade(code, name, dates[i], price, peak=peak, trough=trough)
            entry_i = i
            hold_peak = price
            state = "HOLD"
            pending = None
        elif pending == "SELL":
            cash = shares * price * (1 - cost)
            shares = 0.0
            if cur is not None:
                cur.exit_date, cur.exit_px = dates[i], price
                res.trades.append(cur)
                cur = None
            pending = None

        # ---- 상태 갱신 ----
        if state == "WATCH":
            peak = max(peak, price)
            if price <= peak * (1 - drop):
                state, trough = "ARMED", price
        elif state == "ARMED":
            trough = min(trough, price)
            target = trough + rec * (peak - trough)
            if price >= target and pending is None:
                pending = "BUY"
                res.signals.append((dates[i], i))
        elif state == "HOLD":
            hold_peak = max(hold_peak, price)
            hit = False
            reason = ""
            if exit_mode == "trail":
                if price <= hold_peak * (1 - exit_param):
                    hit, reason = True, f"트레일링 -{exit_param:.0%}"
            elif exit_mode == "hold":
                if i - entry_i >= int(exit_param):
                    hit, reason = True, f"{int(exit_param)}일 경과"
            elif exit_mode == "never":
                pass
            if hit and pending is None:
                pending = "SELL"
                if cur is not None:
                    cur.reason = reason
                # 청산 후 재대기 상태 결정
                if exit_mode == "trail":
                    state, peak, trough = "ARMED", hold_peak, price
                else:
                    state, peak = "WATCH", max(hold_peak, price)

        in_mkt[i] = shares > 0
        equity[i] = cash + shares * price

    # 미청산 포지션 정리
    if shares > 0 and cur is not None:
        cur.exit_date, cur.exit_px, cur.reason = dates[-1], p[-1], "기간종료"
        res.trades.append(cur)

    res.in_mkt = in_mkt
    res.equity = pd.Series(equity, index=dates)
    res.bh = pd.Series(p / p[0], index=dates)
    res.exposure = float(in_mkt.mean())
    return res


# ---------------------------------------------------------------- 몬테카를로

def holding_runs(mask: np.ndarray) -> list:
    """보유 구간의 길이 목록."""
    runs, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def sleeve_value(p: np.ndarray, cf: np.ndarray, windows: list, cost: float) -> float:
    """주어진 (시작,길이) 보유 구간들로 운용했을 때의 최종 배수."""
    v = 1.0
    covered = np.zeros(len(p), dtype=bool)
    for st, ln in windows:
        en = min(st + ln, len(p) - 1)
        if en <= st:
            continue
        v *= (p[en] / p[st]) * (1 - cost) ** 2
        covered[st:en] = True
    v *= float(np.prod(cf[~covered]))
    return v


def monte_carlo(sleeve_data: list, n_sims: int, cost: float, rng) -> tuple:
    """실제 전략 대비, 같은 거래횟수·같은 보유일수를 무작위 시점에 배치했을 때의 분포."""
    sims = np.ones(n_sims)
    for p, cf, runs in sleeve_data:
        n = len(p)
        if not runs:
            sims *= float(np.prod(cf))
            continue
        total = sum(runs)
        vals = np.empty(n_sims)
        for k in range(n_sims):
            # 무작위 비중복 배치: 여유공간을 임의로 쪼개 각 구간 앞에 삽입
            slack = max(n - 1 - total, 0)
            cuts = np.sort(rng.integers(0, slack + 1, size=len(runs)))
            starts, prev = [], 0
            for gap, ln in zip(cuts - np.concatenate(([0], cuts[:-1])), runs):
                st = prev + int(gap)
                starts.append((st, ln))
                prev = st + ln
            vals[k] = sleeve_value(p, cf, starts, cost)
        sims *= vals
    return sims


# ---------------------------------------------------------------- 성과 지표

def cagr(series: pd.Series) -> float:
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def mdd(series: pd.Series) -> float:
    return float((series / series.cummax() - 1).min())


def sharpe(series: pd.Series) -> float:
    r = series.pct_change().dropna()
    if r.std() == 0 or len(r) < 20:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def fwd_returns(px: pd.Series, idxs: list, horizons=(20, 60, 120, 250)) -> dict:
    p = px.to_numpy(dtype=float)
    n = len(p)
    out = {}
    for h in horizons:
        vals = [p[min(i + h, n - 1)] / p[i] - 1 for _, i in idxs if i < n - 1]
        base = p[h:] / p[:-h] - 1 if n > h else np.array([])
        out[h] = (np.array(vals), base)
    return out


# ---------------------------------------------------------------- 리포트

def fmt_pct(x: float, w: int = 7) -> str:
    return "  n/a  " if not np.isfinite(x) else f"{x*100:{w}.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, default=0.20, help="고점 대비 하락 트리거")
    ap.add_argument("--rec", type=float, default=0.50, help="저점에서의 회복 비율")
    ap.add_argument("--exit", dest="exit_mode", default="trail",
                    choices=["trail", "hold", "never"])
    ap.add_argument("--exit-param", type=float, default=0.20)
    ap.add_argument("--cost-bps", type=float, default=25.0, help="편도 거래비용(bp)")
    ap.add_argument("--top", type=int, default=7)
    ap.add_argument("--select", default="turnover",
                    choices=["turnover", "exante", "marcap"])
    ap.add_argument("--cash-yield", default="real",
                    help="현금 보유 수익률: 'real'(한국 3M 은행간금리) 또는 연이율 실수(예: 0.0)")
    ap.add_argument("--sweep", action="store_true", help="파라미터 민감도 분석")
    ap.add_argument("--mc", type=int, default=0,
                    help="몬테카를로 시뮬레이션 횟수 (같은 투자비중의 무작위 타이밍과 비교)")
    ap.add_argument("--csv", default=None, help="거래내역 CSV 저장 경로")
    args = ap.parse_args()

    print("데이터 로딩 …", file=sys.stderr)
    df = load_universe()
    rate = load_cash_rate(args.cash_yield)
    kospi = load_kospi()
    leaders = leaders_by_period(df, args.top, args.select)

    # 종목별 수정주가 캐시
    need = {c for picks in leaders.values() for c, _ in picks}
    prices = {}
    for code, g in df[df["Code"].isin(need)].groupby("Code"):
        prices[code] = adjusted_series(g.sort_values("Date"))

    label = {"turnover": "기간 내 일평균 거래대금 상위(시총100위내)",
             "exante": "기간 시작 직전 1년 거래대금 상위(시총100위내)",
             "marcap": "기간 시작일 시가총액 상위"}[args.select]

    print("=" * 96)
    print(f"  기간별 대장주 TOP{args.top} × '고점 -{args.drop:.0%} 후 {args.rec:.0%} 회복시 매수' 전략")
    cy = "한국 3M 은행간금리" if args.cash_yield == "real" else f"연 {float(args.cash_yield):.1%}"
    print(f"  종목선정: {label}   |   매도: {args.exit_mode} {args.exit_param}   "
          f"|   비용: 편도 {args.cost_bps:.0f}bp   |   현금수익률: {cy}")
    print("=" * 96)

    all_trades, port_curves, rows, mc_sleeves = [], [], [], []
    ev_all = {h: ([], []) for h in (20, 60, 120, 250)}

    for (a, b), picks in leaders.items():
        sleeves, bh_sleeves = [], []
        print(f"\n■ {a[:4]}~{b[:4]}  대장주 TOP{args.top}")
        print(f"  {'종목':<16}{'전략':>9}{'B&H':>9}{'초과':>9}{'거래':>5}"
              f"{'승률':>7}{'투자비중':>9}{'전략MDD':>9}{'B&H MDD':>9}")
        for code, name in picks:
            px = prices.get(code)
            if px is None:
                continue
            w = px[(px.index >= a) & (px.index <= b)]
            if len(w) < 250:
                continue
            r = run_signal(w, args.drop, args.rec, args.exit_mode,
                           args.exit_param, args.cost_bps, code, name,
                           daily_cash_factor(rate, w.index))
            all_trades.extend(r.trades)
            sleeves.append(r.equity / r.equity.iloc[0])
            bh_sleeves.append(r.bh)

            wins = [t for t in r.trades if np.isfinite(t.ret) and t.ret > 0]
            wr = len(wins) / len(r.trades) if r.trades else float("nan")
            s_tot = r.equity.iloc[-1] / r.equity.iloc[0] - 1
            b_tot = r.bh.iloc[-1] - 1
            print(f"  {name[:14]:<16}{fmt_pct(s_tot,8)}{fmt_pct(b_tot,8)}"
                  f"{fmt_pct(s_tot-b_tot,8)}{len(r.trades):5d}{fmt_pct(wr,6)}"
                  f"{fmt_pct(r.exposure,8)}{fmt_pct(mdd(r.equity),8)}{fmt_pct(mdd(r.bh),8)}")

            if args.mc:
                mc_sleeves.append(((a, b), w.to_numpy(dtype=float),
                                   daily_cash_factor(rate, w.index),
                                   holding_runs(r.in_mkt),
                                   float(r.equity.iloc[-1] / r.equity.iloc[0])))
            fr = fwd_returns(w, r.signals)
            for h, (v, base) in fr.items():
                ev_all[h][0].extend(v.tolist())
                ev_all[h][1].extend(base.tolist())

        if not sleeves:
            continue
        idx = sleeves[0].index
        for sl in sleeves[1:]:
            idx = idx.union(sl.index)
        port = pd.concat([s.reindex(idx).ffill() for s in sleeves], axis=1).mean(axis=1)
        bh = pd.concat([s.reindex(idx).ffill() for s in bh_sleeves], axis=1).mean(axis=1)
        port_curves.append((a, b, port, bh))
        ks = float("nan")
        if kospi is not None:
            kw = kospi[(kospi.index >= idx[0]) & (kospi.index <= idx[-1])]
            if len(kw) > 1:
                ks = kw.iloc[-1] / kw.iloc[0] - 1
        rows.append(dict(period=f"{a[:4]}~{b[:4]}", kospi=ks,
                         strat=port.iloc[-1] - 1, bh=bh.iloc[-1] - 1,
                         s_cagr=cagr(port), b_cagr=cagr(bh),
                         s_mdd=mdd(port), b_mdd=mdd(bh),
                         s_shp=sharpe(port), b_shp=sharpe(bh)))
        print(f"  {'─'*86}")
        print(f"  {'TOP7 동일비중 포트':<16}{fmt_pct(port.iloc[-1]-1,8)}"
              f"{fmt_pct(bh.iloc[-1]-1,8)}{fmt_pct(port.iloc[-1]-bh.iloc[-1],8)}"
              f"{'':5}{'':7}{'':9}{fmt_pct(mdd(port),8)}{fmt_pct(mdd(bh),8)}")

    # ---------------- 기간 요약 ----------------
    print("\n" + "=" * 96)
    print(f"  기간별 포트폴리오 요약 (TOP7 동일비중, 각 종목 독립 운용, 현금수익률: {cy})")
    print("=" * 96)
    print(f"  {'기간':<12}{'전략누적':>10}{'B&H누적':>10}{'KOSPI':>10}{'전략CAGR':>10}"
          f"{'B&H CAGR':>10}{'전략MDD':>10}{'B&H MDD':>10}{'전략샤프':>9}{'B&H샤프':>9}")
    for r in rows:
        print(f"  {r['period']:<12}{fmt_pct(r['strat'],9)}{fmt_pct(r['bh'],9)}"
              f"{fmt_pct(r['kospi'],9)}{fmt_pct(r['s_cagr'],9)}{fmt_pct(r['b_cagr'],9)}"
              f"{fmt_pct(r['s_mdd'],9)}{fmt_pct(r['b_mdd'],9)}"
              f"{r['s_shp']:9.2f}{r['b_shp']:9.2f}")

    # 전 기간 체인 (기간 넘어갈 때 대장주 교체, 자본 승계)
    chain_s = np.prod([1 + r["strat"] for r in rows])
    chain_b = np.prod([1 + r["bh"] for r in rows])
    first = next(a for (a, b) in leaders if f"{a[:4]}~{b[:4]}" == rows[0]["period"])
    yrs = (pd.Timestamp(PERIODS[-1][0]) - pd.Timestamp(first)).days / 365.25
    yrs += (port_curves[-1][2].index[-1] - pd.Timestamp(PERIODS[-1][0])).days / 365.25
    print(f"  {'─'*94}")
    chain_k = np.prod([1 + r["kospi"] for r in rows if np.isfinite(r["kospi"])])
    span = f"{rows[0]['period'][:4]}~현재 연결"
    print(f"  {span:<12}{fmt_pct(chain_s-1,9)}{fmt_pct(chain_b-1,9)}"
          f"{fmt_pct(chain_k-1,9)}{fmt_pct(chain_s**(1/yrs)-1,9)}"
          f"{fmt_pct(chain_b**(1/yrs)-1,9)}")
    print(f"  {'(연 CAGR)':<12}{'':9}{'':9}{'':9}"
          f"  KOSPI {fmt_pct(chain_k**(1/yrs)-1,6)}")

    # ---------------- 이벤트 스터디 ----------------
    print("\n" + "=" * 96)
    print("  매수 시그널의 예측력 — 시그널 발생 후 수익률 vs 같은 종목 아무 날이나 샀을 때")
    print("=" * 96)
    print(f"  {'보유기간':<10}{'시그널 평균':>12}{'기준 평균':>12}{'차이':>10}"
          f"{'시그널 중앙':>12}{'기준 중앙':>12}{'시그널 승률':>12}{'기준 승률':>11}{'N':>7}")
    for h, (v, base) in ev_all.items():
        v, base = np.array(v), np.array(base)
        if len(v) == 0:
            continue
        print(f"  {h}거래일{'':<4}{fmt_pct(v.mean(),11)}{fmt_pct(base.mean(),11)}"
              f"{fmt_pct(v.mean()-base.mean(),9)}{fmt_pct(np.median(v),11)}"
              f"{fmt_pct(np.median(base),11)}{fmt_pct((v>0).mean(),11)}"
              f"{fmt_pct((base>0).mean(),10)}{len(v):7d}")

    # ---------------- 거래 통계 ----------------
    rets = np.array([t.ret for t in all_trades if np.isfinite(t.ret)])
    if len(rets):
        print("\n" + "=" * 96)
        print(f"  전체 거래 {len(rets)}건 | 승률 {(rets>0).mean()*100:.1f}% | "
              f"평균 {rets.mean()*100:+.1f}% | 중앙값 {np.median(rets)*100:+.1f}% | "
              f"최대 {rets.max()*100:+.0f}% | 최소 {rets.min()*100:+.0f}%")
        win, loss = rets[rets > 0], rets[rets <= 0]
        if len(loss):
            pf = win.sum() / abs(loss.sum())
            print(f"  평균이익 {win.mean()*100:+.1f}% | 평균손실 {loss.mean()*100:+.1f}% | "
                  f"손익비 {win.mean()/abs(loss.mean()):.2f} | Profit Factor {pf:.2f}")
        days = np.array([t.days for t in all_trades if t.exit_date is not None])
        print(f"  평균 보유일수 {days.mean():.0f}일 | 중앙값 {np.median(days):.0f}일")

    if args.csv:
        pd.DataFrame([dict(code=t.code, name=t.name, entry=t.entry_date,
                           entry_px=t.entry_px, exit=t.exit_date, exit_px=t.exit_px,
                           ret=t.ret, days=t.days, reason=t.reason,
                           peak=t.peak, trough=t.trough) for t in all_trades]
                     ).to_csv(args.csv, index=False)
        print(f"\n  거래내역 저장: {args.csv}")

    # ---------------- 몬테카를로 ----------------
    if args.mc and mc_sleeves:
        print("\n" + "=" * 96)
        print(f"  몬테카를로 — 같은 종목·같은 거래횟수·같은 보유일수를 '무작위 시점'에 배치 ({args.mc}회)")
        print("=" * 96)
        rng = np.random.default_rng(20260920)
        cost = args.cost_bps / 10000.0
        by_period: dict = {}
        for key, p_, cf_, runs_, act in mc_sleeves:
            by_period.setdefault(key, []).append((p_, cf_, runs_, act))

        sims_total = np.ones(args.mc)
        actual_total = 1.0
        for key, sleeves in by_period.items():
            actual_total *= float(np.mean([a for *_, a in sleeves]))
            mat = []
            for p_, cf_, runs_, _ in sleeves:
                n, total = len(p_), sum(runs_)
                vals = np.empty(args.mc)
                if not runs_:
                    vals[:] = float(np.prod(cf_))
                else:
                    slack = max(n - 1 - total, 0)
                    for k in range(args.mc):
                        cuts = np.sort(rng.integers(0, slack + 1, size=len(runs_)))
                        gaps = cuts - np.concatenate(([0], cuts[:-1]))
                        starts, prev = [], 0
                        for gap, ln in zip(gaps, runs_):
                            st = prev + int(gap)
                            starts.append((st, ln))
                            prev = st + ln
                        vals[k] = sleeve_value(p_, cf_, starts, cost)
                mat.append(vals)
            sims_total *= np.vstack(mat).mean(axis=0)

        pct = float((sims_total < actual_total).mean() * 100)
        print(f"  실제 전략 2000~현재 연결 누적 : {(actual_total-1)*100:+.0f}%")
        print(f"  무작위 타이밍(동일 투자비중)  : 중앙값 {(np.median(sims_total)-1)*100:+.0f}%  "
              f"| 5% {(np.percentile(sims_total,5)-1)*100:+.0f}%  "
              f"| 95% {(np.percentile(sims_total,95)-1)*100:+.0f}%")
        print(f"  → 실제 전략은 무작위 타이밍 분포의 백분위 {pct:.0f} 지점")
        print("  * 백분위 50 근처 = 진입 타이밍 자체에는 정보가 없고, 성과는 '얼마나 투자했나'에서 나왔다는 뜻")

    # ---------------- 파라미터 민감도 ----------------
    if args.sweep:
        print("\n" + "=" * 96)
        print("  파라미터 민감도 — 2000~현재 연결 누적수익률(전 기간 TOP7 동일비중)")
        print("=" * 96)
        header = "  회복비율 →" + "".join(f"{r:>12.0%}" for r in (0.0, 0.33, 0.50, 0.66, 1.0))
        print(header)
        for drop in (0.15, 0.20, 0.25, 0.30):
            cells = []
            for rec in (0.0, 0.33, 0.50, 0.66, 1.0):
                tot = 1.0
                for (a, b), picks in leaders.items():
                    sl = []
                    for code, name in picks:
                        px = prices.get(code)
                        if px is None:
                            continue
                        w = px[(px.index >= a) & (px.index <= b)]
                        if len(w) < 250:
                            continue
                        rr = run_signal(w, drop, rec, args.exit_mode,
                                        args.exit_param, args.cost_bps, code, name,
                                        daily_cash_factor(rate, w.index))
                        sl.append(rr.equity / rr.equity.iloc[0])
                    if sl:
                        idx = sl[0].index
                        for x in sl[1:]:
                            idx = idx.union(x.index)
                        pc = pd.concat([s.reindex(idx).ffill() for s in sl], axis=1).mean(axis=1)
                        tot *= pc.iloc[-1]
                cells.append(f"{(tot-1)*100:>11.0f}%")
            print(f"  하락 -{drop:.0%}  " + "".join(cells))


if __name__ == "__main__":
    main()
