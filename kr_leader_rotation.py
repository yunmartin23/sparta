"""
대장주 로테이션 전략 — "언제 사느냐"가 아니라 "무엇을 들고 있느냐"

kr_leader_dip_backtest.py 에서 나온 결론:
  진입 타이밍(고점 -20% 후 절반 회복)에는 정보가 없었고(몬테카를로 백분위 50),
  성과 차이의 대부분은 '어떤 종목을 들고 있었나'에서 나왔다.

이 스크립트는 그 종목선정을 실제로 돌릴 수 있는 규칙으로 만든다.
  유니버스 : 매 리밸런싱 시점 시가총액 상위 N (보통주, 상장 1년 이상)
  점수     : 거래대금 / 모멘텀 / 거래대금 급증 / 조합
  운용     : 상위 K종목 동일비중, M개월마다 교체
모든 점수는 리밸런싱 시점 '이전' 데이터만 사용한다(후행편향 없음).

기본값(2000~2026 검증 결과 채택):
  12-1개월 모멘텀 / 시총 100위 유니버스 / 상위 10종목 / 월간 리밸런싱 / KOSPI MA200 추세필터
  → CAGR 17.9%, MDD -51%, 샤프 0.77  (KOSPI 7.2% / -56% / 0.41)
  전액 투입 대신 주식자산의 ~70%만 배분하면 MDD를 -40% 수준으로 낮출 수 있다.
  주의: 알파의 상당 부분이 2000~2013에서 나왔다.
  2013-07 이후로는 CAGR 9.3% / 샤프 0.49 로 KOSPI(10.2% / 0.57)에 뒤진다.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from kr_leader_dip_backtest import (
    PERIODS, TRADING_DAYS, load_universe, load_cash_rate, load_kospi,
    daily_cash_factor, cagr, mdd, sharpe, fmt_pct,
)

MIN_HISTORY = 250          # 상장 후 최소 거래일
MATRIX_RANK = 300          # 행렬로 만들 종목 범위(시총 300위 이내 경험)


# ---------------------------------------------------------------- 행렬 구축

def build_matrices(df: pd.DataFrame) -> tuple:
    """수정주가 / 시가총액 / 거래대금 을 (날짜 × 종목) 행렬로."""
    rank = df.groupby("Date")["Marcap"].rank(ascending=False)
    keep = set(df.loc[rank <= MATRIX_RANK, "Code"].unique())
    d = df[df["Code"].isin(keep)].sort_values(["Code", "Date"]).copy()

    grp = d.groupby("Code", sort=False)
    s = d["Stocks"] / grp["Stocks"].shift(1)
    c = d["Close"] / grp["Close"].shift(1)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    c = c.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # 주식수는 크게 변했는데 시가총액은 거의 그대로 → 액면분할/병합
    split = ((s * c - 1.0).abs() < 0.15) & ((s > 1.4) | (s < 0.72))
    d["ret"] = c * np.where(split, s, 1.0) - 1.0
    d.loc[grp.cumcount() == 0, "ret"] = 0.0
    d["adj"] = grp["ret"].transform(lambda x: (1 + x).cumprod())

    px = d.pivot_table(index="Date", columns="Code", values="adj")
    mc = d.pivot_table(index="Date", columns="Code", values="Marcap")
    am = d.pivot_table(index="Date", columns="Code", values="Amount")
    names = d.groupby("Code")["Name"].last()
    return px, mc, am, names


# ---------------------------------------------------------------- 점수 산출

def score_matrix(kind: str, px: pd.DataFrame, am: pd.DataFrame,
                 look: int) -> pd.DataFrame:
    """리밸런싱 시점까지의 정보만으로 계산한 '주도주 점수'."""
    if kind == "turnover":                      # 최근 거래대금 = 시장의 관심
        return am.rolling(look, min_periods=look // 2).mean()
    if kind == "turnover_growth":               # 관심이 '늘고 있는' 종목
        recent = am.rolling(look, min_periods=look // 2).mean()
        base = am.rolling(look * 4, min_periods=look * 2).mean()
        return recent / base
    if kind == "mom":                           # 절대 모멘텀 (최근 1개월 제외)
        return px.shift(20) / px.shift(look) - 1.0
    if kind == "mom_raw":                       # 최근 1개월 포함 모멘텀
        return px / px.shift(look) - 1.0
    if kind == "combo":                         # 거래대금 순위 + 모멘텀 순위 합산
        t = am.rolling(look, min_periods=look // 2).mean().rank(axis=1, pct=True)
        mo = (px.shift(20) / px.shift(look) - 1.0).rank(axis=1, pct=True)
        return t + mo
    raise ValueError(kind)


# ---------------------------------------------------------------- 백테스트

def regime_filter(kospi: pd.Series | None, idx: pd.DatetimeIndex,
                  ma: int) -> np.ndarray:
    """KOSPI가 ma일 이동평균 위일 때만 주식 보유. 전일 종가까지의 정보만 사용."""
    if kospi is None or ma <= 0:
        return np.ones(len(idx), dtype=bool)
    ok = (kospi > kospi.rolling(ma, min_periods=ma).mean()).shift(1)
    return ok.reindex(idx.union(ok.index)).ffill().reindex(idx).fillna(True).to_numpy(bool)


def run_rotation(px, mc, am, kind: str, top_k: int, uni: int, months: int,
                 look: int, cost_bps: float, start: str, end: str,
                 names=None, verbose: bool = False,
                 kospi=None, ma: int = 0, rate=0.0, stop: float = 0.0,
                 cache: dict | None = None) -> tuple:
    """모멘텀/거래대금 상위 K종목 동일비중 로테이션.

    ma   : KOSPI 이동평균 추세필터 (0이면 미사용)
    stop : 개별 종목 트레일링 스탑 (0이면 미사용). 걸리면 다음 리밸런싱까지 현금.
    """
    dates = px.index
    idx = dates[(dates >= start) & (dates <= end)]
    if len(idx) < 60:
        return None, None

    key = (kind, look)
    if cache is not None and key in cache:
        SC = cache[key]
    else:
        SC = score_matrix(kind, px, am, look).reindex(idx).to_numpy(np.float32)
        if cache is not None:
            cache[key] = SC
    if cache is not None and "base" in cache:
        R, MR, AGE, OKPX = cache["base"]
    else:
        R = px.pct_change().reindex(idx).to_numpy(np.float32)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        MR = mc.rank(axis=1, ascending=False).reindex(idx).to_numpy(np.float32)
        AGE = px.notna().cumsum().reindex(idx).to_numpy(np.int32)
        OKPX = px.reindex(idx).notna().to_numpy()
        if cache is not None:
            cache["base"] = (R, MR, AGE, OKPX)

    month_first = pd.Series(idx).groupby(idx.to_period("M")).first().sort_values()
    rebal_days = set(pd.DatetimeIndex(month_first.to_numpy())[::months])
    is_rebal = np.array([d in rebal_days for d in idx])

    ok_regime = regime_filter(kospi, idx, ma)
    cf = daily_cash_factor(rate, idx)
    cost = cost_bps / 10000.0
    n_assets = px.shape[1]

    w = np.zeros(n_assets, dtype=np.float64)
    peak = np.zeros(n_assets)          # 보유 종목별 진입 후 고점(수익률 배수)
    mult = np.ones(n_assets)
    equity = np.empty(len(idx))
    invested = np.zeros(len(idx), dtype=bool)
    val = 1.0
    last_pick = np.array([], dtype=int)
    picks_log = []

    for i in range(len(idx)):
        if i > 0:
            if w.sum() > 0:
                gross = w * (1.0 + R[i])
                tot = gross.sum()
                if tot > 0:
                    val *= tot
                    w = gross / tot
                held = w > 0
                mult[held] *= (1.0 + R[i][held])
                peak[held] = np.maximum(peak[held], mult[held])
            else:
                val *= cf[i]

        if is_rebal[i]:
            sc_i = SC[i]
            elig = (MR[i] <= uni) & (AGE[i] >= MIN_HISTORY) & OKPX[i] & np.isfinite(sc_i)
            cand = np.flatnonzero(elig)
            if len(cand) >= max(3, top_k // 2):
                order = cand[np.argsort(-sc_i[cand])][:top_k]
                last_pick = order
                picks_log.append((idx[i], [px.columns[j] for j in order]))

        # 개별 트레일링 스탑
        if stop > 0 and w.sum() > 0:
            hit = (w > 0) & (mult < peak * (1 - stop))
            if hit.any():
                keep = w.copy()
                keep[hit] = 0.0
                val *= (1.0 - float(w[hit].sum()) * cost)
                w = keep / keep.sum() if keep.sum() > 0 else keep

        want = bool(ok_regime[i])
        new = None
        if want and len(last_pick) and (is_rebal[i] or w.sum() == 0):
            new = np.zeros(n_assets)
            new[last_pick] = 1.0 / len(last_pick)
        elif not want and w.sum() > 0:
            new = np.zeros(n_assets)

        if new is not None:
            val *= (1.0 - float(np.abs(new - w).sum()) * cost)
            entering = (new > 0) & (w == 0)
            mult[entering] = 1.0
            peak[entering] = 1.0
            w = new

        invested[i] = w.sum() > 0
        equity[i] = val

    eq = pd.Series(equity, index=idx)
    eq.attrs["exposure"] = float(invested.mean())
    if verbose and names is not None:
        for day, codes in picks_log:
            print(f"    {day.date()}  " + ", ".join(names.get(c, c) for c in codes))
    return eq, picks_log


def bh_basket(px: pd.DataFrame, codes: list, start: str, end: str) -> pd.Series | None:
    """동일비중 매수 후 보유 (기간 시작~끝)."""
    w = px.loc[(px.index >= start) & (px.index <= end), codes].dropna(how="all")
    if w.empty:
        return None
    norm = w.div(w.bfill().iloc[0])
    return norm.ffill().mean(axis=1)


def stats_row(label: str, eq: pd.Series) -> dict:
    return dict(label=label, tot=eq.iloc[-1] / eq.iloc[0] - 1, cagr=cagr(eq),
                mdd=mdd(eq), shp=sharpe(eq))


def print_table(rows: list, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)
    print(f"  {'전략':<34}{'누적':>12}{'CAGR':>10}{'MDD':>10}{'샤프':>9}")
    for r in rows:
        print(f"  {r['label']:<34}{fmt_pct(r['tot'],11)}{fmt_pct(r['cagr'],9)}"
              f"{fmt_pct(r['mdd'],9)}{r['shp']:9.2f}")


# ---------------------------------------------------------------- 실행

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--uni", type=int, default=100, help="시총 상위 N을 유니버스로")
    ap.add_argument("--months", type=int, default=1, help="리밸런싱 주기(개월)")
    ap.add_argument("--look", type=int, default=250, help="점수 산출 룩백(거래일)")
    ap.add_argument("--kind", default="mom",
                    choices=["turnover", "turnover_growth", "mom", "mom_raw", "combo"])
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--ma", type=int, default=200,
                    help="KOSPI 이동평균 추세필터 일수 (0=사용 안함, 예: 200)")
    ap.add_argument("--cash-yield", default="real")
    ap.add_argument("--stop", type=float, default=0.0,
                    help="개별 종목 트레일링 스탑 (예: 0.20)")
    ap.add_argument("--grid", action="store_true", help="주기 × 기준 격자 탐색")
    ap.add_argument("--attrib", action="store_true", help="종목선정 vs 타이밍 기여도 분해")
    ap.add_argument("--picks", action="store_true", help="리밸런싱마다 편입종목 출력")
    ap.add_argument("--live", action="store_true", help="현재 시점 편입종목·추세상태 출력")
    args = ap.parse_args()

    print("데이터 로딩 …", file=sys.stderr)
    df = load_universe()
    px, mc, am, names = build_matrices(df)
    kospi = load_kospi()
    rate = load_cash_rate(args.cash_yield)
    cache: dict = {}
    end = str(px.index[-1].date())
    print(f"  유니버스 {px.shape[1]}종목 / {px.shape[0]}거래일 "
          f"({px.index[0].date()} ~ {px.index[-1].date()})", file=sys.stderr)

    # ---------------- 기여도 분해 ----------------
    if args.attrib:
        from kr_leader_dip_backtest import leaders_by_period
        print("\n" + "=" * 92)
        print("  기여도 분해 — '어떤 종목'과 '언제 사느냐' 중 무엇이 성과를 만들었나")
        print("=" * 92)
        modes = [("사후 주도주 TOP7 (기간내 거래대금 상위)", "turnover"),
                 ("사전 주도주 TOP7 (직전1년 거래대금 상위)", "exante"),
                 ("시총 상위 TOP7", "marcap")]
        for label, mode in modes:
            L = leaders_by_period(df, args.top, mode)
            chain, parts = 1.0, []
            for (a, b), picks in L.items():
                codes = [c for c, _ in picks if c in px.columns]
                eq = bh_basket(px, codes, a, b)
                if eq is None or len(eq) < 250:
                    continue
                chain *= eq.iloc[-1]
                parts.append(f"{a[:4]}:{(eq.iloc[-1]-1)*100:+.0f}%")
            yrs = (px.index[-1] - pd.Timestamp(list(L.keys())[0][0])).days / 365.25
            print(f"  {label:<40}{fmt_pct(chain-1,11)}  CAGR {fmt_pct(chain**(1/yrs)-1,6)}")
            print(f"    {'  '.join(parts)}")
        if kospi is not None:
            kw = kospi[(kospi.index >= args.start)]
            print(f"  {'KOSPI 매수후보유':<40}{fmt_pct(kw.iloc[-1]/kw.iloc[0]-1,11)}"
                  f"  CAGR {fmt_pct(cagr(kw),6)}")
        print("\n  * 위 세 줄의 차이가 '종목선정' 효과다. 같은 종목에 타이밍 전략을 얹었을 때의")
        print("    차이(kr_leader_dip_backtest.py)는 이보다 훨씬 작았다.")

    # ---------------- 격자 탐색 ----------------
    if args.grid:
        print("\n" + "=" * 92)
        print(f"  로테이션 격자탐색 — 시총 상위 {args.uni} 중 TOP{args.top} 동일비중"
              f"{args.start[:4]}~{end[:4]} 연CAGR")
        print("=" * 92)
        kinds = ["turnover", "turnover_growth", "mom", "mom_raw", "combo"]
        print(f"  {'주기':<8}" + "".join(f"{k:>18}" for k in kinds))
        for months in (1, 3, 6, 12):
            cells = []
            for kind in kinds:
                eq, _ = run_rotation(px, mc, am, kind, args.top, args.uni, months,
                                     args.look, args.cost_bps, args.start, end,
                                     kospi=kospi, ma=args.ma, rate=rate,
                                     stop=args.stop, cache=cache)
                cells.append(f"{cagr(eq)*100:>10.1f}% ({mdd(eq)*100:>4.0f})"
                             if eq is not None else f"{'-':>18}")
            print(f"  {months:>2}개월  " + "".join(cells))
        print("  괄호 안은 MDD(%)")

    # ---------------- 본 전략 ----------------
    eq, picks = run_rotation(px, mc, am, args.kind, args.top, args.uni, args.months,
                             args.look, args.cost_bps, args.start, end,
                             names=names, verbose=args.picks,
                             kospi=kospi, ma=args.ma, rate=rate,
                             stop=args.stop, cache=cache)
    tag = f" +MA{args.ma}" if args.ma else ""
    rows = [stats_row(f"로테이션 {args.kind} {args.months}개월 TOP{args.top}{tag}", eq)]
    if args.ma:
        eq0, _ = run_rotation(px, mc, am, args.kind, args.top, args.uni, args.months,
                              args.look, args.cost_bps, args.start, end, rate=rate,
                              stop=args.stop, cache=cache)
        rows.append(stats_row(f"  └ 추세필터 없음", eq0))

    # 비교군
    mrank = mc.rank(axis=1, ascending=False)
    d0 = px.index[px.index >= args.start][0]
    big7 = mrank.loc[d0].nsmallest(args.top).index.tolist()
    b = bh_basket(px, big7, args.start, end)
    if b is not None:
        rows.append(stats_row(f"2000년 시총 TOP{args.top} 매수후보유", b))
    if kospi is not None:
        kw = kospi[(kospi.index >= args.start)]
        rows.append(stats_row("KOSPI 매수후보유", kw))

    print_table(rows, f"로테이션 전략 vs 비교군  ({args.start[:10]} ~ {end})")

    # 기간별
    print(f"\n  {'기간':<12}{'로테이션':>12}{'KOSPI':>12}{'로테MDD':>11}{'KOSPI MDD':>11}")
    for a, b_ in PERIODS:
        w = eq[(eq.index >= a) & (eq.index <= b_)]
        if len(w) < 60:
            continue
        kv = float("nan")
        kmdd = float("nan")
        if kospi is not None:
            kw = kospi[(kospi.index >= w.index[0]) & (kospi.index <= w.index[-1])]
            if len(kw) > 1:
                kv, kmdd = kw.iloc[-1] / kw.iloc[0] - 1, mdd(kw)
        print(f"  {a[:4]}~{b_[:4]}   {fmt_pct(w.iloc[-1]/w.iloc[0]-1,11)}"
              f"{fmt_pct(kv,11)}{fmt_pct(mdd(w),10)}{fmt_pct(kmdd,10)}")

    if picks:
        turn = np.mean([len(set(picks[i][1]) ^ set(picks[i - 1][1])) / 2 / args.top
                        for i in range(1, len(picks))])
        print(f"\n  리밸런싱 {len(picks)}회 | 회당 평균 교체율 {turn:.0%} "
              f"| 주식 보유일 비율 {eq.attrs['exposure']:.0%}")

    # ---------------- 현재 시점 ----------------
    if args.live:
        print("\n" + "=" * 92)
        print(f"  현재 시점 실행안  (기준일 {end})")
        print("=" * 92)
        if kospi is not None and args.ma:
            k, m = float(kospi.iloc[-1]), float(kospi.rolling(args.ma).mean().iloc[-1])
            state = "주식 보유" if k > m else "전량 현금"
            print(f"  추세필터: KOSPI {k:,.0f} vs MA{args.ma} {m:,.0f} "
                  f"({k/m-1:+.1%})  →  \033[1m{state}\033[0m")
        day = px.index[-1]
        sc = score_matrix(args.kind, px, am, args.look)
        mrank = mc.rank(axis=1, ascending=False)
        age = px.notna().cumsum()
        elig = (mrank.loc[day] <= args.uni) & (age.loc[day] >= MIN_HISTORY) \
               & px.loc[day].notna() & sc.loc[day].notna()
        cand = sc.loc[day][elig].sort_values(ascending=False).head(args.top)
        mom12 = (px.loc[day] / px.shift(args.look).loc[day] - 1.0)
        print(f"\n  다음 리밸런싱 편입 후보 (동일비중 {100/args.top:.1f}%씩)")
        print(f"  {'#':<4}{'종목':<18}{'시총순위':>9}{'12개월수익률':>14}{'점수':>9}")
        for n_, (code, sv) in enumerate(cand.items(), 1):
            print(f"  {n_:<4}{names.get(code, code)[:16]:<18}"
                  f"{int(mrank.loc[day, code]):>9}{mom12.get(code, np.nan)*100:>13.1f}%{sv:>9.2f}")


if __name__ == "__main__":
    main()
