#!/usr/bin/env python3
"""
대장주 로테이션 — 현재 시점 실행안만 뽑는 단독 스크립트

kr_leader_rotation.py 의 백테스트는 2000년부터 490MB를 받지만,
"오늘 뭘 들고 있어야 하나"에는 최근 몇 년치만 있으면 된다.
이 스크립트는 필요한 연도만 받아서(기본 3년, 약 75MB) 아래 두 가지만 출력한다.

  1) 추세필터 상태  — KOSPI가 MA200 위인가 (위면 주식 보유, 아래면 전량 현금)
  2) 편입 종목      — 시총 상위 100위 안에서 12-1개월 모멘텀 상위 10종목 동일비중

의존성 없이 단독 실행된다(백테스터를 import 하지 않는다). 대신 전략 파라미터
기본값은 kr_leader_rotation.py 와 같아야 하므로, 한쪽을 바꾸면 다른 쪽도 맞출 것.

  로컬   : python kr_leader_now.py
  Colab  : !pip -q install FinanceDataReader pyarrow
           !curl -sO https://raw.githubusercontent.com/yunmartin23/sparta/main/kr_leader_now.py
           !python kr_leader_now.py

데이터: FinanceData/marcap (KRX 전종목 일별 시가총액·주가)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".marcap_cache")
RAW = "https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-{}.parquet"
EXCLUDE_NAME = re.compile(r"우[BC]?\)?$|우선|스팩|SPAC|리츠$")
MIN_HISTORY = 250          # 상장 후 최소 거래일
BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"


# ---------------------------------------------------------------- 데이터

def fetch_years(years: list[int]) -> list[str]:
    os.makedirs(CACHE, exist_ok=True)

    def one(year: int) -> tuple[int, str]:
        path = os.path.join(CACHE, f"marcap-{year}.parquet")
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return year, path
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(RAW.format(year), path)
                return year, path
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        return year, path

    t0 = time.time()
    with cf.ThreadPoolExecutor(min(6, len(years))) as ex:
        paths = [p for _, p in sorted(ex.map(one, years))]
    mb = sum(os.path.getsize(p) for p in paths) / 1e6
    print(f"  데이터 {len(years)}개 연도 · {mb:.0f}MB · {time.time() - t0:.0f}초",
          file=sys.stderr)
    return paths


def load_frame(years: list[int]) -> pd.DataFrame:
    cols = ["Code", "Name", "Close", "Marcap", "Stocks", "Market", "Date"]
    frames = []
    for path in fetch_years(years):
        d = pd.read_parquet(path, columns=cols)
        d = d[d["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
        frames.append(d[~d["Name"].str.contains(EXCLUDE_NAME, na=False, regex=True)])
    df = pd.concat(frames, ignore_index=True)
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df.sort_values(["Code", "Date"])


def build_matrices(df: pd.DataFrame, uni: int) -> tuple:
    """수정주가·시가총액 행렬. 시총 상위 (uni*3)위 안에 든 적 있는 종목만."""
    rank = df.groupby("Date")["Marcap"].rank(ascending=False)
    keep = set(df.loc[rank <= uni * 3, "Code"].unique())
    d = df[df["Code"].isin(keep)].copy()

    grp = d.groupby("Code", sort=False)
    s = (d["Stocks"] / grp["Stocks"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    c = (d["Close"] / grp["Close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # 주식수는 크게 변했는데 시가총액은 그대로 → 액면분할/병합
    split = ((s * c - 1.0).abs() < 0.15) & ((s > 1.4) | (s < 0.72))
    d["ret"] = c * np.where(split, s, 1.0) - 1.0
    d.loc[grp.cumcount() == 0, "ret"] = 0.0
    d["adj"] = grp["ret"].transform(lambda x: (1 + x).cumprod())

    return (d.pivot_table(index="Date", columns="Code", values="adj"),
            d.pivot_table(index="Date", columns="Code", values="Marcap"),
            d.groupby("Code")["Name"].last(),
            d.pivot_table(index="Date", columns="Code", values="Close"))


def load_kospi():
    try:
        import FinanceDataReader as fdr
    except ImportError:
        print("  (FinanceDataReader 미설치 → 추세필터 생략. pip install FinanceDataReader)",
              file=sys.stderr)
        return None
    try:
        k = fdr.DataReader("KS11", "2023-01-01")["Close"]
        return pd.Series(k.to_numpy(dtype=float), index=pd.to_datetime(k.index)).dropna()
    except Exception as exc:
        print(f"  (KOSPI 조회 실패 → 추세필터 생략: {exc})", file=sys.stderr)
        return None


# ---------------------------------------------------------------- 전략

def score_at(px: pd.DataFrame, day, kind: str, look: int) -> pd.Series:
    """리밸런싱 점수. 해당일까지의 정보만 사용."""
    i = px.index.get_loc(day)
    if kind == "mom":            # 12-1개월 모멘텀 (최근 1개월 제외)
        if i - look < 0:
            raise SystemExit(f"데이터 부족: 모멘텀 {look}일을 계산하려면 --years 를 늘리세요")
        return px.iloc[i - 20] / px.iloc[i - look] - 1.0
    if kind == "mom_raw":
        return px.iloc[i] / px.iloc[i - look] - 1.0
    raise SystemExit(f"이 스크립트는 mom / mom_raw 만 지원합니다 (받은 값: {kind})")


def picks_at(px, mc, day, kind, look, uni, top_k) -> pd.Series:
    sc = score_at(px, day, kind, look)
    mrank = mc.loc[day].rank(ascending=False)
    age = px.loc[:day].notna().sum()
    elig = (mrank <= uni) & (age >= MIN_HISTORY) & px.loc[day].notna() & sc.notna()
    return sc[elig].sort_values(ascending=False).head(top_k)


def month_first_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index).groupby(index.to_period("M")).first().sort_values()
    return pd.DatetimeIndex(s.to_numpy())


# ---------------------------------------------------------------- 출력

def main() -> None:
    ap = argparse.ArgumentParser(description="대장주 로테이션 현재 시점 실행안")
    ap.add_argument("--top", type=int, default=10, help="편입 종목 수")
    ap.add_argument("--uni", type=int, default=100, help="시총 상위 N을 유니버스로")
    ap.add_argument("--look", type=int, default=250, help="모멘텀 룩백(거래일)")
    ap.add_argument("--kind", default="mom", choices=["mom", "mom_raw"])
    ap.add_argument("--ma", type=int, default=200, help="KOSPI 추세필터 이동평균 (0=미사용)")
    ap.add_argument("--years", type=int, default=3, help="내려받을 최근 연도 수")
    ap.add_argument("--json", metavar="경로", default=None, help="결과를 JSON으로 저장")
    args = ap.parse_args()

    this_year = date.today().year
    years = list(range(this_year - args.years + 1, this_year + 1))
    df = load_frame(years)
    px, mc, names, raw = build_matrices(df, args.uni)
    kospi = load_kospi()

    day = px.index[-1]
    mfd = month_first_trading_days(px.index)
    cur_rebal = mfd[mfd <= day][-1]          # 이번 달 리밸런싱일 = 현재 보유 기준일
    nxt_month = (day.to_period("M") + 1).to_timestamp()

    w = 100.0 / args.top
    head = (f"대장주 로테이션 — 현재 시점 실행안")
    print()
    print("═" * 72)
    print(f"  {BOLD}{head}{OFF}")
    print(f"  {DIM}데이터 최종 {day.date()} · {args.kind} {args.look}일 · "
          f"시총 {args.uni}위 · TOP{args.top} 동일비중 · MA{args.ma}{OFF}")
    print("═" * 72)

    # ---- 1. 추세필터 ----
    regime_on = True
    out = {"as_of": str(day.date()), "params": vars(args)}
    print(f"\n{BOLD}▸ 추세필터{OFF}")
    if kospi is not None and args.ma:
        k_now = float(kospi.iloc[-1])
        k_ma = float(kospi.rolling(args.ma).mean().iloc[-1])
        regime_on = k_now > k_ma
        gap = k_now / k_ma - 1
        print(f"    KOSPI  {k_now:>9,.0f}   ({kospi.index[-1].date()})")
        print(f"    MA{args.ma:<3}  {k_ma:>9,.0f}   {gap:+.1%}")
        verdict = "주식 보유" if regime_on else "전량 현금"
        print(f"\n    → {BOLD}{verdict}{OFF}")
        if not regime_on:
            print(f"      {DIM}KOSPI가 {k_ma:,.0f} 위로 올라오면 편입 재개{OFF}")
        out["regime"] = {"kospi": k_now, "ma": k_ma, "on": regime_on}
    else:
        print("    (미사용 — 항상 주식 보유로 간주)")

    # ---- 2. 이번 달 보유 종목 ----
    cur = picks_at(px, mc, cur_rebal, args.kind, args.look, args.uni, args.top)
    mrank = mc.loc[cur_rebal].rank(ascending=False)     # 편입 판정에 쓰인 순위
    ret12 = px.loc[day] / px.iloc[px.index.get_loc(day) - args.look] - 1.0

    label = "이번 달 보유 종목" if regime_on else "이번 달 보유 종목 (추세필터 OFF — 참고용)"
    print(f"\n{BOLD}▸ {label}{OFF}   {DIM}{cur_rebal.date()} 리밸런싱 기준 · 각 {w:.1f}%{OFF}")
    print(f"    {'#':<4}{'종목':<18}{'현재가':>11}{'시총(편입시)':>13}{'12개월':>11}")
    print(f"    {'─' * 53}")
    rows = []
    for n, (code, _) in enumerate(cur.items(), 1):
        r12 = float(ret12.get(code, np.nan))
        rk = mrank.get(code, np.nan)
        price = float(raw.loc[day, code]) if code in raw.columns else float("nan")
        print(f"    {n:<4}{names.get(code, code)[:16]:<18}{price:>11,.0f}"
              f"{'-' if np.isnan(rk) else int(rk):>13}{r12 * 100:>10.1f}%")
        rows.append(dict(rank=n, code=code, name=names.get(code, code),
                         price=price, marcap_rank=None if np.isnan(rk) else int(rk),
                         ret_12m=r12, weight_pct=w))
    out["holdings"] = rows

    # ---- 3. 다음 리밸런싱 미리보기 ----
    nxt = picks_at(px, mc, day, args.kind, args.look, args.uni, args.top)
    add = [c for c in nxt.index if c not in cur.index]
    drop = [c for c in cur.index if c not in nxt.index]
    print(f"\n{BOLD}▸ 지금 리밸런싱한다면{OFF}   {DIM}다음 예정일 {nxt_month.date()} 전후{OFF}")
    print(f"    유지 {args.top - len(add)} · 신규 {len(add)} · 제외 {len(drop)}")
    if add:
        print(f"    {DIM}신규{OFF}  " + ", ".join(names.get(c, c) for c in add))
    if drop:
        print(f"    {DIM}제외{OFF}  " + ", ".join(names.get(c, c) for c in drop))
    out["next_rebalance"] = {
        "date_hint": str(nxt_month.date()),
        "add": [names.get(c, c) for c in add],
        "drop": [names.get(c, c) for c in drop],
        "picks": [names.get(c, c) for c in nxt.index],
    }

    print(f"\n{DIM}  참고: 2000~2026 백테스트 CAGR 17.9% / MDD -51% / 샤프 0.77 "
          f"(KOSPI 7.2% / -56% / 0.41)\n"
          f"  단 알파는 2000~2013에 몰려 있고 2013-07 이후로는 KOSPI에 뒤진다.\n"
          f"  MDD를 낮추려면 전액이 아니라 주식자산의 ~70%만 배분할 것.{OFF}")
    print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"  JSON 저장: {args.json}\n")


if __name__ == "__main__":
    main()
