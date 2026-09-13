"""1단계: S&P500 주봉 MACD 데드크로스 전수 추출 + 하락장(-20%) 매핑"""
import os, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
S = os.path.dirname(os.path.abspath(__file__))
D = f"{S}/data"

px = pd.read_csv(f"{D}/GSPC.csv", index_col=0, parse_dates=True)["Close"]
wk = px.resample("W-FRI").last().dropna().to_frame("close")
macd = wk.close.ewm(span=12, adjust=False).mean() - wk.close.ewm(span=26, adjust=False).mean()
wk["macd"] = macd
wk["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
wk["dead"] = (wk["hist"].shift(1) >= 0) & (wk["hist"] < 0)
wk["golden"] = (wk["hist"].shift(1) <= 0) & (wk["hist"] > 0)

# ── 하락장 식별: 종가 기준 고점 대비 -20% 이상 하락 구간 (일간) ──
def bear_markets(s, thr=-0.20):
    out, peak_i, peak_v, in_bear, trough_i = [], s.index[0], s.iloc[0], False, None
    for d, v in s.items():
        if not in_bear:
            if v > peak_v: peak_i, peak_v = d, v
            elif v / peak_v - 1 <= thr: in_bear, trough_i, trough_v = True, d, v
        else:
            if v < trough_v: trough_i, trough_v = d, v
            if v >= peak_v:   # 전고점 회복 → 하락장 종료
                out.append((peak_i, peak_v, trough_i, trough_v)); in_bear = False; peak_i, peak_v = d, v
    if in_bear: out.append((peak_i, peak_v, trough_i, trough_v))
    return pd.DataFrame(out, columns=["peak", "peak_v", "trough", "trough_v"])

daily = px.loc["1949-01-01":]
bm = bear_markets(daily)
bm["decline%"] = (bm.trough_v / bm.peak_v - 1) * 100
print("=== S&P500 bear markets (close, -20%) since 1949 ===")
print(bm.round(1).to_string())

# ── 각 하락장 고점 전후의 데드크로스 타임라인 ──
wk50 = wk.loc["1950-01-01":]
print("\n=== Dead crosses from peak-52w to trough, per bear market ===")
for _, b in bm.iterrows():
    lo, hi = b.peak - pd.Timedelta(weeks=52), b.trough
    dc = wk50[(wk50.index >= lo) & (wk50.index <= hi) & wk50.dead]
    print(f"\n[{b.peak.date()} peak {b.peak_v:,.0f} → {b.trough.date()} trough, {b['decline%']:.1f}%]")
    for d, r in dc.iterrows():
        g = wk50.index[(wk50.index > d) & wk50.golden]
        g = g[0] if len(g) else None
        at_cross = (r.close / b.peak_v - 1) * 100
        after = (b.trough_v / r.close - 1) * 100 if d <= b.trough else np.nan
        new_high = wk50.close[(wk50.index > d) & (wk50.index <= d + pd.Timedelta(weeks=104))].max() > wk50.close[:d].max()
        print(f"  DC {d.date()}  close {r.close:>8,.1f}  vs peak {at_cross:+6.1f}%  remaining fall to trough {after:+6.1f}%  "
              f"macd>0={r.macd > 0}  next golden {g.date() if g is not None else '-'}")

# ── 전체 데드크로스 이벤트 테이블 + 선행 52주 최대낙폭 라벨 ──
c = wk.close.values
idx = np.where(wk.dead.values)[0]
rows = []
for i in idx:
    d = wk.index[i]
    if d < pd.Timestamp("1950-01-01") or i + 52 >= len(c): continue
    fwd = c[i + 1:i + 53]
    rows.append((d, c[i], (fwd.min() / c[i] - 1) * 100, (c[i + 52] / c[i] - 1) * 100, (c[i + 26] / c[i] - 1) * 100))
ev = pd.DataFrame(rows, columns=["date", "close", "fwd52_mdd", "fwd52_ret", "fwd26_ret"]).set_index("date")
ev["label"] = pd.cut(ev.fwd52_mdd, [-100, -20, -10, 100], labels=["severe(<=-20%)", "moderate", "benign(>-10%)"])
print("\n=== All dead crosses 1950~2025-09 (n=%d) by forward-52w max drawdown ===" % len(ev))
print(ev.label.value_counts().to_string())
print("\nSevere events:")
print(ev[ev.fwd52_mdd <= -20].round(1).to_string())
ev.to_csv(f"{S}/dc_events.csv"); wk.to_csv(f"{S}/gspc_weekly.csv"); bm.to_csv(f"{S}/bear_markets.csv", index=False)
