"""3단계: 조건별 적중률 + 1950~1999 규칙 선정 → 2000~2026 표본외 검증"""
import os, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 50)
S = os.path.dirname(os.path.abspath(__file__))
D = f"{S}/data"

w = pd.read_csv(f"{S}/gspc_weekly.csv", index_col=0, parse_dates=True)
F = pd.read_csv(f"{S}/weekly_features.csv", index_col=0, parse_dates=True)
X = pd.read_csv(f"{S}/dc_events_features.csv", index_col=0, parse_dates=True)
bm = pd.read_csv(f"{S}/bear_markets.csv", parse_dates=["peak", "trough"])

def flags(df):
    f = pd.DataFrame(index=df.index)
    f["T: MACD<0 at cross"] = df.macd_pct < 0
    f["T: below 40w SMA"] = df.gap40 < 0
    f["T: 40w SMA falling"] = df.slope40 < 0
    f["T: monthly MACD hist<0"] = df.m_hist_neg == 1
    f["T: bearish divergence"] = df.bear_div == 1
    f["M: curve inverted in 24m"] = df.inv24 == 1
    f["M: T-bill -0.5%p from 12m max"] = df.tb3_from_max12 <= -0.5
    f["M: jobless claims YoY>0"] = df.claims_yoy > 0
    f["M: permits YoY<-10%"] = df.permit_yoy < -10
    f["M: credit spread +0.25%p 6m"] = df.credit_chg6 > 0.25
    f["M: Sahm>=0.3"] = df.sahm >= 0.3
    f["M: NFCI>0 (tight)"] = df.nfci > 0
    f["M: IP YoY<0"] = df.ip_yoy < 0
    macro = ["M: curve inverted in 24m", "M: T-bill -0.5%p from 12m max", "M: jobless claims YoY>0",
             "M: permits YoY<-10%", "M: credit spread +0.25%p 6m"]
    f["macro_count"] = f[macro].sum(axis=1)
    return f

fx = flags(X)
sev = X.fwd52_mdd <= -20
base = sev.mean()

# ── A. 조건별 적중률 ─────────────────────────────────────────
rows = []
for c in [c for c in fx.columns if c != "macro_count"]:
    m = fx[c]
    rows.append((c, int(m.sum()), m.mean() * 100, sev[m].mean() * 100, sev[~m].mean() * 100,
                 m[sev].mean() * 100, X.fwd52_ret[m].median(), X.fwd52_ret[~m].median()))
A = pd.DataFrame(rows, columns=["condition", "n_flag", "share%", "P(severe|flag)%", "P(severe|no)%", "recall%",
                                "fwd52 med|flag", "fwd52 med|no"])
print(f"=== A. Condition hit rates, dead crosses 1950~2025-09 (n={len(X)}, severe base rate {base*100:.0f}%) ===")
print(A.round(1).to_string(index=False))

# ── B. 매크로 스트레스 개수별 ────────────────────────────────
print("\n=== B. By macro stress count (of 5) and trend (below 40w SMA) ===")
g = pd.DataFrame({"macro": fx.macro_count.clip(upper=3), "below40": fx["T: below 40w SMA"], "sev": sev,
                  "mod_or_sev": X.fwd52_mdd <= -10, "ret": X.fwd52_ret})
print(g.groupby(["below40", "macro"]).agg(n=("sev", "size"), severe_pct=("sev", "mean"), dd10_pct=("mod_or_sev", "mean"),
                                          fwd52_median=("ret", "median")).assign(severe_pct=lambda d: d.severe_pct * 100,
                                          dd10_pct=lambda d: d.dd10_pct * 100).round(1).to_string())

# ── C. 하락장별 '첫 번째 심각 데드크로스' 조건 ───────────────
print("\n=== C. First severe dead cross of each bear market ===")
firsts = []
for _, b in bm.iterrows():
    s = X[sev & (X.index >= b.peak - pd.Timedelta(weeks=60)) & (X.index <= b.trough)]
    if len(s): firsts.append(s.index[0])
fc = fx.loc[firsts].astype(int)
fc.insert(0, "fwd52_mdd", X.loc[firsts, "fwd52_mdd"].round(1))
print(fc.T.to_string())

# ── D. 전략 백테스트: 1950~1999로 고르고 2000~2026에 검증 ─────
FW = flags(F.fillna({"claims_yoy": -999, "permit_yoy": 999, "nfci": -999}))
spy = pd.read_csv(f"{D}/SPY.csv", index_col=0, parse_dates=True)["Adj Close"].resample("W-FRI").last()
sh = pd.read_csv(f"{D}/data.csv", parse_dates=["Date"]).set_index("Date")
dy = (sh.Dividend / sh.SP500).replace(0, np.nan)
dy.index = dy.index + pd.offsets.MonthEnd(0)
dy = dy.reindex(dy.index.union(w.index)).ffill().reindex(w.index)
r_px = w.close.pct_change() + dy / 52
r_spy = spy.pct_change().reindex(w.index)
ret = r_px.where(w.index < pd.Timestamp("1993-02-05"), r_spy)        # 1993-02 이후는 SPY 총수익
tb3 = pd.read_csv(f"{D}/TB3MS.csv"); tb3.columns = ["d", "v"]; tb3.index = pd.to_datetime(tb3.d)
cash = (tb3.v / 100 / 52).reindex(tb3.index.union(w.index)).ffill().reindex(w.index)

hist_neg = w["hist"] < 0
cond = {
    "C0 MACD dead only": hist_neg,
    "C1 dead & below40": hist_neg & FW["T: below 40w SMA"],
    "C2 dead & curve inv24": hist_neg & FW["M: curve inverted in 24m"],
    "C3 dead & macro>=2": hist_neg & (FW.macro_count >= 2),
    "C4 dead & below40 & macro>=2": hist_neg & FW["T: below 40w SMA"] & (FW.macro_count >= 2),
    "C5 dead & monthly hist<0": hist_neg & FW["T: monthly MACD hist<0"],
    "C6 below40 only": FW["T: below 40w SMA"],
    "C7 below40 & macro>=2": FW["T: below 40w SMA"] & (FW.macro_count >= 2),
}

def positions(name, c):
    """risk-off 진입: 조건 충족 / 복귀: MACD 사용 규칙은 골든크로스(hist>0), 40주선 규칙은 40주선 회복"""
    c = c.values; h = w["hist"].values; g40 = F.gap40.values
    pos = np.ones(len(c)); off = False
    for t in range(len(c)):
        if not off and c[t]: off = True
        elif off:
            back = (g40[t] > 0) if name.startswith(("C6", "C7")) else (h[t] > 0)
            if back and not c[t]: off = False
        pos[t] = 0 if off else 1
    return pd.Series(pos, index=w.index).shift(1).fillna(1)     # 1주 지연 실행

def stats(r, p, lo, hi):
    sl = slice(lo, hi); rr = (p[sl] * r[sl] + (1 - p[sl]) * cash[sl]).dropna()
    eq = (1 + rr).cumprod(); yrs = len(rr) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, MDD=(eq / eq.cummax() - 1).min() * 100,
                in_mkt=p[sl].mean() * 100, switches=int(p[sl].diff().abs().sum()))

periods = [("1950-01-01", "1999-12-31", "IN 1950-99"), ("2000-01-01", "2026-09-11", "OUT 2000-26")]
P = {k: positions(k, v) for k, v in cond.items()}
P["B&H"] = pd.Series(1.0, index=w.index)
rows = []
for k, p in P.items():
    row = {"strategy": k}
    for lo, hi, lbl in periods:
        s = stats(ret, p, lo, hi)
        row.update({f"{lbl} CAGR": s["CAGR"], f"{lbl} MDD": s["MDD"], f"{lbl} in%": s["in_mkt"], f"{lbl} sw": s["switches"]})
    rows.append(row)
R = pd.DataFrame(rows).set_index("strategy")
print("\n=== D. Lump-sum switch strategies (total return, cash earns T-bill, 1-week lag, no tax) ===")
print(R.round(1).to_string())

# ── E. DCA (월 $1,000) 롤링 10년: 적립 보류 vs 전량 전환 ─────
px_idx = (1 + ret.fillna(0)).cumprod().values
cr = cash.fillna(0).values
month = w.index.to_period("M"); contrib = np.r_[True, month[1:] != month[:-1]]

def dca(s, e, pos, mode):
    sh = csh = 0.0
    for t in range(s, e):
        csh *= 1 + cr[t]
        if contrib[t]: csh += 1000
        on = pos[t] == 1
        if mode == "base" or on:
            sh += csh / px_idx[t]; csh = 0
        elif mode == "switch":
            csh += sh * px_idx[t]; sh = 0
    return sh * px_idx[e - 1] + csh

print("\n=== E. Rolling 10y monthly DCA vs plain DCA (median / win-rate / worst / best, %) ===")
L = 520
for lo, hi, lbl in periods:
    starts = [i for i in np.where(contrib)[0] if w.index[i] >= pd.Timestamp(lo) and i + L <= len(w)
              and w.index[i + L - 1] <= pd.Timestamp(hi) + pd.Timedelta(days=7)]
    out = []
    for k in ["C0 MACD dead only", "C1 dead & below40", "C4 dead & below40 & macro>=2", "C6 below40 only", "C7 below40 & macro>=2"]:
        pos = P[k].values
        for mode in ["pause", "switch"]:
            rel = np.array([dca(s, s + L, pos, mode) / dca(s, s + L, pos, "base") - 1 for s in starts]) * 100
            out.append((k, mode, len(starts), np.median(rel), (rel > 0).mean() * 100, rel.min(), rel.max()))
    print(f"\n[{lbl}]")
    print(pd.DataFrame(out, columns=["rule", "mode", "windows", "median", "win%", "worst", "best"]).round(1).to_string(index=False))

# ── F. 현재 상태 ─────────────────────────────────────────────
print("\n=== F. Now (2026-09-11) ===")
now = FW.loc["2026-09-11"]
print(now.to_string())
print("raw:", F.loc["2026-09-11", ["macd_pct", "gap40", "slope40", "curve", "tb3_from_max12", "claims_yoy",
                                    "permit_yoy", "credit_chg6", "sahm", "nfci", "ip_yoy", "vix"]].round(2).to_dict())
