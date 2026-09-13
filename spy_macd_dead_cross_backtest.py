import logging, warnings
import numpy as np, pandas as pd, yfinance as yf
warnings.simplefilter("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
pd.set_option("display.width", 200)

h = yf.Ticker("SPY").history(start="1993-01-01", auto_adjust=False)
h.index = h.index.tz_localize(None)
irx = yf.Ticker("^IRX").history(start="1993-01-01")["Close"]
irx.index = irx.index.tz_localize(None)

wk = h.resample("W-FRI").agg({"Close": "last", "Adj Close": "last"}).dropna()
wk["irx"] = irx.resample("W-FRI").last().reindex(wk.index).ffill().fillna(0)
macd = wk.Close.ewm(span=12, adjust=False).mean() - wk.Close.ewm(span=26, adjust=False).mean()
sig = macd.ewm(span=9, adjust=False).mean()
wk["macd"], wk["hist"] = macd, macd - sig
wk["dd52"] = wk.Close / wk.Close.rolling(52).max() - 1
wk["dead"] = (wk["hist"].shift(1) >= 0) & (wk["hist"] < 0)
wk["golden"] = (wk["hist"].shift(1) <= 0) & (wk["hist"] > 0)
print("last weeks:\n", wk[["Close", "macd", "hist", "dd52", "dead"]].tail(3).round(3))

wk = wk.loc["1994-01-01":]
adj = wk["Adj Close"]

# ── A. 데드크로스 이후 선행 수익률 (배당 포함) ────────────────
print("\n=== A. Forward total return after weekly MACD dead cross (1994~) ===")
rows = []
groups = {
    "all weeks": pd.Series(True, index=wk.index),
    "dead cross": wk.dead,
    "dead & MACD>0": wk.dead & (wk.macd > 0),
    "dead & MACD>0 & <5% off high": wk.dead & (wk.macd > 0) & (wk.dd52 > -0.05),
}
for n in (4, 13, 26, 52):
    fwd = adj.shift(-n) / adj - 1
    for g, mask in groups.items():
        f = fwd[mask].dropna()
        rows.append((f"{n}w", g, len(f), f.mean() * 100, f.median() * 100, (f > 0).mean() * 100))
print(pd.DataFrame(rows, columns=["horizon", "group", "n", "mean%", "median%", "win%"]).round(1).to_string(index=False))

# ── B. 데드크로스 → 다음 골든크로스까지: 기다리면 더 싸게 샀나? ──
print("\n=== B. Dead cross -> next golden cross episodes ===")
ep = []
dead_idx = list(wk.index[wk.dead]); gold_idx = list(wk.index[wk.golden])
for d in dead_idx:
    g = next((x for x in gold_idx if x > d), None)
    if g is None: continue
    ep.append((d, g, (wk.index.get_loc(g) - wk.index.get_loc(d)), adj[g] / adj[d] - 1, wk.macd[d] > 0))
ep = pd.DataFrame(ep, columns=["dead", "golden", "weeks", "ret", "macd_pos"])
for lbl, e in [("all", ep), ("MACD>0 at cross", ep[ep.macd_pos])]:
    print(f"{lbl}: n={len(e)}, median weeks={e.weeks.median():.0f}, "
          f"cheaper at golden (ret<0)={(e.ret < 0).mean()*100:.0f}%, median ret={e.ret.median()*100:.1f}%, "
          f"mean ret={e.ret.mean()*100:.1f}%, worst={e.ret.min()*100:.1f}%, best={e.ret.max()*100:.1f}%, "
          f"<=4w whipsaw={(e.weeks <= 4).mean()*100:.0f}%")

# ── C. DCA 전략 비교 (월 $1,000, 신호는 1주 지연 적용, 현금은 T-bill 이자) ──
px = adj.values
state_dead = (wk["hist"].shift(1) < 0).values          # 지난주 확정 신호 기준
cash_r = (wk.irx.values / 100) / 52
month = wk.index.to_period("M")
contrib = np.r_[True, month[1:] != month[:-1]]      # 매월 첫 주

def sim(s, e, strat):
    sh = cash = 0.0
    for t in range(s, e):
        cash *= 1 + cash_r[t]
        if contrib[t]: cash += 1000
        dead = state_dead[t]
        if strat == "base":
            sh += cash / px[t]; cash = 0
        elif strat == "pause_on_dead":        # 데드 구간엔 적립 보류, 골든 전환 시 몰아서 매수
            if not dead: sh += cash / px[t]; cash = 0
        elif strat == "buy_only_on_dead":     # 반대로 데드 구간에만 매수
            if dead: sh += cash / px[t]; cash = 0
        elif strat == "full_switch":          # 데드면 전량 현금화, 골든이면 전량 매수
            if dead: cash += sh * px[t]; sh = 0
            else: sh += cash / px[t]; cash = 0
    return sh * px[e - 1] + cash

strats = ["base", "pause_on_dead", "buy_only_on_dead", "full_switch"]
starts = np.where(contrib)[0]
for years in (3, 10):
    L = 52 * years
    res = {k: [] for k in strats}
    for s in starts:
        if s + L > len(px): break
        for k in strats: res[k].append(sim(s, s + L, k))
    res = pd.DataFrame(res)
    rel = res.div(res["base"], axis=0) - 1
    print(f"\n=== C. Rolling {years}y DCA windows (n={len(res)}), final value vs base ===")
    print(pd.DataFrame({
        "median_vs_base%": rel.median() * 100,
        "mean_vs_base%": rel.mean() * 100,
        "win_rate%": (rel > 0).mean() * 100,
        "worst_vs_base%": rel.min() * 100,
        "best_vs_base%": rel.max() * 100,
    }).round(2).drop("base").to_string())
