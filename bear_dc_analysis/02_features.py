"""2단계: 각 데드크로스 시점의 기술적·매크로 특징 (발표 지연 반영, 미래 정보 배제)"""
import os, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 50)
S = os.path.dirname(os.path.abspath(__file__))
D = f"{S}/data"

px = pd.read_csv(f"{D}/GSPC.csv", index_col=0, parse_dates=True)["Close"]
w = pd.read_csv(f"{S}/gspc_weekly.csv", index_col=0, parse_dates=True)
w["dead"] = w["dead"].astype(bool); w["golden"] = w["golden"].astype(bool)

def asof(s, idx):
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)

def fred(name):
    d = pd.read_csv(f"{D}/{name}.csv"); d.columns = ["date", name]
    d["date"] = pd.to_datetime(d["date"]); d[name] = pd.to_numeric(d[name], errors="coerce")
    return d.set_index("date")[name].dropna()

def monthly_avail(s, days):   # 월간 지표: 해당 월 종료 후 days일 뒤 사용 가능
    s = s.copy(); s.index = s.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=days); return s

def shift_days(s, days):
    s = s.copy(); s.index = s.index + pd.Timedelta(days=days); return s

idx = w.index
F = pd.DataFrame(index=idx)

# ── 기술적 지표 ──────────────────────────────────────────────
F["macd_pct"] = w.macd / w.close * 100
dl = w.close.diff()
g = dl.clip(lower=0).ewm(alpha=1/9, adjust=False).mean(); l = (-dl.clip(upper=0)).ewm(alpha=1/9, adjust=False).mean()
F["w_rsi"] = 100 - 100 / (1 + g / l)
F["dd52"] = (w.close / w.close.rolling(52).max() - 1) * 100
sma40 = w.close.rolling(40).mean()
F["gap40"] = (w.close / sma40 - 1) * 100                     # 40주(≈200일) 이동평균 대비
F["slope40"] = (sma40 / sma40.shift(13) - 1) * 100           # 40주선 13주 기울기
F["ret13"] = (w.close / w.close.shift(13) - 1) * 100
m26 = F.macd_pct.rolling(26).max(); m_prior = F.macd_pct.shift(26).rolling(78).max()
p26 = w.close.rolling(26).max(); p_prior = w.close.shift(26).rolling(78).max()
F["bear_div"] = ((p26 >= p_prior) & (m26 < m_prior)).astype(float)   # 가격은 신고가, MACD 고점은 낮아짐
F["dc52"] = w.dead.astype(int).rolling(52).sum().shift(1)            # 직전 52주 데드크로스 횟수
mc = px.groupby(px.index.to_period("M")).last()
mmacd = mc.ewm(span=12, adjust=False).mean() - mc.ewm(span=26, adjust=False).mean()
mhist = mmacd - mmacd.ewm(span=9, adjust=False).mean()
mhist.index = mhist.index.to_timestamp(how="start")
F["m_hist_neg"] = (asof(monthly_avail(mhist, 0), idx) < 0).astype(float)   # 직전 완성 월봉 MACD 히스토그램
lr = np.log(px).diff()
F["vol20"] = asof(lr.rolling(20).std() * np.sqrt(252) * 100, idx)
vix = pd.read_csv(f"{D}/VIX.csv", index_col=0, parse_dates=True)["Close"]
F["vix"] = asof(vix, idx)
rut = pd.read_csv(f"{D}/RUT.csv", index_col=0, parse_dates=True)["Close"]
rel = asof(rut, idx) / w.close
F["smallcap_rel26"] = (rel / rel.shift(26) - 1) * 100      # 소형주 상대강도 (1987~)

# ── 매크로 (발표 지연 반영) ───────────────────────────────────
gs10, tb3, baa, un = fred("GS10"), fred("TB3MS"), fred("BAA"), fred("UNRATE")
curve = gs10 - tb3
F["curve"] = asof(monthly_avail(curve, 1), idx)
F["inv24"] = (asof(monthly_avail(curve.rolling(24).min(), 1), idx) < 0).astype(float)     # 24개월 내 역전 이력
F["tb3_from_max12"] = asof(monthly_avail(tb3 - tb3.rolling(12).max(), 1), idx)           # 단기금리 고점 대비 (인하 시작)
F["tb3_chg12"] = asof(monthly_avail(tb3.diff(12), 1), idx)
credit = baa - gs10
F["credit"] = asof(monthly_avail(credit, 1), idx)
F["credit_chg6"] = asof(monthly_avail(credit.diff(6), 1), idx)
u3 = un.rolling(3).mean()
F["sahm"] = asof(monthly_avail(u3 - u3.shift(1).rolling(12).min(), 7), idx)
icsa = fred("ICSA")
F["claims_yoy"] = asof(shift_days(icsa.rolling(4).mean().pct_change(52) * 100, 5), idx)
F["nfci"] = asof(shift_days(fred("NFCI"), 5), idx)
F["nfci_chg13"] = asof(shift_days(fred("NFCI").diff(13), 5), idx)
F["ip_yoy"] = asof(monthly_avail(fred("INDPRO").pct_change(12) * 100, 17), idx)
F["permit_yoy"] = asof(monthly_avail(fred("PERMIT").pct_change(12) * 100, 18), idx)
F["cpi_yoy"] = asof(monthly_avail(fred("CPIAUCSL").pct_change(12) * 100, 14), idx)
sh = pd.read_csv(f"{D}/data.csv", parse_dates=["Date"]).set_index("Date")
eps = sh.Earnings.replace(0, np.nan); cape = sh.PE10.replace(0, np.nan)
e = eps.pct_change(12) * 100; e.index = e.index + pd.DateOffset(months=4)   # 실적 발표 지연 ~1분기
F["eps_yoy"] = asof(e, idx)
F["cape"] = asof(monthly_avail(cape, 1), idx)
F.to_csv(f"{S}/weekly_features.csv")

# ── 데드크로스 이벤트와 결합 ─────────────────────────────────
ev = pd.read_csv(f"{S}/dc_events.csv", index_col=0, parse_dates=True)
X = ev.join(F)
X.to_csv(f"{S}/dc_events_features.csv")

cols = ["macd_pct", "w_rsi", "dd52", "gap40", "slope40", "ret13", "bear_div", "dc52", "m_hist_neg", "vol20", "vix",
        "smallcap_rel26", "curve", "inv24", "tb3_from_max12", "tb3_chg12", "credit", "credit_chg6", "sahm",
        "claims_yoy", "nfci", "nfci_chg13", "ip_yoy", "permit_yoy", "cpi_yoy", "eps_yoy", "cape"]
print("=== Median (flags: mean) by outcome, dead crosses 1950~2025-09 ===")
agg = X.groupby("label")[cols].median().T
flag_cols = ["bear_div", "m_hist_neg", "inv24"]
agg.loc[flag_cols] = X.groupby("label")[flag_cols].mean().T
agg["n_available(severe/all)"] = [f"{X.loc[X.label.str.startswith('severe'), c].notna().sum()}/{X[c].notna().sum()}" for c in cols]
print(agg.round(2).to_string())

cases = ["2000-04-14", "2000-07-28", "2000-09-22", "2007-06-29", "2007-07-27", "2007-11-09", "2008-06-27"]
show = ["macd_pct", "dd52", "gap40", "slope40", "bear_div", "m_hist_neg", "curve", "inv24", "tb3_from_max12",
        "credit_chg6", "sahm", "claims_yoy", "nfci", "permit_yoy", "ip_yoy", "eps_yoy", "cape", "vix"]
print("\n=== 2000 / 2007-08 dead crosses vs now ===")
tbl = pd.concat([X.loc[pd.to_datetime(cases), ["fwd52_mdd"] + show],
                 F.loc[[pd.Timestamp("2026-09-11")], show].assign(fwd52_mdd=np.nan)[["fwd52_mdd"] + show]])
tbl.loc["benign median"] = X[X.label.str.startswith("benign")][["fwd52_mdd"] + show].median()
tbl.loc["severe median"] = X[X.label.str.startswith("severe")][["fwd52_mdd"] + show].median()
print(tbl.round(2).T.to_string())
