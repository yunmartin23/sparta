import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
import re
import io
import os
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
warnings.simplefilter(action='ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # 없는 티커 조회 시 404 로그 숨김

# ============================================================
# 파라미터
# ============================================================
SLOPE_LOOKBACK = 5
DAILY_MONTHS   = 18   # 일봉·주봉 지표 계산 기간
MONTHLY_YEARS  = 3    # 월봉 MACD 계산 기간
FRED_API_KEY   = "a74da742f0b44f8db5825eeab97328a1"   # FRED API 키 (32자리)를 따옴표 안에 입력. https://fred.stlouisfed.org/docs/api/api_key.html

# 한국 종목코드: 숫자로 시작하는 6자리 (005930, 0089D0 같은 영문 포함 신규 코드 포함)
KR_CODE_RE   = re.compile(r'^\d[0-9A-Z]{5}$')
# 미국 티커: 알파벳으로 시작 (AAPL), 클래스주는 BRK-B, 지수는 ^GSPC 형태
US_TICKER_RE = re.compile(r'^\^?[A-Z][A-Z0-9]{0,5}(-[A-Z]{1,2})?$')

# ============================================================
# 핵심 계산 함수
# ============================================================
def calc_price_slope_series(df, lookback=SLOPE_LOOKBACK):
    return (df["Close"] / df["Close"].shift(lookback) - 1) * 100

def calc_super_rsi(df, rsi_col="RSI", span=7):
    return df[rsi_col].ewm(span=span, adjust=False).mean()

# ============================================================
# 시장 판별 / 종목 조회
# ============================================================
def has_hangul(s):
    return any('가' <= c <= '힣' for c in s)

def detect_market(s):
    """한글 종목명이거나 숫자로 시작하는 6자리 코드면 KR, 그 외(영문 티커/영문 종목명)는 US"""
    s = s.strip().upper()
    if has_hangul(s) or KR_CODE_RE.match(s):
        return "KR"
    return "US"

def fmt_price(v, market):
    return f"${v:,.2f}" if market == "US" else f"{v:,.0f}원"

def price_hover(market):
    return "$%{y:,.2f}" if market == "US" else "%{y:,.0f}원"

def kr_name_to_ticker(name):
    from pykrx import stock
    name = name.strip()
    tickers = []
    for market in ["KOSPI", "KOSDAQ", "KONEX"]:
        tickers.extend(stock.get_market_ticker_list(market=market))
    for t in tickers:
        if stock.get_market_ticker_name(t) == name: return t
    for t in tickers:
        if name in stock.get_market_ticker_name(t): return t
    raise ValueError(f"종목명을 찾을 수 없습니다: {name}")

def kr_ticker_name(ticker, fallback):
    try:
        from pykrx import stock
        name = stock.get_market_ticker_name(ticker)
        if isinstance(name, str) and name:   # 미존재 시 pykrx는 빈 DataFrame 반환
            return name
    except Exception:
        pass
    return fallback

def us_name_to_ticker(name):
    """영문 종목명 → 미국 티커 (yfinance 검색, 해외거래소 상장분 제외)"""
    try:
        import yfinance as yf
    except ImportError:
        raise ValueError("미국 종목명 검색에는 yfinance가 필요합니다 (pip install yfinance). 티커(예: AAPL)로 입력해 주세요.")
    for q in yf.Search(name, max_results=10, news_count=0).quotes:
        sym = str(q.get("symbol", "")).upper()
        if q.get("quoteType") in ("EQUITY", "ETF") and US_TICKER_RE.match(sym):
            return sym
    raise ValueError(f"미국 종목명을 찾을 수 없습니다: {name}")

def us_ticker_name(ticker):
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker

def resolve_ticker(query, market=None):
    """입력값(코드/티커/종목명) → (ticker, name, market)"""
    q = query.strip()
    market = (market or detect_market(q)).upper()

    if market == "KR":
        ticker = q.upper() if KR_CODE_RE.match(q.upper()) else kr_name_to_ticker(q)
        return ticker, kr_ticker_name(ticker, q), "KR"

    sym = q.upper().replace(".", "-")   # BRK.B → BRK-B (Yahoo 표기)
    ticker = sym if US_TICKER_RE.match(sym) else us_name_to_ticker(q)
    return ticker, us_ticker_name(ticker), "US"

# ============================================================
# 데이터 로드
# ============================================================
def _fetch_pykrx(ticker, start, end):
    from pykrx import stock
    df = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    return df.rename(columns={
        '시가': 'Open', '고가': 'High',
        '저가': 'Low',  '종가': 'Close',
        '거래량': 'Volume'
    })

def _fetch_yfinance(ticker, start, end):
    import yfinance as yf
    return yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )

def fetch_ohlcv(ticker, market, start, end):
    """fdr 우선 → 실패 시 한국은 pykrx, 미국은 yfinance로 폴백"""
    fallback = _fetch_pykrx if market == "KR" else _fetch_yfinance
    df = None
    for fetch in (fdr.DataReader, fallback):
        try:
            df = fetch(ticker, start, end)
        except Exception:
            df = None
        if df is not None and len(df) >= 50:
            break

    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:          # yfinance는 뉴욕 시간대 포함 → 제거
        df.index = df.index.tz_localize(None)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna(subset=["Close"]).sort_index()

def load_all_indicators(ticker, market):
    end = pd.Timestamp.now()
    start_daily = (end - pd.DateOffset(months=DAILY_MONTHS)).normalize()
    start_long  = (end - pd.DateOffset(years=MONTHLY_YEARS)).normalize()

    raw = fetch_ohlcv(ticker, market, start_long, end)
    if raw is None:
        return None

    # 월봉 MACD — 일봉 종가를 월 단위로 묶어 계산 (한국/미국 공통)
    m_close = raw["Close"].groupby(raw.index.to_period("M")).last()
    m_exp1 = m_close.ewm(span=12, adjust=False).mean()
    m_exp2 = m_close.ewm(span=26, adjust=False).mean()
    monthly_macd = m_exp1 - m_exp2

    df = raw.loc[raw.index >= start_daily].copy()
    if len(df) < 50:
        return None

    # 일봉 RSI (기간 7)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(7).mean()
    avg_loss = loss.rolling(7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).bfill()

    # 주봉
    df_weekly = df.resample('W-FRI').agg({
        'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'
    }).dropna()

    w_period = 9
    w_delta = df_weekly["Close"].diff()
    w_gain = w_delta.clip(lower=0)
    w_loss = -w_delta.clip(upper=0)
    w_avg_gain = w_gain.ewm(alpha=1/w_period, adjust=False).mean()
    w_avg_loss = w_loss.ewm(alpha=1/w_period, adjust=False).mean()
    w_rs = w_avg_gain / w_avg_loss.replace(0, np.nan)
    df_weekly["Weekly_RSI"] = 100 - (100 / (1 + w_rs))
    df_weekly["Weekly_Signal"] = df_weekly["Weekly_RSI"].rolling(5).mean()

    exp1 = df_weekly["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df_weekly["Close"].ewm(span=26, adjust=False).mean()
    df_weekly["MACD"] = exp1 - exp2
    df_weekly["MACD_Signal"] = df_weekly["MACD"].ewm(span=9, adjust=False).mean()
    df_weekly["MACD_Hist"] = df_weekly["MACD"] - df_weekly["MACD_Signal"]

    return df.dropna(subset=["Close", "RSI"]), df_weekly.dropna(), monthly_macd


# ============================================================
# Plotly 차트 생성 (가독성 & 마커 버그 수정)
# ============================================================
def draw_sfp_chart_with_plotly(ticker, name, df, df_weekly, monthly_macd, market="KR"):
    df = df.copy()
    df["Slope"] = calc_price_slope_series(df)
    super_rsi   = calc_super_rsi(df)

    print(f"현재 PRICE     : {fmt_price(df['Close'].iloc[-1], market)}")
    print(f"어제 PRICE     : {fmt_price(df['Close'].iloc[-2], market)}")
    print(f"오늘 RSI       : {df['RSI'].iloc[-1]:.2f}")
    print(f"어제 RSI       : {df['RSI'].iloc[-2]:.2f}")
    print(f"오늘 Super RSI : {super_rsi.iloc[-1]:.2f}")
    print(f"어제 Super RSI : {super_rsi.iloc[-2]:.2f}")
    print(f"Super RSI 차이 : {super_rsi.iloc[-1] - super_rsi.iloc[-2]:.4f}")
    print(f"주봉 MACD 차이  : {df_weekly['MACD'].iloc[-1] - df_weekly['MACD'].iloc[-2]:.4f}")
    print(f"월봉 MACD 차이  : {monthly_macd.iloc[-1] - monthly_macd.iloc[-2]:.4f}")


    # ── 서브플롯 생성 ──────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.34, 0.155, 0.155, 0.175, 0.175],
        subplot_titles=(
            f"Price ({'USD' if market == 'US' else 'KRW'})",
            "Daily RSI (7)",
            "Daily Super RSI",
            "Weekly RSI (9)",
            "Weekly MACD"
        )
    )

    # ── 공통 색상 팔레트 ───────────────────────────────────────
    C_PRICE   = "#1a1a2e"   # 거의 검정에 가까운 네이비
    C_RED     = "#E53935"   # 선명한 빨강
    C_BLUE    = "#1565C0"   # 선명한 파랑
    C_GOLD    = "#FFD600"   # 골드 별
    C_GREEN   = "#2E7D32"   # 초록
    C_ORANGE  = "#F57C00"   # 오렌지
    C_GRAY    = "#9E9E9E"   # 회색
    C_PURPLE  = "#9C27B0"   # 퍼플

    # ══════════════════════════════════════════════════════════
    # Row 1 : PRICE
    # ══════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode='lines', name='Price',
        line=dict(color=C_PRICE, width=1.8),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>종가: " + price_hover(market) + "<extra></extra>"
    ), row=1, col=1)

    # 시그널 분류 버킷
    sv_x, sv_y, sv_hover   = [], [], []   # V-Pivot
    sg_x, sg_y, sg_hover   = [], [], []   # Gold V-Pivot
    si_x, si_y, si_hover   = [], [], []   # Inv-V-Pivot

    # RSI / Super-RSI 마커
    rv_x, rv_y = [], []
    ri_x, ri_y = [], []
    sv2_x, sv2_y = [], []
    si2_x, si2_y = [], []

    last_pivot = None

    for i in range(2, len(super_rsi)):
        r2, r1, r0 = super_rsi.iloc[i-2], super_rsi.iloc[i-1], super_rsi.iloc[i]
        is_v    = (r2 > r1 and r1 < r0)
        is_inv  = (r2 < r1 and r1 > r0)
        if not (is_v or is_inv):
            continue

        idx       = df.index[i-1]
        val       = df["Close"].iloc[i-1]
        rsi_val   = df["RSI"].iloc[i-1]
        slope_val = df["Slope"].iloc[i-1]
        srsi_val  = r1

        tip = (
            f"<b>{'[V-Pivot]' if is_v else '[Inv-V-Pivot]'}</b><br>"
            f"<b>날짜:</b> {idx.date()}<br>"
            f"<b>종가:</b> {fmt_price(val, market)}<br>"
            f"<b>RSI:</b> {rsi_val:.2f}<br>"
            f"<b>S-RSI:</b> {srsi_val:.2f}"
        )

        if is_v:
            if not pd.isna(slope_val) and slope_val <= -6.0:
                sg_x.append(idx); sg_y.append(val); sg_hover.append(tip)
            else:
                sv_x.append(idx); sv_y.append(val); sv_hover.append(tip)
            rv_x.append(idx);  rv_y.append(rsi_val)
            sv2_x.append(idx); sv2_y.append(srsi_val)
            last_pivot = ('V', idx, val, rsi_val)
        else:
            si_x.append(idx); si_y.append(val); si_hover.append(tip)
            ri_x.append(idx);  ri_y.append(rsi_val)
            si2_x.append(idx); si2_y.append(srsi_val)
            last_pivot = ('Inv-V', idx, val, rsi_val)

    # ── V-Pivot 마커 (오류 수정: 내부 흰색 + 빨간 테두리) ────────────────
    if sv_x:
        fig.add_trace(go.Scatter(
            x=sv_x, y=sv_y, mode='markers', name='V-Pivot',
            marker=dict(
                symbol='triangle-up', size=13,
                color='white',
                line=dict(color=C_RED, width=2.5)
            ),
            hoverinfo='text', hovertext=sv_hover
        ), row=1, col=1)

    # ── Gold V-Pivot 마커 (금별) ────────────────────────────
    if sg_x:
        fig.add_trace(go.Scatter(
            x=sg_x, y=sg_y, mode='markers', name='Gold V-Pivot',
            marker=dict(
                symbol='star', size=20,
                color=C_GOLD,
                line=dict(color='#795548', width=1.5)
            ),
            hoverinfo='text', hovertext=sg_hover
        ), row=1, col=1)

    # ── Inv-V-Pivot 마커 (오류 수정: 내부 흰색 + 파란 테두리) ────────────
    if si_x:
        fig.add_trace(go.Scatter(
            x=si_x, y=si_y, mode='markers', name='Inv-V-Pivot',
            marker=dict(
                symbol='triangle-down', size=13,
                color='white',
                line=dict(color=C_BLUE, width=2.5)
            ),
            hoverinfo='text', hovertext=si_hover
        ), row=1, col=1)

    # ── 화살표 마커: 2중 확인 신호 ──────────────────────────────
    # 조건 1: Weekly RSI > Weekly Signal (시그널선 골든크로스 상황)
    # 조건 2: Weekly MACD 히스토그램이 음→양으로 전환 (prev_hist <= 0 and curr_hist > 0)
    w_rsi   = df_weekly["Weekly_RSI"]
    w_sig   = df_weekly["Weekly_Signal"]
    w_hist  = df_weekly["MACD_Hist"]
    heart_x, heart_y, heart_hover = [], [], []
    for i in range(1, len(w_rsi)):
        curr_r    = w_rsi.iloc[i]
        curr_sig  = w_sig.iloc[i]
        prev_hist = w_hist.iloc[i-1]
        curr_hist = w_hist.iloc[i]
        if pd.isna(curr_r) or pd.isna(curr_sig) or pd.isna(prev_hist) or pd.isna(curr_hist):
            continue
        if (curr_r > curr_sig) and (prev_hist <= 0 and curr_hist > 0):
            target = df_weekly.index[i]
            mask = df.index <= target
            if mask.any():
                h_idx = df.index[mask][-1]
                h_val = df["Close"].loc[h_idx]
            else:
                h_idx = target
                h_val = df_weekly["Close"].iloc[i]
            tip = (
                f"<b>[Signal Cross + MACD↑]</b><br>"
                f"<b>날짜:</b> {h_idx.date()}<br>"
                f"<b>종가:</b> {fmt_price(h_val, market)}<br>"
                f"<b>W-RSI:</b> {curr_r:.2f}<br>"
                f"<b>W-Signal:</b> {curr_sig:.2f}<br>"
                f"<b>MACD Hist:</b> {curr_hist:.4f}"
            )
            heart_x.append(h_idx)
            heart_y.append(h_val)
            heart_hover.append(tip)

    if heart_x:
        fig.add_trace(go.Scatter(
            x=heart_x, y=heart_y,
            mode='markers',
            marker=dict(
                symbol='arrow-up',
                size=24,
                color='white',
                line=dict(color=C_PURPLE, width=4)
            ),
            name='Signal Cross + MACD↑',
            hoverinfo='text', hovertext=heart_hover
        ), row=1, col=1)

    # ── Latest Pivot Annotation (자동 배치) ───────────────────
    if last_pivot:
        p_type, p_idx, p_val, p_rsi = last_pivot
        c = C_RED if p_type == 'V' else C_BLUE
        lbl = 'V-Pivot' if p_type == 'V' else 'Inv-V-Pivot'

        # ── 자동 배치 로직 ─────────────────────────────────────
        # 1) 가격 범위 내 상대적 위치 계산 (0.0 = 최저, 1.0 = 최고)
        price_min = df["Close"].min()
        price_max = df["Close"].max()
        price_range = price_max - price_min if price_max != price_min else 1
        rel_pos = (p_val - price_min) / price_range  # 0~1

        # 2) 수직 방향: 가격이 차트 상단 40% 이상이면 아래로, 아니면 위로
        if rel_pos >= 0.6:
            ay = 65   # 아래쪽
        elif rel_pos <= 0.4:
            ay = -65  # 위쪽
        else:
            # 중간대: pivot 타입 기본값 유지
            ay = -55 if p_type == 'V' else 55

        # 3) 수평 방향: 시계열 끝 25% 구간이면 왼쪽으로, 아니면 오른쪽으로
        total_bars = len(df)
        pivot_pos  = df.index.get_loc(p_idx) if p_idx in df.index else total_bars - 1
        if pivot_pos >= total_bars * 0.75:
            ax = -100  # 오른쪽 끝 근처 → 말풍선을 왼쪽으로
        else:
            ax = 60    # 그 외 → 말풍선을 오른쪽으로

        # 4) 피벗 주변 ±10봉 가격과의 겹침 추가 보정
        win_start = max(0, pivot_pos - 10)
        win_end   = min(total_bars, pivot_pos + 11)
        local_prices = df["Close"].iloc[win_start:win_end]
        local_max = local_prices.max()
        local_min = local_prices.min()
        local_mid = (local_max + local_min) / 2

        if ay < 0 and p_val < local_mid:
            ay = abs(ay)
        elif ay > 0 and p_val > local_mid:
            ay = -abs(ay)

        fig.add_annotation(
            x=p_idx, y=p_val,
            text=f"<b>Latest {lbl}</b><br>Date: {p_idx.date()}<br>Close: {fmt_price(p_val, market)}<br>RSI: {p_rsi:.1f}",
            showarrow=True, arrowhead=2,
            arrowcolor=c, arrowsize=1.2, arrowwidth=2,
            bgcolor="white", bordercolor=c, borderwidth=2,
            font=dict(size=11, color=c),
            ax=ax, ay=ay,
            row=1, col=1
        )

    # ══════════════════════════════════════════════════════════
    # Row 2 : DAILY RAW RSI
    # ══════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        mode='lines', name='Daily RSI (7)',
        line=dict(color=C_PRICE, width=1.5),
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>RSI: %{y:.2f}<extra></extra>"
    ), row=2, col=1)

    if rv_x:
        fig.add_trace(go.Scatter(
            x=rv_x, y=rv_y, mode='markers', showlegend=False,
            marker=dict(symbol='triangle-up', size=10, color=C_RED, line=dict(color='darkred', width=1)),
            hoverinfo='skip'
        ), row=2, col=1)
    if ri_x:
        fig.add_trace(go.Scatter(
            x=ri_x, y=ri_y, mode='markers', showlegend=False,
            marker=dict(symbol='triangle-down', size=10, color=C_BLUE, line=dict(color='darkblue', width=1)),
            hoverinfo='skip'
        ), row=2, col=1)

    for lvl, clr, dash in [(70, C_RED, 'dash'), (30, C_GREEN, 'dash')]:
        fig.add_hline(y=lvl, line=dict(color=clr, width=1, dash=dash), opacity=0.6, row=2, col=1)

    # ══════════════════════════════════════════════════════════
    # Row 3 : DAILY SUPER RSI
    # ══════════════════════════════════════════════════════════
    upper_q = df["RSI"].rolling(20).quantile(0.7)
    lower_q = df["RSI"].rolling(20).quantile(0.3)

    fig.add_trace(go.Scatter(x=df.index, y=super_rsi, mode='lines', name='Daily Super RSI', line=dict(color=C_PRICE, width=2.0), hovertemplate="날짜: %{x|%Y-%m-%d}<br>S-RSI: %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=upper_q, mode='lines', name='Dynamic Upper(70%)', line=dict(color=C_RED, width=1.2, dash='dash'), opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=lower_q, mode='lines', name='Dynamic Lower(30%)', line=dict(color=C_GREEN, width=1.2, dash='dash'), opacity=0.7), row=3, col=1)

    if sv2_x:
        fig.add_trace(go.Scatter(x=sv2_x, y=sv2_y, mode='markers', showlegend=False, marker=dict(symbol='triangle-up', size=10, color=C_RED, line=dict(color='darkred', width=1)), hoverinfo='skip'), row=3, col=1)
    if si2_x:
        fig.add_trace(go.Scatter(x=si2_x, y=si2_y, mode='markers', showlegend=False, marker=dict(symbol='triangle-down', size=10, color=C_BLUE, line=dict(color='darkblue', width=1)), hoverinfo='skip'), row=3, col=1)

    # ══════════════════════════════════════════════════════════
    # Row 4 : WEEKLY RSI
    # ══════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly["Weekly_RSI"], mode='lines', name='Weekly RSI (9)', line=dict(color=C_RED, width=2.2), hovertemplate="날짜: %{x|%Y-%m-%d}<br>W-RSI: %{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly["Weekly_Signal"], mode='lines', name='W-Signal (5)', line=dict(color=C_PRICE, width=1.5), hovertemplate="날짜: %{x|%Y-%m-%d}<br>Signal: %{y:.2f}<extra></extra>"), row=4, col=1)

    for lvl, clr, dash, op in [(70, C_RED, 'dot', 0.4), (50, C_GRAY, 'dash', 0.5), (30, C_BLUE, 'dot', 0.4)]:
        fig.add_hline(y=lvl, line=dict(color=clr, width=1, dash=dash), opacity=op, row=4, col=1)

    # ══════════════════════════════════════════════════════════
    # Row 5 : WEEKLY MACD
    # ══════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly["MACD"], mode='lines', name='MACD', line=dict(color=C_BLUE, width=2.0), hovertemplate="날짜: %{x|%Y-%m-%d}<br>MACD: %{y:.2f}<extra></extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly["MACD_Signal"], mode='lines', name='Signal', line=dict(color=C_ORANGE, width=1.8), hovertemplate="날짜: %{x|%Y-%m-%d}<br>Signal: %{y:.2f}<extra></extra>"), row=5, col=1)

    hist_colors = [C_RED if v > 0 else C_BLUE for v in df_weekly["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df_weekly.index, y=df_weekly["MACD_Hist"], marker_color=hist_colors, opacity=0.55, name='Hist'), row=5, col=1)
    fig.add_hline(y=0, line=dict(color=C_GRAY, width=1, dash='dash'), opacity=0.6, row=5, col=1)

    # ══════════════════════════════════════════════════════════
    # 축 / 레이아웃 정리
    # ══════════════════════════════════════════════════════════
    for row, title in [(1, "Price"), (2, "Daily RSI"), (3, "Super RSI"), (4, "Weekly RSI"), (5, "Weekly MACD")]:
        fig.update_yaxes(title_text=title, title_font=dict(size=11, color="#333"), tickfont=dict(size=10), gridcolor="#e8e8e8", gridwidth=1, row=row, col=1)

    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    if market == "US":
        fig.update_yaxes(tickprefix="$", row=1, col=1)

    fig.update_xaxes(showspikes=True, spikecolor=C_GRAY, spikesnap="cursor", spikemode="across", spikethickness=1, spikedash="dot", tickfont=dict(size=10), gridcolor="#e8e8e8")

    for ann in fig.layout.annotations:
        ann.font.size  = 12
        ann.font.color = "#222"

    fig.update_layout(
        height=1100,
        template="plotly_white",
        hovermode="closest",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11), bgcolor="rgba(255,255,255,0.85)", bordercolor="#ddd", borderwidth=1),
        margin=dict(l=70, r=30, t=80, b=40),
        title=dict(text=f"<b>[{ticker}] {name} ({market})   Super RSI & Weekly MACD</b>", x=0.5, xanchor="center", font=dict(size=16, color="#1a1a2e"))
    )
    fig.show()


# ============================================================
# 하락장 경계 체크리스트
# ============================================================
# 근거: S&P500 1950~2026 주봉 MACD 데드크로스 171건 분석 (dca_strategy.md 7.4절)
#   데드크로스 전체                                   → 1년 내 -20% 이상 하락 14%
#   데드크로스 + 40주선 아래 + 미국 매크로 악화 3개 이상 → 17건 중 71% (침체형 하락장에서만 발생)
# 매도 신호로 쓰면 수익 우위가 기간마다 뒤집혔으므로 모니터링용으로만 사용
FRED_SERIES      = ["GS10", "TB3MS", "BAA", "ICSA", "PERMIT"]
FRED_API_URL     = "https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json&observation_start={}"
FRED_CSV_URL     = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}&cosd={}"
FRED_CACHE_DIR   = os.path.expanduser("~/.cache/fred")   # Colab에서 세션이 끝나도 유지하려면 Google Drive 경로로 변경
FRED_CACHE_HOURS = 12
FRED_TIMEOUT     = 15
FRED_RETRIES     = 2

def get_fred_api_key():
    """코드 상단 FRED_API_KEY → 환경변수 FRED_API_KEY → Colab 보안 비밀(Secrets) 순으로 조회. 없으면 None"""
    key = FRED_API_KEY.strip() or os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata
        return userdata.get("FRED_API_KEY")
    except Exception:
        return None

def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(FRED_RETRIES):
        try:
            return urllib.request.urlopen(req, timeout=FRED_TIMEOUT).read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == FRED_RETRIES - 1:   # 잘못된 키 등 4xx는 재시도하지 않음
                raise
        except Exception:
            if attempt == FRED_RETRIES - 1:
                raise
        time.sleep(2 ** attempt)

def _download_fred(series_id, api_key, years=5):
    """API 키가 있으면 공식 API, 없거나 실패하면 웹사이트 CSV로 다운로드 → DataFrame(date, value)"""
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    errors = []
    if api_key:
        try:
            obs = json.loads(_http_get(FRED_API_URL.format(series_id, api_key, start)))["observations"]
            return pd.DataFrame({"date": [o["date"] for o in obs], "value": [o["value"] for o in obs]})
        except Exception as e:
            errors.append(f"API {e}")
    try:
        d = pd.read_csv(io.BytesIO(_http_get(FRED_CSV_URL.format(series_id, start))))
        if d.shape[1] != 2 or series_id not in d.columns[1]:
            raise ValueError("CSV 형식 오류")
        d.columns = ["date", "value"]
        return d
    except Exception as e:
        errors.append(f"CSV {e}")
    raise RuntimeError(f"{series_id} — " + " / ".join(errors))

def fetch_fred(series_id, api_key):
    """12시간 이내 캐시는 그대로 사용 → 아니면 다운로드 → 실패 시 오래된 캐시로 대체
       반환: (Series, 오래된 캐시를 썼으면 저장 후 경과 시간, 아니면 None)"""
    path  = os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")
    age_h = (time.time() - os.path.getmtime(path)) / 3600 if os.path.exists(path) else None
    stale = None
    if age_h is None or age_h > FRED_CACHE_HOURS:
        try:
            d = _download_fred(series_id, api_key)
            os.makedirs(FRED_CACHE_DIR, exist_ok=True)
            d.to_csv(path, index=False)
        except Exception:
            if age_h is None:
                raise
            stale = age_h
    d = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(d["value"], errors="coerce").values, index=pd.to_datetime(d["date"])).dropna()
    return s, stale

def calc_us_macro_flags():
    """미국 매크로 악화 5개 항목 → ([(항목, 충족 여부, 현재값)], 기준일 설명, 오래된 캐시 사용 목록)"""
    api_key = get_fred_api_key()   # Colab 권한 팝업이 뜰 수 있어 스레드 밖에서 조회
    with ThreadPoolExecutor(max_workers=len(FRED_SERIES)) as ex:
        res = dict(zip(FRED_SERIES, ex.map(lambda sid: fetch_fred(sid, api_key), FRED_SERIES)))
    gs10, tb3, baa, icsa, permit = (res[sid][0] for sid in FRED_SERIES)
    stale = [f"{sid} {res[sid][1]:.0f}시간 전" for sid in FRED_SERIES if res[sid][1] is not None]

    curve = (gs10 - tb3).dropna()
    curve_min24 = curve.iloc[-24:].min()
    tb3_off = tb3.iloc[-1] - tb3.iloc[-12:].max()
    claims = icsa.rolling(4).mean()
    claims_yoy = (claims.iloc[-1] / claims.iloc[-53] - 1) * 100
    permit_yoy = (permit.iloc[-1] / permit.iloc[-13] - 1) * 100
    credit = (baa - gs10).dropna()
    credit_chg6 = credit.iloc[-1] - credit.iloc[-7]

    flags = [
        ("장단기 금리차(10년-3개월) 24개월 내 역전", curve_min24 < 0,
         f"현재 {curve.iloc[-1]:+.2f}%p / 24개월 최저 {curve_min24:+.2f}%p"),
        ("금리 인하 시작 (3개월물, 12개월 고점 대비 -0.5%p 이하)", tb3_off <= -0.5, f"{tb3_off:+.2f}%p"),
        ("실업수당 청구 전년 대비 증가 (4주 평균)", claims_yoy > 0, f"{claims_yoy:+.1f}%"),
        ("주택 인허가 전년 대비 -10% 이하", permit_yoy < -10, f"{permit_yoy:+.1f}%"),
        ("신용스프레드(BAA-10년물) 6개월 +0.25%p 이상 확대", credit_chg6 > 0.25, f"{credit_chg6:+.2f}%p"),
    ]
    asof = f"금리 {curve.index[-1]:%Y-%m}, 실업수당 {icsa.index[-1]:%Y-%m-%d}, 인허가 {permit.index[-1]:%Y-%m}"
    return flags, asof, stale

def print_bear_checklist(df_weekly, market):
    hist    = df_weekly["MACD_Hist"]
    w_close = df_weekly["Close"]
    dead       = hist.iloc[-1] < 0
    dead_weeks = int((hist[::-1] < 0).astype(int).cumprod().sum())
    sma40   = w_close.rolling(40).mean().iloc[-1]
    gap40   = (w_close.iloc[-1] / sma40 - 1) * 100
    below40 = bool(gap40 < 0)   # 데이터 부족(NaN)이면 False

    def mark(ok):
        return "●" if ok else "○"

    print("\n───── 하락장 경계 체크리스트 ─────")
    print(f" {mark(dead)} 주봉 MACD 데드크로스 구간 : {f'예 ({dead_weeks}주째)' if dead else '아니오'}")
    print(f" {mark(below40)} 40주 이동평균선 아래      : {'데이터 부족' if pd.isna(gap40) else f'{gap40:+.1f}%'}")

    n_macro = None
    try:
        macro, asof, stale = calc_us_macro_flags()
        n_macro = sum(ok for _, ok, _ in macro)
        print(f" {mark(n_macro >= 3)} 미국 매크로 악화 3개 이상 : {n_macro}/5  ({asof})")
        if stale:
            print(f"     ※ FRED 접속 실패로 저장된 데이터 사용: {', '.join(stale)}")
        for label, ok, val in macro:
            print(f"     {mark(ok)} {label}: {val}")
    except Exception as e:
        print(f" ? 미국 매크로 데이터 조회 실패: {e}")
        if not get_fred_api_key():
            print("   → Colab 등 클라우드 환경에서는 코드 상단 FRED_API_KEY에 API 키를 입력하세요")

    if not dead:
        verdict = "데드크로스 아님"
    elif n_macro is None:
        verdict = "판정 보류 — 매크로 데이터 없음"
    elif below40 and n_macro >= 3:
        verdict = "강한 경고 — 과거 17건 중 71%가 1년 내 -20% 이상 하락 (침체형 하락장에서만 발생)"
    else:
        verdict = "일반 데드크로스 — 과거 1년 내 -20% 이상 하락 14%, 평소 수준"
    print(f" ▶ 판정: {verdict}")
    print("   ※ S&P500 지수 기준 검증. 매도 신호로 쓰면 수익 우위가 기간마다 뒤집혔음 (dca_strategy.md 7.4)")
    if market == "KR":
        print("   ※ 매크로 항목은 미국 경제 기준이며 한국 종목에는 검증되지 않음")


# ============================================================
# MAIN
# ============================================================
def analyze_single_ticker(ticker_or_name, market=None):
    """ticker_or_name: 한국 코드(069500, 0089D0) / 한글 종목명 / 미국 티커(AAPL, BRK.B) / 영문 종목명(Apple)
       market: None이면 자동 판별, 'KR' 또는 'US'로 강제 지정 가능"""
    try:
        ticker, name, market = resolve_ticker(ticker_or_name, market)
    except Exception as e:
        print(f"종목 조회 실패: {e}")
        return

    result = load_all_indicators(ticker, market)

    # 'Apple'처럼 티커 형태로 보이는 영문 종목명 → 데이터가 없으면 종목명 검색 후 재시도
    if result is None and market == "US":
        try:
            alt = us_name_to_ticker(ticker_or_name)
        except Exception:
            alt = None
        if alt and alt != ticker:
            ticker, name = alt, us_ticker_name(alt)
            result = load_all_indicators(ticker, market)

    print(f"\n===== [{ticker}] {name} ({market}) 분석 =====")
    if result is None:
        print("데이터 부족으로 분석 불가")
        return

    df_daily, df_weekly, monthly_macd = result
    draw_sfp_chart_with_plotly(ticker, name, df_daily, df_weekly, monthly_macd, market)
    print_bear_checklist(df_weekly, market)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    # ── 한국 ──
    #analyze_single_ticker("498400")  #코스피200 타겟위클리커버드콜
    #analyze_single_ticker("493810")  #미국AI빅테크
    #analyze_single_ticker("498410")  #KODEX 금융고배당TOP10 커버드콜
    #analyze_single_ticker("0048J0")  #KODEX 미국머니마켓액티브
    #analyze_single_ticker("132030")  #KODEX 골드선물
    #analyze_single_ticker("261220")  #KODEX 원유선물
    #analyze_single_ticker("138910")  #KODEX 구리선물
    #analyze_single_ticker("069500")  #KODEX 200
    #analyze_single_ticker("229200")  #KODEX 코스닥150
    #analyze_single_ticker("0052D0")  #TIGER 코리아배당다우존스
    #analyze_single_ticker("458730")  #TIGER 미국배당다우존스
    #analyze_single_ticker("352540")  #KODEX 일본부동산리츠
    analyze_single_ticker("0089D0")   #개별종목

    # ── 미국 ──
    #analyze_single_ticker("PSQ")      #ProShares Short QQQ
    #analyze_single_ticker("QQQ")      #Invesco QQQ
    #analyze_single_ticker("AAPL")     #Apple
    #analyze_single_ticker("BRK.B")    #Berkshire Hathaway B (BRK-B로 자동 변환)
    #analyze_single_ticker("Apple")    #영문 종목명 검색 (yfinance 필요)
