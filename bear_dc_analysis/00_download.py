"""0단계: S&P500·VIX·SPY·러셀2000 (Yahoo), 매크로 (FRED), Shiller 배당·이익 데이터 다운로드 → data/"""
import io, logging, os, urllib.request
import pandas as pd, yfinance as yf
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(D, exist_ok=True)

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()

for t in ["^GSPC", "^VIX", "SPY", "^RUT"]:
    h = yf.Ticker(t).history(period="max", auto_adjust=False)
    h.index = h.index.tz_localize(None)
    h.to_csv(f"{D}/{t.strip('^')}.csv"); print(t, h.index[0].date(), h.index[-1].date())
for s in ["GS10", "TB3MS", "BAA", "UNRATE", "ICSA", "NFCI", "USREC", "INDPRO", "PERMIT", "CPIAUCSL"]:
    pd.read_csv(io.BytesIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"))).to_csv(f"{D}/{s}.csv", index=False)
    print(s)
open(f"{D}/data.csv", "wb").write(get("https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"))
print("Shiller data.csv")
