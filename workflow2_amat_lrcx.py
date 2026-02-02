import yfinance as yf
import pandas as pd
import numpy as np

# =============================
# WORKFLOW 2 – PARAMETRY
# =============================
PAIRS = [("AMAT", "LRCX")]

Z_WINDOW_LONG = 60
Z_WINDOW_SHORT = 30

Z60_ENTRY = 2.0
Z30_CONFIRM = 1.3
AVG_ABS_Z_LIMIT = 1.5

# =============================
# FUNCTIONS
# =============================
def compute_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def half_life(spread):
    spread = spread.dropna()

    if len(spread) < 20:
        return np.inf

    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()

    beta = np.polyfit(lag.values, delta.values, 1)[0]

    if beta >= 0:
        return np.inf

    return -np.log(2) / beta


# =============================
# MAIN LOOP
# =============================
results = []

for x, y in PAIRS:
    px = yf.download(x, period="200d", auto_adjust=True, progress=False)["Close"]
    py = yf.download(y, period="200d", auto_adjust=True, progress=False)["Close"]

    df = pd.concat([px, py], axis=1, join="inner")
    df.columns = ["x", "y"]
    df = df.dropna()

    if len(df) < Z_WINDOW_LONG + 5:
        continue

    spread = df["y"] - df["x"]

    z60 = compute_zscore(spread, Z_WINDOW_LONG)
    z30 = compute_zscore(spread, Z_WINDOW_SHORT)
    avg_abs_z60 = z60.abs().rolling(Z_WINDOW_LONG).mean()

    try:
        z60_last = z60.dropna().iloc[-1].item()
        z30_last = z30.dropna().iloc[-1].item()
        avg_abs_last = avg_abs_z60.dropna().iloc[-1].item()
    except Exception:
        continue

    hl = half_life(spread)

    # =============================
    # ENTRY LOGIC (Workflow 2)
    # =============================
    entry = None

    if abs(z60_last) >= Z60_ENTRY and avg_abs_last <= AVG_ABS_Z_LIMIT:
        # SHORT MR
        if z60_last >= Z60_ENTRY and z30_last >= Z30_CONFIRM:
            entry = "SHORT LRCX / LONG AMAT"

        # LONG MR
        elif z60_last <= -Z60_ENTRY and z30_last <= -Z30_CONFIRM:
            entry = "LONG LRCX / SHORT AMAT"

    results.append({
        "Pair": "AMAT/LRCX",
        "Z60": round(z60_last, 2),
        "Z30": round(z30_last, 2),
        "Avg_abs_Z60": round(avg_abs_last, 2),
        "Half_life": round(hl, 1),
        "Entry_signal": entry
    })

# =============================
# OUTPUT
# =============================
df_out = pd.DataFrame(results)
df_out.to_csv("workflow2_amat_lrcx_results.csv", index=False)
print(df_out)
