import yfinance as yf
import pandas as pd
import numpy as np

# =============================
# PARAMETRY WORKFLOW 2
# =============================
PAIRS = [("LEN", "MTH"), ("DHI", "MTH")]

Z_WINDOW_LONG = 60
Z_WINDOW_SHORT = 30

Z60_ENTRY = 2.0
Z30_CONFIRM = 1.3
AVG_ABS_Z_LIMIT = 1.5

# =============================
# FUNKCJE
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
# GŁÓWNA LOGIKA
# =============================
results = []

for x, y in PAIRS:
    # --- download ---
    px = yf.download(x, period="180d", auto_adjust=True, progress=False)["Close"]
    py = yf.download(y, period="180d", auto_adjust=True, progress=False)["Close"]

    # --- align ---
    df = pd.concat([px, py], axis=1, join="inner")
    df.columns = ["x", "y"]
    df = df.dropna()

    if len(df) < Z_WINDOW_LONG + 5:
        continue

    spread = df["y"] - df["x"]

    z60 = compute_zscore(spread, Z_WINDOW_LONG)
    z30 = compute_zscore(spread, Z_WINDOW_SHORT)

    avg_abs_z60 = z60.abs().rolling(Z_WINDOW_LONG).mean()

    # --- WYCIĄGNIĘCIE SCALARÓW (KLUCZOWE) ---
    try:
        latest_z60 = z60.dropna().iloc[-1].item()
        latest_z30 = z30.dropna().iloc[-1].item()
        latest_avg_abs_z60 = avg_abs_z60.dropna().iloc[-1].item()
    except Exception:
        continue

    hl = half_life(spread)

    # --- ENTRY LOGIC ---
    entry_signal = None

    if (
        abs(latest_z60) >= Z60_ENTRY
        and latest_avg_abs_z60 <= AVG_ABS_Z_LIMIT
    ):
        # SHORT MR
        if latest_z60 >= Z60_ENTRY and latest_z30 >= Z30_CONFIRM:
            entry_signal = "SHORT y / LONG x"

        # LONG MR
        elif latest_z60 <= -Z60_ENTRY and latest_z30 <= -Z30_CONFIRM:
            entry_signal = "LONG y / SHORT x"

    results.append({
        "Pair": f"{x}/{y}",
        "Z60": round(latest_z60, 2),
        "Z30": round(latest_z30, 2),
        "Avg_abs_Z60": round(latest_avg_abs_z60, 2),
        "Half_life": round(hl, 1),
        "Entry_signal": entry_signal
    })

# =============================
# OUTPUT
# =============================
df_out = pd.DataFrame(results)
df_out.to_csv("workflow2_results.csv", index=False)
print(df_out)
