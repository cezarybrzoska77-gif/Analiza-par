import yfinance as yf
import pandas as pd
import numpy as np

# --- Parametry --- #
pairs = [("LEN", "MTH"), ("DHI", "MTH")]
Z_WINDOW_LONG = 60
Z_WINDOW_SHORT = 30
AVG_ABS_Z_LIMIT = 1.5
Z60_ENTRY_THRESHOLD = 2.0
Z30_CONFIRM = 1.3
Z60_EXIT_TP = 0.5
Z60_EXIT_SL = 3.2
HL_WINDOW = 30  # half-life window

# --- Funkcje pomocnicze --- #
def compute_zscore(spread, window):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std

def half_life(spread):
    spread = spread.dropna()
    spread_lag = spread.shift(1).dropna()
    spread_ret = (spread - spread_lag).dropna()

    if len(spread_lag) < 2:  # za mało danych
        return np.inf

    beta = np.polyfit(spread_lag.values, spread_ret.values, 1)[0]
    return -np.log(2) / beta if beta != 0 else np.inf

# --- Główny loop --- #
results = []

for x, y in pairs:
    # Pobierz dane z auto_adjust=True
    df_x = yf.download(x, period="180d", auto_adjust=True)['Close']
    df_y = yf.download(y, period="180d", auto_adjust=True)['Close']

    # Usuń NaN i dopasuj wspólne daty
    df_x = df_x.dropna()
    df_y = df_y.dropna()
    common_index = df_x.index.intersection(df_y.index)
    df_x = df_x.loc[common_index]
    df_y = df_y.loc[common_index]

    spread = df_y - df_x
    z60 = compute_zscore(spread, Z_WINDOW_LONG)
    z30 = compute_zscore(spread, Z_WINDOW_SHORT)
    avg_abs_z60 = z60.abs().rolling(Z_WINDOW_LONG).mean()
    hl = half_life(spread)

    latest_z60 = z60.iloc[-1]
    latest_z30 = z30.iloc[-1]
    latest_avg_abs_z60 = avg_abs_z60.iloc[-1]

    # --- WARUNKI WEJŚCIA --- #
    entry_signal = None
    if abs(latest_z60) >= Z60_ENTRY_THRESHOLD and latest_avg_abs_z60 <= AVG_ABS_Z_LIMIT:
        if latest_z60 >= Z60_ENTRY_THRESHOLD and latest_z30 >= Z30_CONFIRM:
            entry_signal = "SHORT y / LONG x"
        elif latest_z60 <= -Z60_ENTRY_THRESHOLD and latest_z30 <= -Z30_CONFIRM:
            entry_signal = "LONG y / SHORT x"

    results.append({
        "Pair": f"{x}/{y}",
        "Z60": latest_z60,
        "Z30": latest_z30,
        "Avg_abs_Z60": latest_avg_abs_z60,
        "Half_life": hl,
        "Entry_signal": entry_signal
    })

# --- Zapis wyników --- #
df_res = pd.DataFrame(results)
df_res.to_csv("workflow2_results.csv", index=False)
print(df_res)
