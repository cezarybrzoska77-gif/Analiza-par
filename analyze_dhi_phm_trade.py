#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workflow 2 - Retail Pair Analysis DHI/PHM

Cel:
- Ocena gotowości pary DHI (y) / PHM (x) do otwarcia transakcji w trybie retail.
- Analiza spreadów: beta-neutral, cash-neutral, Z-score (60/30), ADF, half-life.
- Wyznaczenie trybu hedgingu, kierunku trade, wagi nóg, progi wyjścia.
- Zapis wyników do CSV, JSON i opcjonalny wykres PNG.

Checklist "TRADE_READY = YES" wymaga:
1. ADF (wybrany tryb) ≤ 0.05
2. Half-life ≤ 15 dni
3. Z60 & Z30 spójne: LONG Z60 ≤ -2 & Z30 ≤ -1.5, SHORT Z60 ≥ 2 & Z30 ≥ 1.5
4. Beta stability ≤ 25% (lub w przedziale 25-35% tylko jeśli ADF/HL beta wyraźnie lepsze)
5. Wystarczająca próbka: ≥ 200 dni

Wyjście:
- results_workflow2/DHI_PHM_trade_readiness.csv
- results_workflow2/DHI_PHM_trade_readiness.json
- results_workflow2/DHI_PHM_spread_chart.png (opcjonalnie)
"""

import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---------------------
# Helper functions
# ---------------------
def download_data(tickers, start_date, max_retry=3):
    for attempt in range(max_retry):
        try:
            df = yf.download(
                tickers,
                start=start_date,
                auto_adjust=True,
                group_by='ticker',
                threads=False
            )
            # obsługa struktury zwróconej przez yfinance
            if tickers[0] in df.columns.levels[0]:
                adj_close = pd.DataFrame({t: df[t]['Close'] for t in tickers})
            else:
                adj_close = df['Close'] if 'Close' in df.columns else df
            return adj_close
        except Exception as e:
            print(f"[WARN] download attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download tickers {tickers}")

def log_returns(df, use_percent=False, winsorize=True):
    if use_percent:
        ret = df.pct_change()
    else:
        ret = np.log(df / df.shift(1))
    if winsorize:
        ret = ret.clip(lower=ret.quantile(0.01), upper=ret.quantile(0.99))
    return ret

def rolling_beta(y, x, window):
    beta = y.rolling(window).cov(x.rolling(window)) / x.rolling(window).var()
    return beta

def half_life(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    delta = series.diff().dropna()
    y = delta.values
    X = series.shift(1).dropna().values.reshape(-1,1)
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X[-min_len:]
        y = y[-min_len:]
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        rho = model.params[1]
        if rho <= -1 or rho >= 1:
            return np.nan
        return -np.log(2)/np.log(1+rho)
    except Exception:
        return np.nan

def z_score(series, window):
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    z = (series - roll_mean)/roll_std
    return z

# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2018-01-01')
    parser.add_argument('--auto-adjust', action='store_true', default=True)
    parser.add_argument('--use-percent-returns', action='store_true', default=False)
    parser.add_argument('--winsorize', action='store_true', default=True)
    parser.add_argument('--z-lookbacks', default='60,30')
    parser.add_argument('--min-sample', type=int, default=200)
    parser.add_argument('--out-dir', default='results_workflow2')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', action='store_false', dest='plot')
    parser.add_argument('--help-checklist', action='store_true', help='Show ENTRY checklist only')
    args = parser.parse_args()

    if args.help_checklist:
        print("""
Checklist ENTRY (TRADE_READY = YES):
- ADF p-value (spread_beta/cash) <= 0.05
- Half-life <= 15 dni
- Z60 & Z30 spójne:
    LONG: Z60 <= -2 & Z30 <= -1.5
    SHORT: Z60 >= 2 & Z30 >= 1.5
- Beta stability <= 25% (lub w przedziale 25-35% tylko jeśli beta wyraźnie lepsza)
- Min sample >= 200
Exit: |Z60| <= 0.5
""")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    tickers = ['DHI','PHM']

    try:
        df = download_data(tickers, args.start_date)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    df = df.dropna()
    if df.shape[0] < args.min_sample:
        print(f"[ERROR] Not enough data ({df.shape[0]} < {args.min_sample})")
        trade_ready = "NO"
        return

    ret = log_returns(df, args.use_percent_returns, args.winsorize)
    y = df['DHI']
    x = df['PHM']

    # OLS beta neutral
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha_OLS = model.params['const']
    beta_OLS = model.params['PHM']

    # rolling betas
    beta_60 = rolling_beta(y, x, 60).iloc[-1]
    beta_90 = rolling_beta(y, x, 90).iloc[-1]
    beta_stability_pct = 100 * abs(beta_60-beta_90)/np.mean([abs(beta_60), abs(beta_90)])

    # spreads
    spread_beta = y - (alpha_OLS + beta_OLS*x)
    spread_cash = y - x

    # ADF
    adf_beta_res = adfuller(spread_beta.dropna())
    adf_cash_res = adfuller(spread_cash.dropna())
    adf_p_beta, adf_stat_beta = adf_beta_res[1], adf_beta_res[0]
    adf_p_cash, adf_stat_cash = adf_cash_res[1], adf_cash_res[0]

    # Half-life
    hl_beta = half_life(spread_beta)
    hl_cash = half_life(spread_cash)

    # Z-scores
    z1, z2 = map(int, args.z_lookbacks.split(','))
    z60 = z_score(spread_beta, z1).iloc[-1]
    z30 = z_score(spread_beta, z2).iloc[-1]
    dz3 = z60 - z_score(spread_beta, z2).iloc[-4]

    # Vol ratio
    vol_ratio = spread_beta.std() / (ret['DHI'].std() + ret['PHM'].std())

    # Hedging recommendation
    hedge_reco = "beta_neutral"
    if beta_stability_pct >= 40:
        hedge_reco = "cash_neutral"
    elif 25 < beta_stability_pct < 35:
        if adf_p_beta > adf_p_cash + 0.02 or hl_beta > hl_cash + 3:
            hedge_reco = "cash_neutral"

    # ENTRY criteria
    entry = False
    if hedge_reco=='beta_neutral':
        entry = (adf_p_beta <=0.05 and hl_beta<=15 and args.min_sample<=len(df))
    else:
        entry = (adf_p_cash <=0.05 and hl_cash<=15 and args.min_sample<=len(df))
    direction = None
    if entry:
        if z60 >= 2 and z30 >= 1.5:
            direction = "short_y_long_x"
        elif z60 <= -2 and z30 <= -1.5:
            direction = "long_y_short_x"
        else:
            entry = False

    trade_ready = "YES" if entry else "NO"

    # wagi
    if hedge_reco=="beta_neutral":
        w_y = 1.0
        w_x = -beta_OLS
    else:
        w_y = 1.0
        w_x = -1.0
    w_sum = abs(w_y)+abs(w_x)
    w_y /= w_sum
    w_x /= w_sum

    # Output directory
    out_csv = os.path.join(args.out_dir,'DHI_PHM_trade_readiness.csv')
    out_json = os.path.join(args.out_dir,'DHI_PHM_trade_readiness.json')
    out_png = os.path.join(args.out_dir,'DHI_PHM_spread_chart.png')

    summary = {
        "TRADE_READY": trade_ready,
        "hedge_reco": hedge_reco,
        "direction": direction,
        "weights": {"y": round(w_y,4), "x": round(w_x,4)},
        "exit_rule": "|Z60| <= 0.5",
        "ADF_p_beta": round(adf_p_beta,4),
        "ADF_stat_beta": round(adf_stat_beta,4),
        "ADF_p_cash": round(adf_p_cash,4),
        "ADF_stat_cash": round(adf_stat_cash,4),
        "Half_life_beta": round(hl_beta,1),
        "Half_life_cash": round(hl_cash,1),
        "Z60": round(z60,3),
        "Z30": round(z30,3),
        "dZ3": round(dz3,3),
        "vol_ratio": round(vol_ratio,3),
        "beta_OLS": round(beta_OLS,3),
        "alpha_OLS": round(alpha_OLS,3),
        "beta_stability_pct": round(beta_stability_pct,2)
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(out_csv,index=False)
    df_summary.to_json(out_json, orient='records', indent=2)

    # Wykres
    if args.plot:
        fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)
        axes[0].plot(spread_beta, label='spread_beta')
        axes[0].axhline(spread_beta.rolling(60).mean().iloc[-1], color='green', ls='--', label='MA60')
        axes[0].axhline(spread_beta.rolling(60).mean().iloc[-1]+2*spread_beta.rolling(60).std().iloc[-1], color='red', ls='--')
        axes[0].axhline(spread_beta.rolling(60).mean().iloc[-1]-2*spread_beta.rolling(60).std().iloc[-1], color='red', ls='--')
        axes[0].legend()
        axes[1].plot(z_score(spread_beta,60), label='Z60')
        axes[1].plot(z_score(spread_beta,30), label='Z30')
        axes[1].axhline(2.0,color='red',ls='--')
        axes[1].axhline(1.5,color='orange',ls='--')
        axes[1].axhline(-1.5,color='orange',ls='--')
        axes[1].axhline(-2.0,color='red',ls='--')
        axes[1].axhline(0.5,color='green',ls='--')
        axes[1].axhline(-0.5,color='green',ls='--')
        axes[1].legend()
        axes[2].plot(spread_beta[-250:], label='spread_beta')
        axes[2].plot(spread_cash[-250:], label='spread_cash')
        axes[2].set_title(f"ADF_p_beta={adf_p_beta:.3f}, ADF_p_cash={adf_p_cash:.3f}")
        axes[2].legend()
        plt.tight_layout()
        fig.savefig(out_png)
        plt.close()

    # stdout
    print("\n==== DHI/PHM TRADE READINESS ====")
    for k,v in summary.items():
        print(f"{k}: {v}")

if __name__=="__main__":
    main()
