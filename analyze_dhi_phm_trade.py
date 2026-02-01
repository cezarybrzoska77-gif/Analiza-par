#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dhi_phm_trade.py

Workflow 2 - Retail Trade Readiness for DHI / PHM

RULES & PROGOWE KRITERIA:

1. Dane:
   - Start-date: 2018-01-01
   - auto-adjust ON (Close adjusted by Yahoo Finance)
   - Domyślnie log-returns, opcjonalnie percent-returns (--use-percent-returns)
   - Winsoryzacja 1-99% domyślnie ON (--no-winsorize wyłącza)
   
2. Beta neutral:
   - OLS: y_t = alpha + beta * x_t
   - Beta stability: beta_60 vs beta_90
   - Spready: spread_beta = y - (alpha+beta*x), spread_cash = y - x
   - Half-life na spread_beta

3. Z-score:
   - Z60 i Z30 na spread_beta
   - ΔZ3d = Z(t) - Z(t-3) (informacyjnie)

4. Volatility ratio: std(spread_beta) / (std(ret_y) + std(ret_x))

5. Tryb hedgingu:
   - beta_stability_pct ≤25% -> beta_neutral
   - beta_stability_pct ≥40% -> cash_neutral
   - 25-35% borderline: wybierz beta_neutral tylko jeśli ADF/HL beta lepsze

6. Kryteria wejścia (ENTRY) dla TRADE_READY=YES:
   - ADF p-value ≤0.05
   - Half-life ≤15
   - Z60 & Z30 zgodne z LONG/SHORT (-2.0,-1.5 lub 2.0,1.5)
   - Beta stability w przedziale dopuszczalnym

7. Wyjście: |Z60| ≤0.5

8. Output:
   - CSV + JSON: results_workflow2/DHI_PHM_trade_readiness.*
   - Opcjonalny wykres PNG: results_workflow2/DHI_PHM_spread_chart.png

TEST:
python analyze_dhi_phm_trade.py --start-date 2018-01-01 --auto-adjust --out-dir results_workflow2
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def download_data(tickers, start_date):
    try:
        df = yf.download(tickers, start=start_date, auto_adjust=True)["Close"]
        df = df.rename(columns={tickers[0]:"y", tickers[1]:"x"})
        return df
    except Exception as e:
        print(f"ERROR downloading tickers: {e}")
        return None

def compute_returns(df, use_percent=False, winsorize=True):
    ret = df.pct_change() if use_percent else np.log(df / df.shift(1))
    ret = ret.dropna()
    if winsorize:
        ret = ret.clip(lower=ret.quantile(0.01), upper=ret.quantile(0.99))
    return ret

def rolling_ols(y, x, window):
    betas = []
    for i in range(window, len(y)+1):
        y_win = y[i-window:i]
        x_win = x[i-window:i]
        X = add_constant(x_win)
        model = OLS(y_win, X).fit()
        betas.append(model.params[1])
    if len(betas) == 0:
        return np.nan
    return np.nanmean(betas)

def half_life(spread):
    spread = spread.dropna()
    delta = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    common_index = delta.index.intersection(spread_lag.index)
    delta = delta.loc[common_index]
    spread_lag = spread_lag.loc[common_index]
    if len(delta) < 10:
        return np.nan
    rho = np.corrcoef(spread_lag, delta)[0,1]
    if np.isnan(rho) or abs(rho) < 1e-6 or (1+rho)<=0:
        return np.nan
    try:
        hl = -np.log(2)/np.log(1+rho)
        return hl
    except:
        return np.nan

def zscore(series, window):
    if len(series) < window:
        return np.nan
    mean = series[-window:].mean()
    std = series[-window:].std()
    if std == 0:
        return 0.0
    return (series.iloc[-1]-mean)/std

def adf_test(series):
    try:
        adf_res = adfuller(series, maxlag=1, autolag=None)
        return adf_res[1], adf_res[0]  # p-value, test stat
    except:
        return np.nan, np.nan

def beta_stability(ret_y, ret_x):
    beta_60 = rolling_ols(ret_y, ret_x, 60)
    beta_90 = rolling_ols(ret_y, ret_x, 90)
    if np.isnan(beta_60) or np.isnan(beta_90) or (abs(beta_60)+abs(beta_90))==0:
        return np.nan, beta_60, beta_90
    pct = 100*abs(beta_60-beta_90)/((abs(beta_60)+abs(beta_90))/2)
    return pct, beta_60, beta_90

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--auto-adjust", action="store_true", default=True)
    parser.add_argument("--use-percent-returns", action="store_true", default=False)
    parser.add_argument("--winsorize", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-dir", default="results_workflow2")
    parser.add_argument("--help-checklist", action="store_true")
    args = parser.parse_args()

    if args.help_checklist:
        print("""
Checklist TRADE_READY criteria:
- ADF p-value <=0.05
- Half-life <=15
- Z60/Z30 for LONG or SHORT (-2,-1.5 or 2,1.5)
- Beta stability <=25% or acceptable
- Output: CSV, JSON, optional PNG
""")
        sys.exit(0)

    tickers = ["DHI","PHM"]
    df = download_data(tickers, args.start_date)
    if df is None or df.empty:
        print("ERROR: no data, TRADE_READY=NO")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    ret = compute_returns(df, use_percent=args.use_percent_returns, winsorize=args.winsorize)

    # Beta neutral OLS on levels
    X = add_constant(df['x'])
    model = OLS(df['y'], X).fit()
    alpha_OLS, beta_OLS = model.params
    df['spread_beta'] = df['y'] - (alpha_OLS + beta_OLS * df['x'])
    df['spread_cash'] = df['y'] - df['x']

    adf_p_beta, adf_stat_beta = adf_test(df['spread_beta'])
    adf_p_cash, adf_stat_cash = adf_test(df['spread_cash'])

    hl_beta = half_life(df['spread_beta'])
    hl_cash = half_life(df['spread_cash'])

    Z60 = zscore(df['spread_beta'], 60)
    Z30 = zscore(df['spread_beta'], 30)
    if len(df) >=4:
        Z3d_delta = Z60 - zscore(df['spread_beta'].iloc[:-3], 60)  # informacyjnie
    else:
        Z3d_delta = np.nan

    vol_ratio = df['spread_beta'].std()/ (ret['y'].std() + ret['x'].std())

    beta_stab_pct, beta_60, beta_90 = beta_stability(ret['y'], ret['x'])

    # Hedge recommendation
    hedge = "beta_neutral"
    if beta_stab_pct >=40:
        hedge="cash_neutral"
    elif 25<beta_stab_pct<35:
        if (adf_p_beta > adf_p_cash+0.02) or (hl_beta>hl_cash):
            hedge="cash_neutral"

    # Entry check
    trade_ready = False
    direction = "none"
    exit_rule = "|Z60|<=0.5"
    reason = []

    if hedge=="beta_neutral":
        if adf_p_beta>0.05:
            reason.append(f"ADF_beta {adf_p_beta:.3f} > 0.05")
        if hl_beta>15:
            reason.append(f"Half-life_beta {hl_beta:.1f} > 15")
        if not (-2>=Z60<=-1.5 or 2<=Z60<=1.5):
            reason.append(f"Z60 {Z60:.3f} not triggering")
        if beta_stab_pct>35:
            reason.append(f"Beta stability {beta_stab_pct:.1f}% too high")
        trade_ready = len(reason)==0
    else:
        if adf_p_cash>0.05:
            reason.append(f"ADF_cash {adf_p_cash:.3f} > 0.05")
        if hl_cash>15:
            reason.append(f"Half-life_cash {hl_cash:.1f} > 15")
        trade_ready = len(reason)==0

    if trade_ready:
        if Z60>=2.0:
            direction="short_y_long_x"
        elif Z60<=-2.0:
            direction="long_y_short_x"
        else:
            direction="none"
            trade_ready=False

    # Sizes
    if hedge=="beta_neutral" and trade_ready:
        w_y = 1.0
        w_x = -beta_OLS
        norm = abs(w_y)+abs(w_x)
        w_y/=norm
        w_x/=norm
    elif hedge=="cash_neutral" and trade_ready:
        w_y=1.0
        w_x=-1.0
    else:
        w_y=w_x=0.0

    result = {
        "TRADE_READY": "YES" if trade_ready else "NO",
        "hedge_reco": hedge,
        "direction": direction,
        "weights_y": round(w_y,3),
        "weights_x": round(w_x,3),
        "exit_rule": exit_rule,
        "alpha_OLS": round(alpha_OLS,4),
        "beta_OLS": round(beta_OLS,4),
        "beta_stability_pct": round(beta_stab_pct,2),
        "ADF_p_beta": round(adf_p_beta,4),
        "ADF_p_cash": round(adf_p_cash,4),
        "Half_life_beta": round(hl_beta,1),
        "Half_life_cash": round(hl_cash,1),
        "Z60": round(Z60,3),
        "Z30": round(Z30,3),
        "Z3d_delta": round(Z3d_delta,3),
        "vol_ratio": round(vol_ratio,3),
        "reason_not_ready": reason
    }

    # Save CSV & JSON
    df_csv = pd.DataFrame([result])
    csv_path = os.path.join(args.out_dir,"DHI_PHM_trade_readiness.csv")
    json_path = os.path.join(args.out_dir,"DHI_PHM_trade_readiness.json")
    df_csv.to_csv(csv_path, index=False)
    with open(json_path,"w") as f:
        json.dump(result,f,indent=2)

    # Optional plot
    if not args.no_plot:
        png_path = os.path.join(args.out_dir,"DHI_PHM_spread_chart.png")
        fig, axes = plt.subplots(3,1, figsize=(12,12))
        axes[0].plot(df['spread_beta'], label="spread_beta")
        axes[0].axhline(df['spread_beta'].rolling(60).mean(),color='green',ls='--',label="MA60")
        axes[0].axhline(df['spread_beta'].rolling(60).mean()+2*df['spread_beta'].rolling(60).std(),color='red',ls=':',label="+2σ")
        axes[0].axhline(df['spread_beta'].rolling(60).mean()-2*df['spread_beta'].rolling(60).std(),color='red',ls=':',label="-2σ")
        axes[0].set_title("Spread Beta")
        axes[0].legend()
        axes[1].plot([Z60]*len(df),label="Z60")
        axes[1].plot([Z30]*len(df),label="Z30")
        axes[1].axhline(2,color='red',ls=':')
        axes[1].axhline(-2,color='red',ls=':')
        axes[1].axhline(1.5,color='orange',ls=':')
        axes[1].axhline(-1.5,color='orange',ls=':')
        axes[1].axhline(0.5,color='green',ls='--')
        axes[1].axhline(-0.5,color='green',ls='--')
        axes[1].set_title("Z-scores")
        axes[1].legend()
        axes[2].plot(df['spread_beta'].iloc[-250:],label="spread_beta")
        axes[2].plot(df['spread_cash'].iloc[-250:],label="spread_cash")
        axes[2].set_title(f"Beta vs Cash Spread last 250 sessions\nADF_p_beta={adf_p_beta:.3f},ADF_p_cash={adf_p_cash:.3f}")
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

    # Console summary
    print(f"\nTRADE_READY = {result['TRADE_READY']}")
    print(f"Hedge recommendation = {result['hedge_reco']}")
    print(f"Direction = {result['direction']}")
    print(f"Weights: y={result['weights_y']}, x={result['weights_x']}")
    print(f"Exit rule = {result['exit_rule']}")
    if not trade_ready:
        print("Reasons not ready:", "; ".join(reason))

if __name__=="__main__":
    main()
