#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workflow 2 - Retail pair trading check dla DHI/PHM

Cel: ocenić, czy para DHI/PHM jest gotowa do otwarcia pozycji w retail mode.
Reguły progowe:
- Spread beta vs cash
- ADF ≤ 0.05
- Half-life ≤ 15
- Z-score dual window: Z60 ≥2/≤-2 i Z30 ≥1.5/≤-1.5
- Beta stability ≤25% lub borderline z poprawionym ADF/HL
- Decyzja TRADE_READY: YES/NO
- Kierunek: long_y_short_x / short_y_long_x
- Hedge recommendation: beta_neutral / cash_neutral
- Rozmiary nóg: proporcje wag lub 1:1 w cash mode
- Wyjście: |Z60| ≤0.5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def download_data(tickers, start_date, auto_adjust=True):
    df = yf.download(tickers, start=start_date, auto_adjust=auto_adjust)
    if "Adj Close" in df.columns:
        df = df["Adj Close"]
    else:
        # w nowszych wersjach yfinance 'Close' może być wystarczające
        df = df['Close']
    df = df.dropna()
    return df

def log_returns(df, use_percent=False, winsorize=True):
    if use_percent:
        ret = df.pct_change().dropna()
    else:
        ret = np.log(df / df.shift(1)).dropna()
    if winsorize:
        lower = ret.quantile(0.01, axis=0)
        upper = ret.quantile(0.99, axis=0)
        ret = ret.clip(lower=lower, upper=upper, axis=1)
    return ret

def rolling_beta(y, x, window):
    beta_list = []
    for i in range(len(y)-window+1):
        Y = y[i:i+window]
        X = x[i:i+window]
        model = OLS(Y, add_constant(X)).fit()
        beta_list.append(model.params[1])
    return np.array(beta_list)

def half_life(spread):
    delta_s = spread.diff().dropna()
    s_lag = spread.shift(1).dropna()
    delta_s = delta_s[s_lag.index]
    s_lag = s_lag
    model = OLS(delta_s.values, s_lag.values).fit()
    rho = model.params[0]
    try:
        hl = -np.log(2)/np.log(1 + rho)
        if hl <= 0 or np.isnan(hl):
            hl = np.nan
    except:
        hl = np.nan
    return hl

def z_score(series, window):
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    z = (series - mu)/sigma
    return z

def compute_metrics(df, y, x, z_lookbacks=(60,30)):
    y_prices = df[y]
    x_prices = df[x]
    # OLS beta neutral
    model_ols = OLS(y_prices, add_constant(x_prices)).fit()
    alpha_OLS = model_ols.params[0]
    beta_OLS = model_ols.params[1]
    # Spready
    spread_beta = y_prices - (alpha_OLS + beta_OLS * x_prices)
    spread_cash = y_prices - x_prices
    # Rolling betas
    beta_60 = rolling_beta(y_prices.values, x_prices.values, 60)[-1]
    beta_90 = rolling_beta(y_prices.values, x_prices.values, 90)[-1]
    beta_stability_pct = 100 * abs(beta_60 - beta_90) / np.mean([abs(beta_60), abs(beta_90)])
    # ADF
    adf_beta = adfuller(spread_beta.dropna())
    adf_cash = adfuller(spread_cash.dropna())
    adf_p_beta, adf_stat_beta = adf_beta[1], adf_beta[0]
    adf_p_cash, adf_stat_cash = adf_cash[1], adf_cash[0]
    # Half-life
    hl_beta = half_life(spread_beta)
    hl_cash = half_life(spread_cash)
    # Z-scores
    z60 = z_score(spread_beta, z_lookbacks[0])
    z30 = z_score(spread_beta, z_lookbacks[1])
    z60_last = z60.iloc[-1]
    z30_last = z30.iloc[-1]
    delta_z3d = z60.iloc[-1] - z60.iloc[-4] if len(z60) > 3 else np.nan
    # Volatility ratio
    ret_y = np.log(y_prices/y_prices.shift(1)).dropna()
    ret_x = np.log(x_prices/x_prices.shift(1)).dropna()
    vol_ratio = spread_beta.std() / (ret_y.std() + ret_x.std())
    return {
        'alpha_OLS': alpha_OLS,
        'beta_OLS': beta_OLS,
        'beta_60': beta_60,
        'beta_90': beta_90,
        'beta_stability_pct': beta_stability_pct,
        'spread_beta': spread_beta,
        'spread_cash': spread_cash,
        'adf_p_beta': adf_p_beta,
        'adf_stat_beta': adf_stat_beta,
        'adf_p_cash': adf_p_cash,
        'adf_stat_cash': adf_stat_cash,
        'hl_beta': hl_beta,
        'hl_cash': hl_cash,
        'z60': z60,
        'z30': z30,
        'z60_last': z60_last,
        'z30_last': z30_last,
        'delta_z3d': delta_z3d,
        'vol_ratio': vol_ratio
    }

def decide_hedge(metrics):
    if metrics['beta_stability_pct'] <= 25 or (metrics['adf_p_beta'] <= metrics['adf_p_cash'] and metrics['hl_beta'] <= metrics['hl_cash']):
        return 'beta_neutral'
    elif metrics['beta_stability_pct'] >= 40 or (metrics['adf_p_cash'] < metrics['adf_p_beta'] and metrics['hl_cash'] < metrics['hl_beta']):
        return 'cash_neutral'
    else:
        # borderline
        if metrics['adf_p_beta'] <= metrics['adf_p_cash'] - 0.02 or metrics['hl_beta'] < metrics['hl_cash'] - 3:
            return 'beta_neutral'
        else:
            return 'cash_neutral'

def check_entry(metrics, hedge_mode):
    adf_ok = (metrics['adf_p_beta'] <= 0.05 if hedge_mode=='beta_neutral' else metrics['adf_p_cash']<=0.05)
    hl_ok = (metrics['hl_beta'] <= 15 if hedge_mode=='beta_neutral' else metrics['hl_cash']<=15)
    z_ok_long = metrics['z60_last'] <= -2 and metrics['z30_last'] <= -1.5
    z_ok_short = metrics['z60_last'] >= 2 and metrics['z30_last'] >= 1.5
    trade_ready = adf_ok and hl_ok and (z_ok_long or z_ok_short)
    direction = 'long_y_short_x' if z_ok_long else 'short_y_long_x' if z_ok_short else 'none'
    return trade_ready, direction

def calculate_sizes(metrics, hedge_mode):
    if hedge_mode=='beta_neutral':
        w_y = 1.0
        w_x = -metrics['beta_OLS']
        norm = abs(w_y) + abs(w_x)
        w_y /= norm
        w_x /= norm
        return {'y_weight': w_y, 'x_weight': w_x}
    else:
        return {'y_weight': 1.0, 'x_weight': -1.0}

def plot_spreads(metrics, out_dir, y, x):
    fig, axes = plt.subplots(3,1, figsize=(12,12), sharex=True)
    spread_beta = metrics['spread_beta']
    spread_cash = metrics['spread_cash']
    # Panel 1: spread beta + rolling ±2 sigma
    ma60 = spread_beta.rolling(60).mean()
    sigma60 = spread_beta.rolling(60).std()
    axes[0].plot(spread_beta.index, spread_beta, label='spread_beta')
    axes[0].plot(ma60.index, ma60, '--', color='green', label='MA60')
    axes[0].fill_between(ma60.index, ma60-2*sigma60, ma60+2*sigma60, color='green', alpha=0.2)
    axes[0].legend(); axes[0].set_title('Spread Beta & Rolling MA ±2σ')
    # Panel 2: Z-scores
    axes[1].plot(metrics['z60'].index, metrics['z60'], label='Z60')
    axes[1].plot(metrics['z30'].index, metrics['z30'], label='Z30')
    axes[1].axhline(2, color='red', ls='--'); axes[1].axhline(1.5, color='orange', ls='--')
    axes[1].axhline(-1.5, color='orange', ls='--'); axes[1].axhline(-2, color='red', ls='--')
    axes[1].axhline(0.5, color='gray', ls=':')
    axes[1].axhline(-0.5, color='gray', ls=':')
    axes[1].legend(); axes[1].set_title('Z-scores')
    # Panel3: spread beta vs cash last 250
    axes[2].plot(spread_beta.index[-250:], spread_beta[-250:], label='spread_beta')
    axes[2].plot(spread_cash.index[-250:], spread_cash[-250:], label='spread_cash')
    axes[2].legend(); axes[2].set_title('Spread Beta vs Cash')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{y}_{x}_spread_chart.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2018-01-01')
    parser.add_argument('--auto-adjust', action='store_true', default=True)
    parser.add_argument('--no-auto-adjust', dest='auto_adjust', action='store_false')
    parser.add_argument('--use-percent-returns', action='store_true', default=False)
    parser.add_argument('--winsorize', action='store_true', default=True)
    parser.add_argument('--no-winsorize', dest='winsorize', action='store_false')
    parser.add_argument('--z-lookbacks', default='60,30')
    parser.add_argument('--min-sample', type=int, default=200)
    parser.add_argument('--out-dir', default='results_workflow2')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--help-checklist', action='store_true', help='Show entry/exit checklist only')
    args = parser.parse_args()

    y, x = 'DHI','PHM'

    if args.help_checklist:
        print("Checklist for TRADE_READY:")
        print("1. ADF p-value <= 0.05")
        print("2. Half-life <= 15")
        print("3. Z60 and Z30 aligned: LONG Z60<=-2 & Z30<=-1.5, SHORT Z60>=2 & Z30>=1.5")
        print("4. Beta stability <=25% (or borderline with better beta metrics)")
        print("5. Hedge mode decision: beta_neutral or cash_neutral")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    df = download_data([y,x], args.start_date, args.auto_adjust)
    ret = log_returns(df, args.use_percent_returns, args.winsorize)
    metrics = compute_metrics(df, y, x, tuple(map(int,args.z_lookbacks.split(','))))
    hedge_mode = decide_hedge(metrics)
    trade_ready, direction = check_entry(metrics, hedge_mode)
    sizes = calculate_sizes(metrics, hedge_mode)

    # Output CSV + JSON
    out_csv = os.path.join(args.out_dir, f'{y}_{x}_trade_readiness.csv')
    out_json = os.path.join(args.out_dir, f'{y}_{x}_trade_readiness.json')
    df_out = pd.DataFrame({
        'TRADE_READY':[trade_ready],
        'hedge_mode':[hedge_mode],
        'direction':[direction],
        'sizes':[sizes],
        'exit_rule':['|Z60| <= 0.5'],
        'adf_p_beta':[metrics['adf_p_beta']],
        'hl_beta':[metrics['hl_beta']],
        'beta_stability_pct':[metrics['beta_stability_pct']],
        'z60':[metrics['z60_last']],
        'z30':[metrics['z30_last']],
        'delta_z3d':[metrics['delta_z3d']],
        'vol_ratio':[metrics['vol_ratio']]
    })
    df_out.to_csv(out_csv, index=False)
    df_out.to_json(out_json, orient='records', indent=2)

    # Plot
    if args.plot:
        plot_spreads(metrics, args.out_dir, y, x)

    # Print summary
    print("TRADE_READY:", "YES" if trade_ready else "NO")
    print("hedge_reco:", hedge_mode)
    print("direction:", direction)
    print("sizes:", sizes)
    print("exit_rule: |Z60| <=0.5")
    print("Metrics summary:")
    print(f"ADF p-value beta: {metrics['adf_p_beta']:.4f}")
    print(f"Half-life beta: {metrics['hl_beta']:.1f}")
    print(f"Beta stability %: {metrics['beta_stability_pct']:.2f}")
    print(f"Z60: {metrics['z60_last']:.3f}, Z30: {metrics['z30_last']:.3f}, ΔZ3d: {metrics['delta_z3d']:.3f}")
    print(f"Volatility ratio: {metrics['vol_ratio']:.3f}")

if __name__ == "__main__":
    main()
