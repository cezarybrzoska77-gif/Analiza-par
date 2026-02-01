#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dhi_phm_trade.py

Analiza pary DHI/PHM wg Workflow 2 (retail):
- Beta neutral i cash neutral spreads
- Half-life, Z-score (60,30), vol ratio
- ADF test
- Beta stability (OLS 60 vs 90)
- Wyznaczenie trybu hedgingu
- Ocena gotowości do wejścia: TRADE_READY YES/NO
- Kierunek: long_y_short_x lub short_y_long_x
- Wyjście przy |Z60| ≤ 0.5
- Raport: CSV, JSON
- Opcjonalnie: wykres PNG

Test akceptacyjny:
python analyze_dhi_phm_trade.py --start-date 2018-01-01 --auto-adjust --out-dir results_workflow2
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

def download_data(tickers, start_date, auto_adjust=True):
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=auto_adjust, threads=1)
    # Obsługa MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            df = df.iloc[:, 0:len(tickers)]
            df.columns = tickers
    elif set(tickers).issubset(df.columns):
        df = df[tickers]
    else:
        raise RuntimeError("Nie znaleziono kolumn cen w pobranych danych")
    return df

def log_returns(df, use_percent=False):
    if use_percent:
        return df.pct_change().dropna()
    else:
        return np.log(df / df.shift(1)).dropna()

def winsorize_series(s, lower=0.01, upper=0.99):
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))

def rolling_ols(y, x, window):
    betas = []
    alphas = []
    for i in range(window, len(y)+1):
        Y = y[i-window:i]
        X = x[i-window:i]
        Xc = sm.add_constant(X)
        model = sm.OLS(Y, Xc).fit()
        alphas.append(model.params[0])
        betas.append(model.params[1])
    return np.array(alphas), np.array(betas)

def half_life(spread):
    delta = spread.diff().dropna().values
    spread_lag = spread.shift(1).dropna().values
    spread_lag = spread_lag[:len(delta)]
    if len(delta)<2:
        return np.nan
    rho = np.polyfit(spread_lag, delta, 1)[0]
    if rho <= -1 or abs(rho) < 1e-5:
        return np.nan
    return -np.log(2) / np.log(1 + rho)

def z_score(spread, window):
    if len(spread) < window:
        return np.nan
    return (spread.iloc[-1] - spread.rolling(window).mean().iloc[-1]) / spread.rolling(window).std().iloc[-1]

def analyze_pair(y, x, z_windows=[60,30], use_percent_returns=False, winsorize=True, out_dir="results_workflow2", plot=True):
    os.makedirs(out_dir, exist_ok=True)
    df = download_data([y, x], start_date="2018-01-01", auto_adjust=True)
    if df.isna().sum().sum() > 0:
        df = df.dropna()
    if len(df) < 90:
        return {"TRADE_READY":"NO", "reason":"Insufficient data (<90)"}    
    ret = log_returns(df, use_percent=use_percent_returns)
    if winsorize:
        ret[y] = winsorize_series(ret[y])
        ret[x] = winsorize_series(ret[x])

    # Beta neutral
    Xc = sm.add_constant(df[x])
    ols_model = sm.OLS(df[y], Xc).fit()
    alpha_OLS = ols_model.params[0]
    beta_OLS = ols_model.params[1]

    # Rolling betas
    _, beta_60 = rolling_ols(ret[y], ret[x], 60)
    _, beta_90 = rolling_ols(ret[y], ret[x], 90)
    if len(beta_60)==0 or len(beta_90)==0:
        return {"TRADE_READY":"NO", "reason":"Not enough rolling data"}
    beta_stability_pct = 100 * abs(beta_60[-1] - beta_90[-1]) / (np.mean([abs(beta_60[-1]), abs(beta_90[-1])]))

    # Spready
    spread_beta = df[y] - (alpha_OLS + beta_OLS*df[x])
    spread_cash = df[y] - df[x]

    # ADF
    try:
        adf_beta = adfuller(spread_beta.dropna())
        adf_cash = adfuller(spread_cash.dropna())
        adf_p_beta, adf_stat_beta = adf_beta[1], adf_beta[0]
        adf_p_cash, adf_stat_cash = adf_cash[1], adf_cash[0]
    except:
        adf_p_beta, adf_stat_beta, adf_p_cash, adf_stat_cash = np.nan, np.nan, np.nan, np.nan

    # Half-life
    hl_beta = half_life(spread_beta)
    hl_cash = half_life(spread_cash)

    # Z-scores
    Zs = {}
    for w in z_windows:
        Zs[w] = z_score(spread_beta, w)
    deltaZ3d = Zs[z_windows[0]] - z_score(spread_beta[:-3], z_windows[0]) if len(spread_beta)>z_windows[0]+3 else np.nan

    # Volatility ratio
    vol_ratio = spread_beta.std() / (ret[y].std() + ret[x].std())

    # Hedging recommendation
    hedge_reco = "beta_neutral"
    if beta_stability_pct >= 40 or (hl_cash<hl_beta and adf_p_cash<adf_p_beta):
        hedge_reco = "cash_neutral"
    elif 25<=beta_stability_pct<=35:
        if not ((adf_p_beta <= adf_p_cash-0.02) or (hl_beta+3 <= hl_cash)):
            hedge_reco = "cash_neutral"

    # ENTRY check
    TRADE_READY = "NO"
    direction = "none"
    exit_rule = "|Z60| ≤ 0.5"
    reasons=[]
    if hedge_reco=="beta_neutral":
        use_spread = spread_beta
        use_adf = adf_p_beta
        use_hl = hl_beta
    else:
        use_spread = spread_cash
        use_adf = adf_p_cash
        use_hl = hl_cash
    if use_adf>0.05:
        reasons.append(f"ADF {use_adf:.3f}>0.05")
    if use_hl>15:
        reasons.append(f"Half-life {use_hl:.1f}>15")
    Z60 = z_score(use_spread,60)
    Z30 = z_score(use_spread,30)
    if Z60 is np.nan or Z30 is np.nan:
        reasons.append("Not enough Z-score data")
    if use_adf<=0.05 and use_hl<=15 and Z60 is not np.nan and Z30 is not np.nan:
        if Z60<=-2 and Z30<=-1.5:
            TRADE_READY="YES"
            direction="long_y_short_x"
        elif Z60>=2 and Z30>=1.5:
            TRADE_READY="YES"
            direction="short_y_long_x"
        else:
            reasons.append("Z-score thresholds not met")

    # Sizes
    if TRADE_READY=="YES":
        if hedge_reco=="beta_neutral":
            size_y = 1
            size_x = -beta_OLS
        else:
            size_y = 1
            size_x = -1
        norm = abs(size_y)+abs(size_x)
        size_y/=norm
        size_x/=norm
        sizes = {"y":size_y,"x":size_x}
    else:
        sizes = {"y":0,"x":0}

    # Prepare output
    output = {
        "y": y,
        "x": x,
        "alpha_OLS": round(alpha_OLS,4),
        "beta_OLS": round(beta_OLS,4),
        "beta_stability_pct": round(beta_stability_pct,2),
        "half_life_beta": round(hl_beta,1),
        "half_life_cash": round(hl_cash,1),
        "ADF_p_beta": round(adf_p_beta,4),
        "ADF_stat_beta": round(adf_stat_beta,4),
        "ADF_p_cash": round(adf_p_cash,4),
        "ADF_stat_cash": round(adf_stat_cash,4),
        "Z60": round(Z60,4),
        "Z30": round(Z30,4),
        "deltaZ3d": round(deltaZ3d,4),
        "vol_ratio": round(vol_ratio,4),
        "hedge_reco": hedge_reco,
        "TRADE_READY": TRADE_READY,
        "direction": direction,
        "sizes": sizes,
        "exit_rule": exit_rule,
        "reasons": reasons
    }

    # Save CSV & JSON
    out_csv = os.path.join(out_dir,f"{y}_{x}_trade_readiness.csv")
    out_json = os.path.join(out_dir,f"{y}_{x}_trade_readiness.json")
    pd.DataFrame([output]).to_csv(out_csv,index=False)
    with open(out_json,"w") as f:
        json.dump(output,f,indent=4)

    # Plot
    if plot:
        plt.figure(figsize=(12,8))
        ax1=plt.subplot(3,1,1)
        spread_beta.plot(ax=ax1,label="spread_beta")
        spread_beta.rolling(60).mean().plot(ax=ax1,label="roll_mean_60")
        ax1.fill_between(spread_beta.index,
                         spread_beta.rolling(60).mean()-2*spread_beta.rolling(60).std(),
                         spread_beta.rolling(60).mean()+2*spread_beta.rolling(60).std(),
                         color='gray',alpha=0.3)
        ax1.set_title(f"{y}-{x} spread_beta")
        ax1.legend()
        ax2=plt.subplot(3,1,2)
        pd.Series(Z60,index=use_spread.index[-len(Z60):]).plot(ax=ax2,label="Z60")
        pd.Series(Z30,index=use_spread.index[-len(Z30):]).plot(ax=ax2,label="Z30")
        ax2.axhline(2,color='red',linestyle="--")
        ax2.axhline(1.5,color='orange',linestyle="--")
        ax2.axhline(-1.5,color='orange',linestyle="--")
        ax2.axhline(-2,color='red',linestyle="--")
        ax2.axhline(0.5,color='green',linestyle="--")
        ax2.axhline(-0.5,color='green',linestyle="--")
        ax2.set_title("Z-scores")
        ax2.legend()
        ax3=plt.subplot(3,1,3)
        spread_beta[-250:].plot(ax=ax3,label="spread_beta")
        spread_cash[-250:].plot(ax=ax3,label="spread_cash")
        ax3.set_title("Spread beta vs cash (last 250)")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"{y}_{x}_spread_chart.png"))
        plt.close()

    # Print summary
    print(f"TRADE_READY: {TRADE_READY}")
    print(f"hedge_reco: {hedge_reco}")
    print(f"direction: {direction}")
    print(f"sizes: {sizes}")
    print(f"exit_rule: {exit_rule}")
    if reasons:
        print("Reasons:", reasons)
    return output

def print_checklist():
    print("""
Checklist dla DHI/PHM:
1) ADF_p_beta <= 0.05
2) Half-life_beta <= 15
3) Z60 <= -2.0 i Z30 <= -1.5 → LONG / Z60 >=2 i Z30>=1.5 → SHORT
4) Beta stability <=25% lub 25-35% tylko przy lepszych metrykach beta
5) Wyjście: |Z60| ≤ 0.5
""")

def main():
    parser = argparse.ArgumentParser(description="Analyze DHI/PHM trade readiness")
    parser.add_argument("--start-date",default="2018-01-01")
    parser.add_argument("--auto-adjust",dest="auto_adjust",action="store_true")
    parser.add_argument("--no-auto-adjust",dest="auto_adjust",action="store_false")
    parser.set_defaults(auto_adjust=True)
    parser.add_argument("--use-percent-returns",action="store_true")
    parser.add_argument("--winsorize",action="store_true")
    parser.add_argument("--no-winsorize",dest="winsorize",action="store_false")
    parser.set_defaults(winsorize=True)
    parser.add_argument("--z-lookbacks",default="60,30")
    parser.add_argument("--min-sample",type=int,default=200)
    parser.add_argument("--out-dir",default="results_workflow2")
    parser.add_argument("--plot",dest="plot",action="store_true")
    parser.add_argument("--no-plot",dest="plot",action="store_false")
    parser.set_defaults(plot=True)
    parser.add_argument("--help-checklist",action="store_true")
    args = parser.parse_args()
    if args.help_checklist:
        print_checklist()
        sys.exit(0)
    analyze_pair("DHI","PHM",z_windows=[int(w) for w in args.z_lookbacks.split(",")],
                 use_percent_returns=args.use_percent_returns,
                 winsorize=args.winsorize,
                 out_dir=args.out_dir,
                 plot=args.plot)

if __name__=="__main__":
    main()
