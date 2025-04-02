import pandas as pd
import numpy as np
from src.config.config import *

def calculate_rsi(prices, period=RSI_PERIOD):
    """Calcule le RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calcule le MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_ema(prices, period):
    """Calcule l'EMA"""
    return prices.ewm(span=period, adjust=False).mean()

def analyze_signals(df):
    """Analyse les signaux de trading"""
    signals = []
    strength = 0
    
    # RSI
    rsi = calculate_rsi(df['close'])
    if rsi.iloc[-1] < RSI_OVERSOLD:
        signals.append("RSI: SURVENTE")
        strength += 1
    elif rsi.iloc[-1] > RSI_OVERBOUGHT:
        signals.append("RSI: SURACHAT")
        strength += 1
    
    # MACD
    macd, signal = calculate_macd(df['close'])
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        signals.append("MACD: CROISEMENT HAUSSIER")
        strength += 1
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        signals.append("MACD: CROISEMENT BAISSIER")
        strength += 1
    
    # EMA
    ema_short = calculate_ema(df['close'], EMA_SHORT)
    ema_long = calculate_ema(df['close'], EMA_LONG)
    if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]:
        signals.append("EMA: CROISEMENT HAUSSIER")
        strength += 1
    elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]:
        signals.append("EMA: CROISEMENT BAISSIER")
        strength += 1
    
    # Déterminer la force du signal
    if strength >= SIGNAL_STRENGTH['STRONG']:
        signal_strength = "FORT"
    elif strength >= SIGNAL_STRENGTH['MEDIUM']:
        signal_strength = "MOYEN"
    else:
        signal_strength = "FAIBLE"
    
    return signals, signal_strength, {
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'signal_line': signal.iloc[-1],
        'ema_short': ema_short.iloc[-1],
        'ema_long': ema_long.iloc[-1]
    } 