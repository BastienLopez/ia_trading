import discord
from discord.ext import commands, tasks
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, jsonify
import os
from dotenv import load_dotenv
from threading import Thread

# Charger les variables d'environnement
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

# Flask Server
app = Flask(__name__)

# Configuration du bot Discord
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Pair et URL Investing.com
PAIR = "BTC/USDT"
URL = f"https://www.investing.com/crypto/bitcoin/tether-btc-usdt"

# Récupérer les données de prix depuis Investing.com
def fetch_data():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(URL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Récupérer le prix actuel
    price = float(soup.find("span", {"data-test": "instrument-price-last"}).text.replace(",", ""))
    
    # Simuler le volume (Investing ne l'affiche pas directement ici)
    volume = np.random.uniform(1000, 2000)  # Exemple aléatoire pour simuler
    return {"price": price, "volume": volume}


# Calcul des indicateurs techniques
def calculate_indicators(data):
    df = pd.DataFrame(data)
    
    # EMA
    df['EMA_9'] = df['price'].ewm(span=9).mean()
    df['EMA_21'] = df['price'].ewm(span=21).mean()
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    short_ema = df['price'].ewm(span=12).mean()
    long_ema = df['price'].ewm(span=26).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    
    return df


# Générer des signaux basés sur les indicateurs
def generate_signals(df):
    signals = []
    if df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1] and df['RSI'].iloc[-1] < 30:
        signals.append("BUY")
    if df['EMA_9'].iloc[-1] < df['EMA_21'].iloc[-1] and df['RSI'].iloc[-1] > 70:
        signals.append("SELL")
    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
        signals.append("MACD Bullish Cross")
    if df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1]:
        signals.append("MACD Bearish Cross")
    return signals


# Tâche pour surveiller les signaux
@tasks.loop(minutes=10)
async def check_signals():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    try:
        data = fetch_data()
        df = calculate_indicators(data)
        signals = generate_signals(df)
        for signal in signals:
            await channel.send(f"📊 **Signal détecté pour {PAIR} : {signal}**\n💰 Prix actuel : ${data['price']:.2f}")
    except Exception as e:
        await channel.send(f"Erreur lors de la vérification des signaux : {str(e)}")


# Commande pour forcer la vérification
@bot.command(name="force_signal")
async def force_signal(ctx):
    try:
        data = fetch_data()
        df = calculate_indicators(data)
        signals = generate_signals(df)
        for signal in signals:
            await ctx.send(f"📊 **Signal détecté pour {PAIR} : {signal}**\n💰 Prix actuel : ${data['price']:.2f}")
    except Exception as e:
        await ctx.send(f"Erreur : {str(e)}")


# Serveur Flask
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Bot is running"}), 200


# Lancer le bot et le serveur Flask
if __name__ == "__main__":
    def run_discord_bot():
        bot.run(DISCORD_TOKEN)

    def run_flask_server():
        app.run(port=5000)

    Thread(target=run_discord_bot).start()
    Thread(target=run_flask_server).start()
