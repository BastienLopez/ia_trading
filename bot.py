import discord
from discord.ext import commands, tasks
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, jsonify
import os
from dotenv import load_dotenv
import asyncio

# Charger les variables d'environnement
load_dotenv()

# Récupération des variables nécessaires depuis .env
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

# Initialisation du serveur Flask
app = Flask(__name__)

# Configuration du bot Discord
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Pair et URL pour récupérer les données
PAIR = "BTC/USDT"
URL = f"https://www.investing.com/crypto/bitcoin/tether-btc-usdt"


# Fonction pour récupérer les données de prix depuis Investing.com
def fetch_data():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(URL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Vérifiez si le prix est présent
    price_element = soup.find("span", {"data-test": "instrument-price-last"})
    if price_element is None:
        raise ValueError("Impossible de trouver le prix sur la page.")
    
    # Récupérer le prix actuel
    price = float(price_element.text.replace(",", ""))
    
    # Simuler le volume (Investing ne l'affiche pas directement ici)
    volume = np.random.uniform(1000, 2000)  # Exemple aléatoire pour simuler
    print(f"Prix récupéré : {price}, Volume simulé : {volume}")
    return {"price": price, "volume": volume}


# Fonction pour calculer les indicateurs techniques
def calculate_indicators(df):
    # EMA (Exponential Moving Average)
    df['EMA_9'] = df['price'].ewm(span=9).mean()
    df['EMA_21'] = df['price'].ewm(span=21).mean()
    
    # RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD (Moving Average Convergence Divergence)
    short_ema = df['price'].ewm(span=12).mean()
    long_ema = df['price'].ewm(span=26).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    print("Indicateurs calculés :")
    print(df.tail())
    return df


# Fonction pour générer les signaux à partir des indicateurs
def generate_signals(df):
    signals = []

    # Signaux EMA + RSI
    if df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1] and df['RSI'].iloc[-1] < 30:
        signals.append("BUY")
    if df['EMA_9'].iloc[-1] < df['EMA_21'].iloc[-1] and df['RSI'].iloc[-1] > 70:
        signals.append("SELL")

    # Signaux MACD
    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
        signals.append("MACD Bullish Cross")
    if df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1]:
        signals.append("MACD Bearish Cross")

    print(f"Signaux générés : {signals}")
    return signals


# Commande pour afficher les signaux actuels
@bot.command(name="force_signal")
async def force_signal(ctx):
    # Récupérez le channel configuré
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    print(f"Channel configuré récupéré : {channel}")  # Debugging

    if ctx.channel.id != DISCORD_CHANNEL_ID:
        await ctx.send("⚠️ Utilisez cette commande dans le channel configuré.")
        return

    try:
        print("Commande force_signal appelée")
        
        # Récupérer les données actuelles
        data = fetch_data()
        print(f"Données récupérées : {data}")
        
        # Créer un DataFrame et calculer les indicateurs
        df = pd.DataFrame([data])
        df = calculate_indicators(df)
        
        # Générer les signaux
        signals = generate_signals(df)
        
        # Envoi des signaux dans le channel configuré
        if not signals:
            await channel.send("⚠️ Aucun signal détecté pour le moment.")
        else:
            for signal in signals:
                await channel.send(
                    f"📊 **Signal détecté pour {PAIR} : {signal}**\n💰 Prix actuel : ${data['price']:.2f}"
                )
    except Exception as e:
        print(f"Erreur : {e}")
        await channel.send(f"Erreur : {str(e)}")


# Commande pour vérifier que le bot fonctionne
@bot.command(name="log")
async def log(ctx):
    # Récupérez le channel Discord configuré
    channel = bot.get_channel(DISCORD_CHANNEL_ID)

    if ctx.channel.id != DISCORD_CHANNEL_ID:
        await ctx.send("⚠️ Utilisez cette commande dans le channel configuré.")
        return

    print("Commande log appelée")
    await channel.send("✅ Le bot est bien connecté et opérationnel !")

# Commande pour vérifier que le bot peut repondre 
@bot.event
async def on_ready():
    print(f"Bot connecté en tant que {bot.user}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    print(f"Channel configuré : {channel}")  # Debugging
    if channel:
        await channel.send("✅ Le bot est connecté et prêt à répondre !")


# Route Flask pour vérifier l'état du bot
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Bot is running"}), 200


# Lancer Discord et Flask avec asyncio
async def main():
    loop = asyncio.get_event_loop()
    discord_task = loop.run_in_executor(None, bot.run, DISCORD_TOKEN)
    flask_task = loop.run_in_executor(None, app.run, "0.0.0.0", 5000)
    await asyncio.gather(discord_task, flask_task)

if __name__ == "__main__":
    asyncio.run(main())
