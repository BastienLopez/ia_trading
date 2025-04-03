import os
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/discord_bot.log", mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("discord_bot")

# Charger les variables d'environnement
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Configuration du bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Fonction pour envoyer des requêtes à l'API
async def api_request(endpoint, method="GET", data=None):
    """Envoie une requête à l'API trading"""
    url = f"{API_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Erreur API {response.status_code}: {response.text}")
            return {
                "error": True, 
                "message": f"Erreur API: {response.status_code}", 
                "details": response.text
            }
    except Exception as e:
        logger.error(f"Erreur de connexion à l'API: {str(e)}")
        return {"error": True, "message": f"Erreur de connexion: {str(e)}"}

# Événements du bot
@bot.event
async def on_ready():
    """Exécuté quand le bot est connecté"""
    logger.info(f'{bot.user.name} est connecté au serveur Discord!')
    
    # Créer les dossiers nécessaires
    os.makedirs("logs", exist_ok=True)
    
    # Démarrer la tâche périodique
    check_signals.start()

# Tâche périodique pour vérifier les signaux
@tasks.loop(minutes=30)
async def check_signals():
    """Vérifie les signaux de trading toutes les 30 minutes"""
    # Canaux où envoyer les alertes
    signal_channels = {}
    
    # Pour chaque paire configurée, vérifier les signaux
    pairs = [("BTC/USDT", "1h"), ("ETH/USDT", "1h")]
    
    for symbol, timeframe in pairs:
        try:
            # Vérifier si un canal est configuré pour cette paire
            channel_id = signal_channels.get(f"{symbol}_{timeframe}")
            if not channel_id:
                continue
            
            channel = bot.get_channel(channel_id)
            if not channel:
                logger.warning(f"Canal {channel_id} non trouvé pour {symbol}_{timeframe}")
                continue
            
            # Obtenir une prédiction de l'API
            prediction_data = {
                "exchange": "binance",
                "symbol": symbol,
                "timeframe": timeframe,
                "use_latest": True,
                "days": 1
            }
            
            result = await api_request("predict", method="POST", data=prediction_data)
            
            if result.get("error"):
                await channel.send(f"⚠️ Erreur lors de la vérification des signaux pour {symbol}: {result.get('message')}")
                continue
            
            # Envoyer une alerte si une action est recommandée
            if result.get("recommendation") != "ATTENDRE":
                emoji = "🟢" if result.get("recommendation") == "ACHETER" else "🔴"
                
                embed = discord.Embed(
                    title=f"{emoji} Signal de trading pour {symbol}",
                    description=f"**Recommandation: {result.get('recommendation')}**",
                    color=discord.Color.green() if result.get("recommendation") == "ACHETER" else discord.Color.red()
                )
                
                # Ajouter les détails du marché
                market_data = result.get("market_data", {})
                embed.add_field(name="Prix", value=f"${market_data.get('price', 0):.2f}", inline=True)
                embed.add_field(name="RSI", value=f"{market_data.get('rsi', 0):.2f}", inline=True)
                embed.add_field(name="MACD", value=f"{market_data.get('macd', 0):.4f}", inline=True)
                
                # Ajouter l'horodatage
                embed.set_footer(text=f"Signal généré le {result.get('timestamp')}")
                
                await channel.send(embed=embed)
                logger.info(f"Signal envoyé pour {symbol}: {result.get('recommendation')}")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des signaux pour {symbol}: {str(e)}")

# Commandes du bot
@bot.command(name='aide', help='Affiche la liste des commandes disponibles')
async def help_command(ctx):
    """Affiche l'aide et la liste des commandes"""
    embed = discord.Embed(
        title="Commandes du Bot Trading",
        description="Voici la liste des commandes disponibles:",
        color=discord.Color.blue()
    )
    
    embed.add_field(name="!aide", value="Affiche cette liste de commandes", inline=False)
    embed.add_field(name="!prix <symbole>", value="Affiche le prix actuel d'une crypto (ex: BTC, ETH)", inline=False)
    embed.add_field(name="!predire <symbole> <timeframe>", value="Prédit l'action à prendre (ex: BTC/USDT 1h)", inline=False)
    embed.add_field(name="!backtest <symbole> <timeframe> <jours>", value="Effectue un backtest (ex: BTC/USDT 1h 30)", inline=False)
    embed.add_field(name="!entrainer <symbole> <timeframe> <jours>", value="Entraîne un nouveau modèle (ex: BTC/USDT 1h 60)", inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='prix', help='Affiche le prix actuel d\'une crypto')
async def price(ctx, symbol="BTC"):
    """Affiche le prix actuel d'une cryptomonnaie"""
    await ctx.send(f"Recherche du prix pour {symbol}...")
    
    try:
        # Formater le symbole
        if "/" not in symbol:
            symbol = f"{symbol}/USDT"
        
        # Obtenir le prix via l'API
        prediction_data = {
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": "1h",
            "use_latest": True,
            "days": 1
        }
        
        result = await api_request("predict", method="POST", data=prediction_data)
        
        if result.get("error"):
            await ctx.send(f"⚠️ Erreur lors de la récupération du prix: {result.get('message')}")
            return
        
        # Extraire le prix
        market_data = result.get("market_data", {})
        price = market_data.get('price', 0)
        
        # Envoyer le message
        embed = discord.Embed(
            title=f"Prix de {symbol}",
            description=f"**${price:.2f}**",
            color=discord.Color.gold()
        )
        
        embed.set_footer(text=f"Prix au {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        await ctx.send(embed=embed)
        logger.info(f"Prix affiché pour {symbol}: ${price:.2f}")
    except Exception as e:
        await ctx.send(f"⚠️ Une erreur est survenue: {str(e)}")
        logger.error(f"Erreur lors de la récupération du prix pour {symbol}: {str(e)}")

@bot.command(name='predire', help='Prédit l\'action à prendre pour une crypto')
async def predict(ctx, symbol="BTC/USDT", timeframe="1h"):
    """Prédit l'action à prendre pour une cryptomonnaie"""
    message = await ctx.send(f"Génération de la prédiction pour {symbol} ({timeframe})...")
    
    try:
        # Obtenir une prédiction de l'API
        prediction_data = {
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "use_latest": True,
            "days": 1
        }
        
        result = await api_request("predict", method="POST", data=prediction_data)
        
        if result.get("error"):
            await message.edit(content=f"⚠️ Erreur lors de la prédiction: {result.get('message')}")
            return
        
        # Créer l'embed
        emoji = "⏳"
        if result.get("recommendation") == "ACHETER":
            emoji = "🟢"
            color = discord.Color.green()
        elif result.get("recommendation") == "VENDRE":
            emoji = "🔴"
            color = discord.Color.red()
        else:
            color = discord.Color.light_gray()
        
        embed = discord.Embed(
            title=f"{emoji} Prédiction pour {symbol}",
            description=f"**Recommandation: {result.get('recommendation')}**",
            color=color
        )
        
        # Ajouter les détails du marché
        market_data = result.get("market_data", {})
        embed.add_field(name="Prix", value=f"${market_data.get('price', 0):.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{market_data.get('rsi', 0):.2f}", inline=True)
        embed.add_field(name="MACD", value=f"{market_data.get('macd', 0):.4f}", inline=True)
        
        # Ajouter le timeframe et la confiance
        embed.add_field(name="Timeframe", value=timeframe, inline=True)
        embed.add_field(name="Confiance", value=f"{result.get('confidence', 0) * 100:.1f}%", inline=True)
        
        # Ajouter l'horodatage
        embed.set_footer(text=f"Prédiction générée le {result.get('timestamp')}")
        
        await message.edit(content=None, embed=embed)
        logger.info(f"Prédiction générée pour {symbol}: {result.get('recommendation')}")
    except Exception as e:
        await message.edit(content=f"⚠️ Une erreur est survenue: {str(e)}")
        logger.error(f"Erreur lors de la génération de prédiction pour {symbol}: {str(e)}")

@bot.command(name='backtest', help='Effectue un backtest sur une période donnée')
async def backtest(ctx, symbol="BTC/USDT", timeframe="1h", days="30"):
    """Effectue un backtest sur une période donnée"""
    message = await ctx.send(f"Lancement du backtest pour {symbol} ({timeframe}) sur {days} jours...")
    
    try:
        # Convertir days en entier
        try:
            days = int(days)
        except ValueError:
            await message.edit(content="⚠️ Erreur: Le nombre de jours doit être un nombre entier.")
            return
        
        # Requête de backtest
        backtest_data = {
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days
        }
        
        result = await api_request("backtest", method="POST", data=backtest_data)
        
        if result.get("error"):
            await message.edit(content=f"⚠️ Erreur lors du backtest: {result.get('message')}")
            return
        
        # Créer l'embed avec les résultats
        color = discord.Color.green() if result.get("profit_pct", 0) > 0 else discord.Color.red()
        
        embed = discord.Embed(
            title=f"Résultats du Backtest pour {symbol}",
            description=f"Période: {result.get('period', 'N/A')}",
            color=color
        )
        
        # Ajouter les métriques
        embed.add_field(name="Profit", value=f"{result.get('profit_pct', 0):.2f}%", inline=True)
        embed.add_field(name="Buy & Hold", value=f"{result.get('buy_hold_pct', 0):.2f}%", inline=True)
        embed.add_field(name="Ratio de Sharpe", value=f"{result.get('sharpe_ratio', 0):.2f}", inline=True)
        
        embed.add_field(name="Nombre de trades", value=result.get('num_trades', 0), inline=True)
        embed.add_field(name="Taux de réussite", value=f"{result.get('win_rate', 0) * 100:.1f}%", inline=True)
        
        # Ajouter un résumé des trades
        trades_summary = result.get('trades_summary', [])
        if trades_summary:
            trades_text = "\n".join([
                f"{'✅' if t.get('win') else '❌'} {t.get('profit_pct', 0):.2f}%" 
                for t in trades_summary[:5]
            ])
            embed.add_field(name="Derniers trades", value=trades_text, inline=False)
        
        await message.edit(content=None, embed=embed)
        logger.info(f"Backtest effectué pour {symbol} sur {days} jours: Profit {result.get('profit_pct', 0):.2f}%")
    except Exception as e:
        await message.edit(content=f"⚠️ Une erreur est survenue: {str(e)}")
        logger.error(f"Erreur lors du backtest pour {symbol}: {str(e)}")

@bot.command(name='entrainer', help='Entraîne un nouveau modèle')
async def train(ctx, symbol="BTC/USDT", timeframe="1h", days="60"):
    """Entraîne un nouveau modèle"""
    message = await ctx.send(f"Lancement de l'entraînement pour {symbol} ({timeframe}) sur {days} jours...")
    
    try:
        # Convertir days en entier
        try:
            days = int(days)
        except ValueError:
            await message.edit(content="⚠️ Erreur: Le nombre de jours doit être un nombre entier.")
            return
        
        # Requête d'entraînement
        training_data = {
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "download": True,
            "timesteps": 50000
        }
        
        # Envoyer un message d'attente (l'entraînement peut prendre du temps)
        await message.edit(content=f"Entraînement en cours pour {symbol}... Cela peut prendre plusieurs minutes.")
        
        # Lancer l'entraînement en arrière-plan
        result = await api_request("train", method="POST", data=training_data)
        
        if result.get("error"):
            await message.edit(content=f"⚠️ Erreur lors de l'entraînement: {result.get('message')}")
            return
        
        # Créer l'embed avec les résultats
        embed = discord.Embed(
            title=f"Résultats de l'entraînement pour {symbol}",
            description=f"Modèle sauvegardé: {result.get('model_path', 'N/A')}",
            color=discord.Color.blue()
        )
        
        # Ajouter les métriques
        embed.add_field(name="Échantillons d'entraînement", value=result.get('training_samples', 0), inline=True)
        embed.add_field(name="Échantillons de test", value=result.get('test_samples', 0), inline=True)
        
        # Ajouter les résultats du backtest rapide
        backtest_results = result.get('backtest_results', {})
        embed.add_field(name="Profit (backtest)", value=f"{backtest_results.get('profit_pct', 0):.2f}%", inline=True)
        embed.add_field(name="Buy & Hold", value=f"{backtest_results.get('buy_hold_pct', 0):.2f}%", inline=True)
        embed.add_field(name="Sharpe Ratio", value=f"{backtest_results.get('sharpe_ratio', 0):.2f}", inline=True)
        
        await message.edit(content=None, embed=embed)
        logger.info(f"Entraînement terminé pour {symbol} sur {days} jours")
    except Exception as e:
        await message.edit(content=f"⚠️ Une erreur est survenue: {str(e)}")
        logger.error(f"Erreur lors de l'entraînement pour {symbol}: {str(e)}")

@bot.command(name='configurer', help='Configure les canaux pour les alertes')
@commands.has_permissions(administrator=True)
async def setup(ctx, symbol="BTC/USDT", timeframe="1h"):
    """Configure le canal pour recevoir des alertes de trading"""
    await ctx.send(f"Canal configuré pour recevoir les alertes de {symbol} ({timeframe})")
    logger.info(f"Canal {ctx.channel.id} configuré pour {symbol}_{timeframe}")
    # Cette commande pourrait être améliorée pour stocker la configuration dans une base de données

@bot.event
async def on_command_error(ctx, error):
    """Gestion centralisée des erreurs de commandes"""
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send("⚠️ Vous n'avez pas la permission d'utiliser cette commande.")
    elif isinstance(error, commands.errors.MissingRequiredArgument):
        await ctx.send(f"⚠️ Argument manquant: {error.param.name}")
    elif isinstance(error, commands.errors.CommandNotFound):
        await ctx.send(f"⚠️ Commande inconnue. Utilisez !aide pour voir la liste des commandes.")
    else:
        await ctx.send(f"⚠️ Une erreur est survenue: {str(error)}")
        logger.error(f"Erreur de commande: {str(error)}")

# Lancer le bot
def run_bot():
    """Fonction principale pour démarrer le bot"""
    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"Erreur fatale lors du démarrage du bot: {str(e)}")

if __name__ == "__main__":
    run_bot() 