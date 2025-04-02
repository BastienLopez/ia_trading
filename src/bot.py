import discord
from discord.ext import commands, tasks
import asyncio
from src.config.config import *
from src.utils.data_fetcher import fetch_binance_data, get_current_price
from src.indicators.technical_indicators import analyze_signals

# Configuration du bot Discord
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    """Événement déclenché quand le bot est prêt"""
    print(f"Bot connecté en tant que {bot.user}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send("✅ Le bot est connecté et prêt à analyser le marché !")
    # Démarrer la tâche périodique
    check_signals.start()

@tasks.loop(minutes=15)
async def check_signals():
    """Vérifie périodiquement les signaux de trading"""
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel:
        return

    try:
        # Récupérer les données
        df = fetch_binance_data()
        if df is None:
            return

        # Analyser les signaux
        signals, strength, indicators = analyze_signals(df)
        current_price = get_current_price()

        if signals:
            # Créer un message embed
            embed = discord.Embed(
                title="📊 Nouveaux signaux détectés",
                description=f"Prix actuel: ${current_price:,.2f}",
                color=discord.Color.blue()
            )
            
            # Ajouter les signaux
            embed.add_field(
                name="Signaux",
                value="\n".join(signals),
                inline=False
            )
            
            # Ajouter la force du signal
            embed.add_field(
                name="Force du signal",
                value=strength,
                inline=True
            )
            
            # Ajouter les valeurs des indicateurs
            embed.add_field(
                name="Indicateurs",
                value=f"RSI: {indicators['rsi']:.2f}\n"
                      f"MACD: {indicators['macd']:.2f}\n"
                      f"Signal: {indicators['signal_line']:.2f}\n"
                      f"EMA9: {indicators['ema_short']:.2f}\n"
                      f"EMA21: {indicators['ema_long']:.2f}",
                inline=False
            )
            
            await channel.send(embed=embed)

    except Exception as e:
        print(f"Erreur lors de la vérification des signaux: {str(e)}")
        await channel.send(f"⚠️ Erreur: {str(e)}")

@bot.command(name="force_check")
async def force_check(ctx):
    """Commande pour forcer une vérification des signaux"""
    if ctx.channel.id != DISCORD_CHANNEL_ID:
        await ctx.send("⚠️ Utilisez cette commande dans le channel configuré.")
        return

    await ctx.send("🔄 Vérification des signaux en cours...")
    await check_signals()

@bot.command(name="price")
async def price(ctx):
    """Affiche le prix actuel"""
    current_price = get_current_price()
    if current_price:
        await ctx.send(f"💰 Prix actuel du {TRADING_PAIR}: ${current_price:,.2f}")
    else:
        await ctx.send("⚠️ Impossible de récupérer le prix actuel")

def run_bot():
    """Lance le bot"""
    bot.run(DISCORD_TOKEN) 