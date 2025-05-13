"""
Exemple d'utilisation du collecteur de données blockchain asynchrone.

Ce script montre comment utiliser le collecteur asynchrone pour :
1. Collecter des données de plusieurs sources en parallèle
2. Gérer les rate limits
3. Utiliser le cache distribué
4. Prioriser les sources de données
"""

import asyncio
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.utils.async_blockchain_collector import AsyncBlockchainCollector

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AsyncBlockchainExample")

# Configuration matplotlib
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


async def collect_and_analyze_data():
    """Collecte et analyse les données blockchain de manière asynchrone."""
    collector = AsyncBlockchainCollector()

    # Liste d'adresses Ethereum à analyser
    addresses = [
        "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",  # Ethereum Foundation
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Binance
        "0x8894E0a0c962CB723c1976a4421c95949bE2D4E3",  # Binance 2
    ]

    logger.info("Démarrage de la collecte de données...")
    start_time = asyncio.get_event_loop().time()

    # Collecter les données pour chaque adresse
    tasks = []
    for address in addresses:
        tasks.append(collector.collect_all_async(address))

    # Attendre tous les résultats
    results = await asyncio.gather(*tasks)

    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"Collecte terminée en {duration:.2f} secondes")

    # Analyser les résultats
    all_transactions = []
    all_tvl = []
    all_pools = []
    all_staking = []

    for i, result in enumerate(results):
        address = addresses[i]
        logger.info(f"\nAnalyse des données pour {address}:")

        # Transactions
        if "transactions" in result and not result["transactions"].empty:
            df_tx = result["transactions"]
            df_tx["address"] = address
            all_transactions.append(df_tx)
            logger.info(f"  - {len(df_tx)} transactions trouvées")

        # TVL
        if "tvl" in result and not result["tvl"].empty:
            df_tvl = result["tvl"]
            all_tvl.append(df_tvl)
            logger.info(f"  - Données TVL pour {len(df_tvl)} protocoles")

        # Pools
        if "pools" in result and not result["pools"].empty:
            df_pools = result["pools"]
            all_pools.append(df_pools)
            logger.info(f"  - {len(df_pools)} pools trouvées")

        # Staking
        if "staking" in result and not result["staking"].empty:
            df_staking = result["staking"]
            all_staking.append(df_staking)
            logger.info(f"  - Données de staking récupérées")

    # Combiner les données
    if all_transactions:
        df_transactions = pd.concat(all_transactions, ignore_index=True)
        analyze_transactions(df_transactions)

    if all_tvl:
        df_tvl = pd.concat(all_tvl, ignore_index=True)
        analyze_tvl(df_tvl)

    if all_pools:
        df_pools = pd.concat(all_pools, ignore_index=True)
        analyze_pools(df_pools)

    if all_staking:
        df_staking = pd.concat(all_staking, ignore_index=True)
        analyze_staking(df_staking)


def analyze_transactions(df: pd.DataFrame):
    """Analyse les transactions."""
    logger.info("\nAnalyse des transactions:")

    # Grouper par adresse
    grouped = df.groupby("address")

    # Statistiques par adresse
    stats = grouped.agg(
        {"hash": "count", "ether_value": ["sum", "mean", "median"]}
    ).round(2)

    logger.info("\nStatistiques par adresse:")
    print(stats)

    # Tracer les transactions par adresse
    plt.figure(figsize=(12, 6))
    grouped["hash"].count().plot(kind="bar")
    plt.title("Nombre de transactions par adresse")
    plt.xlabel("Adresse")
    plt.ylabel("Nombre de transactions")
    plt.tight_layout()

    # Sauvegarder le graphique
    plot_path = INFO_RETOUR_DIR / "async_transactions_by_address.png"
    plt.savefig(plot_path)
    plt.close()

    # Sauvegarder les données
    output_path = INFO_RETOUR_DIR / "async_transactions.csv"
    df.to_csv(output_path)
    logger.info(f"Données et graphiques sauvegardés dans {INFO_RETOUR_DIR}")


def analyze_tvl(df: pd.DataFrame):
    """Analyse les données TVL."""
    logger.info("\nAnalyse des TVL:")

    if "tvl" in df.columns:
        # Top protocoles par TVL
        top_protocols = df.nlargest(10, "tvl")

        logger.info("\nTop 10 protocoles par TVL:")
        for idx, row in top_protocols.iterrows():
            tvl_millions = row["tvl"] / 1000000
            logger.info(f"  {row['name']}: ${tvl_millions:.2f}M")

        # Tracer les top protocoles
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_protocols, x="name", y="tvl")
        plt.title("Top 10 protocoles par TVL")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "async_top_protocols_tvl.png"
        plt.savefig(plot_path)
        plt.close()

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / "async_tvl.csv"
        df.to_csv(output_path)
        logger.info(f"Données et graphiques sauvegardés dans {INFO_RETOUR_DIR}")


def analyze_pools(df: pd.DataFrame):
    """Analyse les pools DeFi."""
    logger.info("\nAnalyse des pools:")

    if "tvlUsd" in df.columns and "project" in df.columns:
        # TVL par projet
        tvl_by_project = (
            df.groupby("project")["tvlUsd"].sum().sort_values(ascending=False)
        )

        logger.info("\nTVL par projet:")
        for project, tvl in tvl_by_project.head(10).items():
            tvl_millions = tvl / 1000000
            logger.info(f"  {project}: ${tvl_millions:.2f}M")

        # Tracer la TVL par projet
        plt.figure(figsize=(12, 6))
        tvl_by_project.head(10).plot(kind="bar")
        plt.title("TVL par projet")
        plt.xlabel("Projet")
        plt.ylabel("TVL (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "async_tvl_by_project.png"
        plt.savefig(plot_path)
        plt.close()

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / "async_pools.csv"
        df.to_csv(output_path)
        logger.info(f"Données et graphiques sauvegardés dans {INFO_RETOUR_DIR}")


def analyze_staking(df: pd.DataFrame):
    """Analyse les données de staking."""
    logger.info("\nAnalyse du staking:")

    if not df.empty:
        # Afficher les statistiques de base
        logger.info("\nStatistiques de staking:")
        print(df.describe())

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / "async_staking.csv"
        df.to_csv(output_path)
        logger.info(f"Données sauvegardées dans {INFO_RETOUR_DIR}")


async def main():
    """Point d'entrée principal."""
    try:
        # Créer le répertoire de sortie si nécessaire
        INFO_RETOUR_DIR.mkdir(exist_ok=True)

        logger.info("Démarrage de l'exemple asynchrone...")
        await collect_and_analyze_data()
        logger.info("Exemple terminé avec succès!")

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
