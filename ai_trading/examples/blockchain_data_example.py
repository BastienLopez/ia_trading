"""
Exemple d'utilisation du collecteur de données blockchain.

Ce script montre comment utiliser le collecteur de données blockchain pour :
1. Récupérer des données de transactions blockchain
2. Analyser des métriques de DeFi (TVL, pools)
3. Obtenir des informations sur le staking et la gouvernance
4. Visualiser les flux de capital entre protocoles
"""

import sys
from pathlib import Path

import matplotlib
import pandas as pd

# Utiliser le backend Agg qui ne nécessite pas d'interface graphique
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai_trading.config import INFO_RETOUR_DIR
from ai_trading.utils.blockchain_data_collector import BlockchainDataCollector

# Configuration matplotlib pour de meilleurs graphiques
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def analyze_ethereum_transactions():
    """Exemple d'analyse de transactions Ethereum."""
    print("\n==== Analyse de transactions Ethereum ====\n")

    collector = BlockchainDataCollector()

    # Adresse Ethereum de l'Ethereum Foundation
    eth_address = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"

    print(f"Récupération des transactions pour l'adresse {eth_address}...")
    transactions = collector.get_eth_transactions(address=eth_address)

    if transactions.empty:
        print("Aucune transaction trouvée.")
        return

    print(f"Récupéré {len(transactions)} transactions")

    # Afficher quelques informations de base
    print(f"\nPremières transactions :")
    print(transactions[["hash", "timeStamp", "from", "to", "ether_value"]].head())

    # Analyse des transactions
    metrics = collector.analyze_transactions(transactions)

    print("\nMétriques des transactions :")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Visualisation des transactions par date
    if "timeStamp" in transactions.columns:
        # Regrouper par jour
        transactions["date"] = transactions["timeStamp"].dt.date
        txs_by_date = transactions.groupby("date").size()

        # Regrouper les transactions par mois pour une meilleure visualisation
        txs_by_month = transactions.groupby(
            transactions["timeStamp"].dt.to_period("M")
        ).size()

        # Tracer le graphique
        plt.figure(figsize=(12, 6))
        txs_by_month.plot(kind="bar")
        plt.title(f"Transactions mensuelles pour {eth_address}")
        plt.xlabel("Mois")
        plt.ylabel("Nombre de transactions")
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "eth_transactions_by_month.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nGraphique sauvegardé dans {plot_path}")

    # Sauvegarder les transactions pour une analyse ultérieure
    output_path = INFO_RETOUR_DIR / "eth_transactions.csv"
    transactions.to_csv(output_path)
    print(f"Données de transactions sauvegardées dans {output_path}")


def analyze_defi_tvl():
    """Exemple d'analyse des TVL DeFi."""
    print("\n==== Analyse des TVL DeFi ====\n")

    collector = BlockchainDataCollector()

    print("Récupération des données TVL globales...")
    tvl_data = collector.get_defillama_tvl()

    if tvl_data.empty:
        print("Aucune donnée TVL trouvée.")
        return

    print(f"Récupéré des données pour {len(tvl_data)} protocoles DeFi")

    # Analyser les TVL par blockchain
    if "chains" in tvl_data.columns:
        # Créer une série pour compter les protocoles par blockchain
        chain_counts = {}
        for chains_list in tvl_data["chains"]:
            for chain in chains_list:
                chain_counts[chain] = chain_counts.get(chain, 0) + 1

        chain_series = pd.Series(chain_counts).sort_values(ascending=False)

        print("\nTop 10 blockchains par nombre de protocoles :")
        print(chain_series.head(10))

        # Tracer les top blockchains
        plt.figure(figsize=(12, 6))
        chain_series.head(10).plot(kind="bar")
        plt.title("Top 10 blockchains par nombre de protocoles")
        plt.xlabel("Blockchain")
        plt.ylabel("Nombre de protocoles")
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "defi_chains_count.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nGraphique sauvegardé dans {plot_path}")

    # Analyser les top protocoles par TVL
    if "tvl" in tvl_data.columns:
        top_protocols = tvl_data.sort_values("tvl", ascending=False).head(10)

        print("\nTop 10 protocoles par TVL (USD) :")
        for idx, row in top_protocols.iterrows():
            tvl_millions = row["tvl"] / 1000000
            print(f"  {row['name']}: ${tvl_millions:.2f}M")

        # Tracer les top protocoles
        plt.figure(figsize=(12, 6))
        top_protocols["tvl"].plot(kind="bar")
        plt.title("Top 10 protocoles DeFi par TVL")
        plt.xlabel("Protocole")
        plt.ylabel("TVL (USD)")
        plt.xticks(range(len(top_protocols)), top_protocols["name"], rotation=45)
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "defi_top_protocols.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nGraphique sauvegardé dans {plot_path}")

    # Sauvegarder les données
    output_path = INFO_RETOUR_DIR / "defi_tvl_data.csv"
    tvl_data.to_csv(output_path)
    print(f"Données TVL sauvegardées dans {output_path}")


def analyze_defi_pools():
    """Exemple d'analyse des pools DeFi."""
    print("\n==== Analyse des pools DeFi ====\n")

    collector = BlockchainDataCollector()

    chains_to_analyze = ["ethereum", "bsc", "polygon"]

    for chain in chains_to_analyze:
        print(f"\nRécupération des pools sur {chain}...")
        pools_data = collector.get_defillama_pools(chain=chain)

        if pools_data.empty:
            print(f"Aucune donnée de pool trouvée pour {chain}.")
            continue

        print(f"Récupéré {len(pools_data)} pools pour {chain}")

        # Analyser les projets avec le plus de pools
        if "project" in pools_data.columns:
            project_counts = pools_data["project"].value_counts().head(10)

            print(f"\nTop projets sur {chain} par nombre de pools :")
            for project, count in project_counts.items():
                print(f"  {project}: {count} pools")

        # Analyser la TVL par projet
        if "project" in pools_data.columns and "tvlUsd" in pools_data.columns:
            tvl_by_project = (
                pools_data.groupby("project")["tvlUsd"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            print(f"\nTop projets sur {chain} par TVL (USD) :")
            for project, tvl in tvl_by_project.items():
                tvl_millions = tvl / 1000000
                print(f"  {project}: ${tvl_millions:.2f}M")

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / f"defi_pools_{chain}.csv"
        pools_data.to_csv(output_path)
        print(f"Données des pools {chain} sauvegardées dans {output_path}")


def analyze_staking_and_governance():
    """Exemple d'analyse des données de staking et de gouvernance."""
    print("\n==== Analyse du staking et de la gouvernance ====\n")

    collector = BlockchainDataCollector()

    # Données de staking
    blockchains = ["ethereum", "solana", "polkadot", "cardano"]

    print("Récupération des données de staking...")
    staking_data = {}

    for blockchain in blockchains:
        data = collector.get_staking_data(blockchain)
        if not data.empty:
            staking_data[blockchain] = data
            print(f"  {blockchain}: données récupérées")
        else:
            print(f"  {blockchain}: aucune donnée disponible")

    # Gouvernance
    protocols = ["aave", "uniswap", "compound"]

    print("\nRécupération des données de gouvernance...")
    governance_data = {}

    for protocol in protocols:
        data = collector.get_governance_data(protocol)
        if not data.empty:
            governance_data[protocol] = data
            print(f"  {protocol}: {len(data)} propositions récupérées")
        else:
            print(f"  {protocol}: aucune donnée disponible")

    # Analyser les propositions de gouvernance
    for protocol, proposals in governance_data.items():
        if proposals.empty:
            continue

        print(f"\nPropositions récentes pour {protocol}:")

        if "title" in proposals.columns and "state" in proposals.columns:
            for idx, row in proposals.head(5).iterrows():
                title = row["title"]
                state = row["state"]
                print(f"  - {title} ({state})")

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / f"governance_{protocol}.csv"
        proposals.to_csv(output_path)
        print(f"Données de gouvernance {protocol} sauvegardées dans {output_path}")


def analyze_capital_flows():
    """Exemple d'analyse des flux de capitaux entre protocoles."""
    print("\n==== Analyse des flux de capitaux ====\n")

    collector = BlockchainDataCollector()

    print("Récupération des données de flux de capitaux...")
    flows_data = collector.get_capital_flows(n_days=30, blockchain="ethereum")

    if flows_data.empty:
        print("Aucune donnée de flux de capitaux trouvée.")
        return

    print(f"Récupéré des données pour {len(flows_data)} protocoles")

    # Analyser les flux entrants et sortants
    if "flow_1d" in flows_data.columns and "name" in flows_data.columns:
        # Top gagnants
        top_gainers = flows_data.sort_values("flow_1d", ascending=False).head(5)

        print("\nTop 5 protocoles avec entrées de capitaux (24h):")
        for idx, row in top_gainers.iterrows():
            flow_millions = row["flow_1d"] / 1000000
            print(f"  {row['name']}: +${flow_millions:.2f}M")

        # Top perdants
        top_losers = flows_data.sort_values("flow_1d", ascending=True).head(5)

        print("\nTop 5 protocoles avec sorties de capitaux (24h):")
        for idx, row in top_losers.iterrows():
            flow_millions = abs(row["flow_1d"]) / 1000000
            print(f"  {row['name']}: -${flow_millions:.2f}M")

        # Tracer les flux de capitaux
        plt.figure(figsize=(12, 6))
        combined = pd.concat([top_gainers, top_losers])
        combined = combined.sort_values("flow_1d")
        combined["flow_1d_millions"] = combined["flow_1d"] / 1000000

        # Tracer le graphique
        plt.barh(combined["name"], combined["flow_1d_millions"])
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        plt.title("Flux de capitaux par protocole (24h)")
        plt.xlabel("Flux de capitaux (millions USD)")
        plt.ylabel("Protocole")
        plt.tight_layout()

        # Sauvegarder le graphique
        plot_path = INFO_RETOUR_DIR / "capital_flows_24h.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nGraphique sauvegardé dans {plot_path}")

    # Sauvegarder les données
    output_path = INFO_RETOUR_DIR / "capital_flows_data.csv"
    flows_data.to_csv(output_path)
    print(f"Données de flux de capitaux sauvegardées dans {output_path}")


def get_combined_blockchain_analysis():
    """Exemple d'analyse blockchain combinée."""
    print("\n==== Analyse blockchain combinée ====\n")

    collector = BlockchainDataCollector()

    blockchains = ["ethereum", "bitcoin", "solana"]

    for blockchain in blockchains:
        print(f"\nAnalyse combinée pour {blockchain}...")
        combined_data = collector.get_combined_blockchain_data(
            blockchain=blockchain, days=30
        )

        if combined_data.empty:
            print(f"Aucune donnée combinée disponible pour {blockchain}.")
            continue

        # Afficher les métriques principales
        print(f"Métriques blockchain pour {blockchain}:")
        for column in combined_data.columns:
            if column not in ["stats", "blockchain", "date"]:
                value = combined_data[column].iloc[0]
                if isinstance(value, (int, float)) and value > 1000000:
                    print(f"  {column}: {value/1000000:.2f}M")
                else:
                    print(f"  {column}: {value}")

        # Sauvegarder les données
        output_path = INFO_RETOUR_DIR / f"blockchain_combined_{blockchain}.csv"
        combined_data.to_csv(output_path)
        print(f"Données combinées {blockchain} sauvegardées dans {output_path}")


def gas_usage_analysis():
    """Exemple d'analyse de la consommation de gas."""
    print("\n==== Analyse de la consommation de gas ====\n")

    collector = BlockchainDataCollector()

    print("Récupération des données de consommation de gas...")
    gas_data = collector.get_top_eth_gas_consumers()

    if gas_data.empty:
        print("Aucune donnée de consommation de gas trouvée.")
        return

    print(f"Récupéré des données pour {len(gas_data)} contrats")

    # Afficher les top contrats
    print("\nTop 10 contrats par consommation de gas:")
    for idx, row in gas_data.iterrows():
        print(
            f"  {row['contract']}: {row['gas_used_24h']:,} gas ({row['txs_24h']:,} transactions)"
        )

    # Tracer la consommation de gas
    plt.figure(figsize=(12, 6))

    # Préparer les données
    gas_data = gas_data.sort_values("gas_used_24h", ascending=True)
    gas_data["gas_used_billions"] = gas_data["gas_used_24h"] / 1000000000

    # Tracer le graphique
    plt.barh(gas_data["contract"], gas_data["gas_used_billions"])
    plt.title("Consommation de gas par contrat (24h)")
    plt.xlabel("Gas utilisé (milliards)")
    plt.ylabel("Contrat")
    plt.tight_layout()

    # Sauvegarder le graphique
    plot_path = INFO_RETOUR_DIR / "gas_usage.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nGraphique sauvegardé dans {plot_path}")

    # Sauvegarder les données
    output_path = INFO_RETOUR_DIR / "gas_usage_data.csv"
    gas_data.to_csv(output_path)
    print(f"Données de consommation de gas sauvegardées dans {output_path}")


if __name__ == "__main__":
    # Créer le répertoire de sortie si nécessaire
    INFO_RETOUR_DIR.mkdir(exist_ok=True)

    print("Démarrage de l'analyse des données blockchain...")

    # Analyse des transactions Ethereum
    analyze_ethereum_transactions()

    # Analyse des TVL DeFi
    analyze_defi_tvl()

    # Analyse des pools DeFi
    analyze_defi_pools()

    # Analyse du staking et de la gouvernance
    analyze_staking_and_governance()

    # Analyse des flux de capitaux
    analyze_capital_flows()

    # Analyse combinée
    get_combined_blockchain_analysis()

    # Analyse de la consommation de gas
    gas_usage_analysis()

    print("\nAnalyse des données blockchain terminée!")
