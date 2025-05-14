#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de la classe ModelEnsemble.

Ce script montre comment combiner plusieurs modèles de prédiction
dans un ensemble pour améliorer la robustesse des prédictions.
"""

import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ajouter le répertoire racine au chemin d'importation Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Importer la classe ModelEnsemble
from ai_trading.llm.predictions.model_ensemble import ModelEnsemble


def create_demo_data():
    """Crée un jeu de données de démonstration pour la classification."""
    print("Création des données de démonstration...")

    # Générer un problème de classification synthétique
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Diviser en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(
        f"Données créées : {X_train.shape[0]} échantillons d'entraînement, {X_test.shape[0]} échantillons de test"
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Entraîne plusieurs modèles de classification."""
    print("Entraînement des modèles...")

    # Modèle 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Modèle 2: Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    # Modèle 3: Régression logistique
    lr_model = LogisticRegression(random_state=42, max_iter=200)
    lr_model.fit(X_train, y_train)

    print("Entraînement terminé.")

    return rf_model, gb_model, lr_model


def main():
    """Fonction principale pour démontrer l'utilisation de ModelEnsemble."""
    # Créer des données de démonstration
    X_train, X_test, y_train, y_test = create_demo_data()

    # Entraîner les modèles
    rf_model, gb_model, lr_model = train_models(X_train, y_train)

    # Créer l'ensemble de modèles avec stratégie de fusion pondérée
    print("\nCréation de l'ensemble de modèles avec stratégie pondérée...")
    ensemble = ModelEnsemble(fusion_strategy="weighted", adjust_weights=True)

    # Ajouter les modèles à l'ensemble avec différents poids initiaux
    ensemble.add_model(
        rf_model,
        "RandomForest",
        weight=1.2,
        predict_method="predict",
        confidence_method="predict_proba",
    )

    ensemble.add_model(
        gb_model,
        "GradientBoosting",
        weight=1.0,
        predict_method="predict",
        confidence_method="predict_proba",
    )

    ensemble.add_model(
        lr_model,
        "LogisticRegression",
        weight=0.8,
        predict_method="predict",
        confidence_method="predict_proba",
    )

    # Évaluer les performances des modèles individuels et de l'ensemble
    print("\nÉvaluation des performances...")
    metrics = ensemble.evaluate(X_test, y_test)

    print("\nPerformances des modèles individuels:")
    for model_name in metrics:
        if model_name != "ensemble":
            print(
                f"- {model_name}: Accuracy = {metrics[model_name]['accuracy']:.4f}, "
                f"F1 = {metrics[model_name]['f1']:.4f}"
            )

    print("\nPerformance de l'ensemble:")
    print(
        f"- Ensemble: Accuracy = {metrics['ensemble']['accuracy']:.4f}, "
        f"F1 = {metrics['ensemble']['f1']:.4f}"
    )

    # Afficher les informations sur les modèles après ajustement des poids
    print("\nPoids des modèles après ajustement:")
    for model_info in ensemble.get_models_info():
        print(f"- {model_info['name']}: Poids = {model_info['weight']:.4f}")

    # Démonstration de prédiction avec l'ensemble
    print("\nDémonstration de prédiction sur 5 exemples:")
    for i in range(5):
        # Sélectionner un exemple aléatoire
        idx = np.random.randint(0, X_test.shape[0])

        # Obtenir les prédictions individuelles et de l'ensemble
        rf_pred = rf_model.predict([X_test[idx]])[0]
        gb_pred = gb_model.predict([X_test[idx]])[0]
        lr_pred = lr_model.predict([X_test[idx]])[0]

        # Prédiction de l'ensemble avec métadonnées
        ensemble_result = ensemble.predict([X_test[idx]])
        ensemble_pred = ensemble_result["prediction"]

        # Vérité terrain
        true_label = y_test[idx]

        print(f"\nExemple {i+1}:")
        print(f"- RandomForest: {rf_pred}")
        print(f"- GradientBoosting: {gb_pred}")
        print(f"- LogisticRegression: {lr_pred}")
        print(
            f"- Ensemble: {ensemble_pred} (Ratio de consensus: {ensemble_result.get('consensus_ratio', 'N/A'):.2f})"
        )
        print(f"- Vérité: {true_label}")

    # Démonstration avec différentes stratégies de fusion
    print("\nComparaison des stratégies de fusion:")

    # Créer des ensembles avec différentes stratégies
    strategies = ["majority", "weighted", "confidence"]
    results = {}

    for strategy in strategies:
        # Créer l'ensemble avec la stratégie
        strat_ensemble = ModelEnsemble(fusion_strategy=strategy, adjust_weights=False)

        # Ajouter les mêmes modèles
        strat_ensemble.add_model(
            rf_model,
            "RandomForest",
            weight=1.0,
            predict_method="predict",
            confidence_method="predict_proba",
        )

        strat_ensemble.add_model(
            gb_model,
            "GradientBoosting",
            weight=1.0,
            predict_method="predict",
            confidence_method="predict_proba",
        )

        strat_ensemble.add_model(
            lr_model,
            "LogisticRegression",
            weight=1.0,
            predict_method="predict",
            confidence_method="predict_proba",
        )

        # Évaluer et stocker les résultats
        metrics = strat_ensemble.evaluate(X_test, y_test)
        results[strategy] = metrics["ensemble"]

    # Afficher les résultats comparatifs
    for strategy in strategies:
        if "error" in results[strategy]:
            print(f"- Stratégie '{strategy}': Erreur - {results[strategy]['error']}")
        else:
            print(
                f"- Stratégie '{strategy}': Accuracy = {results[strategy]['accuracy']:.4f}, "
                f"F1 = {results[strategy]['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
