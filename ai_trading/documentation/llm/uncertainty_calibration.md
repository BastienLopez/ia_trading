# Phase 4: Calibration des incertitudes

Ce document décrit l'implémentation de la calibration des incertitudes dans le cadre de la Phase 4 du projet AI Trading.

## 1. Vue d'ensemble

La calibration des incertitudes est une étape cruciale pour quantifier la fiabilité des prédictions générées par les modèles. Nous avons implémenté un module complet qui permet de :
- Calculer des intervalles de confiance pour les prédictions
- Estimer les distributions de probabilité sous-jacentes
- Identifier et gérer les valeurs aberrantes (outliers)
- Valider les prédictions par validation croisée

Ces fonctionnalités permettent de mieux comprendre le niveau de confiance que l'on peut accorder aux prédictions et d'ajuster les stratégies de trading en conséquence.

## 2. Module implémenté

### UncertaintyCalibrator

Le module `uncertainty_calibration.py` contient la classe principale `UncertaintyCalibrator` qui fournit des méthodes pour calibrer les incertitudes des prédictions :

#### Intervalles de confiance
- Calcul d'intervalles de confiance par plusieurs méthodes:
  - Bootstrap: Méthode de rééchantillonnage non-paramétrique
  - Paramétrique: Utilisation de la distribution normale
  - Bayésien: Approche probabiliste incorporant des informations a priori

#### Distributions de probabilité
- Estimation de la distribution de probabilité pour chaque prédiction
- Calcul des probabilités pour chaque direction (bearish, neutral, bullish)
- Mesure de l'entropie pour quantifier l'incertitude globale

#### Détection d'outliers
- Identification des prédictions aberrantes par deux méthodes:
  - Z-score: Détection basée sur les écarts à la moyenne normalisés
  - IQR (écart interquartile): Approche robuste basée sur les quartiles

#### Validation croisée
- Évaluation de la calibration du modèle par validation croisée
- Calcul des courbes de calibration
- Mesure de l'erreur de calibration

## 3. Technologies utilisées

Le module de calibration des incertitudes s'appuie sur plusieurs bibliothèques statistiques et de machine learning :

- **NumPy/SciPy**: Pour les calculs statistiques de base
- **Scikit-learn**: Pour les méthodes de validation croisée
- **Matplotlib/Seaborn**: Pour les visualisations
- **Pandas**: Pour la manipulation des données

## 4. Exemples d'utilisation

Le module `ai_trading/examples/uncertainty_calibration_example.py` montre comment utiliser ces fonctionnalités :

```python
# Initialiser le calibrateur
calibrator = UncertaintyCalibrator(market_data=market_data)

# Calculer des intervalles de confiance
confidence_interval = calibrator.calculate_confidence_intervals(
    prediction, confidence_level=0.95, method="bootstrap")

# Estimer la distribution de probabilité
probability_dist = calibrator.estimate_probability_distribution(prediction)

# Détecter les outliers
outlier_results = calibrator.detect_outliers(predictions, method="z_score")

# Calibrer une prédiction
calibrated = calibrator.calibrate_prediction(prediction, calibration_method="platt")

# Visualiser la distribution
calibrator.plot_probability_distribution(prediction, distribution=probability_dist)
```

## 5. Bénéfices pour le trading

L'implémentation de la calibration des incertitudes apporte plusieurs avantages majeurs :

1. **Gestion du risque améliorée**: Quantification précise des risques associés à chaque prédiction
2. **Décisions plus nuancées**: Prise en compte de l'incertitude dans les stratégies de trading
3. **Détection précoce d'anomalies**: Identification des prédictions potentiellement erronées
4. **Allocation de capital optimisée**: Investissement proportionnel au niveau de confiance
5. **Évaluation de la qualité du modèle**: Mesure objective de la calibration des prédictions

## 6. Méthodes de calibration

### 6.1 Calibration des probabilités

La calibration des probabilités vise à faire correspondre les probabilités prédites avec les fréquences observées. Par exemple, parmi les prédictions avec une confiance de 0.8, environ 80% devraient effectivement se réaliser.

Nous utilisons deux méthodes principales de calibration :
- **Calibration de Platt**: Applique une régression logistique pour transformer les scores bruts
- **Calibration isotonique**: Utilise une régression isotonique (non-paramétrique) pour calibrer les probabilités

### 6.2 Gestion des outliers

Les outliers peuvent fortement impacter la qualité des prédictions et induire en erreur les stratégies de trading. Notre approche pour les gérer comprend :

1. **Détection automatique**: Identification des prédictions anormales
2. **Évaluation du contexte**: Analyse des conditions de marché entourant l'outlier
3. **Ajustement adaptatif**: Modification de la confiance associée ou exclusion si nécessaire

## 7. Validation et évaluation

Pour évaluer la qualité de la calibration, nous utilisons plusieurs métriques :

- **Brier score**: Mesure l'exactitude des probabilités prédites
- **Erreur de calibration**: Calcule l'écart entre probabilités prédites et fréquences observées
- **Diagrammes de fiabilité**: Visualisent graphiquement la qualité de la calibration

La validation croisée permet d'obtenir des estimations robustes de ces métriques sur différents sous-ensembles de données.

## 8. Intégration avec les autres modules

Le module de calibration des incertitudes s'intègre harmonieusement avec les autres composants du système :

- Avec le `PredictionExplainer` pour fournir des explications enrichies d'intervalles de confiance
- Avec le `PredictionVisualizer` pour créer des visualisations incluant les incertitudes
- Avec le `PredictionReporter` pour générer des rapports détaillés sur les incertitudes

## 9. Perspectives d'amélioration

Plusieurs axes d'amélioration peuvent être envisagés :

1. **Calibration par ensemble**: Utilisation de techniques d'ensemble pour améliorer la calibration
2. **Modèles de séries temporelles**: Prise en compte de la dépendance temporelle dans l'estimation des incertitudes
3. **Calibration spécifique au contexte**: Adaptation de la calibration selon les conditions de marché
4. **Propagation d'incertitude**: Traçage de l'incertitude à travers toute la chaîne de prédiction

## 10. Conclusion

L'implémentation du module de calibration des incertitudes représente une avancée significative pour le projet AI Trading. En quantifiant précisément la fiabilité des prédictions et en identifiant les cas problématiques, ce module permet aux traders de prendre des décisions plus informées et de gérer efficacement le risque, contribuant ainsi à améliorer la performance globale du système de trading. 