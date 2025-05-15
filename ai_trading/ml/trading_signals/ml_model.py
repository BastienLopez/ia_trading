import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ai_trading.data.technical_indicators import TechnicalIndicators
from ai_trading.ml.trading_signals.signal_generator import SignalGenerator, Signal, SignalType

# Configuration du logging
logger = logging.getLogger(__name__)

class MLSignalModel:
    """
    Modèle ML pour la prédiction des signaux de trading.
    Utilise un ensemble d'algorithmes pour générer des prédictions
    plus robustes basées sur les indicateurs techniques.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le modèle ML.
        
        Args:
            config: Configuration des paramètres du modèle
        """
        self.config = config or {}
        self.signal_generator = SignalGenerator(config)
        self.tech_indicators = TechnicalIndicators()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.market_regime = "normal"  # 'normal', 'bullish', 'bearish', 'volatile'
        
        # Paramètres par défaut
        self.default_params = {
            "prediction_horizon": 5,  # Nombre de périodes pour prédire
            "training_window": 500,  # Nombre de périodes pour l'entraînement
            "feature_selection": True,  # Activer la sélection des caractéristiques
            "min_confidence_threshold": 0.6,  # Seuil minimal de confiance
            "ensemble_weights": {
                "random_forest": 0.6,
                "gradient_boosting": 0.4
            }
        }
        
        # Fusionner avec la configuration fournie
        for key, default_value in self.default_params.items():
            self.config[key] = self.config.get(key, default_value)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les caractéristiques pour le modèle ML.
        
        Args:
            data: DataFrame contenant les données OHLCV
        
        Returns:
            DataFrame avec toutes les caractéristiques calculées
        """
        if data.empty:
            logger.warning("Données vides fournies à prepare_features")
            return pd.DataFrame()
        
        # Vérification des colonnes requises
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Colonnes manquantes dans les données. Requises: {required_columns}")
            return pd.DataFrame()
        
        # Calculer les indicateurs techniques
        self.tech_indicators.set_data(data)
        features = data.copy()
        
        # Ajouter tous les indicateurs techniques comme caractéristiques
        features = self.tech_indicators.add_all_indicators(features)
        
        # Ajouter des caractéristiques de prix
        features['price_change'] = features['close'].pct_change()
        features['price_change_lag1'] = features['price_change'].shift(1)
        features['price_change_lag2'] = features['price_change'].shift(2)
        
        # Ajouter des caractéristiques de volume
        features['volume_change'] = features['volume'].pct_change()
        features['volume_change_lag1'] = features['volume_change'].shift(1)
        features['vol_price_corr'] = features['volume'].rolling(window=10).corr(features['close'])
        
        # Ajouter des caractéristiques de volatilité
        features['volatility'] = features['close'].rolling(window=20).std() / features['close'].rolling(window=20).mean()
        
        # Suppression des lignes avec des valeurs NaN
        features = features.dropna()
        
        return features
    
    def prepare_labels(self, data: pd.DataFrame, horizon: int = None) -> pd.Series:
        """
        Prépare les labels pour l'entraînement supervisé.
        
        Args:
            data: DataFrame contenant les données OHLCV
            horizon: Nombre de périodes futures pour le calcul des labels
            
        Returns:
            Série avec les labels (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        if horizon is None:
            horizon = self.config["prediction_horizon"]
        
        # Calculer le rendement futur sur l'horizon spécifié
        future_returns = data['close'].shift(-horizon) / data['close'] - 1
        
        # Définir des seuils pour les signaux d'achat/vente
        # Utiliser la volatilité récente pour déterminer les seuils adaptatifs
        volatility = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
        buy_threshold = volatility * 1.5  # Seuil adaptatif basé sur la volatilité
        sell_threshold = -volatility * 1.5
        
        # Créer les labels
        labels = pd.Series(0, index=future_returns.index)  # Par défaut: neutre
        labels[future_returns > buy_threshold] = 1  # Acheter
        labels[future_returns < sell_threshold] = -1  # Vendre
        
        return labels
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Détecte le régime de marché actuel pour adapter les prédictions.
        
        Args:
            data: DataFrame contenant les données OHLCV récentes
            
        Returns:
            Régime de marché identifié ('normal', 'bullish', 'bearish', 'volatile')
        """
        if data.empty or len(data) < 20:
            return "normal"
        
        # Calculer les rendements et la volatilité
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        trend = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
        
        # Définir des seuils pour la classification
        high_volatility_threshold = 0.02  # 2% de volatilité quotidienne
        strong_trend_threshold = 0.1  # 10% de tendance sur la période
        
        # Classifier le régime
        if volatility > high_volatility_threshold:
            regime = "volatile"
        elif trend > strong_trend_threshold:
            regime = "bullish"
        elif trend < -strong_trend_threshold:
            regime = "bearish"
        else:
            regime = "normal"
        
        logger.info(f"Régime de marché détecté: {regime}")
        return regime
    
    def train(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """
        Entraîne le modèle sur les données historiques.
        
        Args:
            data: DataFrame contenant les données OHLCV
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
        """
        logger.info(f"Entraînement du modèle pour {symbol} sur timeframe {timeframe}")
        
        if data.empty or len(data) < 100:
            logger.warning(f"Données insuffisantes pour l'entraînement ({len(data)} points)")
            return False
        
        try:
            # Préparation des caractéristiques et labels
            features_df = self.prepare_features(data)
            labels = self.prepare_labels(data)
            
            # Alignement des données
            common_index = features_df.index.intersection(labels.index)
            features_df = features_df.loc[common_index]
            labels = labels.loc[common_index]
            
            # Sélection des colonnes numériques uniquement
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            X = features_df[numeric_cols].values
            y = labels.values
            
            if len(X) < 50 or len(np.unique(y)) < 2:
                logger.warning("Données insuffisantes ou déséquilibrées pour l'entraînement")
                return False
            
            # Normalisation des caractéristiques
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Division en ensembles d'entraînement et de validation avec validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Entraînement des modèles
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = {}
            self.scalers[model_key] = scaler
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            # Entraînement avec validation croisée temporelle
            best_score = {
                "random_forest": 0,
                "gradient_boosting": 0
            }
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Random Forest
                rf_model.fit(X_train, y_train)
                rf_score = rf_model.score(X_val, y_val)
                if rf_score > best_score["random_forest"]:
                    self.models[model_key]["random_forest"] = rf_model
                    best_score["random_forest"] = rf_score
                
                # Gradient Boosting
                gb_model.fit(X_train, y_train)
                gb_score = gb_model.score(X_val, y_val)
                if gb_score > best_score["gradient_boosting"]:
                    self.models[model_key]["gradient_boosting"] = gb_model
                    best_score["gradient_boosting"] = gb_score
            
            # Caractéristiques importantes
            rf_feature_importance = self.models[model_key]["random_forest"].feature_importances_
            self.feature_importance[model_key] = dict(zip(numeric_cols, rf_feature_importance))
            
            # Détecter le régime de marché actuel
            self.market_regime = self.detect_market_regime(data.tail(30))
            
            logger.info(f"Entraînement réussi pour {symbol}_{timeframe}")
            logger.info(f"Scores: RF={best_score['random_forest']:.3f}, GB={best_score['gradient_boosting']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """
        Génère des prédictions et signaux de trading pour les données fournies.
        
        Args:
            data: DataFrame contenant les données OHLCV récentes
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
            
        Returns:
            Liste des signaux générés avec leur confiance
        """
        model_key = f"{symbol}_{timeframe}"
        
        # Si le modèle n'est pas entraîné, utiliser uniquement le générateur de signaux
        if model_key not in self.models:
            logger.warning(f"Modèle non entraîné pour {symbol}_{timeframe}, utilisation du générateur de signaux uniquement")
            return self.signal_generator.generate_signals(data, symbol, timeframe)
        
        try:
            # Préparation des caractéristiques
            features_df = self.prepare_features(data)
            if features_df.empty:
                return []
            
            # Sélection des colonnes numériques
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            X = features_df[numeric_cols].values
            
            # Normalisation
            X_scaled = self.scalers[model_key].transform(X)
            
            # Prédictions de chaque modèle de l'ensemble
            predictions = {}
            probabilities = {}
            
            # Random Forest
            if "random_forest" in self.models[model_key]:
                rf_model = self.models[model_key]["random_forest"]
                predictions["random_forest"] = rf_model.predict(X_scaled)
                probabilities["random_forest"] = rf_model.predict_proba(X_scaled)
            
            # Gradient Boosting
            if "gradient_boosting" in self.models[model_key]:
                gb_model = self.models[model_key]["gradient_boosting"]
                predictions["gradient_boosting"] = gb_model.predict(X_scaled)
                probabilities["gradient_boosting"] = gb_model.predict_proba(X_scaled)
            
            # Combiner les prédictions avec les poids de l'ensemble
            weights = self.config["ensemble_weights"]
            
            # Initialiser les probabilités finales
            ensemble_probs = np.zeros((len(X_scaled), 3))  # 3 classes: -1, 0, 1
            
            # Agréger les probabilités pondérées
            for model_name, model_probs in probabilities.items():
                if model_name in weights:
                    # Appliquer le poids du modèle
                    ensemble_probs += model_probs * weights[model_name]
            
            # Normaliser les probabilités
            ensemble_probs /= sum(weights.values())
            
            # Convertir en prédictions de classe
            ensemble_preds = np.argmax(ensemble_probs, axis=1) - 1  # -1, 0, 1
            
            # Obtenir les probabilités maximales comme mesure de confiance
            confidences = np.max(ensemble_probs, axis=1)
            
            # Générer des signaux basés sur les prédictions ML
            signals = []
            last_idx = features_df.index[-1]
            last_price = data.loc[data.index[-1], 'close']
            
            # Signal basé sur la dernière prédiction
            last_pred = ensemble_preds[-1]
            last_conf = confidences[-1]
            
            # Appliquer le seuil de confiance minimum
            if last_conf >= self.config["min_confidence_threshold"]:
                signal_type = None
                
                if last_pred == 1:  # Signal d'achat
                    signal_type = SignalType.BUY
                elif last_pred == -1:  # Signal de vente
                    signal_type = SignalType.SELL
                
                if signal_type:
                    signals.append(Signal(
                        type=signal_type,
                        symbol=symbol,
                        timestamp=last_idx,
                        price=last_price,
                        confidence=last_conf,
                        source="ML_Ensemble",
                        timeframe=timeframe,
                        metadata={
                            "market_regime": self.market_regime,
                            "ensemble_prob": ensemble_probs[-1].tolist()
                        }
                    ))
            
            # Combiner avec les signaux du générateur de base
            basic_signals = self.signal_generator.generate_signals(data, symbol, timeframe)
            
            # Ajuster la confiance des signaux de base en fonction du régime de marché
            for signal in basic_signals:
                # Modifier la confiance en fonction du régime de marché
                if self.market_regime == "volatile":
                    # Réduire la confiance en période volatile
                    signal.confidence *= 0.8
                elif self.market_regime == "bullish" and signal.type == SignalType.BUY:
                    # Augmenter la confiance des signaux d'achat en marché haussier
                    signal.confidence = min(1.0, signal.confidence * 1.2)
                elif self.market_regime == "bearish" and signal.type == SignalType.SELL:
                    # Augmenter la confiance des signaux de vente en marché baissier
                    signal.confidence = min(1.0, signal.confidence * 1.2)
            
            # Fusionner tous les signaux
            signals.extend(basic_signals)
            
            # Trier par confiance
            signals.sort(key=lambda x: -x.confidence)
            
            return signals
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return self.signal_generator.generate_signals(data, symbol, timeframe)
    
    def evaluate(self, test_data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Évalue les performances du modèle sur un jeu de données de test.
        
        Args:
            test_data: DataFrame contenant les données OHLCV de test
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            logger.warning(f"Modèle non entraîné pour {symbol}_{timeframe}, impossible d'évaluer")
            return {}
        
        try:
            # Préparation des caractéristiques et labels
            features_df = self.prepare_features(test_data)
            true_labels = self.prepare_labels(test_data)
            
            # Alignement des données
            common_index = features_df.index.intersection(true_labels.index)
            features_df = features_df.loc[common_index]
            true_labels = true_labels.loc[common_index]
            
            # Sélection des colonnes numériques
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            X = features_df[numeric_cols].values
            y = true_labels.values
            
            # Normalisation
            X_scaled = self.scalers[model_key].transform(X)
            
            # Prédictions de chaque modèle
            ensemble_weights = self.config["ensemble_weights"]
            ensemble_preds = np.zeros_like(y, dtype=float)
            
            for model_name, model in self.models[model_key].items():
                if model_name in ensemble_weights:
                    weight = ensemble_weights[model_name]
                    model_preds = model.predict(X_scaled)
                    ensemble_preds += model_preds * weight
            
            # Convertir en prédictions de classe (-1, 0, 1)
            threshold = 0.5
            final_preds = np.zeros_like(y)
            final_preds[ensemble_preds > threshold] = 1
            final_preds[ensemble_preds < -threshold] = -1
            
            # Calcul des métriques (en considérant chaque classe séparément)
            metrics = {}
            
            # Accuracy globale
            metrics["accuracy"] = accuracy_score(y, final_preds)
            
            # Métriques pour les signaux d'achat (classe 1)
            buy_precision = precision_score(y == 1, final_preds == 1, zero_division=0)
            buy_recall = recall_score(y == 1, final_preds == 1, zero_division=0)
            buy_f1 = f1_score(y == 1, final_preds == 1, zero_division=0)
            
            metrics["buy"] = {
                "precision": buy_precision,
                "recall": buy_recall,
                "f1": buy_f1
            }
            
            # Métriques pour les signaux de vente (classe -1)
            sell_precision = precision_score(y == -1, final_preds == -1, zero_division=0)
            sell_recall = recall_score(y == -1, final_preds == -1, zero_division=0)
            sell_f1 = f1_score(y == -1, final_preds == -1, zero_division=0)
            
            metrics["sell"] = {
                "precision": sell_precision,
                "recall": sell_recall,
                "f1": sell_f1
            }
            
            # Métrique de balance des prédictions
            pred_counts = {
                "buy": np.sum(final_preds == 1),
                "neutral": np.sum(final_preds == 0),
                "sell": np.sum(final_preds == -1)
            }
            metrics["prediction_balance"] = pred_counts
            
            logger.info(f"Évaluation pour {symbol}_{timeframe}: Accuracy={metrics['accuracy']:.3f}")
            logger.info(f"Buy signals: Precision={buy_precision:.3f}, Recall={buy_recall:.3f}")
            logger.info(f"Sell signals: Precision={sell_precision:.3f}, Recall={sell_recall:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation: {str(e)}")
            return {}
    
    def auto_adjust_weights(self, historical_performance: List[Dict], symbol: str, timeframe: str) -> bool:
        """
        Ajuste automatiquement les poids des modèles dans l'ensemble en fonction 
        des performances historiques.
        
        Args:
            historical_performance: Liste des métriques d'évaluation historiques
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
            
        Returns:
            True si l'ajustement a été effectué, False sinon
        """
        if not historical_performance or len(historical_performance) < 3:
            logger.warning("Données de performance insuffisantes pour l'auto-ajustement")
            return False
        
        model_key = f"{symbol}_{timeframe}"
        if model_key not in self.models:
            logger.warning(f"Modèle non entraîné pour {symbol}_{timeframe}, impossible d'ajuster")
            return False
        
        try:
            # Calculer les performances moyennes de chaque modèle
            model_performances = {}
            
            # Extraire les performances par modèle si disponibles
            for perf in historical_performance:
                if "model_metrics" in perf:
                    for model_name, model_perf in perf["model_metrics"].items():
                        if model_name not in model_performances:
                            model_performances[model_name] = []
                        model_performances[model_name].append(model_perf["accuracy"])
            
            # Si pas de métriques par modèle, impossible d'ajuster
            if not model_performances:
                logger.warning("Métriques par modèle non disponibles pour l'auto-ajustement")
                return False
            
            # Calculer les performances moyennes
            avg_performances = {}
            for model_name, perfs in model_performances.items():
                avg_performances[model_name] = np.mean(perfs)
            
            # Normaliser les performances en poids
            total_perf = sum(avg_performances.values())
            if total_perf <= 0:
                logger.warning("Performances totales négatives ou nulles, impossible d'ajuster")
                return False
            
            new_weights = {}
            for model_name, perf in avg_performances.items():
                # Assurer un poids minimum de 0.1 pour maintenir la diversité
                new_weights[model_name] = max(0.1, perf / total_perf)
            
            # Normaliser les poids pour qu'ils somment à 1
            weight_sum = sum(new_weights.values())
            for model_name in new_weights:
                new_weights[model_name] /= weight_sum
            
            # Mettre à jour les poids
            old_weights = self.config["ensemble_weights"].copy()
            self.config["ensemble_weights"] = new_weights
            
            logger.info(f"Poids ajustés: {old_weights} -> {new_weights}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'auto-ajustement: {str(e)}")
            return False
    
    def evaluate_with_model_breakdown(self, test_data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Évalue les performances de chaque modèle dans l'ensemble séparément.
        
        Args:
            test_data: DataFrame contenant les données OHLCV de test
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
            
        Returns:
            Dictionnaire contenant les métriques pour chaque modèle
        """
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            logger.warning(f"Modèle non entraîné pour {symbol}_{timeframe}, impossible d'évaluer")
            return {}
        
        try:
            # Préparation des caractéristiques et labels
            features_df = self.prepare_features(test_data)
            true_labels = self.prepare_labels(test_data)
            
            # Alignement des données
            common_index = features_df.index.intersection(true_labels.index)
            features_df = features_df.loc[common_index]
            true_labels = true_labels.loc[common_index]
            
            # Sélection des colonnes numériques
            numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
            X = features_df[numeric_cols].values
            y = true_labels.values
            
            # Normalisation
            X_scaled = self.scalers[model_key].transform(X)
            
            # Résultats par modèle
            model_metrics = {}
            ensemble_metrics = self.evaluate(test_data, symbol, timeframe)
            
            # Évaluer chaque modèle individuellement
            for model_name, model in self.models[model_key].items():
                preds = model.predict(X_scaled)
                
                accuracy = accuracy_score(y, preds)
                buy_precision = precision_score(y == 1, preds == 1, zero_division=0)
                buy_recall = recall_score(y == 1, preds == 1, zero_division=0)
                sell_precision = precision_score(y == -1, preds == -1, zero_division=0)
                sell_recall = recall_score(y == -1, preds == -1, zero_division=0)
                
                model_metrics[model_name] = {
                    "accuracy": accuracy,
                    "buy_precision": buy_precision,
                    "buy_recall": buy_recall,
                    "sell_precision": sell_precision,
                    "sell_recall": sell_recall
                }
                
                logger.info(f"Modèle {model_name}: Accuracy={accuracy:.3f}")
            
            # Résultat global
            result = {
                "ensemble": ensemble_metrics,
                "model_metrics": model_metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation détaillée: {str(e)}")
            return {}
    
    def adapt_to_market_conditions(self, recent_data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Adapte le modèle aux conditions actuelles du marché.
        
        Args:
            recent_data: DataFrame contenant les données OHLCV récentes
            symbol: Symbole de l'actif
            timeframe: Timeframe des données
            
        Returns:
            True si l'adaptation a été effectuée, False sinon
        """
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            logger.warning(f"Modèle non entraîné pour {symbol}_{timeframe}, impossible d'adapter")
            return False
        
        try:
            # Détecter le nouveau régime de marché
            new_regime = self.detect_market_regime(recent_data)
            
            # Si le régime a changé, ajuster les hyperparamètres
            if new_regime != self.market_regime:
                self.market_regime = new_regime
                logger.info(f"Régime de marché mis à jour: {self.market_regime}")
                
                # Ajuster le seuil de confiance en fonction du régime
                if new_regime == "volatile":
                    # Augmenter le seuil de confiance en période volatile
                    self.config["min_confidence_threshold"] = min(0.8, self.config["min_confidence_threshold"] * 1.2)
                elif new_regime in ["bullish", "bearish"]:
                    # Seuil normal pour les tendances claires
                    self.config["min_confidence_threshold"] = self.default_params["min_confidence_threshold"]
                else:  # régime normal
                    # Seuil légèrement réduit pour le régime normal
                    self.config["min_confidence_threshold"] = max(0.5, self.config["min_confidence_threshold"] * 0.9)
                
                logger.info(f"Seuil de confiance ajusté: {self.config['min_confidence_threshold']:.2f}")
                
                # Ajuster l'horizon de prédiction
                if new_regime == "volatile":
                    # Horizon plus court en période volatile
                    self.config["prediction_horizon"] = max(1, self.default_params["prediction_horizon"] // 2)
                elif new_regime in ["bullish", "bearish"]:
                    # Horizon normal pour les tendances claires
                    self.config["prediction_horizon"] = self.default_params["prediction_horizon"]
                else:  # régime normal
                    # Horizon légèrement plus long pour le régime normal
                    self.config["prediction_horizon"] = self.default_params["prediction_horizon"] + 1
                
                logger.info(f"Horizon de prédiction ajusté: {self.config['prediction_horizon']}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de l'adaptation au marché: {str(e)}")
            return False
    
    def calibrate_confidence(self, historical_signals: List[Signal], actual_outcomes: pd.DataFrame) -> None:
        """
        Calibre les scores de confiance des signaux en fonction des résultats réels.
        
        Args:
            historical_signals: Liste des signaux générés historiquement
            actual_outcomes: DataFrame contenant les résultats réels
        """
        if not historical_signals or actual_outcomes.empty:
            logger.warning("Données insuffisantes pour la calibration de confiance")
            return
        
        try:
            # Créer un mapping des signaux par source
            signals_by_source = {}
            for signal in historical_signals:
                if signal.source not in signals_by_source:
                    signals_by_source[signal.source] = []
                signals_by_source[signal.source].append(signal)
            
            # Calculer le taux de réussite par source
            success_rates = {}
            for source, signals in signals_by_source.items():
                successes = 0
                total = 0
                
                for signal in signals:
                    # Trouver les données correspondant à la période après le signal
                    future_data = actual_outcomes[actual_outcomes.index > signal.timestamp]
                    if len(future_data) >= 5:  # Vérifier au moins 5 périodes futures
                        # Calculer le rendement après le signal
                        future_return = future_data['close'].iloc[4] / actual_outcomes.loc[actual_outcomes.index <= signal.timestamp, 'close'].iloc[-1] - 1
                        
                        if signal.type == SignalType.BUY and future_return > 0.01:
                            successes += 1
                        elif signal.type == SignalType.SELL and future_return < -0.01:
                            successes += 1
                        
                        total += 1
                
                if total > 0:
                    success_rates[source] = successes / total
            
            # Ajuster les facteurs de confiance dans la configuration
            if not hasattr(self, 'confidence_factors'):
                self.confidence_factors = {}
            
            for source, success_rate in success_rates.items():
                # Facteur entre 0.5 et 1.5 basé sur le taux de réussite
                # - 0.5 si taux de 0% (réduire la confiance de moitié)
                # - 1.0 si taux de 50% (confiance inchangée)
                # - 1.5 si taux de 100% (augmenter la confiance de 50%)
                self.confidence_factors[source] = 0.5 + success_rate
                logger.info(f"Facteur de confiance pour {source}: {self.confidence_factors[source]:.2f} (taux de réussite: {success_rate:.2f})")
            
        except Exception as e:
            logger.error(f"Erreur lors de la calibration de confiance: {str(e)}")
    
    def apply_confidence_calibration(self, signals: List[Signal]) -> List[Signal]:
        """
        Applique la calibration de confiance aux signaux générés.
        
        Args:
            signals: Liste des signaux générés
            
        Returns:
            Liste des signaux avec confiance calibrée
        """
        if not hasattr(self, 'confidence_factors') or not self.confidence_factors:
            return signals
        
        calibrated_signals = []
        for signal in signals:
            # Copie du signal pour éviter de modifier l'original
            calibrated = Signal(
                type=signal.type,
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                price=signal.price,
                confidence=signal.confidence,
                source=signal.source,
                timeframe=signal.timeframe,
                metadata=signal.metadata.copy() if signal.metadata else None
            )
            
            # Appliquer le facteur de calibration
            if signal.source in self.confidence_factors:
                factor = self.confidence_factors[signal.source]
                calibrated.confidence = min(1.0, calibrated.confidence * factor)
                
                # Ajouter l'information de calibration aux métadonnées
                if calibrated.metadata is None:
                    calibrated.metadata = {}
                calibrated.metadata["original_confidence"] = signal.confidence
                calibrated.metadata["calibration_factor"] = factor
            
            calibrated_signals.append(calibrated)
        
        return calibrated_signals 