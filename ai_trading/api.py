import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_trading.data_processor import DataProcessor
from ai_trading.rl_agent import CryptoTradingEnv, RLAgent

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ai_trading/info_retour/logs/api.log", mode="a"), logging.StreamHandler()],
)

logger = logging.getLogger("trading_api")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Trading RL",
    description="API pour l'agent de trading par renforcement",
    version="1.0.0",
)


# Modèles de données
class PredictionRequest(BaseModel):
    exchange: str = Field(default="binance", description="Exchange à utiliser")
    symbol: str = Field(default="BTC/USDT", description="Paire de trading")
    timeframe: str = Field(default="1h", description="Intervalle de temps")
    use_latest: bool = Field(
        default=True, description="Utiliser les dernières données disponibles"
    )
    days: Optional[int] = Field(
        default=1, description="Nombre de jours de données à analyser"
    )


class TrainingRequest(BaseModel):
    exchange: str = Field(default="binance", description="Exchange à utiliser")
    symbol: str = Field(default="BTC/USDT", description="Paire de trading")
    timeframe: str = Field(default="1h", description="Intervalle de temps")
    days: int = Field(default=60, description="Nombre de jours de données historiques")
    download: bool = Field(default=True, description="Télécharger de nouvelles données")
    timesteps: int = Field(default=50000, description="Nombre d'étapes d'entraînement")


class BacktestRequest(BaseModel):
    exchange: str = Field(default="binance", description="Exchange à utiliser")
    symbol: str = Field(default="BTC/USDT", description="Paire de trading")
    timeframe: str = Field(default="1h", description="Intervalle de temps")
    days: int = Field(
        default=30, description="Nombre de jours de données pour le backtest"
    )
    model_path: Optional[str] = None


# Dépendances
def get_data_processor():
    return DataProcessor(
        data_dir=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ai_trading/info_retour/data"
        )
    )


def get_agent():
    return RLAgent(
        model_dir=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ai_trading/info_retour/models"
        )
    )


# Routes API
@app.get("/")
async def root():
    return {"message": "API Trading RL - Bienvenue!"}


@app.post("/predict")
async def predict(
    request: PredictionRequest,
    data_processor: DataProcessor = Depends(get_data_processor),
    agent: RLAgent = Depends(get_agent),
):
    try:
        logger.info(f"Prédiction demandée pour {request.symbol} sur {request.exchange}")

        # Charger ou télécharger les données
        if request.use_latest:
            # Télécharger les dernières données
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.days)
            df = data_processor.download_historical_data(
                exchange_id=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date,
                save=False,
            )
        else:
            # Utiliser des données existantes
            data_path = f"ai_trading/info_retour/data/{request.exchange}_{request.symbol.replace('/', '_')}_{request.timeframe}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(
                    status_code=404, detail=f"Données non trouvées: {data_path}"
                )
            df = data_processor.load_data(data_path)

        # Ajouter les indicateurs techniques
        df = data_processor.add_indicators(df)

        # Trouver le modèle approprié
        model_name = (
            f"{request.symbol.replace('/', '_')}_{request.timeframe}_rl_model.zip"
        )
        model_path = f"ai_trading/info_retour/models/rl/{model_name}"

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"Modèle non trouvé: {model_path}"
            )

        # Charger le modèle
        agent.load(model_path)

        # Créer un environnement pour la dernière donnée
        env = CryptoTradingEnv(df.iloc[-20:])  # Utiliser les 20 dernières observations
        observation, _ = env.reset()

        # Prédire l'action
        action = agent.predict(observation)

        # Convertir l'action en recommandation
        recommendation = "ATTENDRE"
        if action == 1:
            recommendation = "ACHETER"
        elif action == 2:
            recommendation = "VENDRE"

        # Récupérer les dernières informations du marché
        latest_data = df.iloc[-1].to_dict()

        # Formater le résultat
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "action": int(action),
            "recommendation": recommendation,
            "confidence": 0.8,  # À améliorer avec la probabilité réelle
            "market_data": {
                "price": latest_data["close"],
                "rsi": latest_data["rsi"],
                "macd": latest_data["macd"],
                "signal": latest_data["signal_line"],
                "ema9": latest_data["ema9"],
                "ema21": latest_data["ema21"],
            },
        }

        logger.info(f"Prédiction pour {request.symbol}: {recommendation}")
        return result

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(
    request: TrainingRequest,
    data_processor: DataProcessor = Depends(get_data_processor),
    agent: RLAgent = Depends(get_agent),
):
    try:
        logger.info(
            f"Entraînement demandé pour {request.symbol} sur {request.exchange}"
        )

        # Préparation des données
        if request.download:
            # Télécharger de nouvelles données
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.days)
            df = data_processor.download_historical_data(
                exchange_id=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # Utiliser des données existantes
            data_path = f"ai_trading/info_retour/data/{request.exchange}_{request.symbol.replace('/', '_')}_{request.timeframe}.csv"
            if not os.path.exists(data_path):
                raise HTTPException(
                    status_code=404, detail=f"Données non trouvées: {data_path}"
                )
            df = data_processor.load_data(data_path)

        # Ajouter les indicateurs techniques
        df = data_processor.add_indicators(df)

        # Préparer les données pour l'entraînement
        train_data, test_data = data_processor.preprocess_for_training(df)

        # Entraîner l'agent
        model_name = f"{request.symbol.replace('/', '_')}_{request.timeframe}_rl_model"
        save_path = f"ai_trading/info_retour/models/rl/{model_name}.zip"

        logger.info(
            f"Début de l'entraînement sur {len(train_data)} points de données..."
        )
        metrics = agent.train(
            train_data, total_timesteps=request.timesteps, save_path=save_path
        )

        logger.info(f"Entraînement terminé. Modèle sauvegardé dans {save_path}")

        # Effectuer un backtest rapide
        backtest_results = agent.backtest(test_data)

        return {
            "status": "success",
            "model_path": save_path,
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "backtest_results": {
                "profit_pct": backtest_results["profit_pct"],
                "buy_hold_pct": backtest_results["bh_profit_pct"],
                "sharpe_ratio": backtest_results["sharpe_ratio"],
                "num_trades": len(backtest_results["trades"]),
            },
        }

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def backtest(
    request: BacktestRequest,
    data_processor: DataProcessor = Depends(get_data_processor),
    agent: RLAgent = Depends(get_agent),
):
    try:
        logger.info(f"Backtest demandé pour {request.symbol} sur {request.exchange}")

        # Télécharger les données pour le backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        df = data_processor.download_historical_data(
            exchange_id=request.exchange,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date,
            save=False,
        )

        # Ajouter les indicateurs techniques
        df = data_processor.prepare_backtesting_data(df)

        # Charger le modèle
        if request.model_path:
            model_path = request.model_path
        else:
            model_name = (
                f"{request.symbol.replace('/', '_')}_{request.timeframe}_rl_model.zip"
            )
            model_path = f"ai_trading/info_retour/models/rl/{model_name}"

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"Modèle non trouvé: {model_path}"
            )

        agent.load(model_path)

        # Effectuer le backtest
        backtest_results = agent.backtest(df)

        # Formater les résultats
        trades_summary = []
        for trade in backtest_results["trades"]:
            trades_summary.append({"profit_pct": float(trade * 100), "win": trade > 0})

        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "model": os.path.basename(model_path),
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "profit_pct": float(backtest_results["profit_pct"]),
            "buy_hold_pct": float(backtest_results["bh_profit_pct"]),
            "sharpe_ratio": float(backtest_results["sharpe_ratio"]),
            "num_trades": len(backtest_results["trades"]),
            "win_rate": (
                len([t for t in backtest_results["trades"] if t > 0])
                / len(backtest_results["trades"])
                if len(backtest_results["trades"]) > 0
                else 0
            ),
            "trades_summary": trades_summary[:10],  # Limiter à 10 trades pour la sortie
        }

    except Exception as e:
        logger.error(f"Erreur lors du backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.route("/api/ema_metrics", methods=["GET"])
def get_ema_metrics():
    periods = request.args.getlist("periods", type=int) or [5, 10, 15, 20, 25, 30, 50]
    data = fetch_market_data()
    processed = add_ema_features(data.copy(), periods)
    return jsonify(
        {
            "timestamps": processed["timestamp"].tolist(),
            "emas": {f"ema_{p}": processed[f"ema_{p}"].tolist() for p in periods},
            "ribbon_width": processed["ema_ribbon_width"].tolist(),
        }
    )


if __name__ == "__main__":
    # Créer le dossier logs s'il n'existe pas
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "info_retour/logs"), exist_ok=True
    )
    uvicorn.run("ai_trading.api:app", host="0.0.0.0", port=8000, reload=True)
