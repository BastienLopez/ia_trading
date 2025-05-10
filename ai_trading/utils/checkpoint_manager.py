"""
Module de gestion des points de contrôle (checkpoints) pour les modèles et sessions de trading.

Ce module fournit des fonctionnalités pour :
- Sauvegarder et charger l'état des modèles
- Gérer les checkpoints avec rotation automatique
- Journaliser les sauvegardes et les reprises
- Gérer les différentes versions des modèles
"""

import datetime
import json
import os
import shutil
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ai_trading.utils.advanced_logging import get_logger

# Logger pour ce module
logger = get_logger("ai_trading.utils.checkpoint_manager")

# Répertoire de base pour les checkpoints
BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / "info_retour" / "checkpoints"

# S'assurer que le répertoire de checkpoints existe
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class CheckpointType(Enum):
    """Types de points de contrôle."""

    MODEL = "model"  # Modèle d'apprentissage
    SESSION = "session"  # Session de trading
    STATE = "state"  # État interne d'un agent
    OPTIMIZER = "optimizer"  # État d'un optimiseur
    BUFFER = "buffer"  # Buffer de replay
    FULL = "full"  # Sauvegarde complète


@dataclass
class CheckpointMetadata:
    """Métadonnées pour un point de contrôle."""

    checkpoint_id: str  # Identifiant unique du checkpoint
    type: CheckpointType  # Type de checkpoint
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    version: str = "1.0"  # Version du format de checkpoint
    description: str = ""  # Description du checkpoint
    metrics: Dict[str, Any] = field(default_factory=dict)  # Métriques associées
    custom_data: Dict[str, Any] = field(default_factory=dict)  # Données personnalisées

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métadonnées en dictionnaire."""
        result = asdict(self)
        result["type"] = self.type.value  # Convertir l'enum en chaîne
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Crée une instance de CheckpointMetadata à partir d'un dictionnaire."""
        # Convertir la chaîne en enum
        data["type"] = CheckpointType(data["type"])
        return cls(**data)


class CheckpointManager:
    """
    Gestionnaire de points de contrôle pour sauvegarder et charger les modèles et sessions.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        max_checkpoints: int = 10,
        save_interval_minutes: Optional[int] = None,
        compression_level: int = 5,
        auto_save: bool = False,
    ):
        """
        Initialise le gestionnaire de points de contrôle.

        Args:
            root_dir: Répertoire racine pour les checkpoints
            max_checkpoints: Nombre maximum de checkpoints à conserver
            save_interval_minutes: Intervalle de sauvegarde automatique en minutes
            compression_level: Niveau de compression (0-9)
            auto_save: Activer la sauvegarde automatique
        """
        self.root_dir = root_dir or CHECKPOINT_DIR
        self.max_checkpoints = max_checkpoints
        self.save_interval_minutes = save_interval_minutes
        self.compression_level = compression_level
        self.auto_save = auto_save

        # État interne
        self._autosave_timer = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # S'assurer que le répertoire existe
        os.makedirs(self.root_dir, exist_ok=True)

        logger.info(f"Gestionnaire de checkpoints initialisé dans {self.root_dir}")
        logger.info(
            f"Configuration: max_checkpoints={max_checkpoints}, "
            f"save_interval={save_interval_minutes} min, "
            f"compression={compression_level}, "
            f"auto_save={auto_save}"
        )

        # Démarrer la sauvegarde automatique si demandée
        if self.auto_save and self.save_interval_minutes:
            self.start_auto_save()

    def start_auto_save(self) -> None:
        """Démarre la sauvegarde automatique des checkpoints."""
        if self._autosave_timer is not None:
            logger.warning("La sauvegarde automatique est déjà activée")
            return

        self._stop_event.clear()
        self._autosave_timer = threading.Thread(
            target=self._auto_save_loop, daemon=True
        )
        self._autosave_timer.start()
        logger.info(
            f"Sauvegarde automatique démarrée (intervalle: {self.save_interval_minutes} min)"
        )

    def stop_auto_save(self) -> None:
        """Arrête la sauvegarde automatique des checkpoints."""
        if self._autosave_timer is None:
            logger.warning("La sauvegarde automatique n'est pas activée")
            return

        self._stop_event.set()
        self._autosave_timer.join(timeout=1.0)
        self._autosave_timer = None
        logger.info("Sauvegarde automatique arrêtée")

    def _auto_save_loop(self) -> None:
        """Boucle de sauvegarde automatique."""
        while not self._stop_event.is_set():
            # Attendre l'intervalle spécifié
            wait_time = self.save_interval_minutes * 60
            if self._stop_event.wait(wait_time):
                break

            # Effectuer la sauvegarde automatique
            try:
                # Cette méthode doit être remplacée par une implémentation concrète
                # dans une classe dérivée ou par un callback
                self._on_auto_save()
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde automatique: {str(e)}")

    def _on_auto_save(self) -> None:
        """Méthode appelée pour effectuer la sauvegarde automatique."""
        logger.info(
            "Sauvegarde automatique (cette méthode doit être remplacée par une implémentation concrète)"
        )

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """
        Retourne le chemin d'un checkpoint.

        Args:
            checkpoint_id: Identifiant du checkpoint

        Returns:
            Chemin du fichier de checkpoint
        """
        return self.root_dir / f"{checkpoint_id}.zip"

    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        """
        Retourne le chemin des métadonnées d'un checkpoint.

        Args:
            checkpoint_id: Identifiant du checkpoint

        Returns:
            Chemin du fichier de métadonnées
        """
        return self.root_dir / f"{checkpoint_id}.json"

    def _generate_checkpoint_id(self, prefix: str, type: CheckpointType) -> str:
        """
        Génère un identifiant unique pour un checkpoint.

        Args:
            prefix: Préfixe pour l'identifiant
            type: Type de checkpoint

        Returns:
            Identifiant unique
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{type.value}_{timestamp}"

    def list_checkpoints(
        self, type: Optional[CheckpointType] = None
    ) -> List[Dict[str, Any]]:
        """
        Liste tous les checkpoints disponibles.

        Args:
            type: Filtrer par type de checkpoint

        Returns:
            Liste des métadonnées des checkpoints
        """
        checkpoints = []

        # Parcourir tous les fichiers JSON dans le répertoire
        for metadata_file in self.root_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Vérifier si le checkpoint existe
                checkpoint_path = self._get_checkpoint_path(metadata["checkpoint_id"])
                if not checkpoint_path.exists():
                    logger.warning(
                        f"Fichier de checkpoint manquant pour {metadata['checkpoint_id']}"
                    )
                    continue

                # Filtrer par type si spécifié
                if type and metadata["type"] != type.value:
                    continue

                # Ajouter le checkpoint à la liste
                checkpoints.append(metadata)

            except Exception as e:
                logger.error(
                    f"Erreur lors de la lecture des métadonnées de {metadata_file}: {str(e)}"
                )

        # Trier par date (du plus récent au plus ancien)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        return checkpoints

    def _cleanup_old_checkpoints(self, type: Optional[CheckpointType] = None) -> None:
        """
        Supprime les checkpoints les plus anciens lorsqu'il y en a trop.

        Args:
            type: Filtrer par type de checkpoint
        """
        checkpoints = self.list_checkpoints(type)

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Supprimer les checkpoints les plus anciens
        for checkpoint in checkpoints[self.max_checkpoints :]:
            self.delete_checkpoint(checkpoint["checkpoint_id"])

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Supprime un checkpoint et ses métadonnées.

        Args:
            checkpoint_id: Identifiant du checkpoint

        Returns:
            True si la suppression a réussi
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        metadata_path = self._get_metadata_path(checkpoint_id)

        success = True

        # Supprimer le fichier de checkpoint
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception as e:
                logger.error(
                    f"Erreur lors de la suppression du checkpoint {checkpoint_id}: {str(e)}"
                )
                success = False

        # Supprimer le fichier de métadonnées
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                logger.error(
                    f"Erreur lors de la suppression des métadonnées du checkpoint {checkpoint_id}: {str(e)}"
                )
                success = False

        if success:
            logger.info(f"Checkpoint {checkpoint_id} supprimé")

        return success

    def get_latest_checkpoint(
        self, type: Optional[CheckpointType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retourne le checkpoint le plus récent.

        Args:
            type: Filtrer par type de checkpoint

        Returns:
            Métadonnées du checkpoint le plus récent, ou None si aucun n'est disponible
        """
        checkpoints = self.list_checkpoints(type)

        if not checkpoints:
            return None

        return checkpoints[0]

    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les métadonnées d'un checkpoint.

        Args:
            checkpoint_id: Identifiant du checkpoint

        Returns:
            Métadonnées du checkpoint, ou None si le checkpoint n'existe pas
        """
        metadata_path = self._get_metadata_path(checkpoint_id)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(
                f"Erreur lors de la lecture des métadonnées de {checkpoint_id}: {str(e)}"
            )
            return None

    def save_checkpoint(
        self,
        obj: Any,
        type: CheckpointType,
        prefix: str,
        description: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Sauvegarde un objet dans un checkpoint.

        Args:
            obj: Objet à sauvegarder
            type: Type de checkpoint
            prefix: Préfixe pour l'identifiant du checkpoint
            description: Description du checkpoint
            metrics: Métriques associées au checkpoint
            custom_data: Données personnalisées

        Returns:
            Identifiant du checkpoint créé, ou None en cas d'erreur
        """
        # Générer un identifiant unique
        checkpoint_id = self._generate_checkpoint_id(prefix, type)

        # Préparer les métadonnées
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            type=type,
            description=description,
            metrics=metrics or {},
            custom_data=custom_data or {},
        )

        # Créer un répertoire temporaire pour préparer les fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Sauvegarder l'objet
                if isinstance(obj, torch.nn.Module):
                    self._save_torch_model(obj, temp_path, checkpoint_id)
                elif isinstance(obj, dict):
                    self._save_dict(obj, temp_path, checkpoint_id)
                else:
                    logger.error(f"Type d'objet non pris en charge: {type(obj)}")
                    return None

                # Créer le fichier ZIP de checkpoint
                checkpoint_path = self._get_checkpoint_path(checkpoint_id)
                metadata_path = self._get_metadata_path(checkpoint_id)

                # Sauvegarder les métadonnées
                with open(metadata_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2)

                # Compresser les fichiers
                with zipfile.ZipFile(
                    checkpoint_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=self.compression_level,
                ) as zipf:
                    # Ajouter tous les fichiers du répertoire temporaire
                    for file_path in temp_path.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(temp_path)
                            zipf.write(file_path, rel_path)

                logger.info(f"Checkpoint {checkpoint_id} créé avec succès")

                # Nettoyer les anciens checkpoints
                self._cleanup_old_checkpoints(type)

                return checkpoint_id

            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du checkpoint: {str(e)}")

                # Supprimer les fichiers en cas d'erreur
                metadata_path = self._get_metadata_path(checkpoint_id)
                if metadata_path.exists():
                    metadata_path.unlink()

                checkpoint_path = self._get_checkpoint_path(checkpoint_id)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

                return None

    def _save_torch_model(
        self, model: torch.nn.Module, temp_path: Path, checkpoint_id: str
    ) -> None:
        """
        Sauvegarde un modèle PyTorch.

        Args:
            model: Modèle PyTorch
            temp_path: Répertoire temporaire
            checkpoint_id: Identifiant du checkpoint
        """
        model_path = temp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Sauvegarder aussi les informations sur l'architecture
        model_info = {
            "class_name": model.__class__.__name__,
            "module": model.__class__.__module__,
        }

        info_path = temp_path / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

    def _save_dict(
        self, data: Dict[str, Any], temp_path: Path, checkpoint_id: str
    ) -> None:
        """
        Sauvegarde un dictionnaire.

        Args:
            data: Dictionnaire à sauvegarder
            temp_path: Répertoire temporaire
            checkpoint_id: Identifiant du checkpoint
        """
        # Séparer les données en fonction de leur type
        torch_tensors = {}
        numpy_arrays = {}
        json_data = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                torch_tensors[key] = value
            elif isinstance(value, np.ndarray):
                numpy_arrays[key] = value
            else:
                # Essayer de sérialiser en JSON
                try:
                    json.dumps({key: value})  # Test de sérialisation
                    json_data[key] = value
                except (TypeError, OverflowError):
                    logger.warning(
                        f"Impossible de sérialiser la valeur pour la clé '{key}', elle sera ignorée"
                    )

        # Sauvegarder les tenseurs PyTorch
        if torch_tensors:
            torch_path = temp_path / "tensors.pt"
            torch.save(torch_tensors, torch_path)

        # Sauvegarder les tableaux NumPy
        if numpy_arrays:
            numpy_path = temp_path / "arrays.npz"
            np.savez_compressed(numpy_path, **numpy_arrays)

        # Sauvegarder les données JSON
        if json_data:
            json_path = temp_path / "data.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

    def load_checkpoint(
        self,
        checkpoint_id: str,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Charge un checkpoint.

        Args:
            checkpoint_id: Identifiant du checkpoint
            model: Modèle PyTorch à remplir (optionnel)

        Returns:
            Dictionnaire contenant les données chargées
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} non trouvé")

        # Créer un répertoire temporaire pour extraire les fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Extraire le checkpoint
                with zipfile.ZipFile(checkpoint_path, "r") as zipf:
                    zipf.extractall(temp_path)

                # Charger les données
                result = {}

                # Charger le modèle PyTorch si demandé
                model_path = temp_path / "model.pt"
                if model_path.exists() and model is not None:
                    model.load_state_dict(torch.load(model_path))
                    result["model"] = model

                # Charger les tenseurs PyTorch
                tensors_path = temp_path / "tensors.pt"
                if tensors_path.exists():
                    tensors = torch.load(tensors_path)
                    result.update(tensors)

                # Charger les tableaux NumPy
                arrays_path = temp_path / "arrays.npz"
                if arrays_path.exists():
                    with np.load(arrays_path) as arrays:
                        for key in arrays:
                            result[key] = arrays[key]

                # Charger les données JSON
                json_path = temp_path / "data.json"
                if json_path.exists():
                    with open(json_path, "r") as f:
                        json_data = json.load(f)
                        result.update(json_data)

                # Charger les métadonnées
                metadata = self.get_checkpoint_metadata(checkpoint_id)
                if metadata:
                    result["metadata"] = metadata

                logger.info(f"Checkpoint {checkpoint_id} chargé avec succès")

                return result

            except Exception as e:
                logger.error(
                    f"Erreur lors du chargement du checkpoint {checkpoint_id}: {str(e)}"
                )
                raise

    def export_checkpoint(self, checkpoint_id: str, export_path: Path) -> bool:
        """
        Exporte un checkpoint vers un emplacement spécifié.

        Args:
            checkpoint_id: Identifiant du checkpoint
            export_path: Chemin d'exportation

        Returns:
            True si l'exportation a réussi
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        metadata_path = self._get_metadata_path(checkpoint_id)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint {checkpoint_id} non trouvé")
            return False

        try:
            # Créer le répertoire d'exportation si nécessaire
            os.makedirs(export_path.parent, exist_ok=True)

            # Copier le fichier de checkpoint
            shutil.copy(checkpoint_path, export_path)

            # Copier les métadonnées si elles existent
            if metadata_path.exists():
                export_metadata_path = export_path.with_suffix(".json")
                shutil.copy(metadata_path, export_metadata_path)

            logger.info(f"Checkpoint {checkpoint_id} exporté vers {export_path}")

            return True

        except Exception as e:
            logger.error(
                f"Erreur lors de l'exportation du checkpoint {checkpoint_id}: {str(e)}"
            )
            return False

    def import_checkpoint(self, import_path: Path) -> Optional[str]:
        """
        Importe un checkpoint depuis un emplacement spécifié.

        Args:
            import_path: Chemin d'importation

        Returns:
            Identifiant du checkpoint importé, ou None en cas d'erreur
        """
        if not import_path.exists():
            logger.error(f"Fichier d'importation {import_path} non trouvé")
            return None

        try:
            # Générer un identifiant unique
            checkpoint_id = f"imported_{import_path.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Chemins de destination
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)

            # Copier le fichier de checkpoint
            shutil.copy(import_path, checkpoint_path)

            # Copier les métadonnées si elles existent
            import_metadata_path = import_path.with_suffix(".json")
            if import_metadata_path.exists():
                shutil.copy(import_metadata_path, metadata_path)
            else:
                # Créer des métadonnées minimales
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    type=CheckpointType.FULL,
                    description=f"Importé depuis {import_path}",
                )

                with open(metadata_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2)

            logger.info(f"Checkpoint importé avec succès: {checkpoint_id}")

            return checkpoint_id

        except Exception as e:
            logger.error(f"Erreur lors de l'importation du checkpoint: {str(e)}")
            return None

    def create_session_snapshot(
        self,
        session_data: Dict[str, Any],
        name: str,
        description: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Crée un snapshot de session.

        Args:
            session_data: Données de la session
            name: Nom de la session
            description: Description de la session
            metrics: Métriques associées à la session

        Returns:
            Identifiant du snapshot créé, ou None en cas d'erreur
        """
        return self.save_checkpoint(
            obj=session_data,
            type=CheckpointType.SESSION,
            prefix=name,
            description=description,
            metrics=metrics,
        )

    def save_model(
        self,
        model: torch.nn.Module,
        name: str,
        description: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Sauvegarde un modèle PyTorch.

        Args:
            model: Modèle PyTorch
            name: Nom du modèle
            description: Description du modèle
            metrics: Métriques associées au modèle

        Returns:
            Identifiant du checkpoint créé, ou None en cas d'erreur
        """
        return self.save_checkpoint(
            obj=model,
            type=CheckpointType.MODEL,
            prefix=name,
            description=description,
            metrics=metrics,
        )

    def load_model(
        self,
        checkpoint_id: str,
        model: torch.nn.Module,
    ) -> bool:
        """
        Charge un modèle PyTorch depuis un checkpoint.

        Args:
            checkpoint_id: Identifiant du checkpoint
            model: Modèle PyTorch à remplir

        Returns:
            True si le chargement a réussi
        """
        try:
            result = self.load_checkpoint(checkpoint_id, model)
            return "model" in result
        except Exception as e:
            logger.error(
                f"Erreur lors du chargement du modèle {checkpoint_id}: {str(e)}"
            )
            return False


# Gestionnaire global de checkpoints
_checkpoint_manager = None


def get_checkpoint_manager() -> CheckpointManager:
    """
    Retourne le gestionnaire global de checkpoints.

    Returns:
        Gestionnaire de checkpoints
    """
    global _checkpoint_manager

    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()

    return _checkpoint_manager
