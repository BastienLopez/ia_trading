"""
Module d'optimisation par checkpointing efficace.

Ce module fournit des utilitaires pour sauvegarder les modèles de manière efficace,
en économisant de l'espace disque et en minimisant l'impact sur l'entraînement.
"""

import os
import time
import logging
import threading
import queue
import torch
from typing import Dict, Optional, Union, Callable, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def save_model_weights_only(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    create_dir: bool = True
) -> Path:
    """
    Sauvegarde uniquement les poids du modèle, sans l'état de l'optimiseur.
    
    Args:
        model: Le modèle PyTorch à sauvegarder
        save_path: Chemin où sauvegarder le modèle
        create_dir: Si True, crée le répertoire parent si nécessaire
        
    Returns:
        Le chemin du fichier sauvegardé
    """
    save_path = Path(save_path)
    
    # Créer le répertoire parent si nécessaire
    if create_dir:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Obtenir uniquement l'état des poids
    model_state = model.state_dict()
    
    # Sauvegarder les poids
    torch.save(model_state, save_path)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"Modèle sauvegardé (poids uniquement) à {save_path} ({size_mb:.2f} MB)")
    
    return save_path


class AsyncCheckpointSaver:
    """
    Classe pour sauvegarder les modèles de manière asynchrone.
    
    Cela permet de continuer l'entraînement pendant que la sauvegarde s'effectue
    dans un thread séparé, évitant ainsi de bloquer le processus principal.
    """
    
    def __init__(self, max_queue_size: int = 3):
        """
        Initialise le sauvegardeur asynchrone.
        
        Args:
            max_queue_size: Taille maximale de la file d'attente (nombre de sauvegardes en attente)
        """
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.active = False
        self.pending_tasks = 0
    
    def _worker_loop(self):
        """Boucle principale du thread de sauvegarde."""
        logger.info("Thread de sauvegarde asynchrone démarré")
        
        while not self.stop_event.is_set() or not self.save_queue.empty():
            try:
                # Récupérer une tâche avec timeout pour pouvoir vérifier stop_event
                task = self.save_queue.get(timeout=1.0)
                if task is not None:
                    save_fn, args, kwargs, callback, save_path = task
                    
                    # Exécuter la sauvegarde
                    try:
                        result = save_fn(*args, **kwargs)
                        if callback and save_path:
                            callback(save_path)
                    except Exception as e:
                        logger.error(f"Erreur lors de la sauvegarde asynchrone: {e}")
                    
                    # Marquer la tâche comme terminée
                    self.save_queue.task_done()
                    self.pending_tasks -= 1
            except queue.Empty:
                # File d'attente vide, continuer la boucle
                pass
        
        logger.info("Thread de sauvegarde asynchrone arrêté")
    
    def start(self):
        """Démarre le thread de sauvegarde."""
        if self.active:
            return
        
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.active = True
        logger.info("Sauvegardeur asynchrone démarré")
    
    def stop(self, wait_for_completion: bool = True):
        """
        Arrête le thread de sauvegarde.
        
        Args:
            wait_for_completion: Si True, attend que toutes les tâches soient terminées
        """
        if not self.active:
            return
        
        # Signaler l'arrêt
        self.stop_event.set()
        
        # Attendre que toutes les tâches soient terminées si demandé
        if wait_for_completion and self.worker_thread is not None:
            logger.info("Attente de la fin des sauvegardes en cours...")
            self.worker_thread.join()
        
        self.active = False
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        save_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        weights_only: bool = True,
        callback: Optional[Callable[[Path], None]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """
        Enqueue une sauvegarde asynchrone.
        
        Args:
            model: Le modèle à sauvegarder
            save_path: Chemin où sauvegarder le checkpoint
            optimizer: L'optimiseur (optionnel, ignoré si weights_only=True)
            weights_only: Si True, sauvegarde uniquement les poids
            callback: Fonction à appeler après la sauvegarde
            extra_data: Données supplémentaires à inclure dans le checkpoint
        """
        if not self.active:
            self.start()
        
        save_path = Path(save_path)
        
        # Créer le répertoire parent si nécessaire
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cloner les états pour éviter les modifications pendant la sauvegarde
        model_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        
        if weights_only:
            # Sauvegarde des poids uniquement
            save_fn = torch.save
            save_args = (model_state, save_path)
            save_kwargs = {}
        else:
            # Sauvegarde complète avec optimiseur et données supplémentaires
            checkpoint = {
                'model_state_dict': model_state,
                'epoch': extra_data.get('epoch', 0) if extra_data else 0,
                'timestamp': time.time()
            }
            
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            if extra_data:
                checkpoint.update({k: v for k, v in extra_data.items() if k != 'epoch'})
            
            save_fn = torch.save
            save_args = (checkpoint, save_path)
            save_kwargs = {}
        
        try:
            # Ajouter la tâche à la file d'attente
            self.save_queue.put((save_fn, save_args, save_kwargs, callback, save_path), block=False)
            self.pending_tasks += 1
            logger.info(f"Sauvegarde asynchrone enfilée: {save_path} (poids uniquement: {weights_only})")
            return True
        except queue.Full:
            logger.warning("File d'attente de sauvegarde pleine, ignoration de la demande")
            return False
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Attend que toutes les sauvegardes soient terminées.
        
        Args:
            timeout: Timeout en secondes, None pour attendre indéfiniment
            
        Returns:
            True si toutes les sauvegardes sont terminées, False si timeout
        """
        if not self.active or self.pending_tasks == 0:
            return True
        
        start_time = time.time()
        while self.pending_tasks > 0:
            if timeout is not None and time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        
        return True


def compare_checkpoint_sizes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    base_path: Union[str, Path]
) -> Dict[str, float]:
    """
    Compare les tailles des différents modes de sauvegarde.
    
    Args:
        model: Le modèle à sauvegarder
        optimizer: L'optimiseur
        base_path: Chemin de base pour les sauvegardes
        
    Returns:
        Dictionnaire contenant les tailles en Mo des différents types de sauvegarde
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde complète (modèle + optimiseur)
    full_path = base_path / "full_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, full_path)
    full_size = os.path.getsize(full_path) / (1024 * 1024)
    
    # Sauvegarde du modèle uniquement
    model_path = base_path / "model_only.pt"
    torch.save(model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    # Sauvegarde de l'optimiseur uniquement
    optim_path = base_path / "optim_only.pt"
    torch.save(optimizer.state_dict(), optim_path)
    optim_size = os.path.getsize(optim_path) / (1024 * 1024)
    
    logger.info(f"Taille checkpoint complet: {full_size:.2f} MB")
    logger.info(f"Taille modèle uniquement: {model_size:.2f} MB")
    logger.info(f"Taille optimiseur uniquement: {optim_size:.2f} MB")
    logger.info(f"Ratio taille optimiseur/modèle: {optim_size/model_size:.2f}x")
    
    return {
        "full_checkpoint_mb": full_size,
        "model_only_mb": model_size,
        "optimizer_only_mb": optim_size,
        "optimizer_model_ratio": optim_size / model_size if model_size > 0 else 0
    }


class CheckpointManager:
    """
    Gestionnaire de points de sauvegarde pour un modèle d'IA.
    
    Offre des fonctionnalités comme la rotation automatique des sauvegardes,
    le stockage asynchrone, et l'optimisation de l'espace de stockage.
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        max_checkpoints: int = 5,
        async_save: bool = True,
        weights_only: bool = True,
        save_format: str = "epoch_{epoch:03d}.pt"
    ):
        """
        Initialise le gestionnaire de checkpoints.
        
        Args:
            base_dir: Répertoire de base pour les sauvegardes
            max_checkpoints: Nombre maximum de checkpoints à conserver
            async_save: Si True, utilise des sauvegardes asynchrones
            weights_only: Si True, sauvegarde uniquement les poids du modèle
            save_format: Format de nom de fichier pour les sauvegardes
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.weights_only = weights_only
        self.save_format = save_format
        
        # Tracker pour les checkpoints existants
        self.checkpoints = []
        self._scan_existing_checkpoints()
        
        # Sauvegardes asynchrones
        self.async_save = async_save
        self.async_saver = AsyncCheckpointSaver() if async_save else None
    
    def _scan_existing_checkpoints(self):
        """Scanne les checkpoints existants dans le répertoire."""
        self.checkpoints = sorted([f for f in self.base_dir.glob("*.pt")])
        
        if self.checkpoints:
            logger.info(f"Trouvé {len(self.checkpoints)} checkpoints existants")
            for cp in self.checkpoints[-3:]:  # Afficher les 3 derniers
                size_mb = os.path.getsize(cp) / (1024 * 1024)
                logger.info(f"  - {cp.name} ({size_mb:.2f} MB)")
    
    def _enforce_max_checkpoints(self):
        """Supprime les checkpoints les plus anciens si le maximum est dépassé."""
        while len(self.checkpoints) > self.max_checkpoints:
            to_remove = self.checkpoints.pop(0)  # Le plus ancien
            try:
                os.remove(to_remove)
                logger.info(f"Checkpoint supprimé: {to_remove.name}")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du checkpoint {to_remove}: {e}")
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        extra_data: Optional[Dict[str, Any]] = None,
        custom_filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Sauvegarde un checkpoint.
        
        Args:
            model: Le modèle à sauvegarder
            optimizer: L'optimiseur (ignoré si weights_only=True)
            epoch: Numéro d'époque actuel
            extra_data: Données supplémentaires à inclure
            custom_filename: Nom de fichier personnalisé (optionnel)
            
        Returns:
            Le chemin du fichier sauvegardé ou None si asynchrone
        """
        # Générer le nom de fichier
        if custom_filename:
            filename = custom_filename
        else:
            filename = self.save_format.format(epoch=epoch)
        
        save_path = self.base_dir / filename
        
        # Sauvegarde synchrone ou asynchrone
        if self.async_save and self.async_saver:
            extra = extra_data or {}
            if epoch is not None:
                extra['epoch'] = epoch
            
            def on_save_complete(path):
                # Vérifier si le chemin existe déjà dans la liste
                path_obj = Path(path)
                if path_obj not in self.checkpoints:
                    self.checkpoints.append(path_obj)
                    self.checkpoints.sort()  # Trier par ordre alphabétique (qui correspond souvent à l'ordre chronologique)
                    self._enforce_max_checkpoints()
            
            self.async_saver.save_checkpoint(
                model=model,
                save_path=save_path,
                optimizer=None if self.weights_only else optimizer,
                weights_only=self.weights_only,
                callback=on_save_complete,
                extra_data=extra
            )
            
            # Ajouter à la liste des checkpoints pour l'enforcment immédiat même si asynchrone
            # Cela garantit que si on ajoute plusieurs checkpoints avant que le premier ne soit traité,
            # on ne dépasse pas la limite (au pire temporairement avant nettoyage)
            if save_path not in self.checkpoints:
                self.checkpoints.append(save_path)
                self.checkpoints.sort()
                self._enforce_max_checkpoints()
                
            return None  # Asynchrone, pas de chemin immédiat
        else:
            # Sauvegarde synchrone
            if self.weights_only:
                save_model_weights_only(model, save_path)
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch
                }
                
                if optimizer:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                
                if extra_data:
                    checkpoint.update(extra_data)
                
                torch.save(checkpoint, save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                logger.info(f"Checkpoint complet sauvegardé à {save_path} ({size_mb:.2f} MB)")
            
            self.checkpoints.append(save_path)
            self.checkpoints.sort()
            self._enforce_max_checkpoints()
            
            return save_path
    
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Charge le checkpoint le plus récent.
        
        Args:
            model: Le modèle à charger
            optimizer: L'optimiseur à charger (si disponible dans le checkpoint)
            device: Périphérique sur lequel charger les poids
            
        Returns:
            Dictionnaire contenant les métadonnées du checkpoint
        """
        if not self.checkpoints:
            logger.warning("Aucun checkpoint trouvé à charger")
            return {}
        
        latest_checkpoint = self.checkpoints[-1]
        return self.load(model, optimizer, latest_checkpoint, device)
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Charge un checkpoint spécifique.
        
        Args:
            model: Le modèle à charger
            optimizer: L'optimiseur à charger (si disponible dans le checkpoint)
            checkpoint_path: Chemin du checkpoint à charger
            device: Périphérique sur lequel charger les poids
            
        Returns:
            Dictionnaire contenant les métadonnées du checkpoint
        """
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint non trouvé: {checkpoint_path}")
            return {}
        
        logger.info(f"Chargement du checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Vérifier le format du checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Format complet
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("État de l'optimiseur restauré")
            
            # Extraire les métadonnées
            metadata = {k: v for k, v in checkpoint.items() 
                       if k not in ['model_state_dict', 'optimizer_state_dict']}
            
            logger.info(f"Checkpoint chargé: {checkpoint_path.name}, époque: {metadata.get('epoch', 'inconnu')}")
            return metadata
        else:
            # Format poids uniquement
            model.load_state_dict(checkpoint)
            logger.info(f"Poids du modèle chargés: {checkpoint_path.name}")
            return {'weights_only': True}
    
    def cleanup(self):
        """Nettoie les ressources et arrête les threads en attente."""
        if self.async_save and self.async_saver:
            self.async_saver.stop(wait_for_completion=True)
    
    def __del__(self):
        """Destructeur qui nettoie les ressources."""
        self.cleanup()


if __name__ == "__main__":
    # Configuration du logger
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Exemple d'utilisation
    import torch.nn as nn
    
    # Créer un modèle et un optimiseur de test
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Comparer les tailles de sauvegarde
    compare_checkpoint_sizes(model, optimizer, "checkpoints/comparison")
    
    # Tester le gestionnaire de checkpoints
    checkpoint_mgr = CheckpointManager(
        base_dir="checkpoints/example",
        max_checkpoints=3,
        async_save=True
    )
    
    # Simuler quelques sauvegardes
    for epoch in range(5):
        checkpoint_mgr.save(model, optimizer, epoch=epoch)
        time.sleep(1)  # Attendre un peu entre les sauvegardes
    
    # Attendre la fin des sauvegardes asynchrones
    checkpoint_mgr.async_saver.wait_for_completion()
    
    # Charger le plus récent
    checkpoint_mgr.load_latest(model, optimizer) 