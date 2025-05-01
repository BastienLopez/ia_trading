import torch
import numpy as np
import time
import multiprocessing as mp
import threading
import queue
import uuid
import socket
import pickle
import zmq
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

class ExperienceWorker:
    """
    Travailleur de collecte d'expériences pour l'apprentissage par renforcement distribué.
    
    S'exécute dans un processus ou une machine séparée et collecte des
    expériences à partir d'un environnement.
    """
    
    def __init__(
        self,
        env_creator: Callable,
        policy: Any,
        worker_id: str = None,
        batch_size: int = 32,
        send_freq: int = 10,
        master_address: str = "localhost",
        master_port: int = 5555,
        local_mode: bool = False
    ):
        """
        Initialise un travailleur de collecte d'expériences.
        
        Args:
            env_creator: Fonction qui crée un environnement
            policy: Politique (modèle) à utiliser pour la collecte
            worker_id: Identifiant unique du travailleur
            batch_size: Taille des lots d'expériences à collecter
            send_freq: Fréquence d'envoi des expériences
            master_address: Adresse du serveur maître
            master_port: Port du serveur maître
            local_mode: Mode local (sans communication réseau)
        """
        self.env_creator = env_creator
        self.policy = policy
        self.worker_id = worker_id if worker_id else str(uuid.uuid4())[:8]
        self.batch_size = batch_size
        self.send_freq = send_freq
        self.master_address = master_address
        self.master_port = master_port
        self.local_mode = local_mode
        
        # Créer l'environnement
        self.env = self.env_creator()
        
        # Buffer local pour stocker les expériences avant envoi
        self.local_buffer = []
        
        # File pour communication locale
        if local_mode:
            self.queue = queue.Queue()
        else:
            # Contexte ZMQ pour communication réseau
            self.context = zmq.Context()
            self.socket = None
        
        # État interne
        self.running = False
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.total_steps = 0
        self.metrics = {}
        self.worker_process = None
        
        # Configuration du logging
        self.logger = logging.getLogger(f"ExperienceWorker-{self.worker_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_connection(self):
        """Configure la connexion au serveur maître."""
        if not self.local_mode:
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.connect(f"tcp://{self.master_address}:{self.master_port}")
            self.logger.info(f"Connecté au maître: {self.master_address}:{self.master_port}")
    
    def _send_experiences(self):
        """Envoie les expériences collectées au serveur maître."""
        if not self.local_buffer:
            return
        
        if self.local_mode:
            # Mode local: mettre dans la queue
            self.queue.put(self.local_buffer)
        else:
            # Mode réseau: envoyer via ZMQ
            try:
                # Ajouter des métadonnées
                data = {
                    "worker_id": self.worker_id,
                    "timestamp": time.time(),
                    "experiences": self.local_buffer,
                    "metrics": {
                        "steps": self.total_steps,
                        "episodes": len(self.episode_rewards),
                        "mean_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    }
                }
                
                # Sérialiser et envoyer
                serialized = pickle.dumps(data)
                self.socket.send(serialized)
                
                self.logger.info(f"Envoyé {len(self.local_buffer)} expériences au maître")
            except Exception as e:
                self.logger.error(f"Erreur lors de l'envoi des expériences: {e}")
        
        # Vider le buffer local
        self.local_buffer = []
    
    def collect_experience(self, steps: int):
        """
        Collecte des expériences pendant un nombre défini d'étapes.
        
        Args:
            steps: Nombre d'étapes à collecter
        """
        self.logger.info(f"Démarrage de la collecte pour {steps} étapes")
        
        # Initialiser la connexion si nécessaire
        if not self.local_mode and self.socket is None:
            self._setup_connection()
        
        # Reset initial de l'environnement
        state = self.env.reset()
        
        for step in range(steps):
            # Sélectionner une action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.policy(state_tensor).argmax().item()
            
            # Exécuter l'action
            next_state, reward, done, info = self.env.step(action)
            
            # Stocker l'expérience
            experience = (state, action, reward, next_state, done)
            self.local_buffer.append(experience)
            
            # Mettre à jour les métriques
            self.total_steps += 1
            self.current_episode_reward += reward
            
            # Si l'épisode est terminé
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                state = self.env.reset()
            else:
                state = next_state
            
            # Envoyer périodiquement les expériences
            if len(self.local_buffer) >= self.batch_size and step % self.send_freq == 0:
                self._send_experiences()
        
        # Envoyer les expériences restantes
        if self.local_buffer:
            self._send_experiences()
            
        self.logger.info(f"Collecte terminée: {steps} étapes, {len(self.episode_rewards)} épisodes")
    
    def run_in_process(self, steps: int):
        """
        Exécute la collecte d'expériences dans un processus séparé.
        
        Args:
            steps: Nombre d'étapes à collecter
            
        Returns:
            Process: Le processus créé
        """
        self.worker_process = mp.Process(
            target=self.collect_experience,
            args=(steps,)
        )
        self.worker_process.start()
        self.logger.info(f"Travailleur démarré dans le processus {self.worker_process.pid}")
        return self.worker_process
    
    def stop(self):
        """Arrête le travailleur."""
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
            self.logger.info("Travailleur arrêté")
        
        if not self.local_mode and self.socket:
            self.socket.close()
            self.context.term()
    
    def get_metrics(self) -> Dict:
        """
        Obtient les métriques du travailleur.
        
        Returns:
            Dictionnaire de métriques
        """
        metrics = {
            "worker_id": self.worker_id,
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards)
        }
        
        if self.episode_rewards:
            metrics.update({
                "mean_reward": float(np.mean(self.episode_rewards[-100:])),
                "max_reward": float(np.max(self.episode_rewards)),
                "min_reward": float(np.min(self.episode_rewards)),
                "latest_reward": float(self.episode_rewards[-1]) if self.episode_rewards else 0
            })
        
        return metrics


class ExperienceMaster:
    """
    Maître qui coordonne la collecte d'expériences distribuées.
    
    Reçoit des expériences de plusieurs travailleurs et les stocke
    dans un buffer d'expérience central.
    """
    
    def __init__(
        self,
        replay_buffer: Any,
        port: int = 5555,
        max_workers: int = 10,
        update_freq: int = 1000,
        policy_provider: Optional[Callable] = None,
        local_mode: bool = False
    ):
        """
        Initialise le maître de collecte d'expériences.
        
        Args:
            replay_buffer: Tampon pour stocker les expériences
            port: Port d'écoute pour les connexions entrantes
            max_workers: Nombre maximum de travailleurs
            update_freq: Fréquence de mise à jour de la politique
            policy_provider: Fonction pour obtenir la politique à jour
            local_mode: Mode local (sans communication réseau)
        """
        self.replay_buffer = replay_buffer
        self.port = port
        self.max_workers = max_workers
        self.update_freq = update_freq
        self.policy_provider = policy_provider
        self.local_mode = local_mode
        
        # Tracking des travailleurs
        self.workers = {}  # {worker_id: worker_info}
        self.local_workers = []  # Liste des travailleurs en mode local
        
        # Statistiques
        self.total_experiences = 0
        self.start_time = None
        self.worker_metrics = {}
        
        # État interne
        self.running = False
        self.receive_thread = None
        
        # Pour le mode local
        if local_mode:
            self.queue = queue.Queue()
        else:
            # Contexte ZMQ pour communication réseau
            self.context = zmq.Context()
            self.socket = None
        
        # Configuration du logging
        self.logger = logging.getLogger("ExperienceMaster")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_server(self):
        """Configure le serveur pour recevoir les expériences."""
        if not self.local_mode:
            self.socket = self.context.socket(zmq.PULL)
            self.socket.bind(f"tcp://*:{self.port}")
            self.logger.info(f"Serveur démarré sur le port {self.port}")
    
    def _receive_experiences(self):
        """
        Boucle de réception des expériences des travailleurs.
        Cette méthode s'exécute dans un thread séparé.
        """
        self.logger.info("Démarrage de la réception d'expériences")
        self.start_time = time.time()
        self.running = True
        
        while self.running:
            try:
                if self.local_mode:
                    # Mode local: récupérer de la queue
                    try:
                        experiences = self.queue.get(timeout=1.0)
                        worker_id = "local"
                    except queue.Empty:
                        continue
                else:
                    # Mode réseau: recevoir via ZMQ
                    serialized = self.socket.recv()
                    data = pickle.loads(serialized)
                    
                    experiences = data["experiences"]
                    worker_id = data["worker_id"]
                    self.worker_metrics[worker_id] = data.get("metrics", {})
                
                # Traiter les expériences
                self._process_experiences(experiences, worker_id)
                
            except Exception as e:
                if self.running:  # Ignorer les erreurs pendant l'arrêt
                    self.logger.error(f"Erreur lors de la réception des expériences: {e}")
    
    def _process_experiences(self, experiences, worker_id):
        """
        Traite les expériences reçues d'un travailleur.
        
        Args:
            experiences: Liste d'expériences
            worker_id: ID du travailleur
        """
        # Mettre à jour les statistiques
        exp_count = len(experiences)
        self.total_experiences += exp_count
        
        # Enregistrer le travailleur s'il est nouveau
        if worker_id not in self.workers:
            self.workers[worker_id] = {
                "first_seen": time.time(),
                "last_seen": time.time(),
                "experiences_count": exp_count
            }
        else:
            self.workers[worker_id]["last_seen"] = time.time()
            self.workers[worker_id]["experiences_count"] += exp_count
        
        # Ajouter les expériences au buffer
        for exp in experiences:
            state, action, reward, next_state, done = exp
            
            # Convertir en tensors si nécessaire
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state)
            
            # Ajouter au buffer d'expériences
            if hasattr(self.replay_buffer, "add"):
                self.replay_buffer.add(state, action, reward, next_state, done)
            else:
                # Pour les buffers avec une autre interface
                self.replay_buffer.append((state, action, reward, next_state, done))
        
        self.logger.info(f"Reçu {exp_count} expériences de {worker_id}. Total: {self.total_experiences}")
        
        # Mettre à jour la politique si nécessaire
        if self.total_experiences % self.update_freq == 0 and self.policy_provider:
            self._update_worker_policies()
    
    def _update_worker_policies(self):
        """Met à jour les politiques des travailleurs locaux."""
        if not self.policy_provider or not self.local_workers:
            return
        
        try:
            # Obtenir la politique à jour
            latest_policy = self.policy_provider()
            
            # Mettre à jour les travailleurs locaux
            for worker in self.local_workers:
                worker.policy = latest_policy
            
            self.logger.info(f"Politiques mises à jour pour {len(self.local_workers)} travailleurs locaux")
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des politiques: {e}")
    
    def start(self):
        """Démarre le maître pour recevoir les expériences."""
        if not self.local_mode:
            self._setup_server()
        
        # Démarrer le thread de réception
        self.receive_thread = threading.Thread(target=self._receive_experiences)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        self.logger.info("Maître démarré")
    
    def add_local_worker(self, worker: ExperienceWorker):
        """
        Ajoute un travailleur local.
        
        Args:
            worker: Travailleur à ajouter
        """
        if not worker.local_mode:
            worker.local_mode = True
            worker.queue = self.queue
        
        self.local_workers.append(worker)
        self.logger.info(f"Travailleur local ajouté: {worker.worker_id}")
    
    def stop(self):
        """Arrête le maître et tous les travailleurs locaux."""
        self.running = False
        
        # Arrêter les travailleurs locaux
        for worker in self.local_workers:
            worker.stop()
        
        # Attendre la fin du thread de réception
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        if not self.local_mode and self.socket:
            self.socket.close()
            self.context.term()
        
        self.logger.info("Maître arrêté")
    
    def get_metrics(self) -> Dict:
        """
        Obtient les métriques du maître.
        
        Returns:
            Dictionnaire de métriques
        """
        runtime = time.time() - self.start_time if self.start_time else 0
        
        metrics = {
            "total_experiences": self.total_experiences,
            "active_workers": len(self.workers),
            "runtime_seconds": runtime,
            "experiences_per_second": self.total_experiences / max(1, runtime),
            "workers": self.workers
        }
        
        # Ajouter les métriques des travailleurs
        metrics["worker_metrics"] = self.worker_metrics
        
        return metrics


class DistributedExperienceManager:
    """
    Gestionnaire de haut niveau pour la collecte d'expérience distribuée.
    
    Fournit une interface simplifiée pour configurer et gérer un système
    de collecte d'expérience distribuée.
    """
    
    def __init__(
        self,
        env_creator: Callable,
        policy: Any,
        replay_buffer: Any,
        n_local_workers: int = 4,
        batch_size: int = 32,
        port: int = 5555,
        update_freq: int = 1000,
        policy_provider: Optional[Callable] = None
    ):
        """
        Initialise le gestionnaire d'expérience distribuée.
        
        Args:
            env_creator: Fonction qui crée un environnement
            policy: Politique initiale
            replay_buffer: Tampon d'expérience
            n_local_workers: Nombre de travailleurs locaux
            batch_size: Taille des lots d'expériences
            port: Port pour les connexions
            update_freq: Fréquence de mise à jour de la politique
            policy_provider: Fonction pour obtenir la politique à jour
        """
        self.env_creator = env_creator
        self.initial_policy = policy
        self.replay_buffer = replay_buffer
        self.n_local_workers = n_local_workers
        self.batch_size = batch_size
        self.port = port
        self.update_freq = update_freq
        self.policy_provider = policy_provider
        
        # Créer le maître
        self.master = ExperienceMaster(
            replay_buffer=replay_buffer,
            port=port,
            update_freq=update_freq,
            policy_provider=policy_provider,
            local_mode=True  # Mode local par défaut
        )
        
        # Créer les travailleurs locaux
        self.local_workers = []
        
        # Configuration du logging
        self.logger = logging.getLogger("DistributedExperienceManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def initialize(self):
        """Initialise le système de collecte d'expérience."""
        # Démarrer le maître
        self.master.start()
        
        # Créer et ajouter les travailleurs locaux
        for i in range(self.n_local_workers):
            worker = ExperienceWorker(
                env_creator=self.env_creator,
                policy=self.initial_policy,
                worker_id=f"local-{i}",
                batch_size=self.batch_size,
                local_mode=True
            )
            self.local_workers.append(worker)
            self.master.add_local_worker(worker)
        
        self.logger.info(f"Système initialisé avec {self.n_local_workers} travailleurs locaux")
        return self
    
    def start_collection(self, steps_per_worker: int = 1000):
        """
        Démarre la collecte d'expériences.
        
        Args:
            steps_per_worker: Nombre d'étapes par travailleur
        """
        self.logger.info(f"Démarrage de la collecte: {steps_per_worker} étapes par travailleur")
        
        # Démarrer les travailleurs
        for worker in self.local_workers:
            worker.run_in_process(steps_per_worker)
        
        return self
    
    def wait_until_done(self, timeout: Optional[float] = None):
        """
        Attend que tous les travailleurs aient terminé.
        
        Args:
            timeout: Temps maximum d'attente en secondes
        """
        start_time = time.time()
        all_done = False
        
        while not all_done:
            # Vérifier si tous les processus sont terminés
            all_done = all(not w.worker_process.is_alive() for w in self.local_workers
                          if w.worker_process is not None)
            
            # Vérifier le timeout
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning(f"Timeout atteint après {timeout} secondes")
                break
            
            # Ne pas surcharger le CPU
            time.sleep(0.1)
        
        # Calculer les statistiques
        active_count = sum(1 for w in self.local_workers 
                          if w.worker_process is not None and w.worker_process.is_alive())
        
        self.logger.info(f"Collection terminée. {len(self.local_workers) - active_count}/{len(self.local_workers)} travailleurs terminés")
        return self
    
    def stop(self):
        """Arrête la collecte d'expériences."""
        # Arrêter les travailleurs
        for worker in self.local_workers:
            worker.stop()
        
        # Arrêter le maître
        self.master.stop()
        
        self.logger.info("Système arrêté")
        return self
    
    def get_metrics(self) -> Dict:
        """
        Obtient les métriques du système.
        
        Returns:
            Dictionnaire de métriques
        """
        master_metrics = self.master.get_metrics()
        
        worker_metrics = {}
        for i, worker in enumerate(self.local_workers):
            worker_metrics[f"local-{i}"] = worker.get_metrics()
        
        return {
            "master": master_metrics,
            "workers": worker_metrics,
            "buffer_size": len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else "unknown",
            "n_local_workers": self.n_local_workers
        } 