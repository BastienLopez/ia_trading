import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


class HogwildWorker:
    """
    Travailleur pour l'entraînement parallèle HOGWILD!.

    Réalise l'entraînement asynchrone d'un modèle partagé sans verrous
    pour une efficacité maximale.
    """

    def __init__(
        self,
        shared_model: nn.Module,
        worker_id: int,
        buffer: Any,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        loss_fn: Callable,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        mini_batches_per_update: int = 10,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialise un travailleur HOGWILD!.

        Args:
            shared_model: Modèle partagé entre tous les travailleurs
            worker_id: Identifiant unique du travailleur
            buffer: Tampon d'expérience pour l'échantillonnage
            optimizer_factory: Fonction factory pour créer un optimiseur
            loss_fn: Fonction de perte
            batch_size: Taille des lots d'entraînement
            learning_rate: Taux d'apprentissage
            mini_batches_per_update: Nombre de mini-batchs par mise à jour
            device: Appareil sur lequel exécuter le travailleur
            seed: Graine aléatoire
        """
        self.worker_id = worker_id
        self.shared_model = shared_model
        self.buffer = buffer
        self.optimizer_factory = optimizer_factory
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mini_batches_per_update = mini_batches_per_update

        # Définir l'appareil (CPU uniquement pour HOGWILD!)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            if self.device.type != "cpu":
                logging.warning(
                    f"HOGWILD! ne fonctionne qu'avec CPU. Utilisation de CPU pour le travailleur {worker_id}"
                )
                self.device = torch.device("cpu")

        # Définir la graine aléatoire
        if seed is not None:
            self.seed = seed + worker_id  # Différent pour chaque travailleur
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        else:
            self.seed = None

        # Créer une copie locale du modèle
        self.local_model = type(shared_model)()
        self.local_model.load_state_dict(shared_model.state_dict())
        self.local_model.to(self.device)

        # Créer l'optimiseur
        self.optimizer = self.optimizer_factory(
            self.local_model.parameters(), lr=self.learning_rate
        )

        # Statistiques
        self.updates_done = 0
        self.total_loss = 0.0
        self.start_time = None
        self.metrics = {"worker_id": worker_id}
        self.is_running = False
        self.process = None

        # Configuration du logging
        self.logger = logging.getLogger(f"HogwildWorker-{worker_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def update_local_model(self):
        """Met à jour le modèle local avec les poids du modèle partagé."""
        self.local_model.load_state_dict(self.shared_model.state_dict())

    def update_shared_model(self):
        """Met à jour le modèle partagé avec les poids du modèle local."""
        # Accès sans verrou au modèle partagé
        for shared_param, local_param in zip(
            self.shared_model.parameters(), self.local_model.parameters()
        ):
            if shared_param.grad is None:
                shared_param.grad = local_param.grad.clone()
            else:
                shared_param.grad += local_param.grad.clone()

    def train_batch(self) -> float:
        """
        Entraîne le modèle sur un batch d'expériences.

        Returns:
            Valeur de la perte
        """
        # Échantillonner du buffer
        if hasattr(self.buffer, "sample_batch"):
            batch, _ = self.buffer.sample_batch(self.batch_size)
        else:
            batch = self.buffer.sample(self.batch_size)

        # Décompresser le batch
        if isinstance(batch, tuple) and len(batch) == 5:
            states, actions, rewards, next_states, dones = batch

            # Convertir en tensors si nécessaire
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states)
            if not isinstance(actions, torch.Tensor):
                actions = torch.LongTensor(actions)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.FloatTensor(rewards)
            if not isinstance(next_states, torch.Tensor):
                next_states = torch.FloatTensor(next_states)
            if not isinstance(dones, torch.Tensor):
                dones = torch.FloatTensor(dones)

            # Déplacer sur l'appareil
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            # Calculer la perte
            self.optimizer.zero_grad()
            loss = self.loss_fn(
                self.local_model, states, actions, rewards, next_states, dones
            )
        else:
            # Interface personnalisée
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.local_model, batch)

        # Rétropropagation
        loss.backward()

        return loss.item()

    def train_mini_batches(self) -> float:
        """
        Entraîne le modèle sur plusieurs mini-batches.

        Returns:
            Perte moyenne
        """
        total_loss = 0.0

        # Mettre à jour le modèle local avec les poids du modèle partagé
        self.update_local_model()

        # Entraîner sur plusieurs mini-batches
        for _ in range(self.mini_batches_per_update):
            loss = self.train_batch()
            total_loss += loss

        # Calculer la perte moyenne
        avg_loss = total_loss / self.mini_batches_per_update

        # Mettre à jour le modèle partagé avec les gradients locaux
        self.update_shared_model()

        return avg_loss

    def train(self, num_updates: int = 1000):
        """
        Boucle d'entraînement principale.

        Args:
            num_updates: Nombre de mises à jour à effectuer
        """
        self.logger.info(f"Démarrage de l'entraînement avec {num_updates} mises à jour")
        self.start_time = time.time()
        self.is_running = True

        for update in range(num_updates):
            if not self.is_running:
                break

            # Entraîner sur des mini-batches
            avg_loss = self.train_mini_batches()

            # Mettre à jour les métriques
            self.updates_done += 1
            self.total_loss += avg_loss

            # Afficher la progression
            if (update + 1) % 100 == 0:
                elapsed = time.time() - self.start_time
                updates_per_sec = (update + 1) / elapsed
                self.logger.info(
                    f"Travailleur {self.worker_id}: {update+1}/{num_updates} mises à jour, "
                    f"perte: {avg_loss:.5f}, vitesse: {updates_per_sec:.2f} mises à jour/s"
                )

        runtime = time.time() - self.start_time
        updates_per_sec = self.updates_done / runtime
        avg_loss = self.total_loss / max(1, self.updates_done)

        self.logger.info(
            f"Entraînement terminé. Travailleur {self.worker_id}: {self.updates_done} mises à jour, "
            f"perte moyenne: {avg_loss:.5f}, vitesse: {updates_per_sec:.2f} mises à jour/s"
        )

        # Mettre à jour les métriques finales
        self.metrics.update(
            {
                "updates_done": self.updates_done,
                "avg_loss": avg_loss,
                "runtime": runtime,
                "updates_per_sec": updates_per_sec,
            }
        )

    def run_in_process(self, num_updates: int = 1000):
        """
        Exécute l'entraînement dans un processus séparé.

        Args:
            num_updates: Nombre de mises à jour à effectuer

        Returns:
            Le processus créé
        """
        # Réinitialiser les statistiques
        self.updates_done = 0
        self.total_loss = 0.0

        # Créer et démarrer le processus
        self.process = mp.Process(target=self.train, args=(num_updates,))
        self.process.start()

        self.logger.info(f"Travailleur démarré dans le processus {self.process.pid}")
        return self.process

    def stop(self):
        """Arrête le travailleur."""
        self.is_running = False

        if self.process and self.process.is_alive():
            self.process.terminate()
            self.logger.info(f"Travailleur {self.worker_id} arrêté")

    def get_metrics(self) -> Dict:
        """
        Obtient les métriques du travailleur.

        Returns:
            Dictionnaire de métriques
        """
        if self.start_time:
            runtime = time.time() - self.start_time
            updates_per_sec = self.updates_done / max(1, runtime)
            avg_loss = self.total_loss / max(1, self.updates_done)

            self.metrics.update(
                {
                    "updates_done": self.updates_done,
                    "avg_loss": avg_loss,
                    "runtime": runtime,
                    "updates_per_sec": updates_per_sec,
                    "mini_batches_processed": self.updates_done
                    * self.mini_batches_per_update,
                }
            )

        return self.metrics


class HogwildTrainer:
    """
    Coordonnateur d'entraînement HOGWILD!.

    Gère plusieurs travailleurs qui entraînent en parallèle un modèle
    partagé sans verrous.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer: Any,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        loss_fn: Callable,
        num_workers: int = 4,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        mini_batches_per_update: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialise le coordonnateur d'entraînement HOGWILD!.

        Args:
            model: Modèle à entraîner
            buffer: Tampon d'expérience
            optimizer_factory: Fonction factory pour l'optimiseur (ex: lambda params: torch.optim.Adam(params))
            loss_fn: Fonction de perte
            num_workers: Nombre de travailleurs parallèles
            batch_size: Taille des lots d'entraînement
            learning_rate: Taux d'apprentissage
            mini_batches_per_update: Nombre de mini-batchs par mise à jour
            seed: Graine aléatoire de base
        """
        self.model = model
        self.buffer = buffer
        self.optimizer_factory = optimizer_factory
        self.loss_fn = loss_fn
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mini_batches_per_update = mini_batches_per_update
        self.seed = seed

        # Vérifier que le modèle est partageable
        self.model.share_memory()

        # Créer les travailleurs
        self.workers = []
        for i in range(num_workers):
            worker = HogwildWorker(
                shared_model=self.model,
                worker_id=i,
                buffer=self.buffer,
                optimizer_factory=self.optimizer_factory,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                mini_batches_per_update=self.mini_batches_per_update,
                seed=self.seed,
            )
            self.workers.append(worker)

        # Pour le suivi
        self.start_time = None
        self.processes = []
        self.is_running = False

        # Optimiseur central (pour étapes finales optionnelles)
        self.optimizer = self.optimizer_factory(
            self.model.parameters(), lr=self.learning_rate
        )

        # File pour les résultats
        self.results_queue = mp.Queue()

        # Configuration du logging
        self.logger = logging.getLogger("HogwildTrainer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start(self, num_updates_per_worker: int = 1000):
        """
        Démarre l'entraînement parallèle.

        Args:
            num_updates_per_worker: Nombre de mises à jour par travailleur
        """
        self.logger.info(
            f"Démarrage de l'entraînement HOGWILD! avec {self.num_workers} travailleurs"
        )
        self.start_time = time.time()
        self.is_running = True

        # Démarrer tous les travailleurs
        for worker in self.workers:
            process = worker.run_in_process(num_updates_per_worker)
            self.processes.append(process)

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
            active_processes = [p for p in self.processes if p.is_alive()]
            all_done = len(active_processes) == 0

            # Vérifier le timeout
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning(f"Timeout atteint après {timeout} secondes")
                break

            # Afficher la progression
            if len(active_processes) > 0:
                self.logger.info(
                    f"{len(active_processes)}/{len(self.processes)} travailleurs encore actifs"
                )

            # Ne pas surcharger le CPU
            time.sleep(1.0)

        # Calculer le temps total
        runtime = time.time() - self.start_time
        self.logger.info(f"Entraînement terminé en {runtime:.2f} secondes")

        return self

    def stop(self):
        """Arrête tous les travailleurs."""
        self.is_running = False

        # Arrêter tous les travailleurs
        for worker in self.workers:
            worker.stop()

        # Attendre la fin de tous les processus
        for i, process in enumerate(self.processes):
            if process.is_alive():
                process.terminate()
                self.logger.info(f"Processus {i} terminé")

        self.logger.info("Tous les travailleurs ont été arrêtés")
        return self

    def get_metrics(self) -> Dict:
        """
        Obtient les métriques d'entraînement.

        Returns:
            Dictionnaire de métriques
        """
        runtime = time.time() - self.start_time if self.start_time else 0

        # Collecter les métriques de chaque travailleur
        worker_metrics = {}
        total_updates = 0
        total_mini_batches = 0

        for worker in self.workers:
            metrics = worker.get_metrics()
            worker_metrics[f"worker_{worker.worker_id}"] = metrics
            total_updates += metrics.get("updates_done", 0)
            total_mini_batches += metrics.get("mini_batches_processed", 0)

        # Calculer les métriques globales
        global_metrics = {
            "num_workers": self.num_workers,
            "total_updates": total_updates,
            "total_mini_batches": total_mini_batches,
            "runtime": runtime,
            "updates_per_second": total_updates / max(1, runtime),
            "mini_batches_per_second": total_mini_batches / max(1, runtime),
        }

        return {"global": global_metrics, "workers": worker_metrics}

    def save_model(self, path: str):
        """
        Sauvegarde le modèle partagé.

        Args:
            path: Chemin de sauvegarde
        """
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Modèle sauvegardé à {path}")


class AsyncAdvantageActorCritic(nn.Module):
    """
    Implémentation d'Asynchronous Advantage Actor-Critic (A3C) avec HOGWILD!.

    Modèle spécifique conçu pour être utilisé avec l'entraînement HOGWILD!.
    Cette implémentation permet d'appliquer la technique de manière optimale
    pour des problèmes d'apprentissage par renforcement.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        shared_backbone: bool = True,
    ):
        """
        Initialise le modèle A3C.

        Args:
            state_dim: Dimension de l'état d'entrée
            action_dim: Dimension de l'espace d'action
            hidden_dim: Dimension des couches cachées
            shared_backbone: Utiliser un backbone partagé entre acteur et critique
        """
        super(AsyncAdvantageActorCritic, self).__init__()

        # Configurations
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.shared_backbone = shared_backbone

        # Couches communes
        if shared_backbone:
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Têtes séparées
            self.policy_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            # Réseaux séparés
            self.policy_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

            self.value_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, state):
        """
        Propage l'état à travers le réseau.

        Args:
            state: État d'entrée

        Returns:
            Tuple (logits de politique, valeur)
        """
        if self.shared_backbone:
            features = self.backbone(state)
            policy_logits = self.policy_head(features)
            value = self.value_head(features)
        else:
            policy_logits = self.policy_net(state)
            value = self.value_net(state)

        return policy_logits, value

    def get_action(self, state, deterministic: bool = False):
        """
        Obtient une action à partir d'un état.

        Args:
            state: État d'entrée
            deterministic: Si True, retourne l'action la plus probable
                           Sinon, échantillonne selon la distribution

        Returns:
            Action sélectionnée
        """
        with torch.no_grad():
            policy_logits, _ = self.forward(state)

            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                # Échantillonner de la distribution
                probs = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()

        return action.item() if action.dim() == 0 else action.cpu().numpy()


def a3c_loss_fn(model, states, actions, rewards, next_states, dones, gamma=0.99):
    """
    Fonction de perte pour A3C.

    Args:
        model: Modèle ActorCritic
        states: États courants
        actions: Actions prises
        rewards: Récompenses reçues
        next_states: États suivants
        dones: Indicateurs de fin d'épisode
        gamma: Facteur d'actualisation

    Returns:
        Perte totale
    """
    # Obtenir les prédictions du modèle
    policy_logits, values = model(states)

    # Calculer les prédictions pour les états suivants
    with torch.no_grad():
        _, next_values = model(next_states)
        next_values = next_values.squeeze(-1)

        # Calculer les retours (récompenses cumulées actualisées)
        targets = rewards + gamma * next_values * (1 - dones.float())

    # Extraire les valeurs prédites
    values = values.squeeze(-1)

    # Calculer l'avantage
    advantages = targets - values

    # Perte de la critique (MSE)
    value_loss = advantages.pow(2).mean()

    # Perte de l'acteur (entropie croisée pondérée par l'avantage)
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    policy_loss = -(advantages.detach() * action_log_probs).mean()

    # Ajouter un terme d'entropie pour encourager l'exploration
    entropy = -(log_probs * torch.softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

    # Perte totale (équilibrée entre politique et valeur, avec bonus d'entropie)
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    return total_loss
