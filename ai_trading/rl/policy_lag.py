import copy
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch


class PolicyLag:
    """
    Implémente un mécanisme de découplage entre la politique d'action (model)
    et la collecte d'expérience pour optimiser la latence CPU/GPU et les performances.

    Permet d'avoir:
    1. Un modèle de "collecte" qui reste stable pendant plusieurs étapes
    2. Un modèle d'"entraînement" qui est mis à jour plus fréquemment
    3. Une mise à jour périodique du modèle de collecte depuis le modèle d'entraînement
    """

    def __init__(
        self,
        model: torch.nn.Module,
        update_frequency: int = 100,
        target_update_freq: int = 1000,
        async_update: bool = True,
        copy_fn: Optional[Callable] = None,
    ):
        """
        Initialise le gestionnaire de policy lag.

        Args:
            model: Modèle initial (policy network)
            update_frequency: Fréquence de mise à jour du modèle de collecte (en étapes)
            target_update_freq: Fréquence de mise à jour du réseau cible (en étapes)
            async_update: Effectuer les mises à jour de façon asynchrone
            copy_fn: Fonction personnalisée de copie de modèle (par défaut: copie profonde)
        """
        self.train_model = model  # Le modèle qui sera entraîné

        # Créer une copie profonde du modèle pour la collecte d'expérience
        if copy_fn is not None:
            self.collect_model = copy_fn(model)
        else:
            self.collect_model = copy.deepcopy(model)

        # De même pour le modèle cible (target network)
        if copy_fn is not None:
            self.target_model = copy_fn(model)
        else:
            self.target_model = copy.deepcopy(model)

        # Configuration
        self.update_frequency = update_frequency
        self.target_update_freq = target_update_freq
        self.async_update = async_update
        self.copy_fn = copy_fn

        # Compteurs
        self.collect_steps = 0  # Nombre d'étapes de collecte effectuées
        self.train_steps = 0  # Nombre d'étapes d'entraînement effectuées

        # Thread et Queue pour mise à jour asynchrone
        self.update_queue = queue.Queue()
        self.update_thread = None
        self.running = False

        # État du modèle
        self.is_frozen = False

        # Statistiques
        self.metrics = {
            "collect_fps": 0,
            "train_fps": 0,
            "update_count": 0,
            "last_update_time": 0,
            "avg_update_time": 0,
        }

        # Démarrer le thread asynchrone si demandé
        if self.async_update:
            self._start_async_thread()

    def _copy_model_attributes(self, src_model, dst_model):
        """
        Copie manuellement les attributs d'un modèle à un autre.
        Cette méthode est utilisée pour s'assurer que tous les attributs sont copiés,
        pas seulement les paramètres du modèle.

        Args:
            src_model: Modèle source
            dst_model: Modèle de destination
        """
        # Copier les attributs non-standard
        for attr_name in dir(src_model):
            # Ignorer les attributs spéciaux ou méthodes
            if attr_name.startswith("_") or callable(getattr(src_model, attr_name)):
                continue

            # Ignorer les attributs standard de Module
            if attr_name in [
                "training",
                "modules",
                "parameters",
                "buffers",
                "named_parameters",
                "named_buffers",
                "named_modules",
                "state_dict",
                "load_state_dict",
            ]:
                continue

            # Copier l'attribut
            try:
                attr_value = getattr(src_model, attr_name)

                # Si c'est une valeur simple, copier directement
                if (
                    isinstance(attr_value, (int, float, str, bool))
                    or attr_value is None
                ):
                    setattr(dst_model, attr_name, attr_value)
            except Exception as e:
                print(f"Erreur lors de la copie de l'attribut {attr_name}: {e}")

    def _update_collect_model(self):
        """
        Met à jour le modèle de collecte à partir du modèle d'entraînement.
        """
        start_time = time.time()

        # Bloquer les gradients pendant la copie
        with torch.no_grad():
            if self.copy_fn is not None:
                self.collect_model = self.copy_fn(self.train_model)
            else:
                # Copie standard par état du dictionnaire
                if hasattr(self.collect_model, "load_state_dict") and hasattr(
                    self.train_model, "state_dict"
                ):
                    self.collect_model.load_state_dict(self.train_model.state_dict())
                else:
                    self.collect_model = copy.deepcopy(self.train_model)

                # Copier les attributs non-standard
                self._copy_model_attributes(self.train_model, self.collect_model)

        # Mettre à jour les métriques
        update_time = time.time() - start_time
        self.metrics["last_update_time"] = update_time

        # Mettre à jour la moyenne mobile
        n = self.metrics["update_count"]
        if n == 0:
            self.metrics["avg_update_time"] = update_time
        else:
            self.metrics["avg_update_time"] = (
                self.metrics["avg_update_time"] * n + update_time
            ) / (n + 1)

        self.metrics["update_count"] += 1

    def _update_target_model(self):
        """
        Met à jour le modèle cible à partir du modèle d'entraînement.
        """
        # Bloquer les gradients pendant la copie
        with torch.no_grad():
            if self.copy_fn is not None:
                self.target_model = self.copy_fn(self.train_model)
            else:
                # Copie standard par état du dictionnaire
                if hasattr(self.target_model, "load_state_dict") and hasattr(
                    self.train_model, "state_dict"
                ):
                    self.target_model.load_state_dict(self.train_model.state_dict())
                else:
                    self.target_model = copy.deepcopy(self.train_model)

                # Copier les attributs non-standard
                self._copy_model_attributes(self.train_model, self.target_model)

    def _async_update_worker(self):
        """
        Thread travailleur pour les mises à jour asynchrones.
        """
        self.running = True

        while self.running:
            try:
                # Attendre une commande dans la queue
                cmd, model_type = self.update_queue.get(timeout=0.5)

                if cmd == "update":
                    if model_type == "collect":
                        self._update_collect_model()
                    elif model_type == "target":
                        self._update_target_model()
                elif cmd == "stop":
                    break

                # Indiquer que la tâche est terminée
                self.update_queue.task_done()

            except queue.Empty:
                # Rien dans la queue, continuer
                continue
            except Exception as e:
                print(f"Erreur dans le thread de mise à jour asynchrone: {e}")

    def _start_async_thread(self):
        """
        Démarre le thread de mise à jour asynchrone.
        """
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(
                target=self._async_update_worker, daemon=True
            )
            self.update_thread.start()

    def collect_step(self) -> bool:
        """
        Enregistre une étape de collecte et met à jour le modèle si nécessaire.

        Returns:
            True si une mise à jour a été déclenchée, False sinon
        """
        self.collect_steps += 1
        update_triggered = False

        # Si temps de mettre à jour le modèle de collecte
        if self.collect_steps % self.update_frequency == 0 and not self.is_frozen:
            if self.async_update:
                # Ajouter une commande de mise à jour à la queue
                self.update_queue.put(("update", "collect"))
            else:
                # Mise à jour synchrone
                self._update_collect_model()

            update_triggered = True

        return update_triggered

    def train_step(self) -> bool:
        """
        Enregistre une étape d'entraînement et met à jour le modèle cible si nécessaire.

        Returns:
            True si une mise à jour du modèle cible a été déclenchée, False sinon
        """
        self.train_steps += 1
        target_update_triggered = False

        # Si temps de mettre à jour le modèle cible
        if self.train_steps % self.target_update_freq == 0:
            if self.async_update:
                # Ajouter une commande de mise à jour à la queue
                self.update_queue.put(("update", "target"))
            else:
                # Mise à jour synchrone
                self._update_target_model()

            target_update_triggered = True

        return target_update_triggered

    def get_collect_model(self) -> torch.nn.Module:
        """
        Récupère le modèle de collecte.

        Returns:
            Le modèle utilisé pour la collecte d'expérience
        """
        return self.collect_model

    def get_train_model(self) -> torch.nn.Module:
        """
        Récupère le modèle d'entraînement.

        Returns:
            Le modèle utilisé pour l'entraînement
        """
        return self.train_model

    def get_target_model(self) -> torch.nn.Module:
        """
        Récupère le modèle cible.

        Returns:
            Le modèle cible (target network)
        """
        return self.target_model

    def freeze_collect_model(self, freeze: bool = True):
        """
        Gèle le modèle de collecte pour éviter qu'il ne soit mis à jour.
        Utile pour l'évaluation.

        Args:
            freeze: True pour geler, False pour dégeler
        """
        self.is_frozen = freeze

    def force_update(self, model_type: str = "collect"):
        """
        Force une mise à jour immédiate du modèle spécifié.

        Args:
            model_type: "collect" pour le modèle de collecte, "target" pour le modèle cible
        """
        if model_type == "collect":
            if self.async_update:
                self.update_queue.put(("update", "collect"))
                # Pour les tests, attendre un peu que la mise à jour soit terminée
                time.sleep(0.1)
            else:
                self._update_collect_model()
        elif model_type == "target":
            if self.async_update:
                self.update_queue.put(("update", "target"))
                # Pour les tests, attendre un peu que la mise à jour soit terminée
                time.sleep(0.1)
            else:
                self._update_target_model()
        else:
            raise ValueError(
                f"Type de modèle inconnu: {model_type}. Utiliser 'collect' ou 'target'."
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de performance.

        Returns:
            Dictionnaire de métriques
        """
        # Calculer les FPS instantanés si possible
        if hasattr(self, "_last_collect_time") and hasattr(self, "_last_collect_steps"):
            now = time.time()
            time_diff = now - self._last_collect_time
            steps_diff = self.collect_steps - self._last_collect_steps

            if time_diff > 0:
                self.metrics["collect_fps"] = steps_diff / time_diff

            self._last_collect_time = now
            self._last_collect_steps = self.collect_steps
        else:
            self._last_collect_time = time.time()
            self._last_collect_steps = self.collect_steps

        # Ajouter les compteurs de pas
        self.metrics["collect_steps"] = self.collect_steps
        self.metrics["train_steps"] = self.train_steps
        self.metrics["update_frequency"] = self.update_frequency
        self.metrics["target_update_freq"] = self.target_update_freq
        self.metrics["is_async"] = self.async_update
        self.metrics["is_frozen"] = self.is_frozen

        return self.metrics

    def shutdown(self):
        """
        Arrête proprement les threads de mise à jour.
        """
        if self.running and self.update_thread and self.update_thread.is_alive():
            self.running = False
            # Signal d'arrêt dans la queue
            self.update_queue.put(("stop", None))
            # Attente du thread
            self.update_thread.join(timeout=1.0)

    def __del__(self):
        """
        Destructeur propre.
        """
        self.shutdown()


class DecoupledPolicyTrainer:
    """
    Gestionnaire complet pour l'entraînement de policy avec découplage.
    Inclut une boucle d'entraînement optimisée.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        update_frequency: int = 100,
        target_update_freq: int = 1000,
        async_update: bool = True,
        device: torch.device = None,
        batch_queue_size: int = 10,
    ):
        """
        Initialise le gestionnaire d'entraînement découplé.

        Args:
            model: Modèle de policy
            optimizer: Optimiseur pour l'entraînement
            update_frequency: Fréquence de mise à jour du modèle de collecte
            target_update_freq: Fréquence de mise à jour du modèle cible
            async_update: Activer les mises à jour asynchrones
            device: Appareil sur lequel exécuter l'entraînement
            batch_queue_size: Taille de la queue de batchs
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Déplacer le modèle sur l'appareil
        self.model = model.to(self.device)
        self.optimizer = optimizer

        # Créer le gestionnaire de policy lag
        self.policy_lag = PolicyLag(
            model=self.model,
            update_frequency=update_frequency,
            target_update_freq=target_update_freq,
            async_update=async_update,
        )

        # Queue de batchs pour l'entraînement asynchrone
        self.batch_queue = queue.Queue(maxsize=batch_queue_size)
        self.batch_thread = None
        self.is_training = False

        # Métriques
        self.metrics = {
            "train_loss": [],
            "batch_process_time": [],
            "queue_wait_time": [],
            "queue_size": 0,
        }

    def _process_batch_worker(self, loss_fn: Callable):
        """
        Thread travailleur pour traiter les batchs d'entraînement.

        Args:
            loss_fn: Fonction de perte pour l'entraînement
        """
        while self.is_training:
            try:
                # Récupérer un batch de la queue
                start_wait = time.time()
                batch = self.batch_queue.get(timeout=0.5)
                wait_time = time.time() - start_wait

                # Traiter le batch
                start_process = time.time()

                # Extraire les données du batch
                states, actions, rewards, next_states, dones = batch

                # Déplacer sur le device si nécessaire
                if isinstance(states, torch.Tensor):
                    states = states.to(self.device)
                if isinstance(actions, torch.Tensor):
                    actions = actions.to(self.device)
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.to(self.device)
                if isinstance(next_states, torch.Tensor):
                    next_states = next_states.to(self.device)
                if isinstance(dones, torch.Tensor):
                    dones = dones.to(self.device)

                # Calculer la perte
                loss = loss_fn(
                    self.policy_lag.get_train_model(),
                    self.policy_lag.get_target_model(),
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                )

                # Mise à jour du modèle
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Enregistrer l'étape d'entraînement
                self.policy_lag.train_step()

                # Mesurer le temps de traitement
                process_time = time.time() - start_process

                # Mettre à jour les métriques
                self.metrics["batch_process_time"].append(process_time)
                self.metrics["queue_wait_time"].append(wait_time)
                self.metrics["train_loss"].append(loss.item())

                # Limiter la taille des historiques
                max_history = 100
                if len(self.metrics["batch_process_time"]) > max_history:
                    self.metrics["batch_process_time"] = self.metrics[
                        "batch_process_time"
                    ][-max_history:]
                if len(self.metrics["queue_wait_time"]) > max_history:
                    self.metrics["queue_wait_time"] = self.metrics["queue_wait_time"][
                        -max_history:
                    ]
                if len(self.metrics["train_loss"]) > max_history:
                    self.metrics["train_loss"] = self.metrics["train_loss"][
                        -max_history:
                    ]

                # Indiquer que le batch est traité
                self.batch_queue.task_done()

            except queue.Empty:
                # Rien dans la queue, continuer
                continue
            except Exception as e:
                print(f"Erreur dans le thread de traitement des batchs: {e}")

    def start_training(self, loss_fn: Callable):
        """
        Démarre l'entraînement asynchrone.

        Args:
            loss_fn: Fonction de perte pour l'entraînement
        """
        if not self.is_training:
            self.is_training = True
            self.batch_thread = threading.Thread(
                target=self._process_batch_worker, args=(loss_fn,), daemon=True
            )
            self.batch_thread.start()

    def stop_training(self):
        """
        Arrête l'entraînement asynchrone.
        """
        self.is_training = False
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=1.0)

    def add_batch(self, batch) -> bool:
        """
        Ajoute un batch à la queue d'entraînement.

        Args:
            batch: Tuple (states, actions, rewards, next_states, dones)

        Returns:
            True si le batch a été ajouté, False si la queue est pleine
        """
        try:
            self.batch_queue.put_nowait(batch)
            self.metrics["queue_size"] = self.batch_queue.qsize()
            return True
        except queue.Full:
            return False

    def collect(self, state) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Utilise le modèle de collecte pour prédire une action.

        Args:
            state: État de l'environnement

        Returns:
            Tuple (action, info)
        """
        # Déplacer l'état sur le device si nécessaire
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)

        # Utiliser le modèle de collecte pour la prédiction
        collect_model = self.policy_lag.get_collect_model()

        with torch.no_grad():
            action = collect_model(state)

        # Enregistrer l'étape de collecte
        updated = self.policy_lag.collect_step()

        # Obtenir les métriques
        metrics = self.policy_lag.get_metrics()
        metrics.update(
            {
                "queue_size": self.batch_queue.qsize() if self.is_training else 0,
            }
        )

        info = {"updated": updated, "metrics": metrics}

        return action, info

    def train_sync(self, batch, loss_fn: Callable) -> float:
        """
        Effectue une étape d'entraînement synchrone.

        Args:
            batch: Tuple (states, actions, rewards, next_states, dones)
            loss_fn: Fonction de perte

        Returns:
            Valeur de la perte
        """
        states, actions, rewards, next_states, dones = batch

        # Déplacer sur le device si nécessaire
        if isinstance(states, torch.Tensor):
            states = states.to(self.device)
        if isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(self.device)
        if isinstance(next_states, torch.Tensor):
            next_states = next_states.to(self.device)
        if isinstance(dones, torch.Tensor):
            dones = dones.to(self.device)

        # Calculer la perte
        train_model = self.policy_lag.get_train_model()
        target_model = self.policy_lag.get_target_model()

        loss = loss_fn(
            train_model, target_model, states, actions, rewards, next_states, dones
        )

        # Mise à jour du modèle
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Enregistrer l'étape d'entraînement
        self.policy_lag.train_step()

        # Mettre à jour les métriques
        loss_value = loss.item()
        self.metrics["train_loss"].append(loss_value)

        return loss_value

    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques combinées.

        Returns:
            Dictionnaire de métriques
        """
        policy_metrics = self.policy_lag.get_metrics()

        # Calculer les moyennes
        train_loss_avg = sum(self.metrics["train_loss"]) / max(
            1, len(self.metrics["train_loss"])
        )
        batch_time_avg = sum(self.metrics["batch_process_time"]) / max(
            1, len(self.metrics["batch_process_time"])
        )
        queue_wait_avg = sum(self.metrics["queue_wait_time"]) / max(
            1, len(self.metrics["queue_wait_time"])
        )

        combined_metrics = {
            **policy_metrics,
            "queue_size": self.metrics["queue_size"],
            "train_loss_avg": train_loss_avg,
            "batch_time_avg": batch_time_avg,
            "queue_wait_avg": queue_wait_avg,
            "is_training": self.is_training,
        }

        return combined_metrics

    def shutdown(self):
        """
        Arrête proprement tous les threads.
        """
        self.stop_training()
        self.policy_lag.shutdown()

    def __del__(self):
        """
        Destructeur propre.
        """
        self.shutdown()
