"""
Gestionnaire de tâches asynchrones robuste basé sur Celery.

Ce module fournit:
- Système de tâches asynchrones robuste pour le traitement en arrière-plan
- Gestion des files d'attente et des priorités
- Surveillance des tâches et reprise sur erreur
- Planification des tâches récurrentes
"""

import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union

from celery import Celery, chain, chord, group
from celery.result import AsyncResult
from celery.schedules import crontab

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AsyncTaskManager")

# Configuration Celery par défaut
DEFAULT_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
DEFAULT_BACKEND_URL = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
)


class AsyncTaskManager:
    """
    Gestionnaire de tâches asynchrones robuste basé sur Celery.
    """

    def __init__(
        self,
        app_name: str = "ai_trading",
        broker_url: str = DEFAULT_BROKER_URL,
        backend_url: str = DEFAULT_BACKEND_URL,
        task_serializer: str = "json",
        result_serializer: str = "json",
        accept_content: List[str] = None,
        task_queues: Dict[str, Dict] = None,
        task_routes: Dict[str, str] = None,
    ):
        """
        Initialise le gestionnaire de tâches asynchrones.

        Args:
            app_name: Nom de l'application Celery
            broker_url: URL du broker de messages (Redis, RabbitMQ)
            backend_url: URL du backend de résultats
            task_serializer: Format de sérialisation des tâches
            result_serializer: Format de sérialisation des résultats
            accept_content: Formats de contenu acceptés
            task_queues: Configuration des files d'attente
            task_routes: Routes des tâches vers les files d'attente
        """
        self.app_name = app_name
        self.broker_url = broker_url
        self.backend_url = backend_url
        self.task_serializer = task_serializer
        self.result_serializer = result_serializer
        self.accept_content = accept_content or ["json", "pickle"]

        # Configuration des files d'attente
        self.task_queues = task_queues or {
            "default": {"exchange": "default", "routing_key": "default"},
            "high_priority": {
                "exchange": "high_priority",
                "routing_key": "high_priority",
            },
            "low_priority": {"exchange": "low_priority", "routing_key": "low_priority"},
            "scheduled": {"exchange": "scheduled", "routing_key": "scheduled"},
        }

        # Configuration des routes
        self.task_routes = task_routes or {
            "ai_trading.tasks.collect_data.*": {"queue": "default"},
            "ai_trading.tasks.analyze_data.*": {"queue": "high_priority"},
            "ai_trading.tasks.train_model.*": {"queue": "low_priority"},
            "ai_trading.tasks.scheduled.*": {"queue": "scheduled"},
        }

        # Initialiser l'application Celery
        self.app = self._create_celery_app()

        # Registre des tâches personnalisées
        self.custom_tasks = {}

        logger.info(
            f"Gestionnaire de tâches asynchrones initialisé avec broker {broker_url}"
        )

    def _create_celery_app(self) -> Celery:
        """
        Crée et configure l'application Celery.

        Returns:
            Celery: Application Celery configurée
        """
        app = Celery(self.app_name, broker=self.broker_url, backend=self.backend_url)

        # Configuration de base
        app.conf.update(
            task_serializer=self.task_serializer,
            result_serializer=self.result_serializer,
            accept_content=self.accept_content,
            timezone="UTC",
            enable_utc=True,
            worker_hijack_root_logger=False,
            task_acks_late=True,  # Acquittement tardif pour éviter la perte de tâches
            worker_prefetch_multiplier=1,  # Contrôle le nombre de tâches préchargées
            task_queues=self.task_queues,
            task_routes=self.task_routes,
            task_default_queue="default",
            result_expires=3600,  # Les résultats expirent après 1 heure
            worker_max_tasks_per_child=1000,  # Redémarrer le worker après 1000 tâches
            broker_connection_retry=True,
            broker_connection_max_retries=10,
            task_compression="gzip",  # Compression des tâches pour réduire la bande passante
        )

        # Configuration des battements de coeur pour la surveillance
        app.conf.beat_schedule = {
            "delete-expired-results": {
                "task": f"{self.app_name}.tasks.maintenance.cleanup_expired_results",
                "schedule": crontab(hour=2, minute=0),  # Exécuter à 2h du matin
                "args": (),
            },
            "monitor-worker-health": {
                "task": f"{self.app_name}.tasks.maintenance.check_worker_health",
                "schedule": 60.0,  # Toutes les minutes
                "args": (),
            },
        }

        return app

    def create_base_task(
        self, name: str = None, bind: bool = True, **options
    ) -> Callable:
        """
        Crée une tâche de base avec gestion d'erreur et retry intégrés.

        Args:
            name: Nom de la tâche (None = auto-généré)
            bind: Lier la tâche à l'instance (permet d'accéder à self)
            **options: Options supplémentaires pour la tâche

        Returns:
            Callable: Décorateur de tâche
        """
        default_options = {
            "name": name,
            "bind": bind,
            "autoretry_for": (Exception,),
            "retry_kwargs": {"max_retries": 3, "countdown": 10},
            "retry_backoff": True,
            "retry_jitter": True,
            "track_started": True,
            "ignore_result": False,
            "acks_late": True,
        }

        # Fusionner les options par défaut avec les options fournies
        task_options = {**default_options, **options}

        def task_decorator(func):
            # Créer la tâche Celery
            task = self.app.task(**task_options)(func)

            # Enregistrer la tâche personnalisée
            if name:
                self.custom_tasks[name] = task

            @wraps(func)
            def wrapper(*args, **kwargs):
                return task.delay(*args, **kwargs)

            # Ajouter des méthodes utiles
            wrapper.task = task
            wrapper.apply_async = task.apply_async
            wrapper.apply = task.apply

            return wrapper

        return task_decorator

    def register_task(self, func: Callable, name: str = None, **options) -> Callable:
        """
        Enregistre une fonction comme tâche Celery.

        Args:
            func: Fonction à enregistrer
            name: Nom de la tâche (None = nom de la fonction)
            **options: Options de la tâche

        Returns:
            Callable: Fonction décorée
        """
        name = name or f"{self.app_name}.tasks.{func.__name__}"
        decorator = self.create_base_task(name=name, **options)
        return decorator(func)

    def execute_task(
        self,
        task_name: str,
        args: Tuple = None,
        kwargs: Dict = None,
        queue: str = "default",
        priority: int = None,
        countdown: int = None,
        eta: datetime = None,
        expires: Union[int, datetime] = None,
        retry: bool = True,
        retry_policy: Dict = None,
    ) -> AsyncResult:
        """
        Exécute une tâche enregistrée.

        Args:
            task_name: Nom de la tâche
            args: Arguments positionnels
            kwargs: Arguments nommés
            queue: File d'attente
            priority: Priorité (10=haute, 0=basse)
            countdown: Délai avant exécution (secondes)
            eta: Heure d'exécution
            expires: Expiration
            retry: Réessayer en cas d'échec
            retry_policy: Politique de réessai

        Returns:
            AsyncResult: Résultat asynchrone
        """
        args = args or ()
        kwargs = kwargs or {}
        retry_policy = retry_policy or {
            "max_retries": 3,
            "interval_start": 0,
            "interval_step": 0.2,
        }

        # Vérifier si la tâche existe
        if task_name in self.custom_tasks:
            task = self.custom_tasks[task_name].task
        else:
            task = self.app.tasks.get(task_name)

        if not task:
            raise ValueError(f"Tâche {task_name} non trouvée")

        # Exécuter la tâche
        return task.apply_async(
            args=args,
            kwargs=kwargs,
            queue=queue,
            priority=priority,
            countdown=countdown,
            eta=eta,
            expires=expires,
            retry=retry,
            retry_policy=retry_policy,
        )

    def execute_parallel_tasks(
        self,
        task_name: str,
        data_list: List[Any],
        common_kwargs: Dict = None,
        queue: str = "default",
        collect_results: bool = True,
    ) -> Union[AsyncResult, List[AsyncResult]]:
        """
        Exécute une tâche en parallèle sur une liste de données.

        Args:
            task_name: Nom de la tâche
            data_list: Liste des données
            common_kwargs: Arguments communs à toutes les tâches
            queue: File d'attente
            collect_results: Collecter les résultats

        Returns:
            Union[AsyncResult, List[AsyncResult]]: Résultat(s) asynchrone(s)
        """
        common_kwargs = common_kwargs or {}

        # Vérifier si la tâche existe
        if task_name in self.custom_tasks:
            task = self.custom_tasks[task_name].task
        else:
            task = self.app.tasks.get(task_name)

        if not task:
            raise ValueError(f"Tâche {task_name} non trouvée")

        # Créer un groupe de tâches
        task_group = group(task.s(data, **common_kwargs) for data in data_list)

        # Exécuter le groupe
        if collect_results:
            # Collecter les résultats
            header = task_group
            callback = self.app.tasks.get(
                f"{self.app_name}.tasks.utils.collect_results"
            )

            if not callback:
                # Créer une tâche de collecte des résultats si elle n'existe pas
                @self.app.task(name=f"{self.app_name}.tasks.utils.collect_results")
                def collect_results(results):
                    return results

                callback = collect_results

            # Utiliser un chord pour collecter les résultats
            result = chord(header, callback.s()).apply_async(queue=queue)
            return result
        else:
            # Retourner les résultats individuels
            return task_group.apply_async(queue=queue)

    def execute_task_chain(
        self,
        tasks: List[Dict],
        initial_args: Tuple = None,
        initial_kwargs: Dict = None,
        queue: str = "default",
    ) -> AsyncResult:
        """
        Exécute une chaîne de tâches séquentielles.

        Args:
            tasks: Liste de dictionnaires {name, args, kwargs}
            initial_args: Arguments initiaux
            initial_kwargs: Arguments nommés initiaux
            queue: File d'attente

        Returns:
            AsyncResult: Résultat asynchrone
        """
        initial_args = initial_args or ()
        initial_kwargs = initial_kwargs or {}

        if not tasks:
            raise ValueError("La liste des tâches ne peut pas être vide")

        # Construire la chaîne
        task_chain = None

        for i, task_info in enumerate(tasks):
            task_name = task_info.get("name")
            task_args = task_info.get("args", ())
            task_kwargs = task_info.get("kwargs", {})

            # Vérifier si la tâche existe
            if task_name in self.custom_tasks:
                task = self.custom_tasks[task_name].task
            else:
                task = self.app.tasks.get(task_name)

            if not task:
                raise ValueError(f"Tâche {task_name} non trouvée")

            # Première tâche de la chaîne
            if i == 0:
                task_chain = task.s(
                    *initial_args, *task_args, **initial_kwargs, **task_kwargs
                )
            else:
                # Tâches suivantes
                if task_args or task_kwargs:
                    # Si la tâche a des arguments, utiliser une fonction d'adaptation
                    @self.app.task(name=f"{self.app_name}.tasks.utils.adapt_args_{i}")
                    def adapt_args(
                        previous_result, task_args=task_args, task_kwargs=task_kwargs
                    ):
                        return previous_result, *task_args, task_kwargs

                    task_chain = chain(task_chain, adapt_args.s(), task.s())
                else:
                    # Sinon, chaîner directement
                    task_chain = chain(task_chain, task.s())

        # Exécuter la chaîne
        return task_chain.apply_async(queue=queue)

    def schedule_task(
        self,
        task_name: str,
        args: Tuple = None,
        kwargs: Dict = None,
        crontab_schedule: str = None,
        periodic_seconds: int = None,
        start_time: datetime = None,
        task_id: str = None,
    ) -> str:
        """
        Planifie une tâche récurrente.

        Args:
            task_name: Nom de la tâche
            args: Arguments positionnels
            kwargs: Arguments nommés
            crontab_schedule: Expression crontab (ex: "0 8 * * *" pour tous les jours à 8h)
            periodic_seconds: Période en secondes
            start_time: Heure de début
            task_id: Identifiant unique de la tâche

        Returns:
            str: Identifiant de la tâche planifiée
        """
        args = args or ()
        kwargs = kwargs or {}

        if not crontab_schedule and not periodic_seconds:
            raise ValueError(
                "Vous devez spécifier soit crontab_schedule soit periodic_seconds"
            )

        # Générer un identifiant unique si non fourni
        if not task_id:
            task_id = f"{task_name}_{int(time.time())}"

        # Convertir l'expression crontab
        if crontab_schedule:
            minute, hour, day_of_month, month_of_year, day_of_week = (
                crontab_schedule.split()
            )
            schedule = crontab(
                minute=minute,
                hour=hour,
                day_of_month=day_of_month,
                month_of_year=month_of_year,
                day_of_week=day_of_week,
            )
        else:
            # Utiliser une planification périodique
            schedule = timedelta(seconds=periodic_seconds)

        # Ajouter à la planification
        self.app.conf.beat_schedule[task_id] = {
            "task": task_name,
            "schedule": schedule,
            "args": args,
            "kwargs": kwargs,
            "options": {"queue": "scheduled"},
            "relative": bool(start_time),
        }

        # Si start_time est spécifié, ajouter l'heure de départ
        if start_time:
            self.app.conf.beat_schedule[task_id]["start_time"] = start_time

        logger.info(f"Tâche {task_name} planifiée avec l'ID {task_id}")
        return task_id

    def cancel_scheduled_task(self, task_id: str) -> bool:
        """
        Annule une tâche planifiée.

        Args:
            task_id: Identifiant de la tâche

        Returns:
            bool: True si la tâche a été annulée
        """
        if task_id in self.app.conf.beat_schedule:
            del self.app.conf.beat_schedule[task_id]
            logger.info(f"Tâche planifiée {task_id} annulée")
            return True

        logger.warning(f"Tâche planifiée {task_id} non trouvée")
        return False

    def get_task_result(self, task_id: str, timeout: int = None) -> Any:
        """
        Récupère le résultat d'une tâche.

        Args:
            task_id: Identifiant de la tâche
            timeout: Délai d'attente maximal (secondes)

        Returns:
            Any: Résultat de la tâche
        """
        result = AsyncResult(task_id, app=self.app)

        if timeout is not None:
            return result.get(timeout=timeout)

        return result.get()

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Récupère le statut d'une tâche.

        Args:
            task_id: Identifiant de la tâche

        Returns:
            Dict[str, Any]: Statut de la tâche
        """
        result = AsyncResult(task_id, app=self.app)

        status = {
            "id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful(),
            "failed": result.failed(),
        }

        # Ajouter des détails supplémentaires si disponibles
        if result.ready():
            if result.successful():
                # Tenter de récupérer le résultat s'il est disponible
                try:
                    status["result"] = result.get(timeout=1)
                except Exception as e:
                    status["result_error"] = str(e)
            else:
                # Récupérer l'erreur
                try:
                    status["error"] = str(result.result)
                    status["traceback"] = result.traceback
                except Exception:
                    status["error"] = "Erreur inconnue"

        return status

    def revoke_task(self, task_id: str, terminate: bool = False) -> None:
        """
        Révoque une tâche.

        Args:
            task_id: Identifiant de la tâche
            terminate: Terminer la tâche si elle est en cours d'exécution
        """
        self.app.control.revoke(task_id, terminate=terminate)
        logger.info(f"Tâche {task_id} révoquée (terminate={terminate})")

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Récupère les tâches actives.

        Returns:
            List[Dict[str, Any]]: Liste des tâches actives
        """
        inspector = self.app.control.inspect()
        active_tasks = []

        # Récupérer les tâches actives par worker
        active = inspector.active() or {}

        for worker_name, tasks in active.items():
            for task in tasks:
                active_tasks.append(
                    {
                        "id": task["id"],
                        "name": task["name"],
                        "args": task["args"],
                        "kwargs": task["kwargs"],
                        "worker": worker_name,
                        "time_start": task.get("time_start"),
                    }
                )

        return active_tasks

    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        Récupère les tâches planifiées.

        Returns:
            List[Dict[str, Any]]: Liste des tâches planifiées
        """
        scheduled_tasks = []

        for task_id, task_info in self.app.conf.beat_schedule.items():
            scheduled_tasks.append(
                {
                    "id": task_id,
                    "name": task_info["task"],
                    "schedule": str(task_info["schedule"]),
                    "args": task_info.get("args", ()),
                    "kwargs": task_info.get("kwargs", {}),
                    "options": task_info.get("options", {}),
                    "start_time": task_info.get("start_time"),
                }
            )

        return scheduled_tasks

    def shutdown(self):
        """Arrête le gestionnaire de tâches."""
        logger.info("Arrêt du gestionnaire de tâches asynchrones")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le gestionnaire de tâches
    task_manager = AsyncTaskManager(
        app_name="ai_trading",
        broker_url="redis://localhost:6379/0",
        backend_url="redis://localhost:6379/0",
    )

    # Définir une tâche simple
    @task_manager.create_base_task(name="ai_trading.tasks.example.add")
    def add(self, x, y):
        return x + y

    # Exécuter la tâche
    result = add(2, 3)
    print(f"Résultat: {result.get()}")

    # Exécuter plusieurs tâches en parallèle
    data_list = [(i, i + 1) for i in range(5)]
    group_result = task_manager.execute_parallel_tasks(
        "ai_trading.tasks.example.add", data_list
    )
    print(f"Résultats parallèles: {group_result.get()}")

    # Planifier une tâche récurrente
    task_id = task_manager.schedule_task(
        "ai_trading.tasks.example.add",
        args=(10, 20),
        periodic_seconds=60,  # Toutes les minutes
    )
    print(f"Tâche planifiée: {task_id}")

    # Récupérer les tâches planifiées
    scheduled_tasks = task_manager.get_scheduled_tasks()
    print(f"Tâches planifiées: {scheduled_tasks}")

    # Annuler la tâche planifiée
    task_manager.cancel_scheduled_task(task_id)
    print("Tâche annulée")
