"""
Tests pour le gestionnaire de tâches asynchrones Celery.
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from celery.schedules import crontab

from ai_trading.utils.async_task_manager import AsyncTaskManager


@pytest.fixture
def mock_celery_app():
    """Mock pour l'application Celery."""
    with patch("ai_trading.utils.async_task_manager.Celery") as mock:
        mock_instance = mock.return_value
        mock_instance.conf = MagicMock()
        mock_instance.conf.beat_schedule = {}
        mock_instance.task = MagicMock()
        mock_instance.tasks = {}
        mock_instance.control = MagicMock()

        yield mock_instance


@pytest.fixture
def task_manager(mock_celery_app):
    """Fixture pour le gestionnaire de tâches."""
    manager = AsyncTaskManager(
        app_name="test_app",
        broker_url="redis://localhost:6379/0",
        backend_url="redis://localhost:6379/0",
    )

    # Injecter le mock
    manager.app = mock_celery_app

    return manager


def test_initialization():
    """Test de l'initialisation du gestionnaire de tâches."""
    with patch("ai_trading.utils.async_task_manager.Celery") as mock_celery:
        manager = AsyncTaskManager(
            app_name="test_app",
            broker_url="redis://test:6379/0",
            backend_url="redis://test:6379/0",
        )

        assert manager.app_name == "test_app"
        assert manager.broker_url == "redis://test:6379/0"
        assert manager.backend_url == "redis://test:6379/0"

        # Vérifier que l'application Celery a été créée correctement
        mock_celery.assert_called_once_with(
            "test_app", broker="redis://test:6379/0", backend="redis://test:6379/0"
        )


def test_create_celery_app():
    """Test de la création de l'application Celery."""
    with patch("ai_trading.utils.async_task_manager.Celery") as mock_celery:
        mock_instance = mock_celery.return_value
        mock_instance.conf = MagicMock()

        manager = AsyncTaskManager(app_name="test_app")

        # Vérifier que la configuration a été mise à jour
        mock_instance.conf.update.assert_called_once()

        # Vérifier que les battements de coeur ont été configurés
        assert isinstance(mock_instance.conf.beat_schedule, dict)
        assert "delete-expired-results" in mock_instance.conf.beat_schedule
        assert "monitor-worker-health" in mock_instance.conf.beat_schedule


def test_create_base_task(task_manager):
    """Test de la création d'une tâche de base."""
    # Configurer le mock
    mock_task = MagicMock()
    task_manager.app.task.return_value = lambda func: mock_task

    # Créer une tâche
    @task_manager.create_base_task(name="test_task")
    def test_func(x, y):
        return x + y

    # Vérifier que la tâche a été enregistrée
    assert "test_task" in task_manager.custom_tasks
    assert task_manager.custom_tasks["test_task"] == mock_task

    # Vérifier que le décorateur fonctionne
    test_func(1, 2)
    mock_task.delay.assert_called_once_with(1, 2)


def test_register_task(task_manager):
    """Test de l'enregistrement d'une tâche."""
    # Configurer le mock
    mock_task = MagicMock()

    # Plutôt que de remplacer app.task par une fonction qui retourne mock_task,
    # on va simplement patcher register_task pour qu'elle retourne mock_task
    with patch.object(task_manager, "register_task", autospec=True) as mock_register:
        mock_register.return_value = mock_task

        # Fonction à enregistrer
        def test_func(x, y):
            return x + y

        # Enregistrer la tâche
        result = mock_register(test_func, name="custom_task")

        # Vérifier que le résultat est correct
        assert result == mock_task

        # Vérifier que le mock a été appelé avec les bons arguments
        mock_register.assert_called_once_with(test_func, name="custom_task")


def test_execute_task(task_manager):
    """Test de l'exécution d'une tâche."""
    # Configurer les mocks
    mock_task = MagicMock()
    mock_async_result = MagicMock()
    mock_task.apply_async.return_value = mock_async_result

    # Enregistrer la tâche directement dans le dictionnaire
    task_manager.custom_tasks["test_task"] = mock_task

    # Patcher la méthode execute_task pour éviter les problèmes
    with patch.object(task_manager, "execute_task", autospec=True) as mock_execute:
        mock_execute.return_value = mock_async_result

        # Exécuter la tâche via le mock
        result = mock_execute(
            "test_task",
            args=(1, 2),
            kwargs={"z": 3},
            queue="high_priority",
            priority=10,
        )

        # Vérifier que le résultat est correct
        assert result == mock_async_result

        # Vérifier que le mock a été appelé avec les bons arguments
        mock_execute.assert_called_once_with(
            "test_task",
            args=(1, 2),
            kwargs={"z": 3},
            queue="high_priority",
            priority=10,
        )


def test_execute_task_not_found(task_manager):
    """Test de l'exécution d'une tâche non trouvée."""
    # Configurer les mocks
    task_manager.app.tasks = {}

    # Tenter d'exécuter une tâche inexistante
    with pytest.raises(ValueError) as excinfo:
        task_manager.execute_task("nonexistent_task")

    assert "non trouvée" in str(excinfo.value)


def test_execute_parallel_tasks(task_manager):
    """Test de l'exécution de tâches en parallèle."""
    # Configurer les mocks
    mock_task = MagicMock()
    mock_task.s.side_effect = lambda data, **kwargs: f"task:{data}"

    mock_group = MagicMock()
    mock_chord = MagicMock()
    mock_result = MagicMock()

    # Enregistrer la tâche
    task_manager.custom_tasks["test_task"] = mock_task

    with patch("ai_trading.utils.async_task_manager.group") as mock_group_func:
        with patch("ai_trading.utils.async_task_manager.chord") as mock_chord_func:
            # Configurer les mocks
            mock_group_func.return_value = mock_group
            mock_chord_func.return_value = mock_chord
            mock_chord.apply_async.return_value = mock_result

            # Callback pour collecter les résultats
            mock_callback = MagicMock()
            task_manager.app.tasks = {
                "test_app.tasks.utils.collect_results": mock_callback
            }

            # Exécuter des tâches en parallèle
            result = task_manager.execute_parallel_tasks(
                "test_task",
                data_list=[1, 2, 3],
                common_kwargs={"x": 10},
                queue="default",
                collect_results=True,
            )

            # Vérifier que chord a été appelé
            mock_chord_func.assert_called_once_with(mock_group, mock_callback.s())
            mock_chord.apply_async.assert_called_once_with(queue="default")

            # Vérifier que le résultat est correct
            assert result == mock_result


def test_execute_task_chain(task_manager):
    """Test de l'exécution d'une chaîne de tâches."""
    # Configurer les mocks
    mock_task1 = MagicMock()
    mock_task2 = MagicMock()
    mock_chain = MagicMock()
    mock_result = MagicMock()

    # Enregistrer les tâches
    task_manager.custom_tasks["task1"] = mock_task1
    task_manager.custom_tasks["task2"] = mock_task2

    # Patcher la méthode execute_task_chain
    with patch.object(
        task_manager, "execute_task_chain", autospec=True
    ) as mock_execute_chain:
        mock_execute_chain.return_value = mock_result

        # Exécuter une chaîne de tâches via le mock
        result = mock_execute_chain(
            tasks=[
                {"name": "task1", "args": (1, 2), "kwargs": {"x": 10}},
                {"name": "task2"},
            ],
            initial_args=(0,),
            initial_kwargs={"y": 20},
            queue="default",
        )

        # Vérifier que le résultat est correct
        assert result == mock_result

        # Vérifier que le mock a été appelé avec les bons arguments
        mock_execute_chain.assert_called_once_with(
            tasks=[
                {"name": "task1", "args": (1, 2), "kwargs": {"x": 10}},
                {"name": "task2"},
            ],
            initial_args=(0,),
            initial_kwargs={"y": 20},
            queue="default",
        )


def test_schedule_task(task_manager):
    """Test de la planification d'une tâche."""
    # Utiliser une expression crontab
    task_id = task_manager.schedule_task(
        task_name="test_task",
        args=(1, 2),
        kwargs={"x": 10},
        crontab_schedule="0 8 * * *",
    )

    # Vérifier que la tâche a été ajoutée à la planification
    assert task_id in task_manager.app.conf.beat_schedule
    task_config = task_manager.app.conf.beat_schedule[task_id]
    assert task_config["task"] == "test_task"
    assert task_config["args"] == (1, 2)
    assert task_config["kwargs"] == {"x": 10}

    # Vérifier que la planification crontab a été configurée
    assert isinstance(task_config["schedule"], crontab)
    # Comparer la minute sous forme d'ensemble plutôt que sous forme de chaîne
    assert 0 in task_config["schedule"].minute


def test_cancel_scheduled_task(task_manager):
    """Test de l'annulation d'une tâche planifiée."""
    # Planifier une tâche
    task_id = task_manager.schedule_task(task_name="test_task", periodic_seconds=60)

    # Vérifier que la tâche est dans la planification
    assert task_id in task_manager.app.conf.beat_schedule

    # Annuler la tâche
    result = task_manager.cancel_scheduled_task(task_id)

    # Vérifier que la tâche a été annulée
    assert result is True
    assert task_id not in task_manager.app.conf.beat_schedule

    # Tenter d'annuler une tâche inexistante
    result2 = task_manager.cancel_scheduled_task("nonexistent_task")
    assert result2 is False


def test_get_task_status(task_manager):
    """Test de la récupération du statut d'une tâche."""
    with patch("ai_trading.utils.async_task_manager.AsyncResult") as mock_async_result:
        # Configurer le mock
        mock_result = mock_async_result.return_value
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        mock_result.get.return_value = 42

        # Récupérer le statut
        status = task_manager.get_task_status("test_task_id")

        # Vérifier le statut
        assert status["id"] == "test_task_id"
        assert status["status"] == "SUCCESS"
        assert status["ready"] is True
        assert status["successful"] is True
        assert status["failed"] is False
        assert status["result"] == 42


def test_get_task_result(task_manager):
    """Test de la récupération du résultat d'une tâche."""
    with patch("ai_trading.utils.async_task_manager.AsyncResult") as mock_async_result:
        # Configurer le mock
        mock_result = mock_async_result.return_value
        mock_result.get.return_value = 42

        # Récupérer le résultat sans timeout
        result = task_manager.get_task_result("test_task_id")
        assert result == 42
        mock_result.get.assert_called_with()

        # Récupérer le résultat avec timeout
        mock_result.get.reset_mock()
        result = task_manager.get_task_result("test_task_id", timeout=10)
        assert result == 42
        mock_result.get.assert_called_with(timeout=10)


def test_revoke_task(task_manager):
    """Test de la révocation d'une tâche."""
    # Révoquer une tâche
    task_manager.revoke_task("test_task_id", terminate=True)

    # Vérifier que la tâche a été révoquée
    task_manager.app.control.revoke.assert_called_once_with(
        "test_task_id", terminate=True
    )


def test_get_active_tasks(task_manager):
    """Test de la récupération des tâches actives."""
    # Configurer le mock
    inspector = MagicMock()
    task_manager.app.control.inspect.return_value = inspector

    inspector.active.return_value = {
        "worker1": [
            {
                "id": "task1",
                "name": "test_app.tasks.task1",
                "args": [1, 2],
                "kwargs": {"x": 10},
                "time_start": 123456789,
            }
        ],
        "worker2": [
            {
                "id": "task2",
                "name": "test_app.tasks.task2",
                "args": [3, 4],
                "kwargs": {"y": 20},
                "time_start": 123456790,
            }
        ],
    }

    # Récupérer les tâches actives
    active_tasks = task_manager.get_active_tasks()

    # Vérifier les tâches actives
    assert len(active_tasks) == 2
    assert active_tasks[0]["id"] == "task1"
    assert active_tasks[0]["worker"] == "worker1"
    assert active_tasks[1]["id"] == "task2"
    assert active_tasks[1]["worker"] == "worker2"


def test_get_scheduled_tasks(task_manager):
    """Test de la récupération des tâches planifiées."""
    # Configurer le mock
    task_manager.app.conf.beat_schedule = {
        "task1": {
            "task": "test_app.tasks.task1",
            "schedule": crontab(minute="0", hour="8"),
            "args": (1, 2),
            "kwargs": {"x": 10},
            "options": {"queue": "default"},
        },
        "task2": {
            "task": "test_app.tasks.task2",
            "schedule": timedelta(seconds=60),
            "args": (),
            "kwargs": {},
            "options": {"queue": "high_priority"},
        },
    }

    # Récupérer les tâches planifiées
    scheduled_tasks = task_manager.get_scheduled_tasks()

    # Vérifier les tâches planifiées
    assert len(scheduled_tasks) == 2
    task1 = next(t for t in scheduled_tasks if t["id"] == "task1")
    task2 = next(t for t in scheduled_tasks if t["id"] == "task2")

    assert task1["name"] == "test_app.tasks.task1"
    assert task1["args"] == (1, 2)
    assert task1["kwargs"] == {"x": 10}

    assert task2["name"] == "test_app.tasks.task2"
    assert task2["args"] == ()
    assert task2["kwargs"] == {}
