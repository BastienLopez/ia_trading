"""
Tests pour le processeur parallèle Dask.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_trading.utils.parallel_processor import DaskParallelProcessor


@pytest.fixture
def mock_dask_client():
    """Mock pour le client Dask."""
    with patch("ai_trading.utils.parallel_processor.Client") as mock:
        mock_instance = mock.return_value

        # Mock des méthodes nécessaires
        mock_instance.submit = MagicMock(return_value=MagicMock())
        mock_instance.gather = MagicMock(return_value=[1, 2, 3])
        mock_instance.ncores = MagicMock(return_value={"worker1": 4, "worker2": 4})

        yield mock


@pytest.fixture
def mock_dask_local_cluster():
    """Mock pour le cluster Dask local."""
    with patch("ai_trading.utils.parallel_processor.LocalCluster") as mock:
        mock_instance = mock.return_value
        mock_instance.workers = ["worker1", "worker2"]

        yield mock


@pytest.fixture
def processor(mock_dask_client, mock_dask_local_cluster):
    """Fixture pour le processeur parallèle."""
    processor = DaskParallelProcessor(n_workers=2)

    # Assurer que les mocks sont correctement injectés
    processor.client = mock_dask_client.return_value
    processor.cluster = mock_dask_local_cluster.return_value

    yield processor

    # Nettoyage
    processor.close()


def test_initialization(mock_dask_client, mock_dask_local_cluster):
    """Test de l'initialisation du processeur."""
    processor = DaskParallelProcessor(n_workers=2)

    assert processor.n_workers == 2
    assert processor.threads_per_worker == 2
    assert processor.memory_limit == "4GB"

    mock_dask_local_cluster.assert_called_once()
    mock_dask_client.assert_called_once()


def test_parallelize_dataframe(processor):
    """Test de la parallélisation d'un DataFrame."""
    # Créer un DataFrame de test
    df = pd.DataFrame({"A": range(100), "B": range(100, 200)})

    # Paralléliser le DataFrame
    ddf = processor.parallelize_dataframe(df, partition_size=10)

    # Vérifier que le type retourné est correct
    assert str(type(ddf)).find("dask.dataframe") >= 0


def test_apply_parallel(processor):
    """Test de l'application parallèle de fonction."""
    # Créer un DataFrame de test
    df = pd.DataFrame({"A": range(10), "B": range(10, 20)})

    # Définir une fonction de test
    def test_func(row):
        return row["A"] + row["B"]

    # Appliquer la fonction en parallèle (avec mock)
    with patch("dask.dataframe.from_pandas") as mock_from_pandas:
        # Configurer le mock
        mock_ddf = MagicMock()
        mock_ddf.map_partitions.return_value = mock_ddf
        mock_ddf.compute.return_value = pd.Series(range(30, 40))
        mock_from_pandas.return_value = mock_ddf

        # Exécuter la fonction
        result = processor.apply_parallel(df, test_func)

        # Vérifier les appels
        mock_from_pandas.assert_called_once()
        mock_ddf.map_partitions.assert_called_once()
        mock_ddf.compute.assert_called_once()

        # Vérifier le résultat
        assert isinstance(result, pd.Series)


def test_process_chunks(processor):
    """Test du traitement par morceaux."""
    # Créer un DataFrame de test
    df = pd.DataFrame({"A": range(100), "B": range(100, 200)})

    # Définir une fonction de traitement de morceaux
    def process_chunk(chunk):
        return chunk.assign(C=chunk["A"] + chunk["B"])

    # Configurer les mocks pour éviter les problèmes de Dask Client
    processor.client.submit = MagicMock()
    processor.client.gather = MagicMock()

    # Créer des futurs simulés
    futures = ["future1", "future2"]
    processor.client.submit.side_effect = (
        lambda func, *args, **kwargs: f"future{processor.client.submit.call_count}"
    )

    # Configurer le résultat du gather
    processor.client.gather.return_value = [
        pd.DataFrame({"A": range(50), "B": range(100, 150), "C": range(100, 150)}),
        pd.DataFrame({"A": range(50, 100), "B": range(150, 200), "C": range(150, 200)}),
    ]

    # Patch le ProgressBar pour éviter les erreurs
    with patch("distributed.diagnostics.progressbar.TextProgressBar"):
        # Exécuter la fonction
        result = processor.process_chunks(df, process_chunk, chunk_size=50)

        # Vérifier les appels
        assert processor.client.submit.call_count == 2
        processor.client.gather.assert_called_once()

        # Vérifier le résultat
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 3)


def test_map_reduce(processor):
    """Test de map-reduce."""
    # Données de test
    data = list(range(10))

    # Fonctions de test
    def map_func(x):
        return x * 2

    def reduce_func(x, y):
        return x + y

    # Configurer les mocks
    with patch("dask.bag.from_sequence") as mock_from_sequence:
        # Configurer le mock
        mock_bag = MagicMock()
        mock_bag.map.return_value = mock_bag
        mock_bag.fold.return_value = mock_bag
        mock_bag.compute.return_value = 90  # sum(range(10) * 2) = 90
        mock_from_sequence.return_value = mock_bag

        # Exécuter la fonction
        result = processor.map_reduce(data, map_func, reduce_func)

        # Vérifier les appels
        mock_from_sequence.assert_called_once_with(data, None)
        mock_bag.map.assert_called_once_with(map_func)
        mock_bag.fold.assert_called_once_with(reduce_func)
        mock_bag.compute.assert_called_once()

        # Vérifier le résultat
        assert result == 90


def test_parallelize_computation(processor):
    """Test de la parallélisation de calculs."""
    # Données de test
    data_list = [1, 2, 3, 4, 5]

    # Fonction de test
    def square(x):
        return x * x

    # Configurer les mocks directement sur l'instance processor
    processor.client.submit = MagicMock()
    processor.client.gather = MagicMock()

    # Créer des futurs simulés
    futures = []
    for i in range(5):
        future = MagicMock()
        future.key = f"key{i}"
        futures.append(future)

    processor.client.submit.side_effect = futures
    processor.client.gather.return_value = [1, 4, 9, 16, 25]

    # Patch le ProgressBar pour éviter les erreurs
    with patch("distributed.diagnostics.progressbar.TextProgressBar"):
        # Exécuter la fonction
        result = processor.parallelize_computation(square, data_list)

        # Vérifier les appels
        assert processor.client.submit.call_count == 5
        processor.client.gather.assert_called_once()

        # Vérifier le résultat
        assert result == [1, 4, 9, 16, 25]


def test_context_manager():
    """Test du support du context manager."""
    with patch("ai_trading.utils.parallel_processor.Client") as mock_client:
        with patch("ai_trading.utils.parallel_processor.LocalCluster") as mock_cluster:
            # Configurer les mocks
            mock_client_instance = mock_client.return_value
            mock_cluster_instance = mock_cluster.return_value

            # Utiliser le context manager
            with DaskParallelProcessor(n_workers=2) as processor:
                assert processor is not None

            # Vérifier que close a été appelé
            mock_client_instance.close.assert_called_once()
            mock_cluster_instance.close.assert_called_once()


def test_empty_dataframe(processor):
    """Test du comportement avec un DataFrame vide."""
    # Créer un DataFrame vide
    df = pd.DataFrame()

    # Paralléliser le DataFrame
    ddf = processor.parallelize_dataframe(df)

    # Vérifier que le type retourné est correct
    assert str(type(ddf)).find("dask.dataframe") >= 0


def test_fallback_on_cluster_error():
    """Test du fallback en cas d'erreur de cluster."""
    with patch("ai_trading.utils.parallel_processor.LocalCluster") as mock_cluster:
        # Simuler une erreur lors de la création du cluster
        mock_cluster.side_effect = Exception("Erreur de cluster")

        with patch("ai_trading.utils.parallel_processor.Client") as mock_client:
            # Configurer le mock du client
            processor = DaskParallelProcessor(n_workers=2)

            # Vérifier que le client a été créé en mode fallback
            mock_client.assert_called_with(processes=False)
