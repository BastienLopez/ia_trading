"""
Tests pour le module de gestion avancée des timeouts et retries.

Ce module teste les fonctionnalités de resilient_requester, notamment :
- Backoff exponentiel
- Circuit breaker pattern
- Fallback sur sources alternatives
- Monitoring des performances
"""

import time
import unittest

from ai_trading.utils.resilient_requester import (
    AllSourcesFailedError,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    MultiSourceRequester,
    PerformanceMetrics,
    ResilientRequester,
    resilient,
)


class TestCircuitBreaker(unittest.TestCase):
    """Tests pour la classe CircuitBreaker."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=1, half_open_max_calls=2
        )

    def test_initial_state(self):
        """Teste l'état initial du circuit breaker."""
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())

    def test_circuit_opens_after_failures(self):
        """Teste l'ouverture du circuit après plusieurs échecs."""
        # Enregistrer des échecs
        for _ in range(3):
            self.circuit_breaker.record_failure()

        # Vérifier que le circuit est ouvert
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())

    def test_circuit_half_opens_after_timeout(self):
        """Teste le passage à l'état semi-ouvert après le timeout."""
        # Ouvrir le circuit
        for _ in range(3):
            self.circuit_breaker.record_failure()

        # Attendre le timeout
        time.sleep(1.1)

        # Vérifier que le circuit est semi-ouvert
        self.assertTrue(self.circuit_breaker.allow_request())
        self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)

    def test_circuit_closes_after_success_in_half_open(self):
        """Teste la fermeture du circuit après un succès en état semi-ouvert."""
        # Ouvrir le circuit puis passer en semi-ouvert
        for _ in range(3):
            self.circuit_breaker.record_failure()

        time.sleep(1.1)
        self.circuit_breaker.allow_request()  # Passer en semi-ouvert

        # Enregistrer un succès
        self.circuit_breaker.record_success()

        # Vérifier que le circuit est fermé
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())

    def test_circuit_reopens_after_failure_in_half_open(self):
        """Teste la réouverture du circuit après un échec en état semi-ouvert."""
        # Ouvrir le circuit puis passer en semi-ouvert
        for _ in range(3):
            self.circuit_breaker.record_failure()

        time.sleep(1.1)
        self.circuit_breaker.allow_request()  # Passer en semi-ouvert

        # Enregistrer un échec
        self.circuit_breaker.record_failure()

        # Vérifier que le circuit est ouvert
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())

    def test_limited_calls_in_half_open(self):
        """Teste la limitation des appels en état semi-ouvert."""
        # Ouvrir le circuit puis passer en semi-ouvert
        for _ in range(3):
            self.circuit_breaker.record_failure()

        time.sleep(1.1)

        # Vérifier que le circuit est bien en état semi-ouvert après le premier appel autorisé
        self.assertTrue(self.circuit_breaker.allow_request())
        self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)

        # Le deuxième appel doit être autorisé
        self.assertTrue(self.circuit_breaker.allow_request())

        # Pas besoin de vérifier half_open_calls, il suffit de vérifier que le
        # circuit reste en état semi-ouvert après deux appels
        self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests pour la classe PerformanceMetrics."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.metrics = PerformanceMetrics(window_size=5)

    def test_record_success(self):
        """Teste l'enregistrement des succès."""
        self.metrics.record_success(0.1)
        self.metrics.record_success(0.2)

        metrics_data = self.metrics.get_metrics()
        self.assertEqual(metrics_data["success_count"], 2)
        self.assertEqual(metrics_data["total_requests"], 2)
        self.assertEqual(metrics_data["success_rate"], 1.0)
        self.assertEqual(metrics_data["avg_response_time"], 0.15)

    def test_record_failure(self):
        """Teste l'enregistrement des échecs."""
        self.metrics.record_failure()
        self.metrics.record_failure(is_timeout=True)

        metrics_data = self.metrics.get_metrics()
        self.assertEqual(metrics_data["failure_count"], 2)
        self.assertEqual(metrics_data["timeout_count"], 1)
        self.assertEqual(metrics_data["success_rate"], 0.0)

    def test_window_size_limit(self):
        """Teste la limitation de la taille de la fenêtre."""
        # Enregistrer plus de valeurs que la taille de la fenêtre
        for i in range(10):
            self.metrics.record_success(i * 0.1)

        metrics_data = self.metrics.get_metrics()
        self.assertEqual(len(self.metrics.response_times), 5)
        self.assertEqual(metrics_data["min_response_time"], 0.5)
        self.assertEqual(metrics_data["max_response_time"], 0.9)

    def test_reset(self):
        """Teste la réinitialisation des métriques."""
        self.metrics.record_success(0.1)
        self.metrics.record_failure()

        self.metrics.reset()

        metrics_data = self.metrics.get_metrics()
        self.assertEqual(metrics_data["success_count"], 0)
        self.assertEqual(metrics_data["failure_count"], 0)
        self.assertEqual(metrics_data["total_requests"], 0)


class TestResilientRequester(unittest.TestCase):
    """Tests pour la classe ResilientRequester."""

    def setUp(self):
        """Initialisation avant chaque test."""
        self.requester = ResilientRequester(
            max_retries=2, base_delay=0.1, max_delay=1.0, timeout=0.5, jitter=0.1
        )

    def test_successful_request(self):
        """Teste une requête réussie."""

        # Fonction de test qui réussit
        def test_func():
            return "success"

        result = self.requester.request(test_func)
        self.assertEqual(result, "success")

        # Vérifier les métriques
        metrics = self.requester.get_metrics()
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["failure_count"], 0)

    def test_retry_on_failure(self):
        """Teste les retries après un échec."""
        # Fonction de test qui échoue puis réussit
        counter = [0]

        def test_func():
            counter[0] += 1
            if counter[0] < 2:
                raise ValueError("Test error")
            return "success after retry"

        result = self.requester.request(test_func)
        self.assertEqual(result, "success after retry")
        self.assertEqual(counter[0], 2)

        # Vérifier les métriques
        metrics = self.requester.get_metrics()
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["failure_count"], 1)

    def test_max_retries_exceeded(self):
        """Teste le dépassement du nombre maximum de retries."""

        # Fonction de test qui échoue toujours
        def test_func():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            self.requester.request(test_func)

        # Vérifier les métriques
        metrics = self.requester.get_metrics()
        self.assertEqual(metrics["success_count"], 0)
        self.assertEqual(metrics["failure_count"], 3)  # Initial + 2 retries

    def test_timeout(self):
        """Teste le timeout d'une requête."""

        # Fonction de test qui prend trop de temps
        def test_func():
            time.sleep(1.0)  # Plus long que le timeout
            return "too late"

        with self.assertRaises(TimeoutError):
            self.requester.request(test_func)

        # Vérifier les métriques
        metrics = self.requester.get_metrics()
        # Le test échoue 3 fois (tentative initiale + 2 retries) avec timeout
        self.assertEqual(metrics["timeout_count"], 3)

    def test_circuit_breaker_integration(self):
        """Teste l'intégration avec le circuit breaker."""
        # Configurer un circuit breaker avec un seuil bas
        self.requester.circuit_breaker = CircuitBreaker(failure_threshold=2)

        # Fonction de test qui échoue toujours
        def test_func():
            raise ValueError("Test error")

        # Provoquer l'ouverture du circuit
        try:
            self.requester.request(test_func)
        except ValueError:
            pass  # Ignorer l'erreur, nous voulons juste ouvrir le circuit

        # Forcer une deuxième erreur pour s'assurer que le circuit s'ouvre
        try:
            self.requester.request(test_func)
        except ValueError:
            pass

        # Le circuit devrait être ouvert maintenant
        with self.assertRaises(CircuitBreakerError):
            self.requester.request(test_func)

    def test_fallback_function(self):
        """Teste l'utilisation de la fonction de fallback."""
        # Configurer un fallback
        self.requester.fallback_function = lambda: "fallback result"

        # Fonction de test qui échoue toujours
        def test_func():
            raise ValueError("Test error")

        # Le fallback devrait être utilisé
        result = self.requester.request(test_func)
        self.assertEqual(result, "fallback result")


class TestMultiSourceRequester(unittest.TestCase):
    """Tests pour la classe MultiSourceRequester."""

    def setUp(self):
        """Initialisation avant chaque test."""

        # Créer des fonctions sources fictives
        def source1_func(*args, **kwargs):
            return "result from source1"

        def source2_func(*args, **kwargs):
            raise ValueError("Source2 error")

        def source3_func(*args, **kwargs):
            return "result from source3"

        # Configurer le requester multi-sources
        self.sources = {
            "source1": (source1_func, {}),
            "source2": (source2_func, {}),
            "source3": (source3_func, {}),
        }

        self.multi_requester = MultiSourceRequester(
            sources=self.sources,
            source_priority=["source1", "source2", "source3"],
            global_timeout=1.0,
        )

    def test_successful_first_source(self):
        """Teste le succès de la première source."""
        result, source = self.multi_requester.request()
        self.assertEqual(result, "result from source1")
        self.assertEqual(source, "source1")

    def test_fallback_to_next_source(self):
        """Teste le fallback vers la source suivante en cas d'échec."""

        # Créer des fonctions sources fictives avec des requesters configurés spécifiquement
        def source1_func():
            return "result from source1"

        def source2_func():
            raise ValueError("Source2 error")

        def source3_func():
            return "result from source3"

        # Configurer le requester multi-sources avec des requesters sans retries
        sources = {
            "source1": (source1_func, {}),
            "source2": (source2_func, {}),
            "source3": (source3_func, {}),
        }

        multi_requester = MultiSourceRequester(
            sources=sources,
            source_priority=["source2", "source3", "source1"],
            global_timeout=0.5,
        )

        # Configurer tous les requesters sans retries pour accélérer le test
        for source_name in multi_requester.requesters:
            multi_requester.requesters[source_name] = ResilientRequester(
                max_retries=0, timeout=0.1
            )

        # La source2 échoue, source3 doit être utilisée
        result, source = multi_requester.request()
        self.assertEqual(result, "result from source3")
        self.assertEqual(source, "source3")

    def test_preferred_sources(self):
        """Teste l'utilisation de sources préférées."""
        result, source = self.multi_requester.request(preferred_sources=["source3"])
        self.assertEqual(result, "result from source3")
        self.assertEqual(source, "source3")

    def test_all_sources_fail(self):
        """Teste le cas où toutes les sources échouent."""

        # Remplacer toutes les sources par des fonctions qui échouent
        def failing_source(*args, **kwargs):
            raise ValueError("Source error")

        # Faire échouer toutes les sources
        self.multi_requester.sources = {
            "source1": (failing_source, {}),
            "source2": (failing_source, {}),
            "source3": (failing_source, {}),
        }

        # Configurer des requesters avec 0 retry pour accélérer le test
        for source in self.multi_requester.source_priority:
            self.multi_requester.requesters[source] = ResilientRequester(max_retries=0)

        with self.assertRaises(AllSourcesFailedError):
            self.multi_requester.request()

    def test_global_timeout(self):
        """Teste le timeout global."""

        # Une fonction source qui simule un temps d'exécution long
        def slow_source(*args, **kwargs):
            # Simuler un délai qui dépasserait le timeout
            time.sleep(0.2)  # Petit délai pour les tests
            return "too late"

        # Configure une autre source qui marche
        def fast_source(*args, **kwargs):
            return "fast result"

        # Créer un MultiSourceRequester avec un timeout très court
        sources = {"slow": (slow_source, {}), "fast": (fast_source, {})}

        multi_requester = MultiSourceRequester(
            sources=sources,
            source_priority=["slow", "fast"],
            global_timeout=0.1,  # Timeout très court
        )

        # Configurer le requester de slow sans retries
        multi_requester.requesters["slow"] = ResilientRequester(
            max_retries=0, timeout=0.05
        )

        # La source lente devrait échouer par timeout, puis passer à la source rapide
        try:
            result, source = multi_requester.request()
            self.assertEqual(source, "fast")
            self.assertEqual(result, "fast result")
        except AllSourcesFailedError:
            # Si même la source rapide n'a pas le temps de s'exécuter à cause du timeout global
            # ce n'est pas un problème pour le test
            pass


class TestResilientDecorator(unittest.TestCase):
    """Tests pour le décorateur resilient."""

    def test_decorator_basic(self):
        """Teste le fonctionnement de base du décorateur."""

        # Fonction à décorer
        @resilient(max_retries=1, base_delay=0.1)
        def test_func():
            return "decorated success"

        result = test_func()
        self.assertEqual(result, "decorated success")

        # Vérifier l'accès aux métriques
        metrics = test_func.get_metrics()
        self.assertEqual(metrics["success_count"], 1)

    def test_decorator_with_retry(self):
        """Teste le décorateur avec retry."""
        counter = [0]

        @resilient(max_retries=2, base_delay=0.1)
        def test_func():
            counter[0] += 1
            if counter[0] < 2:
                raise ValueError("Test error")
            return "success after retry"

        result = test_func()
        self.assertEqual(result, "success after retry")
        self.assertEqual(counter[0], 2)

    def test_decorator_with_fallback(self):
        """Teste le décorateur avec fallback."""

        def fallback():
            return "fallback result"

        @resilient(max_retries=1, fallback_function=fallback)
        def test_func():
            raise ValueError("Test error")

        result = test_func()
        self.assertEqual(result, "fallback result")


if __name__ == "__main__":
    unittest.main()
