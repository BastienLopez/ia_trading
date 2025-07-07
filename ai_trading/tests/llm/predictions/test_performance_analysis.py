"""
Tests unitaires pour le module performance_analysis.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import os
import json
import tempfile
from ai_trading.llm.predictions.performance_analysis import PerformanceProfiler, PerformanceProfile

@pytest.fixture
def profiler():
    """Fixture pour créer un PerformanceProfiler."""
    return PerformanceProfiler()

@pytest.fixture
def profile():
    """Fixture pour créer un PerformanceProfile."""
    return PerformanceProfile()

def test_profiler_initialization(profiler):
    """Teste l'initialisation du PerformanceProfiler."""
    assert profiler.start_time is None
    assert all(len(metrics) == 0 for metrics in profiler.metrics.values())

def test_profiler_start_profiling(profiler):
    """Teste le démarrage du profilage."""
    profiler.start_profiling()
    assert profiler.start_time is not None
    assert all(len(metrics) == 0 for metrics in profiler.metrics.values())

def test_profiler_record_metrics(profiler):
    """Teste l'enregistrement des métriques."""
    profiler.start_profiling()
    profiler.record_metrics()
    assert all(len(metrics) == 1 for metrics in profiler.metrics.values())

def test_profiler_get_summary(profiler):
    """Teste l'obtention du résumé des métriques."""
    profiler.start_profiling()
    profiler.record_metrics()
    summary = profiler.get_summary()
    assert all(key in summary for key in ['total_time', 'max_memory', 'max_gpu_memory', 'avg_cpu_usage'])

def test_profile_initialization(profile):
    """Teste l'initialisation du PerformanceProfile."""
    assert len(profile.predictions) == 0
    assert len(profile.actuals) == 0
    assert len(profile.timestamps) == 0
    assert len(profile.metrics) == 0

def test_profile_add_prediction(profile):
    """Teste l'ajout de prédictions."""
    now = datetime.now()
    profile.add_prediction(1.0, 1.1, now)
    assert len(profile.predictions) == 1
    assert len(profile.actuals) == 1
    assert len(profile.timestamps) == 1

def test_profile_calculate_metrics(profile):
    """Teste le calcul des métriques."""
    # Ajouter quelques prédictions
    now = datetime.now()
    for i in range(10):
        profile.add_prediction(i, i + 0.1, now + timedelta(minutes=i))
    
    metrics = profile.calculate_metrics()
    assert all(key in metrics for key in ['mae', 'mse', 'rmse', 'direction_accuracy', 'correlation'])
    assert metrics['mae'] > 0
    assert metrics['correlation'] > 0.9  # Les prédictions sont très corrélées avec les valeurs réelles

def test_profile_save_load_metrics(profile):
    """Teste la sauvegarde et le chargement des métriques."""
    # Ajouter quelques prédictions
    now = datetime.now()
    for i in range(10):
        profile.add_prediction(i, i + 0.1, now + timedelta(minutes=i))
    
    # Calculer et sauvegarder les métriques
    original_metrics = profile.calculate_metrics()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        profile.save_metrics(tmp.name)
        
        # Créer un nouveau profil et charger les métriques
        new_profile = PerformanceProfile()
        new_profile.load_metrics(tmp.name)
        
        assert new_profile.metrics == original_metrics
        
    # Nettoyer
    os.unlink(tmp.name)

def test_empty_data_handling(profile):
    """Teste la gestion des données vides."""
    with pytest.raises(ValueError):
        profile.calculate_metrics()
    
    with pytest.raises(ValueError):
        profile.plot_performance() 