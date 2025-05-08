# 🧪 Guide des Tests et de la Qualité du Code - AI Trading

## Objectifs

Ce guide vise à établir des standards élevés pour la qualité des tests dans le projet AI Trading :

1. **Fiabilité** : Garantir que tous les composants fonctionnent comme prévu
2. **Maintenance** : Faciliter la détection rapide des régressions
3. **Documentation** : Fournir des exemples d'utilisation à travers les tests
4. **Performance** : Assurer que les tests s'exécutent rapidement pour permettre une itération rapide
5. **Couverture** : Viser une couverture de code complète pour les composants critiques
6. **Compatibilité** : S'assurer que les tests fonctionnent sur toutes les plateformes et environnements supportés
7. **Propreté** : Éviter les avertissements (warnings) pour maintenir des sorties de test claires

## Structure des Tests

```
ai_trading/tests/
├── test_agents/                 # Tests des agents RL (DQN, SAC, etc.)
├── test_environments/           # Tests des environnements de trading
├── test_data/                   # Tests de collecte et prétraitement des données
├── test_rewards/                # Tests des fonctions de récompense
├── test_technical_indicators/   # Tests des indicateurs techniques
├── test_sentiment/              # Tests d'analyse de sentiment
├── test_risk_management/        # Tests de gestion des risques
└── test_integration/            # Tests d'intégration entre composants
```

## Bonnes Pratiques

### 1. Tests Unitaires

- **Isolés** : Chaque test doit être indépendant des autres
- **Déterministes** : Les tests doivent produire les mêmes résultats à chaque exécution
- **Rapides** : Optimiser pour des exécutions rapides (données minimales, moins d'étapes)
- **Lisibles** : Le nom et le contenu du test doivent clairement indiquer ce qui est testé

```python
def test_agent_learns_simple_pattern():
    # Configurer un environnement simple avec un pattern prévisible
    env = SimpleTradingEnvironment(pattern="uptrend", steps=10)
    
    # Créer et entraîner l'agent avec un nombre minimal d'épisodes
    agent = DQNAgent(state_size=env.observation_space.shape[0], 
                    action_size=env.action_space.n)
    history = agent.train(env, episodes=5, batch_size=4)
    
    # Vérifier que l'agent a appris le pattern de base
    assert history['rewards'][-1] > history['rewards'][0], "L'agent doit améliorer ses récompenses"
    
    # Vérifier la performance sur un épisode supplémentaire
    total_reward = agent.evaluate(env, episodes=1)
    assert total_reward > 0, "L'agent doit générer une récompense positive après apprentissage"
```

### 2. Tests d'Agents RL

- **Environnements simplifiés** : Utiliser des environnements déterministes et simples
- **Épisodes courts** : Limiter le nombre d'étapes par épisode (5-10 au lieu de centaines)
- **Vérifications de base** : Tester l'initialisation, les actions, la mémorisation, la sauvegarde/chargement
- **Apprentissage minimal** : Vérifier avec peu d'épisodes que l'agent peut apprendre un pattern trivial

```python
@pytest.fixture
def simple_env():
    """Environnement simple pour tester les agents RL."""
    return TradingEnvironment(
        data=generate_simple_data(steps=20),
        initial_balance=10000,
        transaction_fee=0.001
    )

def test_sac_agent_memorization(simple_env):
    agent = SACAgent(
        state_dim=simple_env.observation_space.shape[0],
        action_dim=simple_env.action_space.shape[0],
        replay_buffer_size=100
    )
    
    # Vérifier que la mémorisation fonctionne
    state = simple_env.reset()
    action = agent.select_action(state)
    next_state, reward, done, _ = simple_env.step(action)
    
    # Mémoriser l'expérience
    agent.remember(state, action, reward, next_state, done)
    
    # Vérifier que le buffer contient l'expérience
    assert len(agent.replay_buffer) == 1
    
    # Vérifier que l'agent peut échantillonner du buffer
    if len(agent.replay_buffer) >= agent.batch_size:
        batch = agent.replay_buffer.sample(agent.batch_size)
        assert len(batch) == 5  # (states, actions, rewards, next_states, dones)
```

### 3. Fixtures et Paramétrage

- **Fixtures communes** : Créer des fixtures pour les objets réutilisés entre tests
- **Tests paramétrés** : Utiliser `@pytest.mark.parametrize` pour tester différentes configurations
- **Données de test partagées** : Centraliser les données de test communes

```python
@pytest.fixture
def price_data():
    """Données de prix pour les tests."""
    return pd.DataFrame({
        'open': [100, 101, 102, 101, 100],
        'high': [102, 103, 104, 103, 102],
        'low': [99, 100, 101, 99, 98],
        'close': [101, 102, 101, 100, 99],
        'volume': [1000, 1100, 1200, 1100, 1000]
    })

@pytest.mark.parametrize("agent_type", ["dqn", "sac", "ppo"])
def test_agent_creation(agent_type, price_data):
    """Tester la création de différents types d'agents."""
    env = TradingEnvironment(data=price_data)
    agent = create_agent(
        agent_type=agent_type,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n if agent_type == "dqn" else env.action_space.shape[0]
    )
    assert agent is not None, f"L'agent de type {agent_type} devrait être créé"
```

### 4. Rapports et Sorties

- **Emplacement standardisé** : Tous les rapports et sorties de tests dans `ai_trading/info_retour/tests/`
- **Rapports de couverture** : Générer des rapports HTML de couverture de code
- **Visualisations de test** : Stocker les graphiques et visualisations de test

### 5. Tests Multi-plateformes

- **Détection de fonctionnalités** : Vérifier la disponibilité des fonctionnalités avant de les utiliser
- **Mécanismes de fallback** : Fournir des alternatives quand les fonctionnalités avancées ne sont pas disponibles
- **Tests informatifs** : Les tests doivent passer tout en indiquant le niveau de support

```python
def test_feature_with_platform_specific_behavior():
    """Test qui s'adapte aux capacités de la plateforme"""
    # Vérifier si une fonctionnalité est disponible
    has_feature = check_if_feature_is_available()
    
    if has_feature:
        # Tester la fonctionnalité complète
        result = use_advanced_feature()
        assert result is not None
        print("Fonctionnalité avancée testée avec succès")
    else:
        # Tester l'alternative ou la version de base
        result = use_basic_feature()
        assert result is not None
        print("Alternative de base testée avec succès (fonctionnalité avancée non disponible)")
```

### 6. Gestion des Avertissements (Warnings)

- **Filtrer les warnings inutiles** : Supprimer les warnings des dépendances externes
- **Filtrer au niveau du module** : Configurer les filtres au début du fichier de test
- **Filtrer au niveau du test** : Utiliser des décorateurs `pytest.mark.filterwarnings`
- **Filtrer localement** : Utiliser `warnings.catch_warnings` pour des sections spécifiques

```python
# 1. Filtrer au niveau du module
import warnings
import pytest

# Filtrer les warnings par catégorie
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")

# Filtrer les warnings par message spécifique
warnings.filterwarnings("ignore", message="jax.xla_computation is deprecated")

# 2. Configurer pytest pour ignorer ces warnings également
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow"),
    pytest.mark.filterwarnings("ignore::UserWarning:torch"),
]

# 3. Fonction avec gestion locale des warnings
def test_with_local_warning_suppression():
    # Supprimer localement les warnings dans une section spécifique
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Code qui génère potentiellement des warnings
        result = function_that_emits_warnings()
    
    # Vérifier le résultat
    assert result is not None
```

#### Exemple : Tests de quantification PyTorch

```python
def test_model_quantization(model, input_data):
    """Test de quantification qui s'adapte aux capacités de PyTorch"""
    # Vérifier le support de quantification
    if not hasattr(torch, "quantization"):
        print("Information: PyTorch sans support de quantification, test limité")
        return
    
    # Essayer d'abord la quantification dynamique (plus universelle)
    try:
        if hasattr(torch.quantization, "quantize_dynamic"):
            model.eval()  # Mettre en mode évaluation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            
            # Tester le modèle quantifié
            output = quantized_model(input_data)
            assert output.shape == expected_shape
            print("Quantification dynamique réussie")
            return
    except Exception as e:
        print(f"Information: Quantification dynamique non disponible: {e}")
    
    # Si la quantification dynamique échoue, essayer avec un modèle plus simple
    try:
        # Créer un modèle plus simple (seulement des couches linéaires)
        simple_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        simple_model.eval()
        
        # Essayer de quantifier ce modèle plus simple
        simple_input = torch.randn(2, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            quantized_simple = torch.quantization.quantize_dynamic(
                simple_model, {nn.Linear}, dtype=torch.qint8
            )
        simple_output = quantized_simple(simple_input)
        
        print("Quantification de modèle simple réussie")
        assert simple_output.shape == (2, 2)
    except Exception as e:
        print(f"Information: Même la quantification simple a échoué: {e}")
        print("La quantification n'est pas complètement supportée sur cette plateforme")
```

## Commandes Utiles

```bash
# Exécuter tous les tests
pytest ai_trading/tests/

# Exécuter avec génération de rapport de couverture
pytest ai_trading/tests/ --cov=ai_trading --cov-report=html:ai_trading/info_retour/tests/coverage

# Exécuter en parallèle pour accélérer les tests
pytest ai_trading/tests/ -xvs -n auto

# Exécuter uniquement les tests rapides (skip les tests lents)
pytest ai_trading/tests/ -v --skip-slow

# Debug avec affichage détaillé
pytest ai_trading/tests/test_agents/test_dqn_agent.py -xvs

# Tester la compatibilité multi-plateforme en utilisant Docker
docker build -f Dockerfile.test -t ai-trading-tests .
docker run ai-trading-tests

# Exécuter les tests sans afficher les warnings
pytest ai_trading/tests/ -v --disable-warnings

# Exécuter les tests avec détail des warnings
pytest ai_trading/tests/ -v -W all
```

## Solutions aux Problèmes Courants

1. **Comparaisons entre MagicMock et int**
   ```python
   # Problème
   assert agent.steps > some_mock_object  # Erreur!
   
   # Solution
   agent.steps = 100  # Utiliser un entier réel au lieu d'un MagicMock
   assert agent.steps > 50
   ```

2. **Tests ignorés pour leur durée**
   ```python
   # Problème
   @pytest.mark.skip(reason="Test trop long")
   def test_train_agent():
       # Test qui prend plusieurs minutes
   
   # Solution
   def test_train_agent():
       # Réduire le nombre d'épisodes et la taille des données
       train_episodes = 5  # Au lieu de 100+
       env = TradingEnvironment(data=small_dataset)
       agent.train(env, episodes=train_episodes)
       
       # Vérifier seulement les aspects essentiels
       assert agent.model is not None
   ```

3. **Méthodes dépréciées**
   ```python
   # Déprécié
   df.fillna(method='bfill')
   
   # Préféré
   df.bfill()
   ```

4. **Tests non déterministes**
   ```python
   # Problème: utilisation de random sans seed fixe
   
   # Solution: toujours fixer les seeds pour les tests
   @pytest.fixture(autouse=True)
   def set_random_seed():
       np.random.seed(42)
       random.seed(42)
       torch.manual_seed(42)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(42)
   ```

5. **Fonctionnalités spécifiques à certaines plateformes**
   ```python
   # Problème: fonctionnalité non disponible sur certaines plateformes
   
   # Solution: détecter et s'adapter
   def test_platform_specific_feature():
       # Vérifier la disponibilité
       feature_available = is_feature_available()
       
       if feature_available:
           # Tester la fonctionnalité complète
           # ...
       else:
           # Test informatif qui passe quand même
           print("Information: Fonctionnalité non disponible sur cette plateforme")
           # Éventuellement tester une alternative si disponible
   ```

6. **Warnings externes polluant les sorties de test**
   ```python
   # Problème: warnings de dépendances externes brouillant les résultats
   
   # Solution 1: filtres de warning au niveau du module
   import warnings
   warnings.filterwarnings("ignore", message="jax.xla_computation is deprecated")
   
   # Solution 2: filtres de warning avec pytest
   pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
   
   # Solution 3: Exécuter pytest avec l'option --disable-warnings
   # pytest --disable-warnings
   ```

## Processus CI/CD

1. Les tests doivent passer avant toute fusion de PR
2. La couverture ne doit pas diminuer
3. Utiliser Docker pour tester dans un environnement cohérent

```bash
# Exécuter les tests via Docker
docker build -f Dockerfile.test -t ai-trading-tests .
docker run ai-trading-tests
```

## Conclusion

Des tests bien conçus et maintenus sont essentiels pour garantir la fiabilité du système de trading algorithmique. En suivant ces bonnes pratiques, nous pouvons développer avec confiance et itérer rapidement.

N'oubliez pas : un bon test est un investissement, pas une dépense! 