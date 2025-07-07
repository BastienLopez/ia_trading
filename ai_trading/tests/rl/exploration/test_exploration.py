import numpy as np
import pytest
from ai_trading.rl.exploration import (
    UCBExploration,
    NoveltyExploration,
    HybridExploration,
    AdaptiveExploration,
)


class TestUCBExploration:
    @pytest.fixture
    def ucb(self):
        return UCBExploration(action_size=3, c=2.0)

    def test_initialization(self, ucb):
        """Teste l'initialisation de l'explorateur UCB."""
        assert ucb.action_size == 3
        assert ucb.c == 2.0
        assert np.all(ucb.action_values == 0)
        assert np.all(ucb.action_counts == 0)
        assert ucb.total_steps == 0

    def test_select_action_untried(self, ucb):
        """Teste la sélection d'action pour des actions jamais essayées."""
        q_values = np.array([1.0, 2.0, 3.0])
        action = ucb.select_action(q_values)
        assert action in [0, 1, 2]  # Une des actions non essayées
        assert ucb.action_counts[action] == 0

    def test_select_action_exploitation(self, ucb):
        """Teste la sélection d'action avec exploitation."""
        # Simuler quelques actions déjà essayées
        ucb.action_counts = np.array([5, 3, 1])
        ucb.total_steps = 9
        q_values = np.array([1.0, 2.0, 3.0])
        action = ucb.select_action(q_values)
        assert action in [0, 1, 2]

    def test_update(self, ucb):
        """Teste la mise à jour des statistiques."""
        action = 1
        reward = 2.0
        ucb.update(action, reward)
        assert ucb.action_counts[action] == 1
        assert ucb.action_values[action] == 2.0

    def test_reset(self, ucb):
        """Teste la réinitialisation de l'explorateur."""
        ucb.action_counts = np.array([1, 2, 3])
        ucb.action_values = np.array([1.0, 2.0, 3.0])
        ucb.total_steps = 6
        ucb.reset()
        assert np.all(ucb.action_counts == 0)
        assert np.all(ucb.action_values == 0)
        assert ucb.total_steps == 0


class TestNoveltyExploration:
    @pytest.fixture
    def novelty(self):
        return NoveltyExploration(
            state_size=2,
            action_size=3,
            novelty_scale=1.0,
            decay_rate=0.99,
            hash_bins=10,
        )

    def test_initialization(self, novelty):
        """Teste l'initialisation de l'explorateur basé sur la nouveauté."""
        assert novelty.state_size == 2
        assert novelty.action_size == 3
        assert novelty.novelty_scale == 1.0
        assert novelty.decay_rate == 0.99
        assert novelty.hash_bins == 10
        assert len(novelty.state_counts) == 0

    def test_hash_state(self, novelty):
        """Teste le hachage des états."""
        state = np.array([0.5, 0.7])
        hashed = novelty._hash_state(state)
        assert isinstance(hashed, tuple)
        assert len(hashed) == 2
        assert all(0 <= h < novelty.hash_bins for h in hashed)

    def test_compute_novelty_bonus(self, novelty):
        """Teste le calcul du bonus de nouveauté."""
        state = np.array([0.5, 0.7])
        bonus = novelty._compute_novelty_bonus(state)
        assert bonus == 1.0  # Premier état visité

        # Visiter le même état plusieurs fois
        for _ in range(5):
            novelty.select_action(np.array([1.0, 2.0, 3.0]), state)
        bonus = novelty._compute_novelty_bonus(state)
        assert bonus < 1.0  # Bonus réduit

    def test_select_action(self, novelty):
        """Teste la sélection d'action avec bonus de nouveauté."""
        state = np.array([0.5, 0.7])
        q_values = np.array([1.0, 2.0, 3.0])
        action = novelty.select_action(q_values, state)
        assert action in [0, 1, 2]

    def test_decay_novelty(self, novelty):
        """Teste la décroissance de la nouveauté."""
        initial_scale = novelty.novelty_scale
        novelty.decay_novelty()
        assert novelty.novelty_scale == initial_scale * novelty.decay_rate

    def test_reset(self, novelty):
        """Teste la réinitialisation de l'explorateur."""
        state = np.array([0.5, 0.7])
        novelty.select_action(np.array([1.0, 2.0, 3.0]), state)
        assert len(novelty.state_counts) > 0
        novelty.reset()
        assert len(novelty.state_counts) == 0


class TestHybridExploration:
    @pytest.fixture
    def hybrid(self):
        return HybridExploration(
            state_size=2,
            action_size=3,
            ucb_c=2.0,
            novelty_scale=1.0,
            decay_rate=0.99,
        )

    def test_initialization(self, hybrid):
        """Teste l'initialisation de l'explorateur hybride."""
        assert hybrid.ucb.action_size == 3
        assert hybrid.novelty.state_size == 2
        assert hybrid.novelty.action_size == 3

    def test_select_action(self, hybrid):
        """Teste la sélection d'action hybride."""
        state = np.array([0.5, 0.7])
        q_values = np.array([1.0, 2.0, 3.0])
        action = hybrid.select_action(q_values, state)
        assert action in [0, 1, 2]

    def test_update(self, hybrid):
        """Teste la mise à jour des statistiques."""
        action = 1
        reward = 2.0
        hybrid.update(action, reward)
        assert hybrid.ucb.action_counts[action] == 1
        assert hybrid.ucb.action_values[action] == 2.0

    def test_decay_novelty(self, hybrid):
        """Teste la décroissance de la nouveauté."""
        initial_scale = hybrid.novelty.novelty_scale
        hybrid.decay_novelty()
        assert hybrid.novelty.novelty_scale == initial_scale * hybrid.novelty.decay_rate

    def test_reset(self, hybrid):
        """Teste la réinitialisation de l'explorateur hybride."""
        state = np.array([0.5, 0.7])
        hybrid.select_action(np.array([1.0, 2.0, 3.0]), state)
        hybrid.update(1, 2.0)
        hybrid.reset()
        assert np.all(hybrid.ucb.action_counts == 0)
        assert len(hybrid.novelty.state_counts) == 0


class TestAdaptiveExploration:
    @pytest.fixture
    def adaptive(self):
        return AdaptiveExploration(
            initial_epsilon=0.1,
            min_epsilon=0.01,
            decay=0.995,
        )

    def test_initialization(self, adaptive):
        """Teste l'initialisation de l'explorateur adaptatif."""
        assert adaptive.epsilon == 0.1
        assert adaptive.min_epsilon == 0.01
        assert adaptive.decay == 0.995
        assert len(adaptive.action_counts) == 0
        assert adaptive.total_steps == 0

    def test_should_explore(self, adaptive):
        """Teste la décision d'exploration."""
        state = np.array([0.5, 0.7])
        # Test sans volatilité
        should_explore = adaptive.should_explore(state)
        assert isinstance(should_explore, bool)
        assert adaptive.epsilon < 0.1  # Epsilon a diminué

        # Test avec volatilité
        should_explore = adaptive.should_explore(state, market_volatility=0.5)
        assert isinstance(should_explore, bool)

    def test_get_ucb_action(self, adaptive):
        """Teste la sélection d'action UCB."""
        state_str = "test_state"
        q_values = np.array([1.0, 2.0, 3.0])
        action = adaptive.get_ucb_action(state_str, q_values)
        assert action in [0, 1, 2]
        assert state_str in adaptive.action_counts
        assert len(adaptive.action_counts[state_str]) == 3

    def test_reset(self, adaptive):
        """Teste la réinitialisation de l'explorateur adaptatif."""
        adaptive.total_steps = 10
        adaptive.action_counts["test"] = np.array([1, 2, 3])
        adaptive.reset()
        assert adaptive.total_steps == 0
        assert len(adaptive.action_counts) == 0 