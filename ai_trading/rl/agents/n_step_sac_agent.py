"""Variante SAC n-step basée sur le noyau PyTorch unique."""

from typing import Optional, Tuple

from ai_trading.rl.agents.sac_agent import OptimizedSACAgent


class NStepSACAgent(OptimizedSACAgent):
    """SAC avec retours n-step exacts et compatible CUDA."""

    def __init__(
        self,
        state_size: Optional[int] = None,
        action_size: int = 1,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: Optional[float] = None,
        discount_factor: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        hidden_size: int = 128,
        train_alpha: bool = True,
        target_entropy=None,
        n_steps: int = 3,
        device: str = "cuda",
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        **kwargs,
    ):
        del alpha_learning_rate, target_entropy
        actual_state_dim = state_dim if state_dim is not None else state_size
        if actual_state_dim is None:
            raise ValueError("state_size ou state_dim est requis")
        actual_action_dim = action_dim if action_dim is not None else action_size
        if n_steps < 1:
            raise ValueError("n_steps doit être supérieur ou égal à 1")
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self.state_size = actual_state_dim
        self.action_size = actual_action_dim
        self.n_step_discount_factor = discount_factor ** n_steps
        super().__init__(
            state_dim=actual_state_dim,
            action_dim=actual_action_dim,
            hidden_dim=hidden_size,
            learning_rate=actor_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=discount_factor,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            action_bounds=action_bounds,
            entropy_regularization=0.0 if train_alpha else 0.2,
            n_step=n_steps,
            device=device,
            **kwargs,
        )
        self.replay_buffer.n_steps = n_steps

    def episode_end(self) -> None:
        self.end_episode()
