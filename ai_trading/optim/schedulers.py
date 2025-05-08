"""
Module contenant des schedulers de taux d'apprentissage optimisés.
Ces schedulers permettent d'ajuster automatiquement le taux d'apprentissage
pendant l'entraînement pour améliorer les performances et la stabilité.
"""

import math
from typing import List, Union

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
):
    """
    Crée un scheduler qui combine une période de warm-up avec une décroissance en cosinus.

    Args:
        optimizer: Optimiseur PyTorch dont le taux d'apprentissage sera ajusté
        num_warmup_steps: Nombre d'étapes de warm-up
        num_training_steps: Nombre total d'étapes d'entraînement
        num_cycles: Nombre de cycles de cosinus à effectuer pendant la décroissance
        last_epoch: Dernière époque à partir de laquelle reprendre
        min_lr_ratio: Ratio du taux d'apprentissage minimal par rapport au taux initial

    Returns:
        Scheduler LambdaLR configuré avec la politique d'ajustement
    """

    def lr_lambda(current_step):
        # Warm-up phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # Calculer le facteur de décroissance cosinus
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Ajuster pour le ratio minimal
        cosine_decay = min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        return cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
):
    """
    Crée un scheduler qui combine une période de warm-up avec une décroissance linéaire.

    Args:
        optimizer: Optimiseur PyTorch dont le taux d'apprentissage sera ajusté
        num_warmup_steps: Nombre d'étapes de warm-up
        num_training_steps: Nombre total d'étapes d'entraînement
        last_epoch: Dernière époque à partir de laquelle reprendre
        min_lr_ratio: Ratio du taux d'apprentissage minimal par rapport au taux initial

    Returns:
        Scheduler LambdaLR configuré avec la politique d'ajustement
    """

    def lr_lambda(current_step):
        # Warm-up phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # Calculer le facteur de décroissance linéaire
        linear_decay = 1.0 - progress

        # Ajuster pour le ratio minimal
        linear_decay = min_lr_ratio + (1 - min_lr_ratio) * linear_decay
        return max(0.0, linear_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
):
    """
    Crée un scheduler qui combine une période de warm-up avec une décroissance polynomiale.

    Args:
        optimizer: Optimiseur PyTorch dont le taux d'apprentissage sera ajusté
        num_warmup_steps: Nombre d'étapes de warm-up
        num_training_steps: Nombre total d'étapes d'entraînement
        lr_end: Taux d'apprentissage final
        power: Puissance du polynôme utilisé pour la décroissance
        last_epoch: Dernière époque à partir de laquelle reprendre

    Returns:
        Scheduler LambdaLR configuré avec la politique d'ajustement
    """
    # Récupérer le taux d'apprentissage initial
    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        # Warm-up phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase
        if current_step > num_training_steps:
            return lr_end / lr_init  # clip au taux minimal

        # Calcul polynomial entre lr_init et lr_end
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_end + lr_range * (pct_remaining**power)
        return decay / lr_init  # normaliser par le taux initial

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_one_cycle_schedule(
    optimizer: Optimizer,
    max_lr: Union[float, List[float]],
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    last_epoch: int = -1,
):
    """
    Implémentation du scheduler One Cycle Policy.

    Args:
        optimizer: Optimiseur PyTorch dont le taux d'apprentissage sera ajusté
        max_lr: Taux d'apprentissage maximal (un seul ou liste pour chaque groupe)
        total_steps: Nombre total d'étapes d'entraînement
        pct_start: Pourcentage des étapes consacrées à la phase d'augmentation du taux
        anneal_strategy: Stratégie de décroissance ('cos' pour cosinus, 'linear' pour linéaire)
        div_factor: Facteur pour calculer le taux initial (max_lr/div_factor)
        final_div_factor: Facteur pour calculer le taux final (max_lr/final_div_factor)
        last_epoch: Dernière époque à partir de laquelle reprendre

    Returns:
        Scheduler LambdaLR configuré avec la politique One Cycle
    """
    # Convertir max_lr en liste si c'est un unique nombre
    if not isinstance(max_lr, (list, tuple)):
        max_lr = [max_lr] * len(optimizer.param_groups)

    # Calculer les seuils des phases
    steps_up = int(total_steps * pct_start)
    steps_down = total_steps - steps_up

    # Stocker les informations pour chaque groupe de paramètres
    lr_info = []
    for i, group in enumerate(optimizer.param_groups):
        # Calculer les taux d'apprentissage initial et final
        initial_lr = max_lr[i] / div_factor
        final_lr = max_lr[i] / final_div_factor
        group["initial_lr"] = initial_lr

        # Calculer les amplitudes pour les phases d'augmentation et de diminution
        up_amplitude = max_lr[i] - initial_lr
        down_amplitude = max_lr[i] - final_lr

        lr_info.append(
            {
                "initial_lr": initial_lr,
                "max_lr": max_lr[i],
                "final_lr": final_lr,
                "up_amplitude": up_amplitude,
                "down_amplitude": down_amplitude,
            }
        )

    def lr_lambda(step, group_idx=0):
        info = lr_info[group_idx]
        initial_lr = info["initial_lr"]
        max_lr = info["max_lr"]
        final_lr = info["final_lr"]
        up_amplitude = info["up_amplitude"]
        down_amplitude = info["down_amplitude"]

        # Phase d'augmentation du taux
        if step < steps_up:
            progress = step / steps_up
            if anneal_strategy == "cos":
                factor = 1 - math.cos(progress * math.pi / 2)
            else:  # 'linear'
                factor = progress
            return initial_lr / max_lr + factor * up_amplitude / max_lr

        # Phase de diminution du taux
        else:
            progress = (step - steps_up) / steps_down
            if anneal_strategy == "cos":
                factor = 1 + math.cos(progress * math.pi / 2)
            else:  # 'linear'
                factor = 1 - progress
            return max_lr / max_lr - factor * down_amplitude / max_lr

    # Créer un lambda pour chaque groupe
    lambdas = [
        lambda step, group_idx=i: lr_lambda(step, group_idx)
        for i in range(len(optimizer.param_groups))
    ]

    return LambdaLR(optimizer, lambdas, last_epoch)
