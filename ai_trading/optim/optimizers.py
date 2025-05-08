"""
Module contenant des optimiseurs personnalisés pour PyTorch.
Ces implémentations utilisent diverses optimisations pour améliorer les performances
et la rapidité de convergence des modèles.
"""

import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Implémentation optimisée d'Adam avec support pour le gradient scaler.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        eps_inside_sqrt: bool = False,
        use_fp16: bool = False,
    ):
        """
        Initialise l'optimiseur Adam optimisé.

        Args:
            params: Paramètres à optimiser
            lr: Taux d'apprentissage
            betas: Coefficients pour les moyennes mobiles
            eps: Terme pour la stabilité numérique
            weight_decay: Coefficient de régularisation L2
            amsgrad: Si True, utilise la variante AMSGrad
            eps_inside_sqrt: Si True, place epsilon à l'intérieur de la racine carrée
            use_fp16: Si True, utilise la précision mixte (float16)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Le taux d'apprentissage doit être positif: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon doit être positif: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta 1 doit être dans [0, 1): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta 2 doit être dans [0, 1): {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"La régularisation doit être positive: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps_inside_sqrt=eps_inside_sqrt,
            use_fp16=use_fp16,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict):
        """Restaurer l'état de l'optimiseur."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("eps_inside_sqrt", False)
            group.setdefault("use_fp16", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Effectue une seule étape d'optimisation.

        Args:
            closure: Fonction de calcul de la perte et des gradients

        Returns:
            Valeur de la perte
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Récupérer les hyperparamètres
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                amsgrad = group["amsgrad"]
                eps_inside_sqrt = group["eps_inside_sqrt"]
                use_fp16 = group["use_fp16"]

                # Appliquer la régularisation L2 au gradient
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # Récupérer l'état ou l'initialiser
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Initialiser les estimateurs de moment
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                # Récupérer les états
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Utiliser float16 pour les calculs si demandé
                if use_fp16:
                    grad = p.grad.data.to(torch.float16)
                    exp_avg = exp_avg.to(torch.float16)
                    exp_avg_sq = exp_avg_sq.to(torch.float16)
                    if amsgrad:
                        max_exp_avg_sq = max_exp_avg_sq.to(torch.float16)
                else:
                    grad = p.grad.data

                # Mettre à jour les estimateurs de moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Garder la valeur maximale des variances historiques
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                else:
                    denom = exp_avg_sq.div(bias_correction2).sqrt_()

                # Ajouter epsilon pour la stabilité numérique
                if eps_inside_sqrt:
                    denom.add_(eps)
                else:
                    denom.add_(eps, alpha=1.0)

                # Appliquer le pas d'apprentissage avec correction du biais
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Reconvertir les états si on utilise float16
                if use_fp16:
                    state["exp_avg"] = exp_avg.to(p.dtype)
                    state["exp_avg_sq"] = exp_avg_sq.to(p.dtype)
                    if amsgrad:
                        state["max_exp_avg_sq"] = max_exp_avg_sq.to(p.dtype)

        return loss


class RAdam(Optimizer):
    """
    Implémentation de RAdam (Rectified Adam) avec float16 et autres optimisations.
    Basé sur l'article: "On the Variance of the Adaptive Learning Rate and Beyond"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        use_fp16: bool = False,
    ):
        """
        Initialise l'optimiseur RAdam.

        Args:
            params: Paramètres à optimiser
            lr: Taux d'apprentissage
            betas: Coefficients pour les moyennes mobiles
            eps: Terme pour la stabilité numérique
            weight_decay: Coefficient de régularisation L2
            use_fp16: Si True, utilise la précision mixte (float16)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Le taux d'apprentissage doit être positif: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon doit être positif: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta 1 doit être dans [0, 1): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta 2 doit être dans [0, 1): {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"La régularisation doit être positive: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_fp16=use_fp16,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Effectue une seule étape d'optimisation.

        Args:
            closure: Fonction de calcul de la perte et des gradients

        Returns:
            Valeur de la perte
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Récupérer les hyperparamètres
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                use_fp16 = group["use_fp16"]

                # Appliquer la régularisation L2 au gradient
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # Récupérer l'état ou l'initialiser
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Initialiser les estimateurs de moment
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0

                # Récupérer les états
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                rho_inf = state["rho_inf"]
                state["step"] += 1

                # Utiliser float16 pour les calculs si demandé
                if use_fp16:
                    grad = p.grad.data.to(torch.float16)
                    exp_avg = exp_avg.to(torch.float16)
                    exp_avg_sq = exp_avg_sq.to(torch.float16)
                else:
                    grad = p.grad.data

                # Mettre à jour les estimateurs de moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Calculer la longueur du pas RAdam
                rho_t = rho_inf - 2 * step * beta2**step / bias_correction2

                # RAdam: si la variance est suffisamment fiable (rho_t >= 5)
                if rho_t >= 5.0:
                    # Calculer la longueur du pas adaptative
                    alpha = math.sqrt(bias_correction2) / bias_correction1
                    delta = (1 - alpha) / alpha

                    # Calculer le dénominateur de la correction
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    # Appliquer la mise à jour du paramètre
                    p.data.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                else:
                    # Fallback à SGD avec momentum si la variance n'est pas fiable
                    p.data.add_(exp_avg, alpha=-lr / bias_correction1)

                # Reconvertir les états si on utilise float16
                if use_fp16:
                    state["exp_avg"] = exp_avg.to(p.dtype)
                    state["exp_avg_sq"] = exp_avg_sq.to(p.dtype)

        return loss
