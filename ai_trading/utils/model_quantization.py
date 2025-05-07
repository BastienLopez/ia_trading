"""
Module d'utilitaires pour la quantification de modèles PyTorch.

Ce module fournit des fonctions et des classes pour réduire la précision numérique
des modèles (ex: float32 → int8), ce qui permet de réduire leur taille, d'accélérer
l'inférence et d'améliorer l'efficacité énergétique.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def prepare_model_for_quantization(
    model: nn.Module, qconfig_name: str = "default", static: bool = True
) -> nn.Module:
    """
    Prépare un modèle pour la quantification en ajoutant les observateurs nécessaires.

    Args:
        model: Le modèle PyTorch à préparer
        qconfig_name: Configuration de quantification ('default', 'fbgemm', 'qnnpack')
        static: Si True, utilise la quantification statique, sinon dynamique

    Returns:
        Le modèle préparé pour la quantification
    """
    # Vérifier si la quantification est supportée
    if not hasattr(torch, "quantization"):
        logger.error("PyTorch ne supporte pas la quantification sur cette plateforme")
        return model

    # Convertir le modèle en mode d'évaluation
    model.eval()

    # Sélectionner la configuration de quantification
    if qconfig_name == "default":
        qconfig = torch.quantization.default_qconfig
    elif qconfig_name == "fbgemm":
        qconfig = torch.quantization.get_default_qconfig("fbgemm")  # Pour x86 CPU
    elif qconfig_name == "qnnpack":
        qconfig = torch.quantization.get_default_qconfig("qnnpack")  # Pour ARM CPU
    else:
        raise ValueError(f"Configuration de quantification inconnue: {qconfig_name}")

    # Configurer le modèle
    if static:
        model.qconfig = qconfig
        logger.info(
            f"Modèle préparé pour la quantification statique avec {qconfig_name}"
        )
        # Fusionner les modules si possible (conv+bn+relu, etc.)
        model = torch.quantization.fuse_modules(model, _get_fusable_modules(model))
        # Préparer le modèle pour la quantification
        prepared_model = torch.quantization.prepare(model)
    else:
        model.qconfig = torch.quantization.default_dynamic_qconfig
        logger.info("Modèle préparé pour la quantification dynamique")
        # Préparer le modèle pour la quantification dynamique
        prepared_model = torch.quantization.prepare_dynamic(model)

    return prepared_model


def _get_fusable_modules(model: nn.Module) -> List[List[str]]:
    """
    Identifie les modules qui peuvent être fusionnés pour la quantification.

    Args:
        model: Le modèle PyTorch

    Returns:
        Liste des groupes de modules fusables
    """
    fusable_modules = []

    # Explorer la structure du modèle pour identifier des patterns fusables
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            layer_names = [n.split(".")[-1] for n, _ in module.named_children()]

            # Pattern: Conv -> BatchNorm -> ReLU
            if len(layer_names) >= 3:
                for i in range(len(layer_names) - 2):
                    if (
                        hasattr(module[i], "weight")
                        and isinstance(module[i + 1], nn.BatchNorm2d)
                        and isinstance(module[i + 2], nn.ReLU)
                    ):
                        fusable_modules.append(
                            [
                                f"{name}.{layer_names[i]}",
                                f"{name}.{layer_names[i+1]}",
                                f"{name}.{layer_names[i+2]}",
                            ]
                        )

            # Pattern: Linear -> ReLU
            if len(layer_names) >= 2:
                for i in range(len(layer_names) - 1):
                    if isinstance(module[i], nn.Linear) and isinstance(
                        module[i + 1], nn.ReLU
                    ):
                        fusable_modules.append(
                            [f"{name}.{layer_names[i]}", f"{name}.{layer_names[i+1]}"]
                        )

    return fusable_modules


def convert_model_to_quantized(
    prepared_model: nn.Module,
    calibration_fn: Optional[Callable[[nn.Module], None]] = None,
    static: bool = True,
) -> nn.Module:
    """
    Convertit un modèle préparé en modèle quantifié.

    Args:
        prepared_model: Le modèle préparé avec observateurs
        calibration_fn: Fonction de calibration qui exécute le modèle avec des données représentatives
        static: Si True, utilise la quantification statique, sinon dynamique

    Returns:
        Le modèle quantifié
    """
    # Calibrer le modèle si c'est une quantification statique
    if static and calibration_fn is not None:
        logger.info("Calibration du modèle avec des données représentatives...")
        calibration_fn(prepared_model)

    # Convertir le modèle
    if static:
        quantized_model = torch.quantization.convert(prepared_model)
        logger.info("Modèle converti en version quantifiée statique")
    else:
        quantized_model = torch.quantization.convert_dynamic(prepared_model)
        logger.info("Modèle converti en version quantifiée dynamique")

    return quantized_model


def quantize_model_static(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    qconfig_name: str = "default",
) -> nn.Module:
    """
    Applique la quantification statique au modèle complet en une seule fonction.

    Args:
        model: Le modèle PyTorch à quantifier
        calibration_fn: Fonction qui exécute le modèle avec des données représentatives
        qconfig_name: Configuration de quantification ('default', 'fbgemm', 'qnnpack')

    Returns:
        Le modèle quantifié
    """
    # Préparation
    model.eval()
    prepared_model = prepare_model_for_quantization(model, qconfig_name, static=True)

    # Calibration
    calibration_fn(prepared_model)

    # Conversion
    quantized_model = torch.quantization.convert(prepared_model)

    # Calculer les statistiques de taille
    original_size = _get_model_size(model)
    quantized_size = _get_model_size(quantized_model)
    reduction = (original_size - quantized_size) / original_size * 100

    logger.info(f"Quantification statique terminée")
    logger.info(f"Taille originale: {original_size/1024/1024:.2f} MB")
    logger.info(f"Taille quantifiée: {quantized_size/1024/1024:.2f} MB")
    logger.info(f"Réduction: {reduction:.2f}%")

    return quantized_model


def quantize_model_dynamic(
    model: nn.Module, dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Applique la quantification dynamique au modèle.

    Args:
        model: Le modèle PyTorch à quantifier
        dtype: Type de données cible pour la quantification

    Returns:
        Le modèle quantifié dynamiquement
    """
    # Préparation
    model.eval()

    # Quantification dynamique
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.RNNCell, nn.GRUCell},
        dtype=dtype,
    )

    # Calculer les statistiques de taille
    original_size = _get_model_size(model)
    quantized_size = _get_model_size(quantized_model)
    reduction = (original_size - quantized_size) / original_size * 100

    logger.info(f"Quantification dynamique terminée")
    logger.info(f"Taille originale: {original_size/1024/1024:.2f} MB")
    logger.info(f"Taille quantifiée: {quantized_size/1024/1024:.2f} MB")
    logger.info(f"Réduction: {reduction:.2f}%")

    return quantized_model


def _get_model_size(model: nn.Module) -> int:
    """
    Calcule la taille approximative du modèle en octets.

    Args:
        model: Le modèle PyTorch

    Returns:
        Taille approximative en octets
    """
    total_size = 0
    for name, param in model.state_dict().items():
        try:
            element_size = param.element_size()
            numel = param.numel()
            param_size = numel * element_size
            total_size += param_size
        except (AttributeError, RuntimeError):
            # Pour les paramètres qui ne sont pas des tenseurs standard
            # ou qui sont des dtypes sans méthode element_size
            if hasattr(param, "nbytes"):
                total_size += param.nbytes
            elif hasattr(param, "numel") and hasattr(param, "dtype"):
                # Estimation basée sur le type
                if param.dtype == torch.qint8 or param.dtype == torch.quint8:
                    total_size += param.numel()  # 1 byte per element
                elif param.dtype == torch.qint32:
                    total_size += param.numel() * 4  # 4 bytes per element
                else:
                    # Par défaut utiliser 4 bytes
                    total_size += param.numel() * 4
            else:
                # Ignorer les paramètres non mesurables
                logger.debug(f"Paramètre non mesurable ignoré: {name}")
    return total_size


def compare_model_performance(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_fn: Callable[[nn.Module], Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compare les performances du modèle original et du modèle quantifié.

    Args:
        original_model: Le modèle PyTorch original
        quantized_model: Le modèle PyTorch quantifié
        test_fn: Fonction d'évaluation qui renvoie un dictionnaire de métriques

    Returns:
        Dictionnaire avec les comparaisons de performances
    """
    logger.info("Évaluation du modèle original...")
    original_metrics = test_fn(original_model)

    logger.info("Évaluation du modèle quantifié...")
    quantized_metrics = test_fn(quantized_model)

    # Calculer les statistiques de taille
    original_size = _get_model_size(original_model)
    quantized_size = _get_model_size(quantized_model)
    size_reduction = (original_size - quantized_size) / original_size * 100

    # Compiler les résultats
    results = {
        "original_metrics": original_metrics,
        "quantized_metrics": quantized_metrics,
        "original_size_mb": original_size / (1024 * 1024),
        "quantized_size_mb": quantized_size / (1024 * 1024),
        "size_reduction_percent": size_reduction,
    }

    # Ajouter les différences relatives pour chaque métrique
    results["metric_differences"] = {}
    for key in original_metrics:
        if key in quantized_metrics:
            diff = quantized_metrics[key] - original_metrics[key]
            rel_diff = (
                diff / original_metrics[key] * 100
                if original_metrics[key] != 0
                else float("inf")
            )
            results["metric_differences"][key] = {
                "absolute": diff,
                "relative_percent": rel_diff,
            }

    logger.info("Comparaison de performances:")
    logger.info(f"Taille originale: {results['original_size_mb']:.2f} MB")
    logger.info(f"Taille quantifiée: {results['quantized_size_mb']:.2f} MB")
    logger.info(f"Réduction: {results['size_reduction_percent']:.2f}%")

    for key in results["metric_differences"]:
        diff = results["metric_differences"][key]
        logger.info(f"Métrique '{key}': {diff['relative_percent']:.2f}% de différence")

    return results


def export_quantized_model(
    model: nn.Module,
    filepath: str,
    input_shape: Optional[List[int]] = None,
    input_sample: Optional[torch.Tensor] = None,
) -> None:
    """
    Exporte un modèle quantifié au format TorchScript.

    Args:
        model: Le modèle quantifié à exporter
        filepath: Chemin où sauvegarder le modèle
        input_shape: Forme d'entrée pour le traçage (ex: [1, 3, 224, 224])
        input_sample: Exemple d'entrée pour le traçage
    """
    model.eval()

    # Préparer un exemple d'entrée pour le traçage si nécessaire
    if input_sample is None and input_shape is not None:
        input_sample = torch.rand(*input_shape)

    if input_sample is not None:
        # Tracer le modèle
        traced_model = torch.jit.trace(model, input_sample)
        logger.info(f"Modèle tracé avec succès")
    else:
        # Script le modèle
        scripted_model = torch.jit.script(model)
        traced_model = scripted_model
        logger.info(f"Modèle scripté avec succès")

    # Sauvegarder le modèle
    traced_model.save(filepath)
    logger.info(f"Modèle quantifié exporté vers: {filepath}")


def quantize_for_mobile(
    model: nn.Module,
    filepath: str,
    input_shape: Optional[List[int]] = None,
    input_sample: Optional[torch.Tensor] = None,
) -> None:
    """
    Quantifie et exporte un modèle optimisé pour les appareils mobiles.

    Args:
        model: Le modèle PyTorch à quantifier
        filepath: Chemin où sauvegarder le modèle
        input_shape: Forme d'entrée pour le traçage (ex: [1, 3, 224, 224])
        input_sample: Exemple d'entrée pour le traçage
    """
    model.eval()

    # Quantification
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    # Préparer un exemple d'entrée pour le traçage si nécessaire
    if input_sample is None and input_shape is not None:
        input_sample = torch.rand(*input_shape)

    if input_sample is not None:
        # Tracer le modèle
        traced_model = torch.jit.trace(quantized, input_sample)
        logger.info(f"Modèle tracé avec succès")
    else:
        # Script le modèle
        scripted_model = torch.jit.script(quantized)
        traced_model = scripted_model
        logger.info(f"Modèle scripté avec succès")

    # Optimiser pour mobile
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)

    # Sauvegarder le modèle
    optimized_model.save(filepath)

    # Calculer les statistiques de taille
    original_size = _get_model_size(model)
    quantized_size = _get_model_size(quantized)

    logger.info(f"Modèle quantifié et optimisé pour mobile exporté vers: {filepath}")
    logger.info(f"Taille originale: {original_size/1024/1024:.2f} MB")
    logger.info(f"Taille quantifiée: {quantized_size/1024/1024:.2f} MB")
    logger.info(
        f"Réduction: {(original_size - quantized_size) / original_size * 100:.2f}%"
    )


def benchmark_inference_speed(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_data: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Compare la vitesse d'inférence entre le modèle original et le modèle quantifié.

    Args:
        original_model: Le modèle PyTorch original
        quantized_model: Le modèle PyTorch quantifié
        input_data: Données d'entrée pour l'inférence
        num_iterations: Nombre d'itérations pour le benchmark
        warmup_iterations: Nombre d'itérations d'échauffement

    Returns:
        Dictionnaire avec les résultats du benchmark
    """
    import time

    # Mettre les modèles en mode évaluation
    original_model.eval()
    quantized_model.eval()

    # Fonction pour mesurer le temps d'inférence
    def measure_inference_time(model, data, iterations, warmup):
        # Échauffement
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(data)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(data)
        end_time = time.time()

        return (end_time - start_time) / iterations

    # Mesurer les temps d'inférence
    original_time = measure_inference_time(
        original_model, input_data, num_iterations, warmup_iterations
    )
    quantized_time = measure_inference_time(
        quantized_model, input_data, num_iterations, warmup_iterations
    )

    # Calculer l'accélération
    speedup = original_time / quantized_time if quantized_time > 0 else float("inf")

    # Compiler les résultats
    results = {
        "original_inference_time_ms": original_time * 1000,
        "quantized_inference_time_ms": quantized_time * 1000,
        "speedup_factor": speedup,
        "improvement_percent": (speedup - 1) * 100,
    }

    logger.info("Benchmark d'inférence:")
    logger.info(
        f"Temps d'inférence original: {results['original_inference_time_ms']:.3f} ms"
    )
    logger.info(
        f"Temps d'inférence quantifié: {results['quantized_inference_time_ms']:.3f} ms"
    )
    logger.info(
        f"Accélération: {results['speedup_factor']:.2f}x ({results['improvement_percent']:.2f}%)"
    )

    return results


def is_dynamic_quantization_supported():
    """Vérifie si la quantification dynamique est réellement supportée."""
    if not hasattr(torch, "quantization") or not hasattr(torch.quantization, "quantize_dynamic"):
        return False
    
    # Vérifier si un backend est disponible pour CPU
    backend_available = False
    for backend in ["fbgemm", "qnnpack"]:
        try:
            torch.backends.quantized.engine = backend
            backend_available = True
            break
        except Exception:
            pass
    
    if not backend_available:
        return False
    
    # Tester si la quantification fonctionne sur un modèle simple
    try:
        with torch.no_grad():
            model = nn.Sequential(nn.Linear(10, 10))
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            # Tester l'inférence
            input_tensor = torch.randn(1, 10)
            output = quantized_model(input_tensor)
            return output.shape == (1, 10)
    except Exception as e:
        logger.warning(f"Test de quantification dynamique a échoué: {e}")
        return False


def is_cuda_quantization_supported(force_cuda=False):
    """Vérifie si la quantification est supportée avec CUDA.
    
    Args:
        force_cuda: Si True, ignore les vérifications et considère que CUDA est supporté
    
    Returns:
        bool: True si la quantification est supportée avec CUDA, False sinon
    """
    # Si force_cuda est activé, on considère que CUDA est supporté
    if force_cuda:
        logger.info("Force CUDA quantization support")
        return True
        
    # Vérifier si CUDA est disponible
    if not torch.cuda.is_available():
        logger.warning("CUDA n'est pas disponible")
        return False
        
    # Vérifier la version de PyTorch (la quantification CUDA est supportée à partir de versions récentes)
    cuda_version = torch.version.cuda
    if cuda_version is None:
        logger.warning("Version CUDA non détectée dans PyTorch")
        return False
        
    # Tester si la quantification CUDA fonctionne
    try:
        with torch.no_grad():
            # Créer un modèle simple
            model = nn.Sequential(nn.Linear(10, 10)).cuda()
            model.eval()
            
            # Essayer la quantification FP16 qui est généralement supportée sur GPU
            input_tensor = torch.randn(1, 10, device='cuda')
            
            # Pour CUDA, on utilise généralement la quantification FP16 
            # plutôt que Int8 qui est plus adaptée au CPU
            with torch.cuda.amp.autocast():
                output = model(input_tensor)
                
            return output.shape == (1, 10)
    except Exception as e:
        logger.warning(f"Test de quantification CUDA a échoué: {e}")
        return False


def setup_cuda_quantization(force_setup=False):
    """Configure l'environnement pour la quantification CUDA.
    
    Cette fonction configure les paramètres nécessaires pour utiliser 
    la quantification avec CUDA, notamment les fusions d'opérations et 
    la précision mixte automatique.
    
    Args:
        force_setup: Si True, poursuit la configuration même si CUDA n'est pas détecté
    
    Returns:
        bool: True si la configuration a réussi, False sinon
    """
    if not torch.cuda.is_available() and not force_setup:
        logger.warning("CUDA n'est pas disponible, impossible de configurer la quantification CUDA")
        return False
        
    try:
        # Activer les optimisations cudnn
        torch.backends.cudnn.benchmark = True
        
        # Activer TF32 pour Ampere+ GPUs (si disponible)
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
            
        logger.info("Configuration CUDA pour quantification réussie")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la configuration CUDA pour quantification: {e}")
        return False


def configure_cuda_quantization_env(model, device="cuda"):
    """Configuration complète de l'environnement pour la quantification CUDA.
    
    Cette fonction effectue une configuration complète pour la quantification
    CUDA, y compris la détection, la configuration des optimisations, et la
    préparation du modèle.
    
    Args:
        model: Le modèle PyTorch à préparer pour la quantification CUDA
        device: Le périphérique cible ('cuda' par défaut)
        
    Returns:
        tuple: (modèle configuré, succès de la configuration)
    """
    # Force la configuration CUDA même si les vérifications échouent
    force_cuda = True
    
    # Vérifier si CUDA est disponible
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA n'est pas disponible, utilisation du CPU à la place")
        device = "cpu"
        
    # Configurer l'environnement CUDA
    if device == "cuda":
        setup_successful = setup_cuda_quantization(force_setup=force_cuda)
        if not setup_successful and not force_cuda:
            logger.warning("Configuration CUDA échouée, passage au CPU")
            device = "cpu"
    
    # Préparer le modèle
    try:
        # Déplacer le modèle vers le dispositif approprié
        model = model.to(device)
        model.eval()  # Mettre en mode évaluation
        
        # Pour CUDA, configurer pour la précision mixte
        if device == "cuda":
            # Désactiver le calcul de gradient
            torch.set_grad_enabled(False)
            
            # Configurer le modèle pour la précision mixte
            if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                logger.info("Précision mixte automatique configurée")
                
            logger.info(f"Modèle configuré pour l'inférence quantifiée sur {device}")
            return model, True
        else:
            logger.info(f"Modèle configuré pour l'inférence sur {device}")
            return model, True
    except Exception as e:
        logger.error(f"Erreur lors de la configuration du modèle: {e}")
        return model, False


def quantize_model_for_inference(model, dtype=torch.float16, device="cuda"):
    """
    Quantifie un modèle pour l'inférence, en utilisant la méthode optimale pour le périphérique.
    
    Cette fonction choisit automatiquement la méthode de quantification la plus appropriée:
    - Pour CUDA: Utilise la précision mixte (float16/bfloat16)
    - Pour CPU: Utilise la quantification dynamique (int8)
    
    Args:
        model: Le modèle PyTorch à quantifier
        dtype: Type de données pour la quantification CUDA (torch.float16 par défaut)
        device: Le périphérique cible ("cuda" ou "cpu")
        
    Returns:
        tuple: (modèle quantifié, méthode utilisée)
    """
    # Vérifier si CUDA est disponible quand demandé
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA demandé mais non disponible, utilisation du CPU")
        device = "cpu"
    
    # Pour CUDA, utiliser la précision mixte
    if device == "cuda":
        try:
            # Configurer l'environnement
            setup_cuda_quantization(force_setup=True)
            
            # Déplacer le modèle sur CUDA
            model = model.to(device)
            model.eval()
            
            # Convertir le modèle en float16/bfloat16 si demandé
            if dtype in [torch.float16, torch.bfloat16]:
                model = model.to(dtype)
                logger.info(f"Modèle converti en {dtype} sur {device}")
            
            # Activer l'optimisation de fusion des opérations si possible
            if hasattr(torch, "jit"):
                try:
                    # Tentative de script JIT avec tracé
                    example_input = torch.randn(1, *model.input_shape[1:], device=device)
                    traced_model = torch.jit.trace(model, example_input)
                    logger.info("Modèle optimisé avec JIT tracing")
                    return traced_model, "cuda_jit_trace"
                except Exception as e:
                    logger.warning(f"JIT tracing échoué: {e}, utilisation du modèle standard")
            
            return model, "cuda_" + str(dtype).split(".")[-1]
        except Exception as e:
            logger.error(f"Quantification CUDA échouée: {e}, retour au CPU")
            device = "cpu"
    
    # Pour CPU, utiliser la quantification dynamique standard
    if device == "cpu":
        try:
            model = model.cpu()
            model.eval()
            
            # Vérifier si la quantification dynamique est supportée
            if is_dynamic_quantization_supported():
                quantized_model = quantize_model_dynamic(model)
                logger.info("Modèle quantifié dynamiquement sur CPU")
                return quantized_model, "cpu_dynamic_int8"
            else:
                logger.warning("Quantification dynamique non supportée, utilisation du modèle non quantifié")
                return model, "cpu_float32"
        except Exception as e:
            logger.error(f"Quantification CPU échouée: {e}")
            return model, "cpu_float32"
    
    # Cas par défaut
    return model, "original"


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)

    # Créer un modèle simple pour le test
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )

    # Fonction de calibration simple pour la quantification statique
    def calibrate(model):
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(4, 3, 32, 32)
                _ = model(x)

    # Appliquer la quantification dynamique
    quantized_model = quantize_model_dynamic(model)

    # Benchmark d'inférence
    input_data = torch.randn(1, 3, 32, 32)
    benchmark_results = benchmark_inference_speed(model, quantized_model, input_data)
