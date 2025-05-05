#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour les optimisations système du projet.
Contient des fonctions pour configurer l'environnement système,
gérer le logging efficace et optimiser les paramètres du système.
"""

import json
import logging
import logging.handlers
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

# Configuration du logging de base
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constantes pour les limites système
DEFAULT_FILE_LIMIT = 4096  # Limite par défaut pour les descripteurs de fichiers
DEFAULT_PROCESS_LIMIT = 4096  # Limite par défaut pour le nombre de processus
DEFAULT_MEMORY_LIMIT = 0  # 0 = pas de limite


class SystemOptimizer:
    """Classe pour optimiser les paramètres système pour l'IA trading."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise l'optimiseur système.

        Args:
            config_file: Chemin optionnel vers un fichier de configuration
        """
        self.system_info = self._get_system_info()
        self.config = self._load_config(config_file)
        self.applied_optimizations = {}

    def _get_system_info(self) -> Dict[str, Any]:
        """Récupère les informations système pour adapter les optimisations."""
        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "swap_total": psutil.swap_memory().total,
            "disk_info": {},
            "python_version": platform.python_version(),
            "is_admin": self._check_admin_privileges(),
        }

        # Obtenir des informations sur les disques
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                info["disk_info"][partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "fs_type": partition.fstype,
                }
            except PermissionError:
                # Certains disques peuvent nécessiter des permissions spéciales
                pass

        return info

    def _check_admin_privileges(self) -> bool:
        """Vérifie si le script s'exécute avec des privilèges administrateur."""
        try:
            if platform.system() == "Windows":
                import ctypes

                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                # Pour Unix/Linux/MacOS
                return os.geteuid() == 0
        except:
            return False

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Charge la configuration à partir d'un fichier ou utilise les valeurs par défaut.

        Args:
            config_file: Chemin vers le fichier de configuration

        Returns:
            Dictionnaire de configuration
        """
        default_config = {
            "env_vars": {
                "PYTHONUNBUFFERED": "1",  # Désactiver la mise en tampon pour stdout/stderr
                "PYTHONFAULTHANDLER": "1",  # Activer les logs de segmentation fault
                "PYTHONHASHSEED": "0",  # Rendre les hashs déterministes
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Limiter les fragmentations de mémoire CUDA
                "OMP_NUM_THREADS": str(
                    max(1, psutil.cpu_count(logical=False) // 2)
                ),  # Threads OpenMP optimisés
                "MKL_NUM_THREADS": str(
                    max(1, psutil.cpu_count(logical=False) // 2)
                ),  # Threads MKL optimisés
                "NUMEXPR_NUM_THREADS": str(
                    max(1, psutil.cpu_count(logical=False) // 2)
                ),  # Threads NumExpr optimisés
            },
            "system_limits": {
                "file_limit": DEFAULT_FILE_LIMIT,
                "process_limit": DEFAULT_PROCESS_LIMIT,
                "memory_limit_mb": DEFAULT_MEMORY_LIMIT,  # 0 = pas de limite
            },
            "disk_optimization": {
                "tmp_in_ram": True,  # Utiliser un tmpfs pour les fichiers temporaires
                "use_ssd_for_cache": True,  # Préférer les SSD pour les caches
            },
            "memory_optimization": {
                "swappiness": 10,  # Valeur recommandée pour la BI/ML (moins de swap)
                "cache_pressure": 50,  # Équilibre entre cache et mémoire application
            },
            "logging": {
                "level": "INFO",
                "max_file_size_mb": 10,
                "rotation_count": 5,
                "use_json_format": False,
            },
        }

        if config_file:
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                    # Fusion des configurations
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(
                    f"Erreur lors du chargement du fichier de configuration: {e}"
                )
                logger.info("Utilisation de la configuration par défaut")

        return default_config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Met à jour récursivement un dictionnaire avec les valeurs d'un autre.

        Args:
            base_dict: Dictionnaire de base à mettre à jour
            update_dict: Dictionnaire contenant les nouvelles valeurs
        """
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def optimize_environment_variables(self) -> Dict[str, str]:
        """
        Configure les variables d'environnement pour des performances optimales.

        Returns:
            Dictionnaire des variables d'environnement configurées
        """
        # Appliquer les variables d'environnement depuis la configuration
        for var_name, var_value in self.config["env_vars"].items():
            os.environ[var_name] = str(var_value)

        # Variables spécifiques à l'OS
        if self.system_info["os"] == "Linux":
            # Optimisations spécifiques Linux pour le machine learning
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
                "true"  # Allocation dynamique de la mémoire GPU pour TensorFlow
            )
            os.environ["TF_XLA_FLAGS"] = (
                "--tf_xla_cpu_global_jit"  # Optimisation XLA pour TensorFlow
            )

        elif self.system_info["os"] == "Windows":
            # Optimisations spécifiques Windows
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
                "2"  # Réduire les logs TensorFlow sur Windows
            )
            os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"  # Cache CUDA de 2 Go

        self.applied_optimizations["environment_variables"] = dict(os.environ)
        logger.info("Variables d'environnement optimisées configurées")

        return dict(os.environ)

    def configure_system_limits(self) -> Dict[str, Any]:
        """
        Configure les limites système (ulimit) pour optimiser les performances.

        Returns:
            État des limites système après configuration
        """
        # Les configurations dépendent du système
        current_limits = {}

        if self.system_info["os"] != "Windows":
            try:
                import resource

                # Limites actuelles
                current_limits["file_limit_soft"] = resource.getrlimit(
                    resource.RLIMIT_NOFILE
                )[0]
                current_limits["file_limit_hard"] = resource.getrlimit(
                    resource.RLIMIT_NOFILE
                )[1]

                # Configurer de nouvelles limites si on a les permissions
                if self.system_info["is_admin"]:
                    # Limites de fichiers ouverts
                    file_limit = self.config["system_limits"]["file_limit"]
                    try:
                        resource.setrlimit(
                            resource.RLIMIT_NOFILE, (file_limit, file_limit)
                        )
                        logger.info(
                            f"Limite de fichiers ouverts configurée à {file_limit}"
                        )
                    except (ValueError, resource.error) as e:
                        logger.warning(
                            f"Impossible de définir la limite de fichiers: {e}"
                        )

                    # Limite de processus
                    process_limit = self.config["system_limits"]["process_limit"]
                    try:
                        resource.setrlimit(
                            resource.RLIMIT_NPROC, (process_limit, process_limit)
                        )
                        logger.info(f"Limite de processus configurée à {process_limit}")
                    except (ValueError, resource.error) as e:
                        logger.warning(
                            f"Impossible de définir la limite de processus: {e}"
                        )

                    # Limite de mémoire
                    memory_limit_mb = self.config["system_limits"]["memory_limit_mb"]
                    if memory_limit_mb > 0:
                        memory_limit_bytes = memory_limit_mb * 1024 * 1024
                        try:
                            resource.setrlimit(
                                resource.RLIMIT_AS,
                                (memory_limit_bytes, memory_limit_bytes),
                            )
                            logger.info(
                                f"Limite de mémoire configurée à {memory_limit_mb} MB"
                            )
                        except (ValueError, resource.error) as e:
                            logger.warning(
                                f"Impossible de définir la limite de mémoire: {e}"
                            )

                # Limites après configuration
                current_limits["file_limit_soft_new"] = resource.getrlimit(
                    resource.RLIMIT_NOFILE
                )[0]
                current_limits["file_limit_hard_new"] = resource.getrlimit(
                    resource.RLIMIT_NOFILE
                )[1]

            except ImportError:
                logger.warning("Module 'resource' non disponible")
        else:
            logger.info("Configuration des limites système non disponible sur Windows")

        self.applied_optimizations["system_limits"] = current_limits
        return current_limits

    def optimize_disk_io(self) -> Dict[str, Any]:
        """
        Optimise les opérations d'entrée/sortie disque.

        Returns:
            Configuration des optimisations disque
        """
        disk_config = {}

        # Déplacer le répertoire temporaire vers RAM si configuré et possible
        if (
            self.config["disk_optimization"]["tmp_in_ram"]
            and self.system_info["os"] != "Windows"
        ):
            temp_dir = None

            # Sur Linux, utiliser /dev/shm si disponible (shared memory)
            if self.system_info["os"] == "Linux" and os.path.exists("/dev/shm"):
                project_tmp_dir = "/dev/shm/ai_trading_tmp"
                try:
                    os.makedirs(project_tmp_dir, exist_ok=True)
                    temp_dir = project_tmp_dir
                except:
                    pass

            # Si on n'a pas pu utiliser /dev/shm, utiliser le tmpfs standard
            if not temp_dir:
                temp_dir = tempfile.mkdtemp(prefix="ai_trading_")

            # Configurer le répertoire temporaire
            os.environ["TMPDIR"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

            disk_config["temp_dir"] = temp_dir
            logger.info(f"Répertoire temporaire configuré dans la RAM: {temp_dir}")

        # Identifier les disques SSD vs HDD
        disk_config["storage_info"] = {}
        if self.system_info["os"] == "Linux":
            try:
                # Sur Linux, on peut identifier les SSD via /sys/block
                for disk in os.listdir("/sys/block"):
                    # Vérifier si c'est un SSD
                    try:
                        with open(f"/sys/block/{disk}/queue/rotational", "r") as f:
                            is_rotational = f.read().strip() == "1"
                            disk_config["storage_info"][disk] = {
                                "is_ssd": not is_rotational,
                                "recommended_for_cache": not is_rotational,
                            }
                    except:
                        pass
            except:
                pass
        else:
            # Sur les autres OS, on ne peut pas facilement détecter SSD vs HDD
            # sans outils tiers supplémentaires
            logger.info("Détection SSD vs HDD non disponible sur ce système")

        self.applied_optimizations["disk_optimization"] = disk_config
        return disk_config

    def configure_memory_params(self) -> Dict[str, Any]:
        """
        Configure les paramètres de mémoire pour une utilisation optimale.

        Returns:
            Configuration des paramètres mémoire
        """
        memory_config = {}

        # Sur Linux, on peut configurer swappiness et cache_pressure si on a les droits admin
        if self.system_info["os"] == "Linux" and self.system_info["is_admin"]:
            try:
                # Configurer swappiness (préférence pour utiliser la RAM plutôt que le swap)
                swappiness = self.config["memory_optimization"]["swappiness"]
                try:
                    with open("/proc/sys/vm/swappiness", "w") as f:
                        f.write(str(swappiness))
                    memory_config["swappiness"] = swappiness
                    logger.info(f"Swappiness configuré à {swappiness}")
                except:
                    logger.warning("Impossible de configurer swappiness")

                # Configurer cache_pressure (équilibre entre cache et mémoire applications)
                cache_pressure = self.config["memory_optimization"]["cache_pressure"]
                try:
                    with open("/proc/sys/vm/vfs_cache_pressure", "w") as f:
                        f.write(str(cache_pressure))
                    memory_config["cache_pressure"] = cache_pressure
                    logger.info(f"Cache pressure configuré à {cache_pressure}")
                except:
                    logger.warning("Impossible de configurer cache_pressure")
            except:
                logger.warning("Erreur lors de la configuration des paramètres mémoire")
        else:
            logger.info(
                "Configuration des paramètres mémoire non disponible sur ce système"
            )

        self.applied_optimizations["memory_optimization"] = memory_config
        return memory_config

    def setup_logging(self) -> logging.Logger:
        """
        Configure un système de logging efficace avec rotation des fichiers.

        Returns:
            Logger configuré
        """
        log_config = self.config["logging"]
        log_level = getattr(logging, log_config["level"], logging.INFO)

        # Créer un logger pour l'application
        app_logger = logging.getLogger("ai_trading")
        app_logger.setLevel(log_level)

        # Supprimer les handlers existants
        for handler in app_logger.handlers[:]:
            app_logger.removeHandler(handler)

        # Ajouter un handler pour la console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Définir le format de log
        if log_config["use_json_format"]:
            # Format JSON pour intégration avec systèmes de log centralisés
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                    }
                    # Ajouter les exceptions si présentes
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)

            formatter = JsonFormatter()
        else:
            # Format texte standard
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )

        console_handler.setFormatter(formatter)
        app_logger.addHandler(console_handler)

        # Ajouter un handler pour les fichiers avec rotation
        try:
            log_dir = Path("ai_trading/info_retour/logs")
            log_dir.mkdir(exist_ok=True, parents=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "ai_trading.log",
                maxBytes=log_config["max_file_size_mb"] * 1024 * 1024,
                backupCount=log_config["rotation_count"],
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)

            logger.info(
                f"Système de logging configuré avec rotation, niveau: {log_config['level']}"
            )
        except Exception as e:
            logger.warning(
                f"Impossible de configurer le logger avec rotation de fichiers: {e}"
            )

        self.applied_optimizations["logging"] = {
            "level": log_config["level"],
            "file_rotation": True,
            "json_format": log_config["use_json_format"],
        }

        return app_logger

    def optimize_all(self) -> Dict[str, Any]:
        """
        Applique toutes les optimisations système disponibles.

        Returns:
            Dictionnaire des optimisations appliquées
        """
        # Appliquer les optimisations dans l'ordre logique
        self.optimize_environment_variables()
        self.configure_system_limits()
        self.optimize_disk_io()
        self.configure_memory_params()
        self.setup_logging()

        logger.info("Toutes les optimisations système ont été appliquées")
        return self.applied_optimizations

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel des optimisations système.

        Returns:
            État des optimisations
        """
        status = {
            "system_info": self.system_info,
            "optimizations": self.applied_optimizations,
            "config": self.config,
        }
        return status


def optimize_system(config_file: Optional[str] = None) -> SystemOptimizer:
    """
    Fonction utilitaire pour optimiser le système en une seule étape.

    Args:
        config_file: Chemin optionnel vers un fichier de configuration

    Returns:
        Instance de SystemOptimizer avec les optimisations appliquées
    """
    optimizer = SystemOptimizer(config_file)
    optimizer.optimize_all()
    return optimizer
