import json
from pathlib import Path

# Configuration DeepSpeed
config = {
    "train_batch_size": "auto",
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": 1000,
        },
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True,
    "fp16": {
        "enabled": True,
        "auto_cast": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
    },
}


def create_deepspeed_config(output_path=None, custom_params=None):
    """
    Crée un fichier de configuration DeepSpeed.

    Args:
        output_path: Chemin de sortie pour le fichier de configuration
        custom_params: Paramètres personnalisés pour écraser les valeurs par défaut

    Returns:
        Path: Chemin du fichier de configuration créé
    """
    # Configuration de base
    ds_config = config.copy()

    # Appliquer les paramètres personnalisés si fournis
    if custom_params:
        for key, value in custom_params.items():
            if (
                isinstance(value, dict)
                and key in ds_config
                and isinstance(ds_config[key], dict)
            ):
                ds_config[key].update(value)
            else:
                ds_config[key] = value

    # Chemin de sortie par défaut
    if not output_path:
        # Utiliser le chemin racine du projet pour info_retour
        config_dir = Path("info_retour/config/deepspeed")
        config_dir.mkdir(parents=True, exist_ok=True)
        output_path = config_dir / "ds_config_default.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Écrire le fichier de configuration
    with open(output_path, "w") as f:
        json.dump(ds_config, f, indent=4)

    print(f"Configuration DeepSpeed créée avec succès : {output_path}")
    return output_path


if __name__ == "__main__":
    # Créer la configuration par défaut
    create_deepspeed_config()
