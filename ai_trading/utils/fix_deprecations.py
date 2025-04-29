"""
Module pour corriger automatiquement les dépréciations dans le code du projet.
"""

import os
import re
import logging
from pathlib import Path

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fix_torch_set_default_tensor_type(file_path):
    """
    Remplace torch.set_default_tensor_type() par torch.set_default_dtype() et torch.set_default_device().
    
    Args:
        file_path (str): Chemin du fichier à modifier
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Recherche les appels à set_default_tensor_type
    pattern = r'torch\.set_default_tensor_type\((.*?)\)'
    matches = re.findall(pattern, content)
    
    if not matches:
        return False
    
    # Remplacer par la nouvelle syntaxe
    for match in matches:
        if 'torch.FloatTensor' in match:
            content = re.sub(
                r'torch\.set_default_tensor_type\(torch\.FloatTensor\)',
                'torch.set_default_dtype(torch.float32)',
                content
            )
        elif 'torch.cuda.FloatTensor' in match:
            content = re.sub(
                r'torch\.set_default_tensor_type\(torch\.cuda\.FloatTensor\)',
                'torch.set_default_dtype(torch.float32)\ntorch.set_default_device("cuda")',
                content
            )
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    logger.info(f"Corrigé torch.set_default_tensor_type dans {file_path}")
    return True

def fix_distutils_version(file_path):
    """
    Remplace distutils.version.LooseVersion par packaging.version.
    
    Args:
        file_path (str): Chemin du fichier à modifier
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Recherche les imports de distutils.version
    if 'distutils.version' not in content:
        return False
    
    # Remplacer les imports
    content = re.sub(
        r'from\s+distutils\s+import\s+version',
        'from packaging import version',
        content
    )
    content = re.sub(
        r'import\s+distutils\.version',
        'from packaging import version',
        content
    )
    
    # Remplacer les utilisations
    content = re.sub(
        r'distutils\.version\.LooseVersion\((.*?)\)',
        r'version.parse(\1)',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    logger.info(f"Corrigé distutils.version.LooseVersion dans {file_path}")
    return True

def fix_jax_make_jaxpr(jit(file_path)):
    """
    Remplace jax.xla_computation par les API AOT de JAX.
    
    Args:
        file_path (str): Chemin du fichier à modifier
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Recherche les imports de jax.xla_computation
    if 'xla_computation' not in content:
        return False
    
    # Remplacer les imports
    content = re.sub(
        r'from\s+jax\s+import\s+xla_computation',
        'from jax import jit, make_jaxpr',
        content
    )
    
    # Remplacer les utilisations (cette partie est approximative et peut nécessiter des ajustements manuels)
    content = re.sub(
        r'xla_computation\((.*?)\)',
        r'make_jaxpr(jit(\1))',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    logger.info(f"Corrigé jax.xla_computation dans {file_path}")
    return True

def fix_list_tensor_conversion(file_path):
    """
    Améliore la conversion des listes en tenseurs PyTorch.
    
    Args:
        file_path (str): Chemin du fichier à modifier
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Recherche les conversions de liste en tenseur
    pattern = r'torch\.FloatTensor\(\[(.*?)\]\)'
    if not re.search(pattern, content):
        return False
    
    # Remplacer par numpy.array() avant la conversion
    content = re.sub(
        r'torch\.FloatTensor\(\[(.*?)\]\)',
        r'torch.tensor(np.array([\1]), dtype=torch.float32)',
        content
    )
    
    # Assurer que numpy est importé
    if 'import numpy' not in content and 'from numpy import' not in content:
        content = 'import numpy as np\n' + content
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    logger.info(f"Corrigé conversion de liste en tenseur dans {file_path}")
    return True

def fix_all_files(directory="."):
    """
    Applique toutes les corrections à tous les fichiers Python du répertoire.
    
    Args:
        directory (str): Répertoire à traiter récursivement
    """
    fixed_count = 0
    python_files = Path(directory).rglob("*.py")
    
    for file_path in python_files:
        file_path_str = str(file_path)
        logger.debug(f"Traitement de {file_path_str}")
        
        # Appliquer les corrections
        fixed_tensor_type = fix_torch_set_default_tensor_type(file_path_str)
        fixed_distutils = fix_distutils_version(file_path_str)
        fixed_jax = fix_jax_make_jaxpr(jit(file_path_str))
        fixed_list_tensor = fix_list_tensor_conversion(file_path_str)
        
        if fixed_tensor_type or fixed_distutils or fixed_jax or fixed_list_tensor:
            fixed_count += 1
    
    logger.info(f"Corrections appliquées à {fixed_count} fichiers")
    return fixed_count

if __name__ == "__main__":
    # Exécuter les corrections sur le projet
    fix_all_files("ai_trading") 