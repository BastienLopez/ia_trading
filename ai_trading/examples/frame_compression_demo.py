import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.frame_compression import FrameCompressor, FrameStackWrapper


def generate_sample_frames(
    n_frames: int, frame_size: Tuple[int, int, int] = (84, 84, 3)
) -> List[np.ndarray]:
    """
    Génère des frames de test avec des valeurs aléatoires.

    Args:
        n_frames: Nombre de frames à générer
        frame_size: Taille des frames (height, width, channels)

    Returns:
        Liste de frames générées
    """
    frames = []
    for i in range(n_frames):
        # Générer une frame avec des valeurs aléatoires
        frame = np.random.randint(0, 255, frame_size, dtype=np.uint8)

        # Ajouter un motif pour que ce soit plus visuel
        h, w = frame_size[0], frame_size[1]
        # Ajouter un cercle qui se déplace
        center_x = int(w / 2 + (w / 4) * np.sin(i * 0.2))
        center_y = int(h / 2 + (h / 4) * np.cos(i * 0.2))
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < 10:
                    frame[y, x] = [255, 255, 255]  # Point blanc

        frames.append(frame)

    return frames


def measure_memory(obj: any) -> float:
    """
    Mesure la consommation mémoire d'un objet en Mo.

    Args:
        obj: Objet à mesurer

    Returns:
        Taille en mégaoctets
    """
    size_bytes = sys.getsizeof(obj)

    # Si c'est un tableau numpy, utiliser nbytes
    if isinstance(obj, np.ndarray):
        size_bytes = obj.nbytes

    # Si c'est une liste, récursivement mesurer ses éléments
    elif isinstance(obj, list) and len(obj) > 0:
        size_bytes = sum(measure_memory(item) * 1024 * 1024 for item in obj)

    return size_bytes / (1024 * 1024)  # Convertir en MB


def test_compression_performance(
    frames: List[np.ndarray], compressor: FrameCompressor
) -> Tuple[float, float, float]:
    """
    Teste les performances du compresseur.

    Args:
        frames: Liste de frames à compresser
        compressor: Compresseur à utiliser

    Returns:
        (taille_originale, taille_compressée, temps_traitement)
    """
    orig_size = measure_memory(frames)

    start_time = time.time()
    compressed_data = []

    # Réinitialiser le compresseur
    compressor.reset()

    # Compresser chaque frame
    for frame in frames:
        compressed, metadata = compressor.compress_state(frame)
        compressed_data.append((compressed, metadata))

    proc_time = time.time() - start_time
    comp_size = measure_memory(compressed_data)

    return orig_size, comp_size, proc_time


def test_frame_stack_performance(
    frames: List[np.ndarray], wrapper: FrameStackWrapper
) -> Tuple[float, List[np.ndarray]]:
    """
    Teste les performances du wrapper d'empilement.

    Args:
        frames: Liste de frames à empiler
        wrapper: Wrapper à utiliser

    Returns:
        (temps_traitement, frames_empilées)
    """
    stacked_frames = []

    start_time = time.time()

    # Réinitialiser avec la première frame
    stacked = wrapper.reset(observation=frames[0])
    stacked_frames.append(stacked)

    # Empiler les frames suivantes
    for frame in frames[1:]:
        stacked = wrapper.add_frame(frame)
        stacked_frames.append(stacked)

    proc_time = time.time() - start_time

    return proc_time, stacked_frames


def plot_compression_results(
    sizes: List[float],
    times: List[float],
    labels: List[str],
    title: str = "Compression Performance",
):
    """
    Affiche les résultats de compression sous forme de graphique.

    Args:
        sizes: Liste des tailles
        times: Liste des temps de traitement
        labels: Noms des méthodes
        title: Titre du graphique
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Graphique des tailles
    ax1.bar(labels, sizes)
    ax1.set_title("Taille en mémoire")
    ax1.set_ylabel("Mo")
    ax1.set_ylim(bottom=0)

    # Annotate with percentages
    for i, value in enumerate(sizes):
        if i > 0:
            reduction = (1 - value / sizes[0]) * 100
            ax1.text(
                i,
                value + 0.1,
                f"{reduction:.1f}% de réduction",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Graphique des temps
    ax2.bar(labels, times)
    ax2.set_title("Temps de traitement")
    ax2.set_ylabel("Secondes")
    ax2.set_ylim(bottom=0)

    # Annotate processing times
    for i, value in enumerate(times):
        ax2.text(i, value + 0.01, f"{value:.4f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("compression_results.png")
    plt.show()


def plot_sample_frames(
    orig_frames: List[np.ndarray], comp_frames: List[np.ndarray], n_samples: int = 3
):
    """
    Affiche des exemples d'images avant/après compression.

    Args:
        orig_frames: Frames originales
        comp_frames: Frames compressées
        n_samples: Nombre d'exemples à afficher
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    # Sélectionner des indices d'échantillons
    indices = np.linspace(
        0, min(len(orig_frames), len(comp_frames)) - 1, n_samples, dtype=int
    )

    for i, idx in enumerate(indices):
        # Afficher l'original
        axes[0, i].imshow(orig_frames[idx])
        axes[0, i].set_title(f"Original {idx}")
        axes[0, i].axis("off")

        # Afficher la version compressée
        if comp_frames[idx].shape != orig_frames[idx].shape:
            # Si les formes sont différentes (ex: grayscale vs RGB)
            # Adapter l'affichage
            if len(comp_frames[idx].shape) == 2:
                axes[1, i].imshow(comp_frames[idx], cmap="gray")
            else:
                axes[1, i].imshow(comp_frames[idx])
        else:
            axes[1, i].imshow(comp_frames[idx])

        axes[1, i].set_title(f"Compressé {idx}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("frame_samples.png")
    plt.show()


def main(args):
    """
    Fonction principale pour la démonstration.
    """
    print(f"Génération de {args.n_frames} frames de test...")
    frames = generate_sample_frames(
        args.n_frames, (args.frame_size, args.frame_size, 3)
    )

    # Mesurer la mémoire de base
    orig_size = measure_memory(frames)
    print(f"Taille originale: {orig_size:.2f} Mo pour {len(frames)} frames")

    # Configuration de base
    print("\n1. Test avec configuration de base...")
    basic_compressor = FrameCompressor()
    basic_size, basic_comp_size, basic_time = test_compression_performance(
        frames, basic_compressor
    )
    print(f"Taille de base: {basic_size:.2f} Mo")
    print(f"Taille compressée: {basic_comp_size:.2f} Mo")
    print(f"Réduction: {(1 - basic_comp_size / basic_size) * 100:.2f}%")
    print(f"Temps: {basic_time:.4f} secondes")

    # Configuration optimisée
    print("\n2. Test avec configuration optimisée...")
    optimized_compressor = FrameCompressor(
        compression_level=args.compression_level,
        frame_stack_size=args.stack_size,
        resize_dim=(args.frame_size // 2, args.frame_size // 2),
        use_grayscale=args.grayscale,
        quantize=args.quantize,
        use_delta_encoding=args.delta,
    )
    opti_size, opti_comp_size, opti_time = test_compression_performance(
        frames, optimized_compressor
    )
    print(f"Taille de base: {opti_size:.2f} Mo")
    print(f"Taille compressée: {opti_comp_size:.2f} Mo")
    print(f"Réduction: {(1 - opti_comp_size / opti_size) * 100:.2f}%")
    print(f"Temps: {opti_time:.4f} secondes")

    # Test du FrameStackWrapper
    print("\n3. Test du FrameStackWrapper...")
    basic_wrapper = FrameStackWrapper(n_frames=args.stack_size)
    basic_wrapper_time, basic_stacked = test_frame_stack_performance(
        frames, basic_wrapper
    )

    compress_wrapper = FrameStackWrapper(
        n_frames=args.stack_size,
        compress=True,
        compression_level=args.compression_level,
    )
    compress_wrapper_time, compress_stacked = test_frame_stack_performance(
        frames, compress_wrapper
    )

    print(f"Temps d'empilement de base: {basic_wrapper_time:.4f} secondes")
    print(f"Temps d'empilement avec compression: {compress_wrapper_time:.4f} secondes")

    # Afficher les résultats
    print("\nAffichage des résultats...")
    sizes = [basic_size, basic_comp_size, opti_comp_size]
    times = [0, basic_time, opti_time]
    labels = ["Original", "Compression de base", "Compression optimisée"]

    plot_compression_results(
        sizes, times, labels, "Performance de la compression de frames"
    )

    # Récupérer quelques frames décompressées pour l'affichage
    print("Décompression de quelques frames pour comparaison...")
    decompressed_frames = []

    # Corriger l'erreur de déballage
    compressed_data, metadata = optimized_compressor.compress_state(frames[0])
    decompressed = optimized_compressor.decompress_state(compressed_data, metadata)
    decompressed_frames.append(decompressed)

    # Afficher quelques exemples
    print("Affichage des exemples...")
    plot_sample_frames(frames, [decompressed])

    print("Démonstration terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Démonstration de la compression de frames RL"
    )
    parser.add_argument(
        "--n_frames", type=int, default=1000, help="Nombre de frames à générer"
    )
    parser.add_argument(
        "--frame_size", type=int, default=84, help="Taille des frames (carré)"
    )
    parser.add_argument(
        "--stack_size", type=int, default=4, help="Nombre de frames à empiler"
    )
    parser.add_argument(
        "--compression_level", type=int, default=6, help="Niveau de compression (0-9)"
    )
    parser.add_argument(
        "--grayscale", action="store_true", help="Convertir en niveaux de gris"
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantifier les valeurs"
    )
    parser.add_argument(
        "--delta", action="store_true", help="Utiliser l'encodage delta"
    )

    args = parser.parse_args()
    main(args)
