import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # Exemple de génération de graphique
    plt.figure(figsize=(12,6))
    plt.plot([0,1,2,3,4], [10,20,15,25,30])
    plt.title('Exemple de performance')
    plt.savefig(f"{args.output_dir}/performance_example.png")

if __name__ == "__main__":
    main() 