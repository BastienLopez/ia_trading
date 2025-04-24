import os

import matplotlib.pyplot as plt

from ai_trading.config import VISUALIZATION_DIR


def save_visualization(self, filename):
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    return output_path
