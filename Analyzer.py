import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


def get_dummy_plots():
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(arr, bins=20)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf
