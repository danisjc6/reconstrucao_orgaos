
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

PASTA = (
    BASE / "reconstrucoes"
    / "tigre" / "teste_fdk_nominal"
)

volume = np.load(
    PASTA / "volume_fdk_exploratorio.npy"
)

print("Dimensões do volume:", volume.shape)
print("Valores finitos:", np.isfinite(volume).all())

# Percentis apenas dos valores finitos
validos = volume[np.isfinite(volume)]

vmin, vmax = np.percentile(validos, [2, 99.5])

# Cortes distribuídos ao longo do eixo Z
indices = np.linspace(
    20,
    volume.shape[0] - 21,
    16,
    dtype=int
)

fig, axes = plt.subplots(
    4, 4,
    figsize=(12, 12)
)

for ax, indice in zip(axes.flat, indices):
    ax.imshow(
        volume[indice, :, :],
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"Corte axial {indice}")
    ax.axis("off")

plt.suptitle(
    "Bexiga urinária — cortes axiais FDK",
    fontsize=14
)

plt.tight_layout()

saida = PASTA / "multiplos_cortes_axiais_fdk.png"

plt.savefig(saida, dpi=180)
plt.show()

print("Figura salva em:", saida)
