from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


VOLUME_PATH = Path("resultados/volume_coracao.npy")


def normalizar(volume):
    minimo = np.percentile(volume, 1)
    maximo = np.percentile(volume, 99)

    volume = np.clip(volume, minimo, maximo)
    return (volume - minimo) / (maximo - minimo + 1e-6)


volume = np.load(VOLUME_PATH)
volume = normalizar(volume)

z_inicial = volume.shape[0] // 2
y_inicial = volume.shape[1] // 2
x_inicial = volume.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(bottom=0.27)

imagem_axial = axes[0].imshow(
    volume[z_inicial, :, :],
    cmap="gray",
    origin="lower"
)

imagem_coronal = axes[1].imshow(
    volume[:, y_inicial, :],
    cmap="gray",
    origin="lower"
)

imagem_sagital = axes[2].imshow(
    volume[:, :, x_inicial],
    cmap="gray",
    origin="lower"
)

axes[0].set_title(f"Axial — corte {z_inicial}")
axes[1].set_title(f"Coronal — corte {y_inicial}")
axes[2].set_title(f"Sagital — corte {x_inicial}")

for axis in axes:
    axis.axis("off")

eixo_slider_z = plt.axes([0.20, 0.16, 0.60, 0.025])
eixo_slider_y = plt.axes([0.20, 0.11, 0.60, 0.025])
eixo_slider_x = plt.axes([0.20, 0.06, 0.60, 0.025])

slider_z = Slider(
    eixo_slider_z,
    "Axial",
    0,
    volume.shape[0] - 1,
    valinit=z_inicial,
    valstep=1
)

slider_y = Slider(
    eixo_slider_y,
    "Coronal",
    0,
    volume.shape[1] - 1,
    valinit=y_inicial,
    valstep=1
)

slider_x = Slider(
    eixo_slider_x,
    "Sagital",
    0,
    volume.shape[2] - 1,
    valinit=x_inicial,
    valstep=1
)


def atualizar_axial(valor):
    indice = int(slider_z.val)
    imagem_axial.set_data(volume[indice, :, :])
    axes[0].set_title(f"Axial — corte {indice}")
    fig.canvas.draw_idle()


def atualizar_coronal(valor):
    indice = int(slider_y.val)
    imagem_coronal.set_data(volume[:, indice, :])
    axes[1].set_title(f"Coronal — corte {indice}")
    fig.canvas.draw_idle()


def atualizar_sagital(valor):
    indice = int(slider_x.val)
    imagem_sagital.set_data(volume[:, :, indice])
    axes[2].set_title(f"Sagital — corte {indice}")
    fig.canvas.draw_idle()


slider_z.on_changed(atualizar_axial)
slider_y.on_changed(atualizar_coronal)
slider_x.on_changed(atualizar_sagital)

plt.show()