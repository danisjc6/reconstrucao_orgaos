
from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PASTA_DICOM = BASE / "configuracao" / "dicom"
PASTA_SAIDA = BASE / "relatorios"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

numeros = [13,38]

fig, axes = plt.subplots(1, 2, figsize=(12, 14))

for ax, numero in zip(axes.flat, numeros):
    arquivo = PASTA_DICOM / f"{numero}.dcm"

    ds = pydicom.dcmread(arquivo)
    imagem = ds.pixel_array.astype(np.float32)

    # Ajuste de contraste apenas para visualização
    p1, p99 = np.percentile(imagem, [1, 99])

    ax.imshow(
        imagem,
        cmap="gray",
        vmin=p1,
        vmax=p99
    )

    angulo = (numero - 1) * 7.2

    ax.set_title(
        f"{numero}.dcm — {angulo:.1f}°",
        fontsize=12
    )
    ax.axis("off")

plt.suptitle(
    "Bexiga urinária — projeções radiográficas",
    fontsize=16
)

plt.tight_layout()

saida = PASTA_SAIDA / "projecoes_bexiga.png"
plt.savefig(saida, dpi=150, bbox_inches="tight")
plt.show()

print(f"Figura salva em: {saida}")
