
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PASTA_DICOM = BASE / "configuracao" / "dicom"
PASTA_SAIDA = BASE / "relatorios"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

PROJECOES = list(range(1, 51))
FATOR_REDUCAO = 4


def carregar_imagem(numero):
    arquivo = PASTA_DICOM / f"{numero}.dcm"
    ds = pydicom.dcmread(arquivo)

    imagem = ds.pixel_array.astype(np.float32)

    # Redução somente para detecção
    altura, largura = imagem.shape
    imagem = cv2.resize(
        imagem,
        (largura // FATOR_REDUCAO,
         altura // FATOR_REDUCAO),
        interpolation=cv2.INTER_AREA
    )

    p1, p99 = np.percentile(imagem, [1, 99])

    imagem = np.clip(
        (imagem - p1) / max(p99 - p1, 1e-6),
        0,
        1
    )

    imagem8 = (imagem * 255).astype(np.uint8)
    return imagem8


def detectar_moedas(imagem):
    suavizada = cv2.GaussianBlur(
        imagem, (9, 9), 2
    )

    circulos = cv2.HoughCircles(
        suavizada,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=100,
        param2=25,
        minRadius=8,
        maxRadius=45
    )

    if circulos is None:
        return []

    circulos = np.round(circulos[0]).astype(int)

    altura, largura = imagem.shape

    margem_superior = int(altura * 0.08)
    margem_inferior = int(altura * 0.05)

    candidatos = []

    for x, y, r in circulos:
        if y < margem_superior:
            continue

        if y > altura - margem_inferior:
            continue

        candidatos.append((int(x), int(y), int(r)))

    return candidatos



resultados = []

fig, axes = plt.subplots(10, 5, figsize=(20, 35))
axes = axes.flatten()

for ax, numero in zip(axes, PROJECOES):
    imagem = carregar_imagem(numero)
    circulos = detectar_moedas(imagem)

    visualizacao = cv2.cvtColor(
        imagem, cv2.COLOR_GRAY2RGB
    )

    for indice, (x, y, r) in enumerate(circulos, start=1):
        cv2.circle(
            visualizacao,
            (x, y),
            r,
            (255, 0, 0),
            2
        )

        cv2.circle(
            visualizacao,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )

        resultados.append({
            "arquivo": f"{numero}.dcm",
            "angulo_graus": (numero - 1) * 7.2,
            "candidato": indice,
            "x_reduzido_px": x,
            "y_reduzido_px": y,
            "raio_reduzido_px": r,
            "x_original_aprox_px": x * FATOR_REDUCAO,
            "y_original_aprox_px": y * FATOR_REDUCAO
        })

    ax.imshow(visualizacao)
    ax.set_title(
        f"{numero}.dcm — {len(circulos)} candidatos"
    )
    ax.axis("off")

plt.tight_layout()

figura = PASTA_SAIDA / "deteccao_moedas_inicial.png"
plt.savefig(figura, dpi=150, bbox_inches="tight")

tabela = pd.DataFrame(resultados)
tabela.to_csv(
    PASTA_SAIDA / "candidatos_moedas_inicial.csv",
    index=False
)

print("Candidatos detectados:", len(resultados))
print("Figura salva:", figura)

print("\nResumo por projeção:")

for numero in PROJECOES:
    quantidade = sum(
        r["arquivo"] == f"{numero}.dcm"
        for r in resultados
    )

    print(
        f"{numero:02d}.dcm | "
        f"{(numero - 1) * 7.2:6.1f}° | "
        f"{quantidade} candidatos"
    )

plt.show()
