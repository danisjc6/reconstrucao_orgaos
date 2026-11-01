
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DICOM = BASE / "configuracao" / "dicom"
SAIDA = BASE / "relatorios"
SAIDA.mkdir(parents=True, exist_ok=True)

FATOR = 4
NUM_PROJECOES = 50

# Faixas verticais iniciais, expressas como fração da altura.
# Devem ser conferidas nas imagens.
ROIS = {
    "cranial": (0.08, 0.38),
    "caudal": (0.55, 0.95),
}


def carregar(numero):
    ds = pydicom.dcmread(DICOM / f"{numero}.dcm")
    img = ds.pixel_array.astype(np.float32)

    h, w = img.shape
    img = cv2.resize(
        img,
        (w // FATOR, h // FATOR),
        interpolation=cv2.INTER_AREA
    )

    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)

    return (img * 255).astype(np.uint8)


def detectar_circulos(img):
    suavizada = cv2.GaussianBlur(img, (9, 9), 2)

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

    h, w = img.shape
    candidatos = []

    for x, y, r in np.round(circulos[0]).astype(int):
        if not (0.08 * h <= y <= 0.95 * h):
            continue

        candidatos.append({
            "x": int(x),
            "y": int(y),
            "tamanho": int(r),
            "metodo": "circulo",
        })

    return candidatos


def detectar_segmentos(img, faixa):
    h, w = img.shape
    y0 = int(faixa[0] * h)
    y1 = int(faixa[1] * h)

    roi = img[y0:y1, :]

    suavizada = cv2.GaussianBlur(roi, (5, 5), 1)

    # Detectar regiões de intensidade elevada
    limite = np.percentile(suavizada, 98)
    mascara = (suavizada >= limite).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 7)
    )
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    for contorno in contornos:
        x, y, largura, altura = cv2.boundingRect(contorno)

        if altura < 10 or altura > 90:
            continue

        if largura < 2 or largura > 30:
            continue

        proporcao = altura / max(largura, 1)

        if proporcao < 1.5:
            continue

        candidatos.append({
            "x": int(x + largura / 2),
            "y": int(y0 + y + altura / 2),
            "tamanho": int(max(largura, altura)),
            "metodo": "segmento",
        })

    return candidatos


def pertence_roi(candidato, faixa, altura):
    return faixa[0] * altura <= candidato["y"] <= faixa[1] * altura


registros = []
painel = []

for numero in range(1, NUM_PROJECOES + 1):
    img = carregar(numero)
    h, w = img.shape

    circulos = detectar_circulos(img)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for marcador, faixa in ROIS.items():
        candidatos = [
            c for c in circulos
            if pertence_roi(c, faixa, h)
        ]

        candidatos += detectar_segmentos(img, faixa)

        # Não escolher automaticamente um candidato.
        # Guardar todos para conferência.
        for indice, c in enumerate(candidatos, start=1):
            registros.append({
                "arquivo": f"{numero}.dcm",
                "angulo_graus": (numero - 1) * 7.2,
                "marcador_provavel": marcador,
                "candidato": indice,
                "metodo": c["metodo"],
                "x_reduzido_px": c["x"],
                "y_reduzido_px": c["y"],
                "x_original_aprox_px": c["x"] * FATOR,
                "y_original_aprox_px": c["y"] * FATOR,
            })

            cor = (
                (255, 0, 0)
                if marcador == "cranial"
                else (0, 255, 0)
            )

            cv2.circle(
                vis,
                (c["x"], c["y"]),
                5,
                cor,
                2
            )

    painel.append(vis)

    print(
        f"{numero:02d}.dcm — "
        f"{len([r for r in registros if r['arquivo'] == f'{numero}.dcm'])} candidatos"
    )


df = pd.DataFrame(registros)
for numero in [13, 38]:
    arquivo = f"{numero}.dcm"

    print(f"\nCandidatos de {arquivo}:")

    colunas = [
        "marcador_provavel",
        "metodo",
        "x_reduzido_px",
        "y_reduzido_px"
    ]

    print(
        df.loc[df["arquivo"] == arquivo, colunas]
        .to_string(index=False)
    )

fig, axes = plt.subplots(10, 5, figsize=(20, 35))

for i, ax in enumerate(axes.flat):
    ax.imshow(painel[i])
    ax.set_title(f"{i + 1}.dcm", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    SAIDA / "painel_deteccao_hibrida.png",
    dpi=130
)

plt.show()

print("\nArquivos gerados:")
print(SAIDA / "candidatos_hibridos_moedas.csv")
print(SAIDA / "painel_deteccao_hibrida.png")
