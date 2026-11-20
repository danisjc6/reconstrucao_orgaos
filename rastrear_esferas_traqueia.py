import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom

DICOM = Path("dados/traqueia_caprina_01/dicom")
SAIDA = Path("resultados/traqueia_caprina_01/alinhamento/esferas.csv")
ESFERAS = ("S1", "S2", "I1")
CAMPOS = ("arquivo", "angulo_graus", "esfera", "x_px", "y_px", "qualidade")

arquivos = [DICOM / f"{i}.dcm" for i in range(1, 51)]
faltantes = [p.name for p in arquivos if not p.is_file()]
if faltantes:
    raise SystemExit("Arquivos ausentes: " + ", ".join(faltantes))

SAIDA.parent.mkdir(parents=True, exist_ok=True)
concluidos = set()
ultimas = {}

if SAIDA.exists() and SAIDA.stat().st_size:
    with SAIDA.open(newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            esfera = linha["esfera"]
            concluidos.add((linha["arquivo"], esfera))
            if linha["qualidade"] == "visivel":
                ultimas[esfera] = (float(linha["x_px"]), float(linha["y_px"]))

with SAIDA.open("a", newline="", encoding="utf-8") as f:
    csv_saida = csv.DictWriter(f, fieldnames=CAMPOS)
    if f.tell() == 0:
        csv_saida.writeheader()

    for i, caminho in enumerate(arquivos):
        pendentes = [e for e in ESFERAS if (caminho.name, e) not in concluidos]
        if not pendentes:
            continue

        imagem = pydicom.dcmread(caminho).pixel_array
        minimo, maximo = np.percentile(imagem, [1, 99.9])
        angulo = round(i * 7.2, 1)
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.imshow(imagem, cmap="gray", origin="upper", vmin=minimo, vmax=maximo)

        for esfera in pendentes:
            ax.set_title(
                f"{caminho.name} · {angulo:g}° · {esfera}\n"
                "Clique no centro | Enter: incerta/ausente | Fechar: pausar"
            )
            guia = None
            if esfera in ultimas:
                x, y = ultimas[esfera]
                guia, = ax.plot(x, y, "+", color="cyan", markersize=12)

            ponto = fig.ginput(1, timeout=0, show_clicks=False)
            if not plt.fignum_exists(fig.number):
                raise SystemExit(f"Pausado. Marcações salvas em {SAIDA}")

            if guia is not None:
                guia.remove()

            if ponto:
                x, y = ponto[0]
                ultimas[esfera] = (x, y)
                qualidade = "visivel"
                ax.plot(x, y, "o", color="lime", markersize=4)
                ax.annotate(esfera, (x, y), color="lime")
            else:
                x, y, qualidade = "", "", "incerta_ou_ausente"

            csv_saida.writerow(dict(zip(
                CAMPOS, (caminho.name, angulo, esfera, x, y, qualidade)
            )))
            f.flush()

        plt.close(fig)

print(f"Rastreamento concluído: {SAIDA}")