"""Inspecao de CANDIDATOS a metal nas radiografias da bexiga.

Nao altera DICOM nem gera projecoes corrigidas. As linhas detectadas podem
incluir bordas do PLA, detector e outros objetos: revisar antes de mascarar.
"""
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DICOM = BASE / 'configuracao' / 'dicom'
CSV = BASE / 'relatorios' / 'candidatos_hibridos_moedas.csv'
OUT = BASE / 'relatorios' / 'inspecao_metais'
OUT.mkdir(parents=True, exist_ok=True)
FATOR = 4
AMOSTRAS = [1, 7, 13, 19, 26, 32, 38, 44, 50]


def numero(path):
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else 10**9


def ler_imagem(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    h, w = arr.shape
    arr = cv2.resize(arr, (w // FATOR, h // FATOR), interpolation=cv2.INTER_AREA)
    p1, p99 = np.percentile(arr, [1, 99.5])
    norm = np.clip((arr - p1) / max(p99 - p1, 1), 0, 1)
    return (norm * 255).astype(np.uint8)


def linhas_candidatas(img):
    # Procura apenas segmentos longos e quase verticais. NAO classifica material.
    edges = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 35, 100)
    h, w = img.shape
    linhas = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=65,
                             minLineLength=int(h * 0.28), maxLineGap=28)
    saida = []
    if linhas is not None:
        for linha in linhas.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, linha)
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dy >= int(h * 0.28) and dx <= max(5, int(0.09 * dy)):
                saida.append((int(x1), int(y1), int(x2), int(y2)))
    return saida


def main():
    arquivos = sorted(DICOM.glob('*.dcm'), key=numero)
    if len(arquivos) != 50:
        raise RuntimeError(f'Esperados 50 DICOM em {DICOM}; encontrados {len(arquivos)}')
    moedas = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if moedas.empty:
        print('AVISO: CSV de candidatos a moedas ausente. Exibindo somente linhas.')
    registros = []
    fig, axs = plt.subplots(3, 3, figsize=(15, 16))
    for ax, n in zip(axs.flat, AMOSTRAS):
        path = arquivos[n - 1]
        img = ler_imagem(path)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        linhas = linhas_candidatas(img)
        for x1, y1, x2, y2 in linhas:
            ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=1.4)
            registros.append({'arquivo': path.name, 'tipo': 'linha_candidata',
                              'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        if not moedas.empty:
            selecao = moedas[moedas['arquivo'].astype(str).str.match(rf'^0*{n}\.dcm$', case=False)]
            for _, r in selecao.iterrows():
                # Apenas candidatos circulares; deteccoes podem conter falsos positivos.
                if str(r.get('metodo', '')) != 'circulo':
                    continue
                x, y = float(r['x_reduzido_px']), float(r['y_reduzido_px'])
                ax.add_patch(plt.Circle((x, y), 35, fill=False,
                                        edgecolor='yellow', linewidth=1.5))
                ax.plot(x, y, 'y+', markersize=7)
        ax.set_title(f'{path.name}: {len(linhas)} segmentos verticais candidatos', fontsize=10)
        ax.axis('off')
    fig.suptitle('Inspecao preliminar: ciano = linhas; amarelo = moedas candidatas\n'
                 'NAO SAO MASCARAS VALIDADAS', fontsize=13)
    fig.tight_layout()
    destino = OUT / 'painel_candidatos_metais.png'
    fig.savefig(destino, dpi=160)
    pd.DataFrame(registros, columns=['arquivo', 'tipo', 'x1', 'y1', 'x2', 'y2']).to_csv(
        OUT / 'linhas_candidatas_amostras.csv', index=False)
    print('Painel salvo:', destino)
    print('Nenhuma imagem DICOM foi alterada.')
    plt.show()


if __name__ == '__main__':
    main()
