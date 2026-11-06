"""Avalia um recorte FIXO de detector nas 50 radiografias da bexiga.

Não modifica arquivos DICOM nem projeções de reconstrução.
Execute a partir de qualquer diretório:
  python dados/bexiga_urinaria_01/processamento/avaliar_recorte_bexiga.py

Opcional: --x0 140 --x1 540 --y0 130 --y1 730
As coordenadas são da imagem REDUZIDA por fator 4 (largura 676, altura 851).
"""
from pathlib import Path
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pydicom

BASE = Path(__file__).resolve().parents[1]
DICOM = BASE / 'configuracao' / 'dicom'
OUT = BASE / 'relatorios' / 'avaliacao_recorte'
FATOR = 4

def chave_natural(p):
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else 10**9

def ler_reduzida(path):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    a = a[::FATOR, ::FATOR]
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x0', type=int, default=140)
    ap.add_argument('--x1', type=int, default=540)
    ap.add_argument('--y0', type=int, default=130)
    ap.add_argument('--y1', type=int, default=730)
    args = ap.parse_args()
    arquivos = sorted(DICOM.glob('*.dcm'), key=chave_natural)
    if len(arquivos) != 50:
        raise RuntimeError(f'Esperados 50 DICOM em {DICOM}; encontrados {len(arquivos)}')
    exemplo = ler_reduzida(arquivos[0])
    h, w = exemplo.shape
    if not (0 <= args.x0 < args.x1 <= w and 0 <= args.y0 < args.y1 <= h):
        raise ValueError(f'Recorte fora da imagem reduzida ({w} x {h})')
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(10, 5, figsize=(16, 27))
    for i, (ax, arq) in enumerate(zip(axes.flat, arquivos)):
        a = ler_reduzida(arq)
        ax.imshow(a, cmap='gray', vmin=0, vmax=1, origin='upper')
        ax.add_patch(Rectangle((args.x0, args.y0), args.x1-args.x0,
                               args.y1-args.y0, fill=False, ec='red', lw=1.3))
        ax.set_title(f'{arq.name} — {i * 7.2:.1f}°', fontsize=9)
        ax.axis('off')
    fig.suptitle('Bexiga — recorte FIXO proposto (vermelho); verificar as 50 projeções', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    painel = OUT / 'painel_recorte_50_projecoes.png'
    fig.savefig(painel, dpi=150)
    plt.close(fig)

    indices = [0, 12, 25, 37, 49]
    fig, axs = plt.subplots(1, len(indices), figsize=(15, 5))
    for ax, idx in zip(axs, indices):
        a = ler_reduzida(arquivos[idx])
        ax.imshow(a[args.y0:args.y1, args.x0:args.x1], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{arquivos[idx].name} — {idx*7.2:.1f}°')
        ax.axis('off')
    fig.suptitle('Prévia do conteúdo do recorte (somente visualização)')
    fig.tight_layout()
    previa = OUT / 'previa_recorte_5_projecoes.png'
    fig.savefig(previa, dpi=150)
    plt.close(fig)

    print('Detector reduzido (altura, largura):', (h, w))
    print('Limites propostos x:', args.x0, args.x1, '| y:', args.y0, args.y1)
    print('Dimensão do recorte (altura, largura):', (args.y1-args.y0, args.x1-args.x0))
    print('Centro nominal detector (x, y):', ((w-1)/2, (h-1)/2))
    print('Centro do recorte em coordenadas originais (x, y):',
          ((args.x0+args.x1-1)/2, (args.y0+args.y1-1)/2))
    print('Painel:', painel)
    print('Prévia:', previa)
    print('NENHUM DICOM OU ARQUIVO DE RECONSTRUÇÃO FOI MODIFICADO.')
    print('ATENÇÃO: o recorte é apenas uma hipótese inicial; revise as 50 imagens.')

if __name__ == '__main__':
    main()
