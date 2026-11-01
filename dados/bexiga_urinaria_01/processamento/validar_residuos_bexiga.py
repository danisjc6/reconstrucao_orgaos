from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RELATORIOS = BASE / 'relatorios'
DICOM = BASE / 'configuracao' / 'dicom'
ENTRADA = RELATORIOS / 'avaliacao_conebeam_bexiga.csv'
FATOR = 4
TOP_N = 6  # por marcador, entre observações usadas no ajuste


def imagem_reduzida(numero):
    ds = pydicom.dcmread(DICOM / f'{numero}.dcm')
    img = ds.pixel_array.astype(np.float32)[::FATOR, ::FATOR]
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Execute primeiro ajustar_geometria_conebeam_bexiga.py: {ENTRADA}')
    dados = pd.read_csv(ENTRADA)
    obrigatorias = {'marcador', 'numero', 'angulo_graus', 'usado_no_ajuste',
                    'x_observado_px', 'y_observado_px', 'u_conebeam_px',
                    'v_conebeam_px', 'residuo_conebeam_px'}
    faltantes = obrigatorias - set(dados.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(faltantes)}')

    # Não confundir previsões com observações; usar somente pontos efetivamente
    # medidos e selecionados no ajuste anterior.
    selecionados = dados[
        dados['usado_no_ajuste'].astype(str).str.lower().eq('true')
        & dados['x_observado_px'].notna()
        & dados['y_observado_px'].notna()
    ].copy()
    selecionados['revisao'] = 'pendente'
    selecionados['observacao_revisor'] = ''
    selecionados['residuo_detector_mm'] = selecionados['residuo_conebeam_px'] * 0.086215 * FATOR

    piores = pd.concat([
        grupo.nlargest(TOP_N, 'residuo_conebeam_px')
        for _, grupo in selecionados.groupby('marcador', sort=False)
    ], ignore_index=True)
    piores = piores.sort_values(['marcador', 'residuo_conebeam_px'], ascending=[True, False])
    colunas = ['marcador', 'numero', 'angulo_graus', 'x_observado_px', 'y_observado_px',
               'u_conebeam_px', 'v_conebeam_px', 'residuo_conebeam_px',
               'residuo_detector_mm', 'revisao', 'observacao_revisor']
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho_csv = RELATORIOS / 'revisao_residuos_bexiga.csv'
    piores[colunas].to_csv(caminho_csv, index=False, float_format='%.3f')

    ncol = 3
    nrow = int(np.ceil(len(piores) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.5 * nrow), squeeze=False)
    for ax, (_, r) in zip(axes.flat, piores.iterrows()):
        img = imagem_reduzida(int(r['numero']))
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        xo, yo = float(r['x_observado_px']), float(r['y_observado_px'])
        xp, yp = float(r['u_conebeam_px']), float(r['v_conebeam_px'])
        ax.plot(xo, yo, 'o', color='lime', markersize=7, markerfacecolor='none', markeredgewidth=2)
        ax.plot(xp, yp, '+', color='red', markersize=12, markeredgewidth=2)
        ax.plot([xo, xp], [yo, yp], '-', color='yellow', linewidth=1)
        ax.set_xlim(max(0, min(xo, xp) - 100), min(img.shape[1], max(xo, xp) + 100))
        ax.set_ylim(min(img.shape[0], max(yo, yp) + 130), max(0, min(yo, yp) - 130))
        ax.set_title(f"{r['marcador']} | {int(r['numero'])}.dcm | {r['angulo_graus']:.1f}°\n"
                     f"resíduo = {r['residuo_conebeam_px']:.1f} px", fontsize=10)
        ax.axis('off')
    for ax in axes.flat[len(piores):]:
        ax.axis('off')
    fig.suptitle('Verificação visual — verde: observado | vermelho: previsto | amarelo: diferença', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    caminho_fig = RELATORIOS / 'revisao_residuos_bexiga.png'
    fig.savefig(caminho_fig, dpi=160)
    print('\nMaiores resíduos por marcador (observações provisórias):')
    print(piores[['marcador', 'numero', 'angulo_graus', 'residuo_conebeam_px']].to_string(index=False))
    print('\nArquivos gerados:', caminho_csv, caminho_fig, sep='\n')
    print('Atenção: a posição prevista não é verdade-terreno; distância entre marcas não prova erro de detecção.')
    plt.show()


if __name__ == '__main__':
    main()
