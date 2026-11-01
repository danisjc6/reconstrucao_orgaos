"""Comparação exploratória: duas moedas isoladas vs. ajuste conjunto cone-beam.

Não modifica DICOM, não aplica correções e não demonstra calibração absoluta.
Entrada: relatorios/trajetorias_moedas_bexiga.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'relatorios'
ENTRADA = OUT / 'trajetorias_moedas_bexiga.csv'
DSO = 640.0
DSD = 835.0
PIXEL = 0.086215 * 4  # mm por pixel após redução 4x
U0 = (676 - 1) / 2
V0 = (851 - 1) / 2
# Sinais e origem angular ainda não foram calibrados fisicamente.
SINAL_U = 1.0
SINAL_V = 1.0
MARCADORES = ('cranial', 'caudal')
INTERVALOS = set(range(1, 9)) | set(range(21, 34)) | set(range(46, 51))
ANCORAS = {1, 26}
# Deslocamento horizontal comum permitido, com penalização moderada.
MAX_DESLOCAMENTO_U_PX = 40.0
SIGMA_PRIOR_U_PX = 15.0


def projetar(xyz, theta, u0=U0):
    X, Y, Z = xyz
    profundidade = X * np.cos(theta) - Y * np.sin(theta)
    lateral = X * np.sin(theta) + Y * np.cos(theta)
    escala = DSD / (DSO + profundidade) / PIXEL
    return u0 + SINAL_U * escala * lateral, V0 + SINAL_V * escala * Z


def dados_validos(df):
    obrigatorias = {'marcador', 'numero', 'angulo_graus', 'metodo',
                    'x_observado_px', 'y_observado_px'}
    faltam = obrigatorias - set(df.columns)
    if faltam:
        raise ValueError(f'Colunas ausentes: {sorted(faltam)}')
    df = df.copy()
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
    df = df[df['marcador'].isin(MARCADORES)]
    df = df[df['x_observado_px'].notna() & df['y_observado_px'].notna()]
    df = df[df['numero'].isin(ANCORAS) |
            (df['numero'].isin(INTERVALOS) & df['metodo'].eq('circulo'))].copy()
    df['numero'] = df['numero'].astype(int)
    # Validação separada, com ângulos intercalados; âncoras ficam no treino.
    df['conjunto'] = np.where((df['numero'] % 5 == 0) & ~df['numero'].isin(ANCORAS),
                             'validacao', 'treino')
    for m in MARCADORES:
        sub = df[df['marcador'] == m]
        if sum(sub['conjunto'] == 'treino') < 6 or sum(sub['conjunto'] == 'validacao') < 2:
            raise ValueError(f'{m}: poucos dados para treino/validação. Verifique CSV e filtros.')
    return df.sort_values(['marcador', 'numero']).reset_index(drop=True)


def vetores(df):
    t = np.deg2rad(df['angulo_graus'].to_numpy(float))
    u = df['x_observado_px'].to_numpy(float)
    v = df['y_observado_px'].to_numpy(float)
    return t, u, v


def xyz_inicial(df):
    _, _, v = vetores(df)
    z = (np.median(v) - V0) * PIXEL * DSO / DSD
    return np.array([0., 0., np.clip(z, -200., 200.)])


def rms(df, xyz, u0):
    if df.empty:
        return np.nan
    t, u, v = vetores(df)
    up, vp = projetar(xyz, t, u0)
    return float(np.sqrt(np.mean((up - u)**2 + (vp - v)**2)))


def ajustar_individual(df, marcador):
    sub = df[(df['marcador'] == marcador) & (df['conjunto'] == 'treino')]
    t, u, v = vetores(sub)
    def res(xyz):
        up, vp = projetar(xyz, t)
        return np.r_[up-u, vp-v]
    fit = least_squares(res, xyz_inicial(sub), bounds=(-250., 250.),
                        loss='soft_l1', f_scale=5., max_nfev=5000)
    return fit.x, fit


def ajustar_conjunto(df):
    treino = df[df['conjunto'] == 'treino']
    por_m = {m: treino[treino['marcador'] == m] for m in MARCADORES}
    x0 = np.r_[xyz_inicial(por_m['cranial']),
               xyz_inicial(por_m['caudal']), 0.]
    lower = np.array([-250.]*6 + [-MAX_DESLOCAMENTO_U_PX])
    upper = np.array([250.]*6 + [MAX_DESLOCAMENTO_U_PX])
    def res(par):
        partes = []
        for i, m in enumerate(MARCADORES):
            t, u, v = vetores(por_m[m])
            up, vp = projetar(par[3*i:3*i+3], t, U0 + par[6])
            partes.extend([up-u, vp-v])
        # Prior explícito: evita deslocamento excessivo do centro do detector.
        partes.append(np.array([par[6] / SIGMA_PRIOR_U_PX]))
        return np.concatenate(partes)
    fit = least_squares(res, x0, bounds=(lower, upper),
                        loss='soft_l1', f_scale=5., max_nfev=8000)
    return fit.x, fit


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {ENTRADA}. Execute ajustar_trajetorias_bexiga.py.')
    OUT.mkdir(parents=True, exist_ok=True)
    df = dados_validos(pd.read_csv(ENTRADA))
    individuais = {}
    for m in MARCADORES:
        individuais[m] = ajustar_individual(df, m)[0]
    par, fit = ajustar_conjunto(df)
    u0_conjunto = U0 + par[6]

    registros = []
    for m in MARCADORES:
        sub = df[df['marcador'] == m]
        for nome, xyz, u0 in [('individual', individuais[m], U0),
                              ('conjunto', par[3*MARCADORES.index(m):3*MARCADORES.index(m)+3], u0_conjunto)]:
            for conjunto in ('treino', 'validacao'):
                parte = sub[sub['conjunto'] == conjunto]
                registros.append({'marcador': m, 'modelo': nome, 'conjunto': conjunto,
                                  'n': len(parte), 'rms_2d_px_reduzidos': rms(parte, xyz, u0),
                                  'X_mm': xyz[0], 'Y_mm': xyz[1], 'Z_mm': xyz[2],
                                  'u0_px': u0, 'v0_px': V0})
    resumo = pd.DataFrame(registros)
    resumo.to_csv(OUT / 'comparacao_calibracao_conjunta_bexiga.csv', index=False, float_format='%.4f')

    trajetorias = []
    angulos = np.arange(50) * 7.2
    t = np.deg2rad(angulos)
    for m in MARCADORES:
        for nome, xyz, u0 in [('individual', individuais[m], U0),
                              ('conjunto', par[3*MARCADORES.index(m):3*MARCADORES.index(m)+3], u0_conjunto)]:
            up, vp = projetar(xyz, t, u0)
            for i in range(50):
                trajetorias.append({'marcador': m, 'modelo': nome, 'numero': i+1,
                                    'angulo_graus': angulos[i], 'u_previsto_px': up[i],
                                    'v_previsto_px': vp[i]})
    pd.DataFrame(trajetorias).to_csv(OUT / 'trajetorias_calibracao_conjunta_bexiga.csv',
                                    index=False, float_format='%.4f')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex='col')
    for i, m in enumerate(MARCADORES):
        obs = df[df['marcador'] == m]
        for nome, xyz, u0 in [('Individual', individuais[m], U0),
                              ('Conjunto', par[3*i:3*i+3], u0_conjunto)]:
            up, _ = projetar(xyz, t, u0)
            axes[i, 0].plot(angulos, up, label=nome)
        for grupo, marker in [('treino', 'o'), ('validacao', 'x')]:
            pts = obs[obs['conjunto'] == grupo]
            axes[i, 0].scatter(pts['angulo_graus'], pts['x_observado_px'],
                               c='black', s=22, marker=marker, label=grupo)
        axes[i, 0].set_ylabel(f'{m.capitalize()} — u (px reduzidos)')
        axes[i, 0].legend(fontsize=8)
        sub = resumo[(resumo['marcador'] == m) & (resumo['conjunto'] == 'validacao')]
        axes[i, 1].bar(sub['modelo'], sub['rms_2d_px_reduzidos'])
        axes[i, 1].set_ylabel(f'{m.capitalize()} — RMS validação (px)')
        axes[i, 1].set_title('Observações não usadas no ajuste')
    axes[0, 0].set_title('Trajetórias horizontais — modelos e observações')
    axes[1, 0].set_xlabel('Ângulo (graus)')
    fig.tight_layout()
    fig.savefig(OUT / 'comparacao_calibracao_conjunta_bexiga.png', dpi=160)
    plt.close(fig)

    print('\nComparação RMS 2D (pixels reduzidos):')
    print(resumo[['marcador','modelo','conjunto','n','rms_2d_px_reduzidos']].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print(f'\nCentro horizontal nominal: {U0:.2f} px; conjunto: {u0_conjunto:.2f} px')
    print(f'Deslocamento estimado: {par[6]:+.2f} px reduzidos')
    print(f'Convergência: {fit.success} — {fit.message}')
    print('\nArquivos em:', OUT)
    print('ATENÇÃO: observações ainda provisórias; parâmetros não são calibração absoluta.')
    print('Não aplicar correções aos DICOM nem à geometria TIGRE nesta etapa.')


if __name__ == '__main__':
    main()

