from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Geometria da aquisição
DSO_MM = 640.0
DSD_MM = 835.0
PIXEL_MM = 0.086215 * 4
N_PROJECOES = 50
# Centro geométrico inicial do detector reduzido (676 x 851)
U0_PX = (676 - 1) / 2
V0_PX = (851 - 1) / 2
# Convenção de orientação: pode ser alterada após verificar os DICOM.
SINAL_U = 1.0
SINAL_V = 1.0

BASE = Path(__file__).resolve().parents[1]
RELATORIOS = BASE / 'relatorios'
ENTRADA = RELATORIOS / 'trajetorias_moedas_bexiga.csv'
RELATORIOS.mkdir(parents=True, exist_ok=True)

# Apenas âncoras verificadas visualmente e detecções circulares em intervalos
# favoráveis. Mesmo estas últimas continuam provisórias.
ANCORAS = {'cranial': {1, 26}, 'caudal': {1, 26}}
INTERVALOS_FAVORAVEIS = set(range(1, 9)) | set(range(21, 34)) | set(range(46, 51))


def projetar(params, theta):
    """Modelo cone-beam de eixo vertical, com detector fixo.

    params = (X, Y, Z, u0, v0), mm para XYZ e px para offsets.
    Coordenadas X,Y giram em torno do eixo vertical Z.
    """
    X, Y, Z, u0, v0 = params
    profundidade = X * np.cos(theta) - Y * np.sin(theta)
    lateral = X * np.sin(theta) + Y * np.cos(theta)
    denom = DSO_MM + profundidade
    escala = DSD_MM / denom / PIXEL_MM
    u = u0 + SINAL_U * escala * lateral
    v = v0 + SINAL_V * escala * Z
    return u, v


def matriz_harmonica(theta):
    return np.column_stack([np.ones(len(theta)), np.cos(theta), np.sin(theta)])


def ajustar_marcador(df, marcador):
    dados = df[df['marcador'] == marcador].copy()
    dados['numero'] = dados['numero'].astype(int)
    mascara = (
        dados['x_observado_px'].notna()
        & dados['y_observado_px'].notna()
        & (
            dados['numero'].isin(ANCORAS[marcador])
            | ((dados['metodo'] == 'circulo') & dados['numero'].isin(INTERVALOS_FAVORAVEIS))
        )
    )
    treino = dados.loc[mascara].copy().sort_values('numero')
    if len(treino) < 6:
        raise RuntimeError(f'{marcador}: apenas {len(treino)} observações; insuficiente para ajuste confiável.')
    theta = np.deg2rad(treino['angulo_graus'].to_numpy(float))
    u_obs = treino['x_observado_px'].to_numpy(float)
    v_obs = treino['y_observado_px'].to_numpy(float)

    # Offset de detector fixado inicialmente no centro. Liberar u0/v0 e XYZ
    # simultaneamente pode produzir degenerescência sem calibração externa.
    def residual(xyz):
        u, v = projetar([*xyz, U0_PX, V0_PX], theta)
        return np.concatenate([u - u_obs, v - v_obs])

    # Valores iniciais em milímetros no sistema local da plataforma.
    x0 = np.array([0.0, 0.0, (np.median(v_obs) - V0_PX) * PIXEL_MM * DSO_MM / DSD_MM])
    limites = ([-250., -250., -250.], [250., 250., 250.])
    ajuste = least_squares(residual, x0, bounds=limites, loss='soft_l1', f_scale=5.0, max_nfev=5000)
    params = [*ajuste.x, U0_PX, V0_PX]

    todos_theta = np.deg2rad(dados['angulo_graus'].to_numpy(float))
    u_pred, v_pred = projetar(params, todos_theta)
    dados['u_conebeam_px'] = u_pred
    dados['v_conebeam_px'] = v_pred
    dados['residuo_conebeam_px'] = np.hypot(dados['x_observado_px'] - u_pred, dados['y_observado_px'] - v_pred)
    dados['usado_no_ajuste'] = mascara.to_numpy()

    # Comparação justa: mesmas observações e parâmetros de treino.
    A = matriz_harmonica(theta)
    coef_u = np.linalg.lstsq(A, u_obs, rcond=None)[0]
    coef_v = np.linalg.lstsq(A, v_obs, rcond=None)[0]
    dados['u_harmonico_px'] = matriz_harmonica(todos_theta) @ coef_u
    dados['v_harmonico_px'] = matriz_harmonica(todos_theta) @ coef_v
    dados['residuo_harmonico_px'] = np.hypot(dados['x_observado_px'] - dados['u_harmonico_px'], dados['y_observado_px'] - dados['v_harmonico_px'])

    teste = dados[dados['usado_no_ajuste']]
    rms_cb = np.sqrt(np.mean(teste['residuo_conebeam_px'] ** 2))
    rms_h = np.sqrt(np.mean(teste['residuo_harmonico_px'] ** 2))
    print(f'\n{marcador}: {len(treino)} observações selecionadas (provisórias)')
    print(f'XYZ ajustado (mm): {ajuste.x.round(3)}')
    print(f'RMS cone-beam: {rms_cb:.2f} px reduzidos')
    print(f'RMS harmônico: {rms_h:.2f} px reduzidos')
    print(f'Convergência: {ajuste.success} — {ajuste.message}')
    return dados, ajuste.x


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Execute antes ajustar_trajetorias_bexiga.py: {ENTRADA}')
    df = pd.read_csv(ENTRADA)
    resultados = []
    parametros = []
    for marcador in ('cranial', 'caudal'):
        dados, xyz = ajustar_marcador(df, marcador)
        resultados.append(dados)
        parametros.append({'marcador': marcador, 'X_mm': xyz[0], 'Y_mm': xyz[1], 'Z_mm': xyz[2]})

    resultado = pd.concat(resultados, ignore_index=True)
    resultado.to_csv(RELATORIOS / 'avaliacao_conebeam_bexiga.csv', index=False, float_format='%.4f')
    pd.DataFrame(parametros).to_csv(RELATORIOS / 'parametros_marcadores_conebeam.csv', index=False, float_format='%.4f')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for i, marcador in enumerate(('cranial', 'caudal')):
        dados = resultado[resultado['marcador'] == marcador].sort_values('angulo_graus')
        usados = dados[dados['usado_no_ajuste']]
        axes[i, 0].plot(dados['angulo_graus'], dados['u_conebeam_px'], label='Cone-beam')
        axes[i, 0].plot(dados['angulo_graus'], dados['u_harmonico_px'], '--', label='Harmônico')
        axes[i, 0].scatter(usados['angulo_graus'], usados['x_observado_px'], s=14, c='black', label='Observações selecionadas')
        axes[i, 0].set_ylabel(f'{marcador.capitalize()} — u (px reduzidos)')
        axes[i, 0].legend(fontsize=8)
        axes[i, 1].scatter(usados['angulo_graus'], usados['residuo_conebeam_px'], s=20, label='Cone-beam')
        axes[i, 1].scatter(usados['angulo_graus'], usados['residuo_harmonico_px'], s=20, marker='x', label='Harmônico')
        axes[i, 1].set_ylabel('Resíduo 2D (px reduzidos)')
        axes[i, 1].legend(fontsize=8)
    axes[0, 0].set_title('Trajetórias horizontais')
    axes[0, 1].set_title('Resíduos nas observações selecionadas')
    axes[1, 0].set_xlabel('Ângulo (graus)')
    axes[1, 1].set_xlabel('Ângulo (graus)')
    fig.tight_layout()
    fig.savefig(RELATORIOS / 'comparacao_conebeam_harmonico_bexiga.png', dpi=160)
    plt.show()
    print('\nArquivos salvos em:', RELATORIOS)
    print('ATENÇÃO: centro do detector e sentido angular são hipóteses; não aplicar correções ainda.')


if __name__ == '__main__':
    main()
