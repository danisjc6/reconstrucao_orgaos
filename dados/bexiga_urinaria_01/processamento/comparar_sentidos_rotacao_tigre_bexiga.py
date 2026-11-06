"""Compara convenções angulares e de eixos via projeção real TIGRE.

Não altera DICOM, volumes ou parâmetros de calibração.
Execute no ambiente conda coracao-tigre, a partir de qualquer diretório.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tigre

BASE = Path(__file__).resolve().parents[1]
REL = BASE / 'relatorios'
OUT = REL / 'validacao_geometria_tigre' / 'comparacao_sentidos'
OUT.mkdir(parents=True, exist_ok=True)

DSO, DSD = 640.0, 835.0
PIX = 0.086215 * 4
NDET = np.array([851, 676], dtype=np.int32)
NV = 192
EXT = 220.0
RAIO = 2.0

# Os parâmetros abaixo são do ajuste conjunto; NÃO são calibração absoluta.
params = pd.read_csv(REL / 'comparacao_calibracao_conjunta_bexiga.csv')
params = params[(params['modelo'] == 'conjunto') & (params['conjunto'] == 'treino')]
if set(params['marcador']) != {'cranial', 'caudal'}:
    raise RuntimeError('Não encontrei parâmetros conjuntos de treino para as duas moedas.')

obs = pd.read_csv(REL / 'avaliacao_conebeam_bexiga.csv')
obs = obs[obs['usado_no_ajuste'].astype(str).str.lower() == 'true'].copy()
obs = obs.dropna(subset=['x_observado_px', 'y_observado_px'])
if obs.empty:
    raise RuntimeError('Não há observações válidas em avaliacao_conebeam_bexiga.csv')

geo = tigre.geometry(mode='cone', default=True)
geo.DSO = DSO
geo.DSD = DSD
geo.nDetector = NDET
geo.dDetector = np.array([PIX, PIX], dtype=np.float32)
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array([NV, NV, NV], dtype=np.int32)
geo.sVoxel = np.array([EXT, EXT, EXT], dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.offOrigin = np.zeros(3, dtype=np.float32)
geo.offDetector = np.zeros(2, dtype=np.float32)
geo.rotDetector = np.zeros(3, dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

# TIGRE usa volume na ordem [z,y,x]. O voxel mais próximo é usado
# para evitar interpolação ambígua no posicionamento inicial.
def esfera(x, y, z):
    zz, yy, xx = np.ogrid[:NV, :NV, :NV]
    ix = (x + EXT / 2) / (EXT / NV) - 0.5
    iy = (y + EXT / 2) / (EXT / NV) - 0.5
    iz = (z + EXT / 2) / (EXT / NV) - 0.5
    return (((xx-ix)**2 + (yy-iy)**2 + (zz-iz)**2) <= (RAIO / (EXT / NV))**2).astype(np.float32)

# Centroide com limiar relativo: robusto para uma esfera sintética isolada.
def centros(proj):
    saida = []
    for img in proj:
        img = np.maximum(np.asarray(img, dtype=np.float64), 0)
        lim = img.max() * 0.15
        pesos = np.where(img >= lim, img, 0)
        total = pesos.sum()
        if total <= 0:
            saida.append((np.nan, np.nan))
            continue
        vs = np.arange(img.shape[0], dtype=float)
        us = np.arange(img.shape[1], dtype=float)
        saida.append(((pesos.sum(axis=0) @ us) / total,
                      (pesos.sum(axis=1) @ vs) / total))
    return np.array(saida)

# O ajuste original usa u = u0 + escala*(X*sin(theta)+Y*cos(theta)).
# Testamos theta e -theta, e X/Y com sinal mantido ou invertido.
# Isso evita atribuir antecipadamente um sinal à rotação física.
variantes = [
    ('ang_positivo_XY_original', +1, +1, +1),
    ('ang_negativo_XY_original', -1, +1, +1),
    ('ang_positivo_X_invertido', +1, -1, +1),
    ('ang_negativo_X_invertido', -1, -1, +1),
    ('ang_positivo_Y_invertido', +1, +1, -1),
    ('ang_negativo_Y_invertido', -1, +1, -1),
]

resultados = []
for _, p in params.iterrows():
    marcador = p['marcador']
    dados = obs[obs['marcador'] == marcador].sort_values('angulo_graus')
    if dados.empty:
        continue
    graus = dados['angulo_graus'].to_numpy(dtype=float)
    esperado_u = dados['x_observado_px'].to_numpy(dtype=float)
    esperado_v = dados['y_observado_px'].to_numpy(dtype=float)
    for nome, sinal_ang, sinal_x, sinal_y in variantes:
        vol = esfera(sinal_x * float(p['X_mm']),
                     sinal_y * float(p['Y_mm']), float(p['Z_mm']))
        ang = np.deg2rad(sinal_ang * graus).astype(np.float32)
        print(f'Projetando {marcador} | {nome} | {len(ang)} ângulos', flush=True)
        proj = tigre.Ax(vol, geo, ang)
        uv = centros(proj)
        # Os centros são coordenadas de pixel, sem ajuste de offset.
        for i, (u, v) in enumerate(uv):
            resultados.append(dict(
                marcador=marcador, variante=nome, angulo_graus=graus[i],
                u_observado=esperado_u[i], v_observado=esperado_v[i],
                u_tigre=u, v_tigre=v,
                erro_u=u-esperado_u[i], erro_v=v-esperado_v[i],
                erro_2d=np.hypot(u-esperado_u[i], v-esperado_v[i])
            ))

df = pd.DataFrame(resultados)
df.to_csv(OUT / 'reprojecoes_comparadas.csv', index=False)
resumo = (df.groupby(['marcador', 'variante'], as_index=False)
          .agg(n=('erro_2d', 'count'),
               rms_2d_px=('erro_2d', lambda x: float(np.sqrt(np.mean(np.square(x))))),
               rms_u_px=('erro_u', lambda x: float(np.sqrt(np.mean(np.square(x))))),
               rms_v_px=('erro_v', lambda x: float(np.sqrt(np.mean(np.square(x)))))))
resumo.to_csv(OUT / 'resumo_erros.csv', index=False)
print('\nCOMPARAÇÃO DE ERROS (pixels reduzidos):')
print(resumo.sort_values(['marcador', 'rms_2d_px']).to_string(index=False, float_format=lambda x: f'{x:.2f}'))

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for j, marcador in enumerate(['cranial', 'caudal']):
    sub = df[df['marcador'] == marcador]
    if sub.empty:
        continue
    axu, axv = axs[j]
    base = sub[sub['variante'] == variantes[0][0]]
    axu.scatter(base['angulo_graus'], base['u_observado'], c='black', s=16, label='Observado')
    axv.scatter(base['angulo_graus'], base['v_observado'], c='black', s=16, label='Observado')
    for nome, *_ in variantes:
        a = sub[sub['variante'] == nome]
        axu.plot(a['angulo_graus'], a['u_tigre'], label=nome, linewidth=1)
        axv.plot(a['angulo_graus'], a['v_tigre'], label=nome, linewidth=1)
    axu.set_title(f'{marcador}: horizontal (u)')
    axv.set_title(f'{marcador}: vertical (v)')
    for ax in (axu, axv):
        ax.set_xlabel('Ângulo nominal (graus)')
        ax.set_ylabel('Pixel reduzido')
        ax.grid(alpha=0.2)
axs[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'comparacao_trajetorias_tigre.png', dpi=160)
plt.close(fig)
print('\nResultados:', OUT)
print('AVISO: teste de convenções, não calibração física independente.')
print('O sentido horário ainda precisa ser associado ao ponto de observação.')
