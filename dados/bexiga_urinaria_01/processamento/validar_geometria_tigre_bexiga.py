from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
REL = BASE / 'relatorios'
OUT = REL / 'validacao_geometria_tigre'
OUT.mkdir(parents=True, exist_ok=True)

DSO, DSD = 640.0, 835.0
PIXEL_MM = 0.086215 * 4
U0_NOMINAL = 676 / 2 - 0.5
V0_NOMINAL = 851 / 2 - 0.5

params = pd.read_csv(REL / 'comparacao_calibracao_conjunta_bexiga.csv')
obs = pd.read_csv(REL / 'avaliacao_conebeam_bexiga.csv')
traj = pd.read_csv(REL / 'trajetorias_calibracao_conjunta_bexiga.csv')

params = params[(params['modelo'] == 'conjunto') & (params['conjunto'] == 'treino')].copy()
if set(params['marcador']) != {'cranial', 'caudal'}:
    raise ValueError('Não foi possível localizar parâmetros conjuntos para as duas moedas.')

obs = obs[obs['usado_no_ajuste'].astype(str).str.lower().eq('true')].copy()
obs = obs.dropna(subset=['x_observado_px', 'y_observado_px'])

# Modelo analítico utilizado como conferência de consistência.
# A transformação XYZ -> TIGRE depende da convenção original do ajuste;
# por isso NÃO se assume aqui uma correspondência de eixos não documentada.
# Usamos as trajetórias já exportadas pelo modelo conjunto e comparamos
# às observações que participaram do ajuste.
traj = traj[traj['modelo'] == 'conjunto'].copy()
if traj.empty:
    raise ValueError('Trajetórias do modelo conjunto não encontradas.')

merged = obs.merge(
    traj[['marcador', 'numero', 'u_previsto_px', 'v_previsto_px']],
    on=['marcador', 'numero'], how='inner', validate='many_to_one'
)
if merged.empty:
    raise ValueError('Nenhuma observação corresponde às trajetórias conjuntas.')
merged['residuo_u_px'] = merged['x_observado_px'] - merged['u_previsto_px']
merged['residuo_v_px'] = merged['y_observado_px'] - merged['v_previsto_px']
merged['residuo_2d_px'] = np.hypot(merged['residuo_u_px'], merged['residuo_v_px'])
merged['residuo_2d_mm_detector'] = merged['residuo_2d_px'] * PIXEL_MM
merged.to_csv(OUT / 'residuos_modelo_conjunto.csv', index=False)

print('VALIDAÇÃO INTERNA DAS TRAJETÓRIAS — NÃO É CALIBRAÇÃO INDEPENDENTE')
print(f'DSO={DSO:.1f} mm; DSD={DSD:.1f} mm; pixel reduzido={PIXEL_MM:.6f} mm')
print(f'Centro nominal (u,v)=({U0_NOMINAL:.2f}, {V0_NOMINAL:.2f}) px')
print('\nParâmetros conjuntos exportados:')
print(params[['marcador', 'X_mm', 'Y_mm', 'Z_mm', 'u0_px', 'v0_px']].to_string(index=False))

resumo = []
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex='col')
for row, marker in enumerate(['cranial', 'caudal']):
    p = merged[merged['marcador'] == marker].sort_values('angulo_graus')
    if p.empty:
        continue
    rms = np.sqrt(np.mean(p['residuo_2d_px'] ** 2))
    resumo.append({'marcador': marker, 'n': len(p), 'rms_2d_px': rms,
                   'rms_2d_mm_detector': rms * PIXEL_MM,
                   'media_residuo_u_px': p['residuo_u_px'].mean(),
                   'media_residuo_v_px': p['residuo_v_px'].mean()})
    axs[row, 0].plot(p['angulo_graus'], p['x_observado_px'], 'o', label='u observado')
    axs[row, 0].plot(p['angulo_graus'], p['u_previsto_px'], '-', label='u previsto')
    axs[row, 0].plot(p['angulo_graus'], p['y_observado_px'], 's', label='v observado')
    axs[row, 0].plot(p['angulo_graus'], p['v_previsto_px'], '--', label='v previsto')
    axs[row, 0].set_ylabel(f'{marker}\nCoordenada (px reduzidos)')
    axs[row, 0].legend(fontsize=8)
    axs[row, 1].plot(p['angulo_graus'], p['residuo_u_px'], 'o-', label='resíduo u')
    axs[row, 1].plot(p['angulo_graus'], p['residuo_v_px'], 's-', label='resíduo v')
    axs[row, 1].axhline(0, color='gray', linewidth=0.8)
    axs[row, 1].set_ylabel(f'{marker}\nResíduo (px reduzidos)')
    axs[row, 1].legend(fontsize=8)
axs[0, 0].set_title('Posições observadas e previstas — ajuste conjunto')
axs[0, 1].set_title('Resíduos por ângulo')
axs[1, 0].set_xlabel('Ângulo (graus)')
axs[1, 1].set_xlabel('Ângulo (graus)')
fig.tight_layout()
fig.savefig(OUT / 'validacao_interna_trajetorias.png', dpi=170)
plt.close(fig)

summary = pd.DataFrame(resumo)
summary.to_csv(OUT / 'resumo_residuos.csv', index=False)
print('\nResumo dos resíduos (observações usadas no ajuste):')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nArquivos em:', OUT)
print('ATENÇÃO: não foi feita reprojeção via tigre.Ax nem validada a conversão de eixos XYZ para TIGRE.')
print('Não alterar offDetector, COR ou DICOM com base apenas neste diagnóstico.')
