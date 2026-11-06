"""Comparação exploratória de preparações CR para FDK/TIGRE.

Não produz atenuação calibrada. Preserva entradas e resultados anteriores.
Executar no ambiente coracao-tigre.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tigre
import tigre.algorithms as algs

BASE = Path(__file__).resolve().parents[1]
PREP = BASE / 'reconstrucoes' / 'tigre' / 'preparacao'
ROOT = BASE / 'reconstrucoes' / 'tigre'
OUT = ROOT / 'comparacao_preprocessamentos'
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = PREP / 'projecoes_cr_reduzidas.npy'
LOG_FILE = PREP / 'projecoes_exploratorias_log.npy'
ANGLES_FILE = PREP / 'angulos_rad.npy'
for path in (RAW_FILE, LOG_FILE, ANGLES_FILE):
    if not path.exists():
        raise FileNotFoundError(f'Arquivo necessário ausente: {path}')

raw = np.load(RAW_FILE, mmap_mode='r')
angles = np.load(ANGLES_FILE).astype(np.float32)
if raw.shape != (50, 851, 676) or angles.shape != (50,):
    raise ValueError(f'Dimensões inesperadas: raw={raw.shape}, angles={angles.shape}')
if not np.isfinite(raw).all():
    raise ValueError('Dados CR contêm valores não finitos')

# Os percentis são GLOBAIS, calculados sobre as 50 projeções.
# A inversão é uma hipótese, não uma relação física demonstrada.
# Não normalizamos cada imagem isoladamente.
limits = {
    'B_linear_global': (0.1, 99.9),
    'C_linear_robusta': (1.0, 99.0),
}

geo = tigre.geometry(mode='cone', default=True)
geo.DSO = 640.0
geo.DSD = 835.0
geo.nDetector = np.array([851, 676], dtype=np.int32)
geo.dDetector = np.array([0.34486, 0.34486], dtype=np.float32)
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array([256, 256, 256], dtype=np.int32)
# Mesmo FOV transversal de 150 mm da última reconstrução.
geo.sVoxel = np.array([220.0, 150.0, 150.0], dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.offOrigin = np.zeros(3, dtype=np.float32)
geo.offDetector = np.zeros(2, dtype=np.float32)
geo.rotDetector = np.zeros(3, dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

# Reconstrução A de referência: caso já exista FOV 150, reutilizar.
# Se não existir, recalcular com a mesma geometria desta comparação.
ref_file = ROOT / 'teste_fdk_fov150' / 'volume_fdk_exploratorio.npy'
volumes = {}
if ref_file.exists():
    print('Reutilizando referência A:', ref_file, flush=True)
    volumes['A_log_atual'] = np.load(ref_file, mmap_mode='r')
    if volumes['A_log_atual'].shape != (256, 256, 256):
        raise ValueError('Volume de referência A com dimensões incompatíveis')
else:
    print('Referência A não encontrada; reconstruindo FDK A com FOV 150 mm...', flush=True)
    a = np.load(LOG_FILE).astype(np.float32)
    volumes['A_log_atual'] = np.asarray(algs.fdk(a, geo, angles), dtype=np.float32)
    np.save(OUT / 'volume_A_log_atual.npy', volumes['A_log_atual'])
    del a

metadata = {'geometria': {'DSO_mm': 640, 'DSD_mm': 835,
                         'nDetector': [851, 676], 'dDetector_mm': [0.34486, 0.34486],
                         'nVoxel': [256, 256, 256], 'sVoxel_mm': [220, 150, 150]},
            'aviso': 'Imagens CR processadas: preparações exploratórias, não quantitativas',
            'referencia_A': str(ref_file) if ref_file.exists() else 'recalculada'}

for name, (p_lo, p_hi) in limits.items():
    lo, hi = np.percentile(raw, [p_lo, p_hi])
    if hi <= lo:
        raise ValueError(f'Intervalo inválido para {name}: {lo}, {hi}')
    # Pixels mais escuros no CR recebem maior valor; polaridade provisória.
    proj = np.clip((hi - np.asarray(raw, dtype=np.float32)) / (hi - lo), 0.0, 1.0)
    proj = np.ascontiguousarray(proj, dtype=np.float32)
    np.save(OUT / f'projecoes_{name}.npy', proj)
    metadata[name] = {'percentis_globais': [p_lo, p_hi],
                      'limites_cr': [float(lo), float(hi)],
                      'transformacao': 'clip((hi - CR)/(hi - lo), 0, 1)'}
    print(f'Reconstruindo {name} — limites globais {lo:.2f} a {hi:.2f}', flush=True)
    volume = np.asarray(algs.fdk(proj, geo, angles), dtype=np.float32)
    if not np.isfinite(volume).all():
        raise RuntimeError(f'FDK {name} retornou valores não finitos')
    np.save(OUT / f'volume_{name}.npy', volume)
    volumes[name] = volume
    del proj

with (OUT / 'parametros_comparacao.json').open('w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Os métodos possuem escalas de intensidade diferentes; para comparação
# VISUAL normalizamos cada volume por percentis globais do próprio volume.
# Isto não serve para comparar valores físicos nem amplitudes absolutas.
def normalizar_para_exibicao(volume):
    lo, hi = np.percentile(volume, [2, 99.5])
    return np.clip((np.asarray(volume) - lo) / max(hi - lo, 1e-8), 0, 1)

names = list(volumes)
fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)))
for i, name in enumerate(names):
    v = normalizar_para_exibicao(volumes[name])
    slices = (v[128], v[:, 128, :], v[:, :, 128])
    for j, (sl, plane) in enumerate(zip(slices, ('Axial', 'Coronal', 'Sagital'))):
        axes[i, j].imshow(sl, cmap='gray', vmin=0, vmax=1)
        axes[i, j].set_title(f'{name} — {plane}')
        axes[i, j].axis('off')
fig.suptitle('Bexiga: comparação exploratória de pré-processamentos (FDK)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'comparacao_cortes_fdk.png', dpi=160, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, idx in enumerate((0, 12, 25)):
    lo, hi = np.percentile(raw, [0.1, 99.9])
    axes[0, j].imshow(raw[idx], cmap='gray', vmin=lo, vmax=hi)
    axes[0, j].set_title(f'CR original: {idx + 1}.dcm')
    axes[1, j].imshow(np.load(OUT / 'projecoes_C_linear_robusta.npy', mmap_mode='r')[idx],
                      cmap='gray', vmin=0, vmax=1)
    axes[1, j].set_title('C: inversão linear robusta')
    for ax in (axes[0, j], axes[1, j]):
        ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'controle_projecoes_linear.png', dpi=160)
plt.close(fig)

print('\nComparação concluída. Arquivos em:', OUT)
print('Inspecione comparacao_cortes_fdk.png e controle_projecoes_linear.png')
print('AVISO: FDK exploratório; CR processado não é atenuação calibrada.')
