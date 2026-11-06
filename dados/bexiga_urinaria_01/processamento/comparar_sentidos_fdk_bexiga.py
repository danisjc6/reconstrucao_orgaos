from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tigre
import tigre.algorithms as algs

BASE = Path(__file__).resolve().parents[1]
PREP = BASE / 'reconstrucoes' / 'tigre' / 'preparacao'
OUT = BASE / 'reconstrucoes' / 'tigre' / 'comparacao_sentidos_fdk'
OUT.mkdir(parents=True, exist_ok=True)

# Reutilizar EXATAMENTE o mesmo conjunto exploratorio de projecoes nos dois testes.
proj_path = PREP / 'projecoes_exploratorias_log.npy'
angles_path = PREP / 'angulos_rad.npy'
if not proj_path.exists() or not angles_path.exists():
    raise FileNotFoundError(f'Verifique {proj_path} e {angles_path}')
proj = np.load(proj_path).astype(np.float32)
angles = np.load(angles_path).astype(np.float32)
if proj.shape != (50, 851, 676) or angles.shape != (50,):
    raise ValueError(f'Formato inesperado: proj={proj.shape}, angulos={angles.shape}')
if not np.isfinite(proj).all():
    raise ValueError('Projecoes contem NaN ou infinito')

geo = tigre.geometry(mode='cone', default=True)
geo.DSO = 640.0
geo.DSD = 835.0
geo.nDetector = np.array([851, 676], dtype=np.int32)
geo.dDetector = np.array([0.34486, 0.34486], dtype=np.float32)
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array([256, 256, 256], dtype=np.int32)
geo.sVoxel = np.array([150.0, 150.0, 150.0], dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.offOrigin = np.zeros(3, dtype=np.float32)
geo.offDetector = np.zeros(2, dtype=np.float32)
geo.rotDetector = np.zeros(3, dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

volumes = {}
for nome, ang in [('positivo', angles), ('negativo', -angles)]:
    print(f'Reconstruindo FDK: angulos {nome}...', flush=True)
    vol = np.asarray(algs.fdk(proj, geo, ang), dtype=np.float32)
    if not np.isfinite(vol).all():
        print(f'AVISO: volume {nome} possui valores nao finitos')
    np.save(OUT / f'volume_fdk_{nome}.npy', vol)
    volumes[nome] = vol
    print(f'  volume: {vol.shape}, intervalo: {np.nanmin(vol):.5g} a {np.nanmax(vol):.5g}', flush=True)

# Mesma escala de cinza para ambas as reconstrucoes.
all_finite = np.concatenate([v[np.isfinite(v)].ravel() for v in volumes.values()])
vmin, vmax = np.percentile(all_finite, [2, 99.5])
if vmax <= vmin:
    vmax = vmin + 1e-6

fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
for i, (nome, vol) in enumerate(volumes.items()):
    n0, n1, n2 = [s // 2 for s in vol.shape]
    slices = [vol[n0, :, :], vol[:, n1, :], vol[:, :, n2]]
    for j, (sl, plano) in enumerate(zip(slices, ['Axial', 'Coronal', 'Sagital'])):
        axes[i, j].imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, j].set_title(f'{nome.capitalize()} — {plano}')
        axes[i, j].axis('off')
fig.savefig(OUT / 'comparacao_cortes_centrais.png', dpi=170)
plt.close(fig)

# Cortes axiais multiplos para localizar artefatos e anatomia ao longo do eixo.
indices = np.linspace(25, 230, 12, dtype=int)
fig, axes = plt.subplots(4, 6, figsize=(16, 11), constrained_layout=True)
for row, (nome, vol) in enumerate(volumes.items()):
    for k, idx in enumerate(indices):
        ax = axes[row * 2 + k // 6, k % 6]
        ax.imshow(vol[idx], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{nome} | axial {idx}', fontsize=9)
        ax.axis('off')
fig.savefig(OUT / 'comparacao_cortes_axiais.png', dpi=160)
plt.close(fig)

with (OUT / 'metadados_experimento.txt').open('w', encoding='utf-8') as f:
    f.write('Comparacao exploratoria FDK: angulos positivos versus negativos\n')
    f.write(f'Projecoes: {proj_path}\nAngulos: {angles_path}\n')
    f.write('DSO=640 mm; DSD=835 mm; detector=851x676; pixel=0.34486 mm\n')
    f.write('Volume=256x256x256; FOV=150 mm; COR=0; offDetector=0\n')
    f.write('ATENCAO: radiografias CR nao calibradas como atenuacao.\n')
    f.write('Nao foi feita calibracao geometrica independente nem alteracao dos DICOM.\n')
print(f'Concluido. Resultados: {OUT}')
print('Imagens: comparacao_cortes_centrais.png e comparacao_cortes_axiais.png')
