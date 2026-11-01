from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tigre
import tigre.algorithms as algs

BASE = Path(__file__).resolve().parents[1]
PREP = BASE / 'reconstrucoes' / 'tigre' / 'preparacao'
OUT = BASE / 'reconstrucoes' / 'tigre' / 'teste_fdk_fov150'
OUT.mkdir(parents=True, exist_ok=True)

proj = np.load(PREP / 'projecoes_exploratorias_log.npy').astype(np.float32)
angles = np.load(PREP / 'angulos_rad.npy').astype(np.float32)
assert proj.shape == (50, 851, 676), proj.shape

geo = tigre.geometry(mode='cone', default=True)
geo.DSO = 640.0
geo.DSD = 835.0
geo.nDetector = np.array([851, 676], dtype=np.int32)
geo.dDetector = np.array([0.34486, 0.34486], dtype=np.float32)
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array([256, 256, 256], dtype=np.int32)
geo.sVoxel = np.array([220.0, 150.0, 150.0], dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
geo.rotDetector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

print('FDK: projeções', proj.shape, 'volume', geo.nVoxel)
print('IMPORTANTE: ângulos/sentido e orientação do detector ainda precisam de validação.')
vol = algs.fdk(proj, geo, angles)
vol = np.asarray(vol, dtype=np.float32)
np.save(OUT / 'volume_fdk_exploratorio.npy', vol)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, sl, name in zip(axs,
                        [vol[128], vol[:, 128, :], vol[:, :, 128]],
                        ['Axial', 'Coronal', 'Sagital']):
    ax.imshow(sl, cmap='gray', vmin=np.percentile(vol, 2),
              vmax=np.percentile(vol, 99.5))
    ax.set_title(name)
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'cortes_centrais_fdk.png', dpi=160)
plt.show()
print('Resultado:', OUT)
