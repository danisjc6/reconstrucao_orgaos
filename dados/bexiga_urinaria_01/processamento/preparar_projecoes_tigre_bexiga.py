from pathlib import Path
import json
import re
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

BASE = Path(__file__).resolve().parents[1]
DICOM = BASE / "configuracao" / "dicom"
OUT = BASE / 'reconstrucoes' / 'tigre' / 'preparacao'
OUT.mkdir(parents=True, exist_ok=True)
FACTOR = 4

def natural_key(p):
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else -1

files = sorted(DICOM.glob('*.dcm'), key=natural_key)
if len(files) != 50:
    raise RuntimeError(f'Esperados 50 arquivos .dcm; encontrados {len(files)} em {DICOM}')

stack = []
info = []
for path in files:
    ds = pydicom.dcmread(path)
    im = ds.pixel_array.astype(np.float32)
    if im.shape != (3404, 2704):
        raise ValueError(f'{path.name}: dimensões inesperadas {im.shape}')
    if getattr(ds, 'PhotometricInterpretation', '') != 'MONOCHROME2':
        raise ValueError(f'{path.name}: verificar interpretação fotométrica')
    # Média de blocos 4x4, sem interpolar ou normalizar por imagem.
    reduced = im.reshape(851, 4, 676, 4).mean(axis=(1, 3))
    stack.append(reduced)
    info.append({'arquivo': path.name, 'min': float(im.min()),
                 'max': float(im.max()), 'p1': float(np.percentile(im, 1)),
                 'p99': float(np.percentile(im, 99))})

raw = np.stack(stack).astype(np.float32)
np.save(OUT / 'projecoes_cr_reduzidas.npy', raw)
angles = np.deg2rad(np.arange(50, dtype=np.float32) * 7.2).astype(np.float32)
np.save(OUT / 'angulos_rad.npy', angles)

# Apenas hipótese exploratória: inverter contraste globalmente e obter -log(I/I0).
# Isto NÃO equivale a calibração de imagens CR processadas.
lo, hi = np.percentile(raw, [0.1, 99.9])
trans = np.clip((hi - raw) / max(hi - lo, 1e-6), 1e-3, 1.0)
# Referência por coluna: percentil alto entre projeções; aproximação, não flat-field físico.
i0 = np.percentile(trans, 99, axis=0, keepdims=True)
line_integrals = -np.log(np.clip(trans / np.maximum(i0, 1e-3), 1e-3, 1.0))
line_integrals = np.maximum(line_integrals, 0).astype(np.float32)
np.save(OUT / 'projecoes_exploratorias_log.npy', line_integrals)

with open(OUT / 'metadados.json', 'w', encoding='utf-8') as f:
    json.dump({'n': len(files), 'shape': list(raw.shape), 'files': [p.name for p in files],
               'downsample': FACTOR, 'pixel_mm': 0.086215 * FACTOR,
               'DSO_mm': 640, 'DSD_mm': 835, 'intensity_warning':
               'CR processado; transformacao log hipotetica, nao quantitativa',
               'stats': info}, f, ensure_ascii=False, indent=2)

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for col, i in enumerate([0, 12, 25]):
    axes[0, col].imshow(raw[i], cmap='gray', vmin=lo, vmax=hi)
    axes[0, col].set_title(f'CR original reduzido: {files[i].name}')
    axes[1, col].imshow(line_integrals[i], cmap='gray', vmin=0,
                        vmax=np.percentile(line_integrals, 99))
    axes[1, col].set_title('Hipótese de atenuação (exploratória)')
    for ax in (axes[0, col], axes[1, col]):
        ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'controle_preprocessamento.png', dpi=150)
plt.close(fig)
print('Projeções:', raw.shape)
print('Arquivos salvos em:', OUT)
print('ATENÇÃO: conferir o painel antes de reconstruir; intensidades CR não calibradas.')
