"""Diagnóstico das intensidades CR e do pré-processamento exploratório TIGRE.
Não altera DICOM, projeções ou reconstruções existentes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

BASE = Path(__file__).resolve().parents[1]
PREP = BASE / 'reconstrucoes' / 'tigre' / 'preparacao'
OUT = BASE / 'relatorios' / 'avaliacao_intensidades'
DICOM = BASE / 'configuracao' / 'dicom'
OUT.mkdir(parents=True, exist_ok=True)

raw = np.load(PREP / 'projecoes_cr_reduzidas.npy', mmap_mode='r')
log = np.load(PREP / 'projecoes_exploratorias_log.npy', mmap_mode='r')
with open(PREP / 'metadados.json', encoding='utf-8') as f:
    meta = json.load(f)
files = meta['files']
if raw.shape != log.shape or raw.shape[0] != len(files) or len(files) != 50:
    raise ValueError(f'Dimensões/arquivos incompatíveis: CR={raw.shape}, log={log.shape}, nomes={len(files)}')

# Amostras espaciais fixas: comparações entre ângulos não dependem de janelas automáticas.
h, w = raw.shape[1:]
ys = np.arange(0, h, 8)
xs = np.arange(0, w, 8)
records = []
profiles_raw = []
profiles_log = []
for i, name in enumerate(files):
    a = np.asarray(raw[i], dtype=np.float32)
    b = np.asarray(log[i], dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'Valores não finitos em {name}')
    aa = a[np.ix_(ys, xs)].ravel()
    bb = b[np.ix_(ys, xs)].ravel()
    # Região central fixa; não é uma segmentação da bexiga.
    cy, cx = h // 2, w // 2
    central = b[max(0,cy-100):min(h,cy+100), max(0,cx-100):min(w,cx+100)]
    ds = pydicom.dcmread(DICOM / name, stop_before_pixels=True)
    records.append({
        'arquivo': name, 'angulo_graus': i * 7.2,
        'cr_p1': float(np.percentile(aa, 1)),
        'cr_mediana': float(np.median(aa)),
        'cr_p99': float(np.percentile(aa, 99)),
        'log_p1': float(np.percentile(bb, 1)),
        'log_mediana': float(np.median(bb)),
        'log_p99': float(np.percentile(bb, 99)),
        'log_frac_zero': float(np.mean(bb <= 1e-6)),
        'log_frac_alto': float(np.mean(bb >= 6.9)),
        'log_centro_mediana': float(np.median(central)),
        'photometric': str(getattr(ds, 'PhotometricInterpretation', '')),
        'presentation_lut_shape': str(getattr(ds, 'PresentationLUTShape', '')),
        'pixel_intensity_relationship': str(getattr(ds, 'PixelIntensityRelationship', '')),
        'pixel_intensity_relationship_sign': str(getattr(ds, 'PixelIntensityRelationshipSign', '')),
        'rescale_slope': str(getattr(ds, 'RescaleSlope', '')),
        'rescale_intercept': str(getattr(ds, 'RescaleIntercept', '')),
        'voi_lut_function': str(getattr(ds, 'VOILUTFunction', '')),
        'window_center': str(getattr(ds, 'WindowCenter', '')),
        'window_width': str(getattr(ds, 'WindowWidth', '')),
    })
    profiles_raw.append(np.median(a, axis=1))
    profiles_log.append(np.median(b, axis=1))

stats = pd.DataFrame(records)
stats.to_csv(OUT / 'estatisticas_intensidades_bexiga.csv', index=False)
profiles_raw = np.stack(profiles_raw)
profiles_log = np.stack(profiles_log)

fig, ax = plt.subplots(3, 2, figsize=(14, 13))
angles = stats['angulo_graus'].to_numpy()
for col, (arr, label) in enumerate([(raw, 'CR reduzido'), (log, 'Transformação exploratória')]):
    for j in [0, 12, 25]:
        vals = np.asarray(arr[j])[::5, ::5].ravel()
        ax[0, col].hist(vals, bins=90, histtype='step', label=files[j])
    ax[0, col].set_title(f'Histogramas — {label}')
    ax[0, col].set_xlabel('Valor do pixel')
    ax[0, col].set_ylabel('Contagem amostrada')
    ax[0, col].legend()

for col, prefix in enumerate(['cr', 'log']):
    for q, label in [('p1', 'P1'), ('mediana', 'Mediana'), ('p99', 'P99')]:
        ax[1, col].plot(angles, stats[f'{prefix}_{q}'], marker='.', label=label)
    ax[1, col].set_title(f'Intensidades por ângulo — {prefix.upper()}')
    ax[1, col].set_xlabel('Ângulo (graus)')
    ax[1, col].set_ylabel('Valor do pixel')
    ax[1, col].legend()

for col, (profiles, label) in enumerate([(profiles_raw, 'CR'), (profiles_log, 'Transformação')]):
    for j in [0, 12, 25]:
        ax[2, col].plot(np.arange(h), profiles[j], label=files[j], alpha=0.85)
    ax[2, col].set_title(f'Perfil vertical: mediana de cada linha — {label}')
    ax[2, col].set_xlabel('Linha do detector reduzido (pixel)')
    ax[2, col].set_ylabel('Mediana da linha')
    ax[2, col].legend()
fig.tight_layout()
fig.savefig(OUT / 'diagnostico_intensidades_bexiga.png', dpi=160)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for col, j in enumerate([0, 12]):
    a = np.asarray(raw[j]); b = np.asarray(log[j])
    for row, (im, label) in enumerate([(a, 'CR'), (b, 'Transformação')]):
        # Mesma escala por modalidade, compartilhada entre as duas projeções.
        vmin = float(np.percentile(raw[:, ::16, ::16], 1)) if row == 0 else 0
        vmax = float(np.percentile(raw[:, ::16, ::16], 99)) if row == 0 else float(np.percentile(log[:, ::16, ::16], 99))
        ax[row, col].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        ax[row, col].set_title(f'{files[j]} — {label}')
        ax[row, col].axis('off')
fig.tight_layout()
fig.savefig(OUT / 'comparacao_imagens_escala_fixa.png', dpi=150)
plt.close(fig)

print('Projeções avaliadas:', len(files), '| dimensões:', raw.shape)
print('Fração de pixels zerados na transformação (mín./máx.):',
      f"{stats['log_frac_zero'].min():.1%} / {stats['log_frac_zero'].max():.1%}")
print('Mediana transformada (mín./máx.):',
      f"{stats['log_mediana'].min():.3f} / {stats['log_mediana'].max():.3f}")
print('Metadados DICOM distintos:')
for field in ['photometric', 'presentation_lut_shape', 'pixel_intensity_relationship',
              'pixel_intensity_relationship_sign', 'voi_lut_function']:
    print(f'  {field}: {stats[field].unique().tolist()}')
print('Relatórios:', OUT)
print('ATENÇÃO: resultados diagnósticos; não representam calibração física da atenuação.')
