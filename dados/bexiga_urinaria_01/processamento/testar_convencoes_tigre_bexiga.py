"""Diagnóstico das convenções de projeção TIGRE para a bexiga.

Executar no ambiente conda coracao-tigre. Não altera dados originais.
As posições são sintéticas: este é um teste de convenções, não calibração.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import tigre

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'relatorios' / 'validacao_geometria_tigre' / 'convencoes'
OUT.mkdir(parents=True, exist_ok=True)

DSO, DSD = 640.0, 835.0
PIX = 0.086215 * 4
NROWS, NCOLS = 851, 676

geo = tigre.geometry(mode='cone', default=True)
geo.DSO = DSO
geo.DSD = DSD
geo.nDetector = np.array([NROWS, NCOLS], dtype=np.int32)
geo.dDetector = np.array([PIX, PIX], dtype=np.float32)
geo.sDetector = geo.nDetector * geo.dDetector
# Pequeno volume de teste: eixo TIGRE z, y, x; espaçamento de 1 mm
geo.nVoxel = np.array([128, 128, 128], dtype=np.int32)
geo.dVoxel = np.array([1., 1., 1.], dtype=np.float32)
geo.sVoxel = geo.nVoxel * geo.dVoxel
geo.offOrigin = np.zeros(3, dtype=np.float32)
geo.offDetector = np.zeros(2, dtype=np.float32)
geo.rotDetector = np.zeros(3, dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

# Coordenadas de teste na matriz (z,y,x) em mm, relativas ao centro.
# Evitamos pontos na origem para revelar sinais e orientações.
points = {
    'origem': (0., 0., 0.),
    'mais_x': (0., 0., 30.),
    'mais_y': (0., 30., 0.),
    'mais_z': (30., 0., 0.),
}
angles_deg = [0., 45., 90., 180., 270.]
angles = np.deg2rad(angles_deg).astype(np.float32)

# Colocar um pequeno cubo de 3x3x3 voxels para facilitar a identificação.
def make_phantom(pos):
    arr = np.zeros(tuple(geo.nVoxel), dtype=np.float32)
    idx = [int(round((n - 1) / 2 + p / d)) for n, p, d in zip(geo.nVoxel, pos, geo.dVoxel)]
    for i, n in zip(idx, geo.nVoxel):
        if i < 2 or i >= n - 2:
            raise ValueError('Ponto fora do volume')
    z, y, x = idx
    arr[z-1:z+2, y-1:y+2, x-1:x+2] = 1.0
    return arr

rows = []
fig, axs = plt.subplots(len(points), len(angles), figsize=(15, 10), squeeze=False)
for r, (name, pos) in enumerate(points.items()):
    projections = tigre.Ax(make_phantom(pos), geo, angles, projection_type='interpolated')
    for c, deg in enumerate(angles_deg):
        img = np.asarray(projections[c], dtype=np.float64)
        # Centro de massa dos pixels positivos (sem depender de limiar relativo).
        weights = np.maximum(img, 0)
        mass = weights.sum()
        if mass <= 0:
            u = v = float('nan')
        else:
            rr, cc = np.indices(img.shape)
            u = float((weights * cc).sum() / mass)
            v = float((weights * rr).sum() / mass)
        rows.append({'objeto': name, 'z_mm': pos[0], 'y_mm': pos[1],
                     'x_mm': pos[2], 'angulo_graus': deg,
                     'u_px': u, 'v_px': v})
        ax = axs[r, c]
        # Recorte em torno do pico para manter a figura legível.
        if np.isfinite(u) and np.isfinite(v):
            x0, y0 = int(round(u)), int(round(v))
            ax.imshow(img[max(0,y0-24):min(NROWS,y0+25),
                          max(0,x0-24):min(NCOLS,x0+25)], cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(f'{name} / {deg:g}°\n(u,v)=({u:.1f},{v:.1f})', fontsize=9)
        ax.axis('off')
    print(f'{name}: concluído')

fig.suptitle('TIGRE — reprojeção de pontos sintéticos (centros em pixels)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'convencoes_pontos_sinteticos.png', dpi=160)
plt.close(fig)

with (OUT / 'centros_pontos_sinteticos.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print('\nResultados em:', OUT)
print('Referência: centro do detector =', (NCOLS-1)/2, (NROWS-1)/2)
print('IMPORTANTE: horário depende do ponto de vista; ainda não alteramos o sinal angular.')
print('Envie a tabela CSV ou a saída abaixo para interpretar os eixos:')
for row in rows:
    print(f"{row['objeto']:>7s}  {row['angulo_graus']:>5.0f}°  u={row['u_px']:>8.2f}  v={row['v_px']:>8.2f}")
