"""Reconstrucao exploratoria SIRT da bexiga com TIGRE (50 projecoes CR).

Executar no ambiente coracao-tigre, a partir de qualquer diretorio.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tigre
import tigre.algorithms as algs


def main():
    parser = argparse.ArgumentParser(description='Reconstrucao SIRT TIGRE da bexiga')
    parser.add_argument('--iteracoes', type=int, default=20)
    args = parser.parse_args()
    if args.iteracoes < 1:
        parser.error('--iteracoes deve ser positivo')

    base = Path(__file__).resolve().parents[1]
    prep = base / 'reconstrucoes' / 'tigre' / 'preparacao'
    out = base / 'reconstrucoes' / 'tigre' / f'sirt_{args.iteracoes:03d}_positivo'
    out.mkdir(parents=True, exist_ok=True)

    proj_file = prep / 'projecoes_exploratorias_log.npy'
    ang_file = prep / 'angulos_rad.npy'
    proj = np.ascontiguousarray(np.load(proj_file), dtype=np.float32)
    angles = np.ascontiguousarray(np.load(ang_file), dtype=np.float32)
    if proj.shape != (50, 851, 676) or angles.shape != (50,):
        raise ValueError(f'Dimensoes inesperadas: {proj.shape}, {angles.shape}')
    if not np.isfinite(proj).all() or not np.isfinite(angles).all():
        raise ValueError('Projecoes ou angulos contem NaN/infinito')

    geo = tigre.geometry(mode='cone', default=True)
    geo.DSO = 640.0
    geo.DSD = 835.0
    geo.nDetector = np.array([851, 676], dtype=np.int32)
    geo.dDetector = np.array([0.34486, 0.34486], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([256, 256, 256], dtype=np.int32)
    geo.sVoxel = np.array([150., 150., 150.], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.zeros(3, dtype=np.float32)
    geo.offDetector = np.zeros(2, dtype=np.float32)
    geo.rotDetector = np.zeros(3, dtype=np.float32)
    geo.COR = 0.0
    geo.accuracy = 0.5

    print(f'SIRT TIGRE: {args.iteracoes} iteracoes; projecoes={proj.shape}; volume={tuple(geo.nVoxel)}', flush=True)
    print('Geometria nominal; angulos positivos; intensidades CR exploratorias.', flush=True)
    vol = np.asarray(algs.sirt(proj, geo, angles, niter=args.iteracoes, verbose=True), dtype=np.float32)
    if vol.shape != (256, 256, 256) or not np.isfinite(vol).all():
        raise RuntimeError(f'Volume invalido: shape={vol.shape}; finitos={np.isfinite(vol).all()}')
    np.save(out / 'volume_sirt.npy', vol)
    vmin, vmax = np.percentile(vol, [2, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, sl, titulo in zip(axes, [vol[128], vol[:, 128, :], vol[:, :, 128]],
                              ['Axial', 'Coronal', 'Sagital']):
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'SIRT {args.iteracoes} — {titulo}')
        ax.axis('off')
    fig.savefig(out / 'cortes_centrais_sirt.png', dpi=170)
    plt.close(fig)

    indices = np.linspace(25, 230, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10), constrained_layout=True)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(vol[idx], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Axial {idx}', fontsize=10)
        ax.axis('off')
    fig.savefig(out / 'cortes_axiais_sirt.png', dpi=160)
    plt.close(fig)

    fdk_file = base / 'reconstrucoes' / 'tigre' / 'comparacao_sentidos_fdk' / 'volume_fdk_positivo.npy'
    if fdk_file.exists():
        fdk = np.load(fdk_file, mmap_mode='r')
        if fdk.shape == vol.shape:
            # Escalas independentes: comparar morfologia, nao brilho absoluto.
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            for row, (volume, titulo) in enumerate([(fdk, 'FDK'), (vol, f'SIRT {args.iteracoes}')]):
                lo, hi = np.percentile(volume, [2, 99.5])
                for col, (sl, plano) in enumerate(zip(
                    [volume[128], volume[:, 128, :], volume[:, :, 128]],
                    ['Axial', 'Coronal', 'Sagital'])):
                    axes[row, col].imshow(sl, cmap='gray', vmin=lo, vmax=hi)
                    axes[row, col].set_title(f'{titulo} — {plano}')
                    axes[row, col].axis('off')
            fig.savefig(out / 'comparacao_visual_fdk_sirt.png', dpi=170)
            plt.close(fig)

    info = {
        'algoritmo': 'TIGRE SIRT', 'iteracoes': args.iteracoes,
        'projecoes': str(proj_file), 'angulos': str(ang_file),
        'sentido_angular': 'positivo', 'DSO_mm': 640., 'DSD_mm': 835.,
        'pixel_detector_reduzido_mm': 0.34486, 'detector_px': [851, 676],
        'volume_voxels': [256, 256, 256], 'campo_mm': [150., 150., 150.],
        'COR_mm': 0., 'offDetector_mm': [0., 0.],
        'observacao': 'Projecoes CR exploratorias, nao calibradas em atenuacao; resultado nao quantitativo.',
        'min': float(vol.min()), 'max': float(vol.max())
    }
    (out / 'metadados.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SIRT concluido. Resultados em:', out, flush=True)


if __name__ == '__main__':
    main()
