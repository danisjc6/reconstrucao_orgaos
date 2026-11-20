from pathlib import Path

import numpy as np
import tigre
import tigre.algorithms as algs

base = Path(__file__).resolve().parent / "resultados" / "estomago_canino_01"
arquivo = base / "entrada_tigre_nominal" / "entrada.npz"

if not arquivo.is_file():
    raise SystemExit(f"Entrada não encontrada: {arquivo}")

with np.load(arquivo) as entrada:
    proj = np.ascontiguousarray(entrada["projecoes"], dtype=np.float32)
    angulos = np.ascontiguousarray(entrada["angulos_rad"], dtype=np.float32)
    pixel_mm = np.asarray(entrada["pixel_mm"], dtype=np.float32)
    dso = float(entrada["dso"].item())
    dsd = float(entrada["dsd"].item())

if proj.ndim != 3 or proj.shape[0] != 49 or angulos.shape != (49,):
    raise ValueError(
        f"Esperadas 49 projeções e 49 ângulos: {proj.shape}, {angulos.shape}"
    )

geo = tigre.geometry()
geo.mode = "cone"
geo.DSO, geo.DSD = dso, dsd
geo.nDetector = np.array(proj.shape[1:], dtype=np.int32)
geo.dDetector = pixel_mm
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array([256, 256, 256], dtype=np.int32)
geo.sVoxel = np.full(3, 220.0, dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.offOrigin = np.zeros(3, dtype=np.float32)
geo.offDetector = np.zeros(2, dtype=np.float32)
geo.rotDetector = np.zeros(3, dtype=np.float32)
geo.COR = 0.0
geo.accuracy = 0.5

print(f"Projeções: {proj.shape} | DSO: {dso:g} mm | DSD: {dsd:g} mm")
volume = algs.sirt(proj, geo, angulos, 15)

saida = base / "teste_tigre_sirt15" / "volume.npy"
saida.parent.mkdir(parents=True, exist_ok=True)
np.save(saida, volume)
print("Volume salvo em:", saida)