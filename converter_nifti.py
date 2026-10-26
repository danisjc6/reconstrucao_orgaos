from pathlib import Path

import nibabel as nib
import numpy as np


INPUT_PATH = Path("resultados/volume_coracao.npy")
OUTPUT_PATH = Path("resultados/volume_coracao.nii.gz")

volume = np.load(INPUT_PATH).astype(np.float32)

tamanho_voxel = 220.0 / 256.0

affine = np.array([
    [tamanho_voxel, 0, 0, 0],
    [0, tamanho_voxel, 0, 0],
    [0, 0, tamanho_voxel, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

imagem_nifti = nib.Nifti1Image(volume, affine)
nib.save(imagem_nifti, OUTPUT_PATH)

print("Arquivo criado:", OUTPUT_PATH.resolve())
print("Tamanho do voxel:", tamanho_voxel, "mm")