from pathlib import Path
from datetime import datetime

import argparse
import json
import shutil

import astra
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from pydicom.pixels import apply_modality_lut
from scipy.ndimage import zoom  

PROJECT_DIR = Path(__file__).resolve().parent

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Reconstrução tomográfica 3D de órgãos."
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help=(
            "Pasta do experimento contendo config.json "
            "e a pasta dicom."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome opcional para a execução."
    )

    return parser.parse_args()


def load_config(experiment_dir):
    config_path = experiment_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuração não encontrada: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    return config, config_path


def create_output_dir(config, run_name=None):
    organ_name = config["nome"]

    if run_name is None:
        algorithm = config["reconstrucao"]["algoritmo"]
        iterations = config["reconstrucao"].get(
            "iteracoes",
            1
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        if algorithm == "SIRT3D_CUDA":
            method_name = (
                f"sirt3d_{iterations}iter"
            )
        else:
            method_name = algorithm.lower()

        run_name = (
            f"{method_name}_{timestamp}"
        )

    output_dir = (
    PROJECT_DIR
    / "resultados"
    / organ_name
    / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    return output_dir

# ============================================================
# LEITURA DOS DICOM
# ============================================================

def dicom_paths(dicom_dir, number_of_projections):
    paths = [
        dicom_dir / f"{number}.dcm"
        for number in range(
            1,
            number_of_projections + 1
        )
    ]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Arquivos DICOM ausentes:\n"
            + "\n".join(missing)
        )

    return paths


def read_dicom(path, rotation_k):
    dataset = pydicom.dcmread(path)

    image = dataset.pixel_array
    image = apply_modality_lut(
        image,
        dataset
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    photometric = getattr(
        dataset,
        "PhotometricInterpretation",
        "MONOCHROME2"
    )

    if photometric == "MONOCHROME1":
        image = image.max() - image

    # A mesma rotação é aplicada a todas as projeções.
    if rotation_k != 0:
        image = np.rot90(
            image,
            k=rotation_k
        )

    return image


def load_projections(
    dicom_dir,
    number_of_projections,
    rotation_k
):
    images = []

    paths = dicom_paths(
        dicom_dir,
        number_of_projections
    )

    expected_shape = None

    for path in paths:
        image = read_dicom(
            path,
            rotation_k
        )

        if expected_shape is None:
            expected_shape = image.shape

        if image.shape != expected_shape:
            raise ValueError(
                f"{path.name}: matriz {image.shape}; "
                f"esperada {expected_shape}"
            )

        print(
            f"Lendo {path.name}: "
            f"{image.shape[0]} linhas × "
            f"{image.shape[1]} colunas"
        )

        images.append(image)

    projections = np.stack(
        images,
        axis=0
    ).astype(np.float32)

    print(
        "Matriz original:",
        projections.shape
    )

    return projections, paths

def get_pixel_spacing(dicom_path, rotation_k):
    dataset = pydicom.dcmread(
        dicom_path,
        stop_before_pixels=True
    )

    if hasattr(dataset, "ImagerPixelSpacing"):
        spacing = dataset.ImagerPixelSpacing

    elif hasattr(dataset, "PixelSpacing"):
        spacing = dataset.PixelSpacing

    else:
        raise ValueError(
            f"Pixel Spacing ausente em {dicom_path.name}"
        )

    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1])

    # Uma rotação de 90° ou 270° troca linhas e colunas.
    if rotation_k % 2 != 0:
        row_spacing, col_spacing = (
            col_spacing,
            row_spacing
        )

    print(
        "Pixel Spacing após orientação:",
        f"linha={row_spacing:.6f} mm;",
        f"coluna={col_spacing:.6f} mm"
    )

    return row_spacing, col_spacing

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocess(
    projections,
    downsample_factor,
    lower_percentile,
    upper_percentile
):
    low = np.percentile(
        projections,
        lower_percentile
    )

    high = np.percentile(
        projections,
        upper_percentile
    )

    projections = np.clip(
        projections,
        low,
        high
    )

    projections = (
        projections - low
    ) / (
        high - low + 1e-6
    )

    scale = 1.0 / downsample_factor

    projections = zoom(
        projections,
        zoom=(1.0, scale, scale),
        order=1,
        prefilter=False
    )

    projections = np.ascontiguousarray(
        projections,
        dtype=np.float32
    )

    print(
        "Matriz reduzida:",
        projections.shape
    )

    return projections



# ============================================================
# VISUALIZAÇÃO DAS PROJEÇÕES
# ============================================================

# ============================================================
# VISUALIZAÇÃO DAS PROJEÇÕES
# ============================================================

def save_projection_preview(
    projections,
    angles_deg,
    output_dir
):
    target_angles = [
        0.0,
        90.0,
        180.0,
        270.0
    ]

    indexes = [
        int(
            np.argmin(
                np.abs(angles_deg - target)
            )
        )
        for target in target_angles
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 6)
    )

    for axis, index in zip(
        axes,
        indexes
    ):
        axis.imshow(
            projections[index],
            cmap="gray"
        )

        axis.set_title(
            f"{index + 1}.dcm\n"
            f"{angles_deg[index]:.1f}°"
        )

        axis.axis("off")

    plt.tight_layout()

    output_path = (
        output_dir
        / "projecoes_principais.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Pré-visualização salva em:",
        output_path
    )

def create_angles(config):
    number = config["numero_projecoes"]
    angle_config = config["angulos"]

    if angle_config["tipo"] == "uniforme":
        initial = float(
            angle_config["inicial_graus"]
        )

        total = float(
            angle_config["total_graus"]
        )

        angles_deg = np.linspace(
            initial,
            initial + total,
            number,
            endpoint=False,
            dtype=np.float32
        )

    elif angle_config["tipo"] == "lista":
        angles_deg = np.asarray(
            angle_config["valores_graus"],
            dtype=np.float32
        )

    else:
        raise ValueError(
            "Tipo de ângulo desconhecido."
        )

    if len(angles_deg) != number:
        raise ValueError(
            "O número de ângulos não corresponde "
            "ao número de projeções."
        )

    sign = float(
        angle_config.get("sinal", -1.0)
    )

    angles_rad = (
        sign * np.deg2rad(angles_deg)
    )

    return angles_deg, angles_rad


# ============================================================
# GEOMETRIA E RECONSTRUÇÃO
# ============================================================

def reconstruct(
    projections,
    angles_rad,
    row_spacing,
    col_spacing,
    config
):
    number_of_angles, detector_rows, detector_cols = (
        projections.shape
    )

    if number_of_angles != len(angles_rad):
        raise ValueError(
            "O número de projeções não corresponde "
            "ao número de ângulos."
        )

    processing = config["processamento"]
    geometry = config["geometria"]
    volume_config = config["volume"]
    reconstruction_config = config["reconstrucao"]

    downsample_factor = int(
        processing["fator_reducao"]
    )

    detector_row_spacing = (
        row_spacing * downsample_factor
    )

    detector_col_spacing = (
        col_spacing * downsample_factor
    )

    source_origin = float(
        geometry["distancia_fonte_orgao_mm"]
    )

    origin_detector = float(
        geometry["distancia_orgao_detector_mm"]
    )

    volume_size = int(
        volume_config["tamanho_voxels"]
    )

    volume_extent = float(
        volume_config["extensao_mm"]
    )

    algorithm = reconstruction_config["algoritmo"]

    iterations = int(
        reconstruction_config.get(
            "iteracoes",
            1
        )
    )

    min_constraint = float(
        reconstruction_config.get(
            "restricao_minima",
            0.0
        )
    )

    print(
        "Detector reduzido:",
        f"{detector_rows} × {detector_cols}"
    )

    print(
        "Espaçamento efetivo do detector:",
        f"linha={detector_row_spacing:.6f} mm;",
        f"coluna={detector_col_spacing:.6f} mm"
    )

    print(
        "Geometria:",
        f"fonte–órgão={source_origin:.2f} mm;",
        f"órgão–detector={origin_detector:.2f} mm"
    )

    proj_geom = astra.create_proj_geom(
        "cone",
        detector_col_spacing,
        detector_row_spacing,
        detector_rows,
        detector_cols,
        angles_rad,
        source_origin,
        origin_detector
    )

    half_extent = volume_extent / 2.0

    vol_geom = astra.create_vol_geom(
        volume_size,
        volume_size,
        volume_size,
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        -half_extent,
        half_extent
    )

    # ASTRA espera:
    # (linhas do detector, ângulos, colunas do detector)
    astra_projections = np.transpose(
        projections,
        (1, 0, 2)
    )

    astra_projections = np.ascontiguousarray(
        astra_projections,
        dtype=np.float32
    )

    projection_id = None
    reconstruction_id = None
    algorithm_id = None

    try:
        projection_id = astra.data3d.create(
            "-proj3d",
            proj_geom,
            astra_projections
        )

        reconstruction_id = astra.data3d.create(
            "-vol",
            vol_geom,
            0
        )

        cfg = astra.astra_dict(algorithm)
        cfg["ProjectionDataId"] = projection_id
        cfg["ReconstructionDataId"] = reconstruction_id

        if algorithm == "SIRT3D_CUDA":
            cfg["option"] = {
                "MinConstraint": min_constraint
            }

        algorithm_id = astra.algorithm.create(cfg)

        print("Executando:", algorithm)

        if algorithm == "SIRT3D_CUDA":
            print("Iterações:", iterations)

            astra.algorithm.run(
                algorithm_id,
                iterations
            )

        else:
            # FDK_CUDA é executado uma única vez.
            astra.algorithm.run(
                algorithm_id
            )

        volume = astra.data3d.get(
            reconstruction_id
        )

        volume = np.asarray(
            volume,
            dtype=np.float32
        )

        print(
            "Volume reconstruído:",
            volume.shape
        )

        print(
            "Mínimo do volume:",
            float(volume.min())
        )

        print(
            "Máximo do volume:",
            float(volume.max())
        )

        return volume

    finally:
        if algorithm_id is not None:
            astra.algorithm.delete(
                algorithm_id
            )

        ids_to_delete = []

        if projection_id is not None:
            ids_to_delete.append(
                projection_id
            )

        if reconstruction_id is not None:
            ids_to_delete.append(
                reconstruction_id
            )

        if ids_to_delete:
            astra.data3d.delete(
                ids_to_delete
            )

# ============================================================
# SALVAR RESULTADOS
# ============================================================

def normalize_for_display(image):
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    return np.clip(
        (image - low) / (high - low + 1e-6),
        0,
        1
    )


def save_volume_results(
    volume,
    output_dir
):
    np.save(
        output_dir / "volume.npy",
        volume
    )

    z = volume.shape[0] // 2
    y = volume.shape[1] // 2
    x = volume.shape[2] // 2

    axial = normalize_for_display(volume[z, :, :])
    coronal = normalize_for_display(volume[:, y, :])
    sagittal = normalize_for_display(volume[:, :, x])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial, cmap="gray")
    axes[0].set_title("Corte axial")

    axes[1].imshow(coronal, cmap="gray")
    axes[1].set_title("Corte coronal")

    axes[2].imshow(sagittal, cmap="gray")
    axes[2].set_title("Corte sagital")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "cortes_centrais.png",
        dpi=200
    )
    plt.close()


def main():
    args = parse_arguments()

    experiment_dir = args.experiment_dir.resolve()
    dicom_dir = experiment_dir / "dicom"

    config, config_path = load_config(
        experiment_dir
    )

    output_dir = create_output_dir(
        config,
        args.run_name
    )

    print("Experimento:", config["nome"])
    print("ASTRA:", astra.__version__)
    print(
        "GPU disponível:",
        astra.use_cuda()
    )

    if not astra.use_cuda():
        raise RuntimeError(
            "O ASTRA não detectou CUDA."
        )

    number = int(
        config["numero_projecoes"]
    )

    rotation_k = int(
        config["orientacao"][
            "rotacao_imagem_90"
        ]
    )

    projections, paths = load_projections(
        dicom_dir,
        number,
        rotation_k
    )

    row_spacing, col_spacing = (
        get_pixel_spacing(
            paths[0],
            rotation_k
        )
    )

    angles_deg, angles_rad = (
        create_angles(config)
    )

    processing = config["processamento"]

    projections = preprocess(
        projections,
        int(processing["fator_reducao"]),
        float(processing["percentil_inferior"]),
        float(processing["percentil_superior"])
    )

    save_projection_preview(
        projections,
        angles_deg,
        output_dir
    )

    volume = reconstruct(
        projections,
        angles_rad,
        row_spacing,
        col_spacing,
        config
    )

    save_volume_results(
        volume,
        output_dir
    )

    shutil.copy2(
        config_path,
        output_dir / "parametros_utilizados.json"
    )

    print("\nReconstrução concluída.")
    print("Resultados:", output_dir.resolve())

if __name__ == "__main__":
    main()