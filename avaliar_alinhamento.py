from pathlib import Path
from datetime import datetime

import argparse
import csv
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from pydicom.pixels import apply_modality_lut


PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# ARGUMENTOS E CONFIGURAÇÃO
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia o alinhamento das projeções usando "
            "uma moeda radiopaca."
        )
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help=(
            "Pasta contendo config.json e a pasta dicom."
        )
    )

    parser.add_argument(
        "--reutilizar",
        action="store_true",
        help=(
            "Reutiliza marcacoes_moeda.json sem abrir "
            "novamente as imagens."
        )
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
        return json.load(file)


def create_angles(config):
    number = int(config["numero_projecoes"])
    angle_config = config["angulos"]

    if angle_config["tipo"] == "uniforme":
        initial = float(
            angle_config["inicial_graus"]
        )

        total = float(
            angle_config["total_graus"]
        )

        return np.linspace(
            initial,
            initial + total,
            number,
            endpoint=False,
            dtype=np.float64
        )

    if angle_config["tipo"] == "lista":
        angles = np.asarray(
            angle_config["valores_graus"],
            dtype=np.float64
        )

        if len(angles) != number:
            raise ValueError(
                "Número de ângulos diferente "
                "do número de projeções."
            )

        return angles

    raise ValueError(
        "Tipo de distribuição angular desconhecido."
    )


def dicom_paths(dicom_dir, number):
    paths = [
        dicom_dir / f"{index}.dcm"
        for index in range(1, number + 1)
    ]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "DICOMs ausentes:\n"
            + "\n".join(missing)
        )

    return paths


# ============================================================
# LEITURA DOS DICOM
# ============================================================

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

    if rotation_k != 0:
        image = np.rot90(
            image,
            k=rotation_k
        )

    return image


def normalize_for_display(image):
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    return np.clip(
        (image - low) / (high - low + 1e-6),
        0,
        1
    )


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
            "Pixel Spacing não encontrado."
        )

    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1])

    if rotation_k % 2 != 0:
        row_spacing, col_spacing = (
            col_spacing,
            row_spacing
        )

    return row_spacing, col_spacing


# ============================================================
# MARCAÇÃO MANUAL
# ============================================================

def request_quality(filename):
    while True:
        response = input(
            f"{filename} — qualidade "
            "[c=completa, p=parcial]: "
        ).strip().lower()

        if response == "c":
            return "completa"

        if response == "p":
            return "parcial"

        print("Digite apenas c ou p.")


def mark_coin(paths, angles_deg, rotation_k):
    markings = {}

    print(
        "\nINSTRUÇÕES:\n"
        "- Clique uma vez no centro da moeda.\n"
        "- Se a moeda estiver ausente ou impossível de medir, "
        "pressione Enter sem clicar.\n"
        "- Depois do clique, informe c ou p no terminal.\n"
    )

    for path, angle in zip(paths, angles_deg):
        image = read_dicom(
            path,
            rotation_k
        )

        display_image = normalize_for_display(
            image
        )

        rows, cols = image.shape
        center_x = (cols - 1) / 2
        center_y = (rows - 1) / 2

        fig, axis = plt.subplots(
            figsize=(10, 9)
        )

        axis.imshow(
            display_image,
            cmap="gray"
        )

        axis.axvline(
            center_x,
            color="red",
            linestyle="--",
            linewidth=0.8,
            label="Centro horizontal"
        )

        axis.axhline(
            center_y,
            color="cyan",
            linestyle="--",
            linewidth=0.8,
            label="Centro vertical"
        )

        axis.set_title(
            f"{path.name} — {angle:.1f}°\n"
            "Clique no centro da moeda ou pressione Enter"
        )

        axis.legend(
            loc="lower right"
        )

        points = plt.ginput(
            1,
            timeout=0,
            show_clicks=True
        )

        plt.close(fig)

        if not points:
            markings[path.name] = {
                "angulo_graus": float(angle),
                "x": None,
                "y": None,
                "qualidade": "ausente"
            }

            print(
                f"{path.name}: moeda ausente."
            )

            continue

        x, y = points[0]

        quality = request_quality(
            path.name
        )

        markings[path.name] = {
            "angulo_graus": float(angle),
            "x": float(x),
            "y": float(y),
            "qualidade": quality
        }

        print(
            f"{path.name}: "
            f"x={x:.2f}, y={y:.2f}, "
            f"qualidade={quality}"
        )

    return markings


def save_markings(
    markings,
    markings_path,
    config
):
    if markings_path.exists():
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_path = markings_path.with_name(
            f"marcacoes_moeda_backup_{timestamp}.json"
        )

        shutil.copy2(
            markings_path,
            backup_path
        )

        print(
            "Backup das marcações anteriores:",
            backup_path
        )

    content = {
        "experimento": config["nome"],
        "criado_em": datetime.now().isoformat(
            timespec="seconds"
        ),
        "marcacoes": markings
    }

    with markings_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Marcações salvas em:",
        markings_path
    )


def load_markings(markings_path):
    if not markings_path.exists():
        raise FileNotFoundError(
            f"Marcações não encontradas: {markings_path}"
        )

    with markings_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        content = json.load(file)

    if "marcacoes" not in content:
        raise ValueError(
            "Formato de marcacoes_moeda.json inválido. "
            "Execute sem --reutilizar para criar novas marcações."
        )

    return content["marcacoes"]


# ============================================================
# AJUSTE DA TRAJETÓRIA
# ============================================================

def quality_weight(quality):
    if quality == "completa":
        return 1.0

    if quality == "parcial":
        return 0.25

    return 0.0


def fit_harmonic(
    angles_deg,
    values,
    weights
):
    """
    Ajusta uma trajetória periódica com dois harmônicos:

        valor(theta) =
            c
            + a1*cos(theta)
            + b1*sin(theta)
            + a2*cos(2*theta)
            + b2*sin(2*theta)

    O segundo harmônico permite representar melhor os efeitos
    de perspectiva da geometria cone-beam.
    """

    angles_rad = np.deg2rad(
        angles_deg
    )

    valid = (
        np.isfinite(values)
        & (weights > 0)
    )

    valid_count = np.count_nonzero(
        valid
    )

    if valid_count < 8:
        raise ValueError(
            "São necessárias pelo menos oito marcações "
            "válidas para o ajuste com dois harmônicos."
        )

    # Matriz usada para calcular os coeficientes
    # somente a partir das marcações válidas.
    design_valid = np.column_stack([
        np.ones(valid_count),
        np.cos(angles_rad[valid]),
        np.sin(angles_rad[valid]),
        np.cos(2.0 * angles_rad[valid]),
        np.sin(2.0 * angles_rad[valid])
    ])

    sqrt_weights = np.sqrt(
        weights[valid]
    )

    weighted_design = (
        design_valid
        * sqrt_weights[:, None]
    )

    weighted_values = (
        values[valid]
        * sqrt_weights
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=None
    )

    # Matriz usada para prever a trajetória
    # em todos os 50 ângulos.
    design_all = np.column_stack([
        np.ones(len(angles_rad)),
        np.cos(angles_rad),
        np.sin(angles_rad),
        np.cos(2.0 * angles_rad),
        np.sin(2.0 * angles_rad)
    ])

    predicted = (
        design_all @ coefficients
    )

    return coefficients, predicted, valid


def robust_threshold(values):
    valid = values[np.isfinite(values)]

    if len(valid) == 0:
        return np.nan

    median = np.median(valid)

    mad = np.median(
        np.abs(valid - median)
    )

    if mad < 1e-9:
        return median + 1.0

    return median + 3.0 * 1.4826 * mad


def analyze_markings(
    paths,
    angles_deg,
    markings,
    row_spacing,
    col_spacing,
    magnification
):
    x_observed = []
    y_observed = []
    qualities = []
    weights = []

    for path in paths:
        marking = markings[path.name]
        quality = marking["qualidade"]

        x = marking["x"]
        y = marking["y"]

        x_observed.append(
            np.nan if x is None else float(x)
        )

        y_observed.append(
            np.nan if y is None else float(y)
        )

        qualities.append(quality)
        weights.append(
            quality_weight(quality)
        )

    x_observed = np.asarray(
        x_observed,
        dtype=np.float64
    )

    y_observed = np.asarray(
        y_observed,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    _, x_predicted, _ = fit_harmonic(
        angles_deg,
        x_observed,
        weights
    )

    _, y_predicted, _ = fit_harmonic(
        angles_deg,
        y_observed,
        weights
    )

    residual_x = (
        x_observed - x_predicted
    )

    residual_y = (
        y_observed - y_predicted
    )

    residual_x_object_mm = (
        residual_x
        * col_spacing
        / magnification
    )

    residual_y_object_mm = (
        residual_y
        * row_spacing
        / magnification
    )

    residual_total_mm = np.sqrt(
        residual_x_object_mm ** 2
        + residual_y_object_mm ** 2
    )

    threshold_mm = robust_threshold(
        residual_total_mm
    )

    outlier = (
        np.isfinite(residual_total_mm)
        & (residual_total_mm > threshold_mm)
    )

    return {
        "arquivos": [
            path.name
            for path in paths
        ],
        "angulos": angles_deg,
        "qualidades": qualities,
        "pesos": weights,
        "x_observado": x_observed,
        "y_observado": y_observed,
        "x_previsto": x_predicted,
        "y_previsto": y_predicted,
        "residuo_x_px": residual_x,
        "residuo_y_px": residual_y,
        "residuo_x_mm": residual_x_object_mm,
        "residuo_y_mm": residual_y_object_mm,
        "residuo_total_mm": residual_total_mm,
        "limite_outlier_mm": threshold_mm,
        "outlier": outlier
    }


# ============================================================
# SALVAR RELATÓRIO
# ============================================================

def save_csv(analysis, output_dir):
    output_path = (
        output_dir
        / "medidas_alinhamento.csv"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "arquivo",
            "angulo_graus",
            "qualidade",
            "peso",
            "x_observado_px",
            "y_observado_px",
            "x_previsto_px",
            "y_previsto_px",
            "residuo_x_px",
            "residuo_y_px",
            "residuo_x_orgao_mm",
            "residuo_y_orgao_mm",
            "residuo_total_orgao_mm",
            "possivel_outlier"
        ])

        for index in range(
            len(analysis["arquivos"])
        ):
            writer.writerow([
                analysis["arquivos"][index],
                analysis["angulos"][index],
                analysis["qualidades"][index],
                analysis["pesos"][index],
                analysis["x_observado"][index],
                analysis["y_observado"][index],
                analysis["x_previsto"][index],
                analysis["y_previsto"][index],
                analysis["residuo_x_px"][index],
                analysis["residuo_y_px"][index],
                analysis["residuo_x_mm"][index],
                analysis["residuo_y_mm"][index],
                analysis["residuo_total_mm"][index],
                bool(analysis["outlier"][index])
            ])

    print("CSV salvo em:", output_path)


def save_plots(analysis, output_dir):
    angles = analysis["angulos"]

    complete = np.array([
        quality == "completa"
        for quality in analysis["qualidades"]
    ])

    partial = np.array([
        quality == "parcial"
        for quality in analysis["qualidades"]
    ])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 13)
    )

    # Trajetória horizontal
    axes[0].plot(
        angles,
        analysis["x_previsto"],
        color="blue",
        label="Trajetória ajustada"
    )

    axes[0].scatter(
        angles[complete],
        analysis["x_observado"][complete],
        color="green",
        label="Moeda completa"
    )

    axes[0].scatter(
        angles[partial],
        analysis["x_observado"][partial],
        color="orange",
        label="Moeda parcial"
    )

    axes[0].set_title(
        "Trajetória horizontal da moeda"
    )

    axes[0].set_xlabel(
        "Ângulo (graus)"
    )

    axes[0].set_ylabel(
        "Posição x (pixels)"
    )

    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Trajetória vertical
    axes[1].plot(
        angles,
        analysis["y_previsto"],
        color="blue",
        label="Trajetória ajustada"
    )

    axes[1].scatter(
        angles[complete],
        analysis["y_observado"][complete],
        color="green",
        label="Moeda completa"
    )

    axes[1].scatter(
        angles[partial],
        analysis["y_observado"][partial],
        color="orange",
        label="Moeda parcial"
    )

    axes[1].set_title(
        "Trajetória vertical da moeda"
    )

    axes[1].set_xlabel(
        "Ângulo (graus)"
    )

    axes[1].set_ylabel(
        "Posição y (pixels)"
    )

    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Resíduo total
    axes[2].plot(
        angles,
        analysis["residuo_total_mm"],
        marker="o",
        color="black",
        label="Resíduo"
    )

    axes[2].axhline(
        analysis["limite_outlier_mm"],
        color="red",
        linestyle="--",
        label="Limite robusto"
    )

    outlier = analysis["outlier"]

    axes[2].scatter(
        angles[outlier],
        analysis["residuo_total_mm"][outlier],
        color="red",
        s=70,
        label="Possível deslocamento"
    )

    axes[2].set_title(
        "Desvio em relação à trajetória esperada"
    )

    axes[2].set_xlabel(
        "Ângulo (graus)"
    )

    axes[2].set_ylabel(
        "Desvio no plano do órgão (mm)"
    )

    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "relatorio_alinhamento.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Gráfico salvo em:", output_path)


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    args = parse_arguments()

    experiment_dir = (
        args.experiment_dir.resolve()
    )

    config = load_config(
        experiment_dir
    )

    organ_name = config["nome"]
    number = int(
        config["numero_projecoes"]
    )

    rotation_k = int(
        config["orientacao"][
            "rotacao_imagem_90"
        ]
    )

    dicom_dir = (
        experiment_dir / "dicom"
    )

    paths = dicom_paths(
        dicom_dir,
        number
    )

    angles_deg = create_angles(
        config
    )

    markings_path = (
        experiment_dir
        / "marcacoes_moeda.json"
    )

    if args.reutilizar:
        markings = load_markings(
            markings_path
        )

        print(
            "Reutilizando:",
            markings_path
        )

    else:
        markings = mark_coin(
            paths,
            angles_deg,
            rotation_k
        )

        save_markings(
            markings,
            markings_path,
            config
        )

    row_spacing, col_spacing = (
        get_pixel_spacing(
            paths[0],
            rotation_k
        )
    )

    geometry = config["geometria"]

    source_origin = float(
        geometry["distancia_fonte_orgao_mm"]
    )

    origin_detector = float(
        geometry[
            "distancia_orgao_detector_mm"
        ]
    )

    magnification = (
        source_origin + origin_detector
    ) / source_origin

    print(
        "Magnificação:",
        f"{magnification:.6f}"
    )

    analysis = analyze_markings(
        paths,
        angles_deg,
        markings,
        row_spacing,
        col_spacing,
        magnification
    )

    output_dir = (
        PROJECT_DIR
        / "resultados"
        / organ_name
        / "alinhamento"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        markings_path,
        output_dir
        / "marcacoes_moeda.json"
    )

    save_csv(
        analysis,
        output_dir
    )

    save_plots(
        analysis,
        output_dir
    )

    valid_count = sum(
        quality != "ausente"
        for quality
        in analysis["qualidades"]
    )

    outlier_count = int(
        np.count_nonzero(
            analysis["outlier"]
        )
    )

    print("\nAvaliação concluída.")
    print(
        "Marcações válidas:",
        f"{valid_count}/{number}"
    )
    print(
        "Possíveis deslocamentos:",
        outlier_count
    )
    print(
        "Limite robusto:",
        f"{analysis['limite_outlier_mm']:.3f} mm"
    )
    print(
        "Resultados:",
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()