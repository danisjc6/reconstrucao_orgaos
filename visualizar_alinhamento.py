from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut


def read_float(value):
    if value is None or value.strip() == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def read_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "sim", "yes"
    }


def load_config(experiment_dir):
    config_path = experiment_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuração não encontrada: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_measurements(csv_path):
    measurements = []

    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            measurements.append({
                "arquivo": row["arquivo"],
                "angulo": read_float(row["angulo_graus"]),
                "qualidade": row["qualidade"],
                "peso": read_float(row["peso"]),
                "x_observado": read_float(row["x_observado_px"]),
                "y_observado": read_float(row["y_observado_px"]),
                "x_previsto": read_float(row["x_previsto_px"]),
                "y_previsto": read_float(row["y_previsto_px"]),
                "residuo_x": read_float(row["residuo_x_px"]),
                "residuo_y": read_float(row["residuo_y_px"]),
                "residuo_mm": read_float(
                    row["residuo_total_orgao_mm"]
                ),
                "outlier": read_bool(row["possivel_outlier"])
            })

    return measurements


def read_dicom(path, rotation_k):
    dataset = pydicom.dcmread(path)

    image = apply_modality_lut(
        dataset.pixel_array,
        dataset
    )

    image = np.asarray(image, dtype=np.float32)

    photometric = getattr(
        dataset,
        "PhotometricInterpretation",
        "MONOCHROME2"
    )

    if photometric == "MONOCHROME1":
        image = image.max() - image

    if rotation_k:
        image = np.rot90(image, k=rotation_k)

    return image


def display_limits(image):
    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))

    return low, high


def point_inside_image(x, y, image):
    if not np.isfinite(x) or not np.isfinite(y):
        return False

    rows, columns = image.shape

    return 0 <= x < columns and 0 <= y < rows


def draw_markers(axis, measurement, image):
    x_obs = measurement["x_observado"]
    y_obs = measurement["y_observado"]
    x_pred = measurement["x_previsto"]
    y_pred = measurement["y_previsto"]

    predicted_inside = point_inside_image(
        x_pred,
        y_pred,
        image
    )

    observed_inside = point_inside_image(
        x_obs,
        y_obs,
        image
    )

    if predicted_inside:
        axis.scatter(
            x_pred,
            y_pred,
            marker="+",
            s=180,
            linewidths=2.5,
            color="deepskyblue",
            label="Previsto",
            zorder=5
        )

    if observed_inside:
        observed_color = (
            "lime"
            if measurement["qualidade"] == "completa"
            else "orange"
        )

        axis.scatter(
            x_obs,
            y_obs,
            marker="o",
            s=90,
            facecolors="none",
            edgecolors=observed_color,
            linewidths=2.5,
            label="Observado",
            zorder=6
        )

    if predicted_inside and observed_inside:
        arrow_color = (
            "red"
            if measurement["outlier"]
            else "yellow"
        )

        axis.annotate(
            "",
            xy=(x_obs, y_obs),
            xytext=(x_pred, y_pred),
            arrowprops={
                "arrowstyle": "->",
                "color": arrow_color,
                "linewidth": 2.0
            },
            zorder=7
        )


def define_zoom(measurement, image, margin=220):
    coordinates_x = []
    coordinates_y = []

    for x, y in [
        (
            measurement["x_observado"],
            measurement["y_observado"]
        ),
        (
            measurement["x_previsto"],
            measurement["y_previsto"]
        )
    ]:
        if point_inside_image(x, y, image):
            coordinates_x.append(x)
            coordinates_y.append(y)

    if not coordinates_x:
        return None

    center_x = float(np.mean(coordinates_x))
    center_y = float(np.mean(coordinates_y))

    rows, columns = image.shape

    x_min = max(0, center_x - margin)
    x_max = min(columns, center_x + margin)
    y_min = max(0, center_y - margin)
    y_max = min(rows, center_y + margin)

    return x_min, x_max, y_min, y_max


def save_control_image(
    image,
    measurement,
    output_path
):
    low, high = display_limits(image)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )

    title = (
        f'{measurement["arquivo"]} — '
        f'{measurement["angulo"]:.1f}° — '
        f'{measurement["qualidade"]}'
    )

    if np.isfinite(measurement["residuo_mm"]):
        title += (
            f' — desvio: '
            f'{measurement["residuo_mm"]:.2f} mm'
        )

    fig.suptitle(title, fontsize=14)

    # Imagem completa
    axes[0].imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high
    )

    draw_markers(
        axes[0],
        measurement,
        image
    )

    axes[0].set_title("Radiografia completa")
    axes[0].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        axes[0].legend(
            handles,
            labels,
            loc="upper right"
        )

    # Ampliação da região do marcador
    axes[1].imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high
    )

    draw_markers(
        axes[1],
        measurement,
        image
    )

    zoom_limits = define_zoom(
        measurement,
        image
    )

    if zoom_limits is None:
        axes[1].set_title(
            "Marcador ausente ou previsto fora do campo"
        )
        axes[1].axis("off")
    else:
        x_min, x_max, y_min, y_max = zoom_limits

        axes[1].set_xlim(x_min, x_max)

        # O eixo vertical da imagem cresce para baixo.
        axes[1].set_ylim(y_max, y_min)

        axes[1].set_title("Ampliação do marcador")
        axes[1].set_xlabel("x (pixels)")
        axes[1].set_ylabel("y (pixels)")
        axes[1].grid(
            color="white",
            alpha=0.25
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza posições observadas e previstas "
            "do marcador radiopaco."
        )
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Pasta do experimento"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV com as medidas de alinhamento"
    )

    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    csv_path = args.csv.resolve()

    config = load_config(experiment_dir)

    # Aceita os dois nomes, caso o config tenha sido reorganizado.
    rotation_k = int(
        config.get(
            "rotacao_imagem_90",
            config.get("rotation_k", 0)
        )
    )

    dicom_dir = experiment_dir / "dicom"

    output_dir = (
        Path("resultados")
        / experiment_dir.name
        / "alinhamento"
        / "controle_visual"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    measurements = load_measurements(csv_path)

    print(f"Projeções no CSV: {len(measurements)}")
    print(f"Rotação aplicada aos DICOMs: k={rotation_k}")

    for index, measurement in enumerate(
        measurements,
        start=1
    ):
        dicom_path = (
            dicom_dir
            / measurement["arquivo"]
        )

        if not dicom_path.exists():
            print(
                f"AVISO: DICOM ausente: {dicom_path}"
            )
            continue

        print(
            f'[{index:02d}/{len(measurements):02d}] '
            f'{measurement["arquivo"]} — '
            f'{measurement["angulo"]:.1f}°'
        )

        image = read_dicom(
            dicom_path,
            rotation_k
        )

        output_name = (
            f'{index:02d}_'
            f'{Path(measurement["arquivo"]).stem}_'
            f'{measurement["angulo"]:.1f}graus.png'
        )

        save_control_image(
            image,
            measurement,
            output_dir / output_name
        )

    print("\nControle visual concluído.")
    print("Resultados em:", output_dir.resolve())


if __name__ == "__main__":
    main()