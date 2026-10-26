from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compara dois volumes ASTRA sem registro adicional."
    )
    parser.add_argument("controle", type=Path)
    parser.add_argument("corrigido", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados/comparacao_controle_cone_vec")
    )
    return parser.parse_args()


def load_volume(path):
    if not path.exists():
        raise FileNotFoundError(path)
    volume = np.load(path, allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"{path}: esperado volume 3D; obtido {volume.shape}")
    if not np.all(np.isfinite(volume)):
        raise ValueError(f"{path}: volume contém NaN ou infinito")
    return np.asarray(volume, dtype=np.float32)


def sampled_values(volume, maximum=2_000_000):
    flat = volume.ravel()
    step = max(1, flat.size // maximum)
    return flat[::step]


def common_window(control, corrected):
    samples = np.concatenate([
        sampled_values(control),
        sampled_values(corrected)
    ])
    low, high = np.percentile(samples, [1.0, 99.5])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def central_slices(volume):
    z, y, x = (size // 2 for size in volume.shape)
    return [
        volume[z, :, :],
        volume[:, y, :],
        volume[:, :, x]
    ]


def save_comparison(control, corrected, output_path):
    low, high = common_window(control, corrected)
    control_slices = central_slices(control)
    corrected_slices = central_slices(corrected)
    plane_names = ["Axial", "Coronal", "Sagital"]

    differences = [
        np.abs(after - before)
        for before, after in zip(control_slices, corrected_slices)
    ]
    difference_samples = np.concatenate([
        image.ravel()[::max(1, image.size // 200_000)]
        for image in differences
    ])
    difference_high = float(np.percentile(difference_samples, 99.0))
    difference_high = max(difference_high, 1e-6)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    for row, plane_name in enumerate(plane_names):
        axes[row, 0].imshow(
            control_slices[row], cmap="gray", vmin=low, vmax=high
        )
        axes[row, 1].imshow(
            corrected_slices[row], cmap="gray", vmin=low, vmax=high
        )
        difference_image = axes[row, 2].imshow(
            differences[row],
            cmap="inferno",
            vmin=0.0,
            vmax=difference_high
        )

        axes[row, 0].set_ylabel(plane_name, fontsize=12)
        for column in range(3):
            axes[row, column].axis("off")

    axes[0, 0].set_title("Controle: cone")
    axes[0, 1].set_title("Corrigido: cone_vec")
    axes[0, 2].set_title("Diferença absoluta")

    fig.colorbar(
        difference_image,
        ax=axes[:, 2].tolist(),
        fraction=0.025,
        pad=0.02,
        label="Diferença de intensidade"
    )

    fig.suptitle(
        f"Comparação com janela comum: {low:.5g} a {high:.5g}",
        fontsize=14
    )
    fig.subplots_adjust(
        left=0.04, right=0.93, bottom=0.03, top=0.93,
        wspace=0.04, hspace=0.08
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return low, high


def central_mask(shape, fraction=0.60):
    mask = np.zeros(shape, dtype=bool)
    slices = []
    for size in shape:
        length = int(round(size * fraction))
        start = (size - length) // 2
        slices.append(slice(start, start + length))
    mask[tuple(slices)] = True
    return mask


def gradient_energy(volume, mask):
    gradients = np.gradient(volume.astype(np.float64, copy=False))
    squared = sum(component * component for component in gradients)
    return float(np.mean(squared[mask]))


def calculate_metrics(control, corrected, low, high):
    difference = corrected.astype(np.float64) - control.astype(np.float64)
    absolute_difference = np.abs(difference)
    central = central_mask(control.shape)
    peripheral = ~central
    dynamic_range = max(high - low, 1e-12)

    control_flat = control.ravel().astype(np.float64)
    corrected_flat = corrected.ravel().astype(np.float64)
    correlation = float(np.corrcoef(control_flat, corrected_flat)[0, 1])

    return {
        "forma_volume": "x".join(map(str, control.shape)),
        "janela_comum_min": low,
        "janela_comum_max": high,
        "media_controle": float(np.mean(control)),
        "media_corrigido": float(np.mean(corrected)),
        "desvio_padrao_controle": float(np.std(control)),
        "desvio_padrao_corrigido": float(np.std(corrected)),
        "mae_global": float(np.mean(absolute_difference)),
        "rmse_global": float(np.sqrt(np.mean(difference * difference))),
        "rmse_normalizado_janela": float(
            np.sqrt(np.mean(difference * difference)) / dynamic_range
        ),
        "correlacao_global": correlation,
        "mae_regiao_central": float(np.mean(absolute_difference[central])),
        "mae_regiao_periferica": float(np.mean(absolute_difference[peripheral])),
        "energia_gradiente_central_controle": gradient_energy(control, central),
        "energia_gradiente_central_corrigido": gradient_energy(corrected, central)
    }


def save_metrics(metrics, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metrica", "valor"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def main():
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    control = load_volume(args.controle.resolve())
    corrected = load_volume(args.corrigido.resolve())

    if control.shape != corrected.shape:
        raise ValueError(
            f"Volumes com formas diferentes: {control.shape} e {corrected.shape}"
        )

    figure_path = output_dir / "comparacao_cortes_centrais.png"
    low, high = save_comparison(control, corrected, figure_path)
    metrics = calculate_metrics(control, corrected, low, high)
    metrics_path = output_dir / "metricas_comparacao.csv"
    save_metrics(metrics, metrics_path)

    print("Comparação concluída.")
    print("Figura:", figure_path)
    print("Métricas:", metrics_path)
    print("Observação: as métricas descrevem diferenças; sem referência externa,")
    print("elas não comprovam fidelidade anatômica superior.")


if __name__ == "__main__":
    main()
