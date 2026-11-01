from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


FRACTIONS = (0.25, 0.375, 0.50, 0.625, 0.75)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compara vários cortes de dois volumes usando "
            "uma janela de intensidade comum."
        )
    )
    parser.add_argument("controle", type=Path)
    parser.add_argument("corrigido", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "resultados/comparacao_multiplos_cortes"
        )
    )
    return parser.parse_args()


def load_volume(path):
    if not path.exists():
        raise FileNotFoundError(path)

    volume = np.load(path, allow_pickle=False)

    if volume.ndim != 3:
        raise ValueError(
            f"{path}: esperado volume 3D; "
            f"obtido {volume.shape}"
        )

    if not np.all(np.isfinite(volume)):
        raise ValueError(
            f"{path}: contém NaN ou infinito"
        )

    return np.asarray(volume, dtype=np.float32)


def sample_volume(volume, maximum=2_000_000):
    flat = volume.ravel()
    step = max(1, flat.size // maximum)
    return flat[::step]


def common_window(control, corrected):
    values = np.concatenate([
        sample_volume(control),
        sample_volume(corrected)
    ])

    low, high = np.percentile(
        values,
        [1.0, 99.5]
    )

    if high <= low:
        high = low + 1e-6

    return float(low), float(high)


def slice_indexes(size):
    return [
        min(
            size - 1,
            max(0, int(round((size - 1) * fraction)))
        )
        for fraction in FRACTIONS
    ]


def extract_slice(volume, axis, index):
    if axis == 0:
        return volume[index, :, :]

    if axis == 1:
        return volume[:, index, :]

    if axis == 2:
        return volume[:, :, index]

    raise ValueError(f"Eixo inválido: {axis}")


def difference_limit(control, corrected, axes):
    samples = []

    for axis in axes:
        for index in slice_indexes(control.shape[axis]):
            difference = np.abs(
                extract_slice(corrected, axis, index)
                - extract_slice(control, axis, index)
            )

            step = max(
                1,
                difference.size // 100_000
            )

            samples.append(
                difference.ravel()[::step]
            )

    values = np.concatenate(samples)
    limit = float(np.percentile(values, 99.0))

    return max(limit, 1e-6)


def save_plane_comparison(
    control,
    corrected,
    axis,
    plane_name,
    output_path,
    low,
    high,
    difference_high
):
    indexes = slice_indexes(control.shape[axis])

    fig, axes = plt.subplots(
        3,
        len(indexes),
        figsize=(18, 10)
    )

    last_difference_image = None

    for column, (fraction, index) in enumerate(
        zip(FRACTIONS, indexes)
    ):
        control_slice = extract_slice(
            control,
            axis,
            index
        )

        corrected_slice = extract_slice(
            corrected,
            axis,
            index
        )

        difference = np.abs(
            corrected_slice - control_slice
        )

        axes[0, column].imshow(
            control_slice,
            cmap="gray",
            vmin=low,
            vmax=high
        )

        axes[1, column].imshow(
            corrected_slice,
            cmap="gray",
            vmin=low,
            vmax=high
        )

        last_difference_image = axes[
            2,
            column
        ].imshow(
            difference,
            cmap="inferno",
            vmin=0.0,
            vmax=difference_high
        )

        axes[0, column].set_title(
            f"{fraction * 100:.1f}%\níndice {index}"
        )

        for row in range(3):
            axes[row, column].axis("off")

    axes[0, 0].set_ylabel(
        "Controle",
        fontsize=13
    )

    axes[1, 0].set_ylabel(
        "cone_vec",
        fontsize=13
    )

    axes[2, 0].set_ylabel(
        "Diferença",
        fontsize=13
    )

    fig.suptitle(
        f"{plane_name}: comparação em múltiplas posições",
        fontsize=15
    )

    fig.colorbar(
        last_difference_image,
        ax=axes[2, :].tolist(),
        fraction=0.018,
        pad=0.015,
        label="Diferença absoluta"
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.94,
        bottom=0.03,
        top=0.90,
        wspace=0.03,
        hspace=0.08
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return indexes


def save_index_table(rows, output_path):
    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "plano_matriz",
                "eixo_numpy",
                "percentual",
                "indice"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()

    control_path = args.controle.resolve()
    corrected_path = args.corrigido.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    control = load_volume(control_path)
    corrected = load_volume(corrected_path)

    if control.shape != corrected.shape:
        raise ValueError(
            "Volumes com formas diferentes: "
            f"{control.shape} e {corrected.shape}"
        )

    low, high = common_window(
        control,
        corrected
    )

    planes = [
        (0, "Axial da matriz", "axial"),
        (1, "Coronal da matriz", "coronal"),
        (2, "Sagital da matriz", "sagital")
    ]

    difference_high = difference_limit(
        control,
        corrected,
        [axis for axis, _, _ in planes]
    )

    index_rows = []

    for axis, plane_name, filename in planes:
        indexes = save_plane_comparison(
            control,
            corrected,
            axis,
            plane_name,
            output_dir
            / f"comparacao_{filename}.png",
            low,
            high,
            difference_high
        )

        for fraction, index in zip(
            FRACTIONS,
            indexes
        ):
            index_rows.append({
                "plano_matriz": plane_name,
                "eixo_numpy": axis,
                "percentual": fraction,
                "indice": index
            })

    save_index_table(
        index_rows,
        output_dir / "indices_avaliados.csv"
    )

    print("Comparação concluída.")
    print("Janela comum:", low, "a", high)
    print("Resultados:", output_dir)
    print(
        "Observação: os nomes dos planos referem-se "
        "aos eixos da matriz e não substituem "
        "orientação anatômica validada."
    )


if __name__ == "__main__":
    main()
