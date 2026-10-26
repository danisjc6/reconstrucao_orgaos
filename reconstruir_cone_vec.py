from pathlib import Path
import argparse
import csv
import json
import shutil

import astra
import numpy as np

import reconstruir as base


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstrução experimental com correções do marcador "
            "representadas na geometria cone_vec."
        )
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Pasta contendo config.json e dicom/."
    )

    parser.add_argument(
        "--corrections-csv",
        type=Path,
        required=True,
        help="CSV definitivo produzido pela avaliação do alinhamento."
    )

    parser.add_argument(
        "--run-name",
        default="teste_cone_vec_correcao_moeda",
        help="Nome da pasta desta execução."
    )

    return parser.parse_args()


def read_float(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return np.nan

    return float(value)


def load_alignment_csv(csv_path):
    required_columns = {
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
        "residuo_total_orgao_mm"
    }

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV sem cabeçalho.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Colunas ausentes no CSV: "
                + ", ".join(sorted(missing))
            )

        records = []

        for row in reader:
            records.append({
                "arquivo": row["arquivo"].strip(),
                "angulo_graus": read_float(row["angulo_graus"]),
                "qualidade": row["qualidade"].strip().lower(),
                "peso": read_float(row["peso"]),
                "x_observado_px": read_float(row["x_observado_px"]),
                "y_observado_px": read_float(row["y_observado_px"]),
                "x_previsto_px": read_float(row["x_previsto_px"]),
                "y_previsto_px": read_float(row["y_previsto_px"]),
                "residuo_x_px": read_float(row["residuo_x_px"]),
                "residuo_y_px": read_float(row["residuo_y_px"]),
                "residuo_total_orgao_mm": read_float(
                    row["residuo_total_orgao_mm"]
                )
            })

    return records


def validate_alignment(records, paths, angles_deg):
    if len(records) != len(paths):
        raise ValueError(
            f"CSV contém {len(records)} linhas, mas o experimento "
            f"possui {len(paths)} projeções."
        )

    for index, (record, path, expected_angle) in enumerate(
        zip(records, paths, angles_deg),
        start=1
    ):
        if record["arquivo"] != path.name:
            raise ValueError(
                f"Linha {index}: arquivo {record['arquivo']!r}; "
                f"esperado {path.name!r}."
            )

        if not np.isclose(
            record["angulo_graus"],
            float(expected_angle),
            atol=1e-3
        ):
            raise ValueError(
                f"{path.name}: ângulo {record['angulo_graus']}°; "
                f"esperado {float(expected_angle)}°."
            )


def prepare_corrections(
    records,
    downsample_factor,
    row_spacing,
    col_spacing
):
    corrections = []

    for record in records:
        residual_x = record["residuo_x_px"]
        residual_y = record["residuo_y_px"]

        finite_residual = (
            np.isfinite(residual_x)
            and np.isfinite(residual_y)
        )

        # Decisão conservadora desta etapa experimental:
        # somente moedas completas entram na correção.
        apply_correction = (
            record["qualidade"] == "completa"
            and finite_residual
        )

        if apply_correction:
            reason = "aplicada_moeda_completa"

            # residual = observado - previsto.
            # Para fazer o raio do pixel observado corresponder à
            # posição prevista, o centro do detector é deslocado pelo
            # valor oposto ao residual.
            correction_col_original_px = -residual_x
            correction_row_original_px = -residual_y
        else:
            correction_col_original_px = 0.0
            correction_row_original_px = 0.0

            if record["qualidade"] == "parcial":
                reason = "nao_aplicada_moeda_parcial"
            elif record["qualidade"] == "ausente":
                reason = "nao_aplicada_moeda_ausente"
            elif not finite_residual:
                reason = "nao_aplicada_residuo_invalido"
            else:
                reason = "nao_aplicada_qualidade_nao_confiavel"

        corrections.append({
            **record,
            "aplicar_correcao": apply_correction,
            "motivo": reason,
            "correcao_coluna_px_original": correction_col_original_px,
            "correcao_linha_px_original": correction_row_original_px,
            "correcao_coluna_px_reduzida": (
                correction_col_original_px / downsample_factor
            ),
            "correcao_linha_px_reduzida": (
                correction_row_original_px / downsample_factor
            ),
            "deslocamento_detector_u_mm": (
                correction_col_original_px * col_spacing
            ),
            "deslocamento_detector_v_mm": (
                correction_row_original_px * row_spacing
            )
        })

    return corrections


def save_correction_table(corrections, output_path):
    fieldnames = [
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
        "residuo_total_orgao_mm",
        "aplicar_correcao",
        "motivo",
        "correcao_coluna_px_original",
        "correcao_linha_px_original",
        "correcao_coluna_px_reduzida",
        "correcao_linha_px_reduzida",
        "deslocamento_detector_u_mm",
        "deslocamento_detector_v_mm"
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for correction in corrections:
            writer.writerow({
                field: correction[field]
                for field in fieldnames
            })


def create_corrected_cone_vec(
    detector_rows,
    detector_cols,
    angles_rad,
    detector_row_spacing,
    detector_col_spacing,
    source_origin,
    origin_detector,
    corrections
):
    base_geometry = astra.create_proj_geom(
        "cone",
        detector_col_spacing,
        detector_row_spacing,
        detector_rows,
        detector_cols,
        angles_rad,
        source_origin,
        origin_detector
    )

    vector_geometry = astra.geom_2vec(base_geometry)
    vectors = np.asarray(
        vector_geometry["Vectors"],
        dtype=np.float64
    ).copy()

    if vectors.shape != (len(corrections), 12):
        raise ValueError(
            f"Matriz cone_vec inesperada: {vectors.shape}."
        )

    for index, correction in enumerate(corrections):
        correction_col = correction[
            "correcao_coluna_px_reduzida"
        ]

        correction_row = correction[
            "correcao_linha_px_reduzida"
        ]

        u_vector = vectors[index, 6:9].copy()
        v_vector = vectors[index, 9:12].copy()

        # d = centro do detector (colunas 3:6).
        vectors[index, 3:6] += (
            correction_col * u_vector
            + correction_row * v_vector
        )

    projection_geometry = astra.create_proj_geom(
        "cone_vec",
        detector_rows,
        detector_cols,
        vectors
    )

    return projection_geometry, vectors


def reconstruct_with_geometry(
    projections,
    projection_geometry,
    config
):
    volume_config = config["volume"]
    reconstruction_config = config["reconstrucao"]

    volume_size = int(volume_config["tamanho_voxels"])
    volume_extent = float(volume_config["extensao_mm"])
    half_extent = volume_extent / 2.0

    algorithm = reconstruction_config["algoritmo"]
    iterations = int(reconstruction_config.get("iteracoes", 1))
    min_constraint = float(
        reconstruction_config.get("restricao_minima", 0.0)
    )

    if algorithm != "SIRT3D_CUDA":
        raise ValueError(
            "Esta execução experimental foi preparada para "
            "SIRT3D_CUDA."
        )

    volume_geometry = astra.create_vol_geom(
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

    astra_projections = np.ascontiguousarray(
        np.transpose(projections, (1, 0, 2)),
        dtype=np.float32
    )

    projection_id = None
    reconstruction_id = None
    algorithm_id = None

    try:
        projection_id = astra.data3d.create(
            "-proj3d",
            projection_geometry,
            astra_projections
        )

        reconstruction_id = astra.data3d.create(
            "-vol",
            volume_geometry,
            0
        )

        cfg = astra.astra_dict(algorithm)
        cfg["ProjectionDataId"] = projection_id
        cfg["ReconstructionDataId"] = reconstruction_id
        cfg["option"] = {
            "MinConstraint": min_constraint
        }

        algorithm_id = astra.algorithm.create(cfg)

        print("Executando:", algorithm)
        print("Geometria: cone_vec corrigida")
        print("Iterações:", iterations)

        astra.algorithm.run(
            algorithm_id,
            iterations
        )

        volume = astra.data3d.get(
            reconstruction_id
        )

        return np.asarray(volume, dtype=np.float32)

    finally:
        if algorithm_id is not None:
            astra.algorithm.delete(algorithm_id)

        identifiers = []

        if projection_id is not None:
            identifiers.append(projection_id)

        if reconstruction_id is not None:
            identifiers.append(reconstruction_id)

        if identifiers:
            astra.data3d.delete(identifiers)


def main():
    args = parse_arguments()

    experiment_dir = args.experiment_dir.resolve()
    corrections_csv = args.corrections_csv.resolve()
    dicom_dir = experiment_dir / "dicom"

    config, config_path = base.load_config(experiment_dir)

    output_dir = base.create_output_dir(
        config,
        args.run_name
    )

    print("Experimento:", config["nome"])
    print("ASTRA:", astra.__version__)
    print("GPU disponível:", astra.use_cuda())

    if not astra.use_cuda():
        raise RuntimeError("O ASTRA não detectou CUDA.")

    number = int(config["numero_projecoes"])
    rotation_k = int(
        config["orientacao"]["rotacao_imagem_90"]
    )

    projections, paths = base.load_projections(
        dicom_dir,
        number,
        rotation_k
    )

    row_spacing, col_spacing = base.get_pixel_spacing(
        paths[0],
        rotation_k
    )

    angles_deg, angles_rad = base.create_angles(config)

    records = load_alignment_csv(corrections_csv)
    validate_alignment(records, paths, angles_deg)

    processing = config["processamento"]
    downsample_factor = int(processing["fator_reducao"])

    corrections = prepare_corrections(
        records,
        downsample_factor,
        row_spacing,
        col_spacing
    )

    correction_table_path = (
        output_dir / "tabela_definitiva_correcoes.csv"
    )

    save_correction_table(
        corrections,
        correction_table_path
    )

    applied_count = sum(
        correction["aplicar_correcao"]
        for correction in corrections
    )

    print(
        f"Correções aplicadas: {applied_count}/{number} "
        "(somente moedas completas)"
    )

    projections = base.preprocess(
        projections,
        downsample_factor,
        float(processing["percentil_inferior"]),
        float(processing["percentil_superior"])
    )

    base.save_projection_preview(
        projections,
        angles_deg,
        output_dir
    )

    _, detector_rows, detector_cols = projections.shape

    detector_row_spacing = row_spacing * downsample_factor
    detector_col_spacing = col_spacing * downsample_factor

    geometry_config = config["geometria"]
    source_origin = float(
        geometry_config["distancia_fonte_orgao_mm"]
    )
    origin_detector = float(
        geometry_config["distancia_orgao_detector_mm"]
    )

    projection_geometry, vectors = create_corrected_cone_vec(
        detector_rows,
        detector_cols,
        angles_rad,
        detector_row_spacing,
        detector_col_spacing,
        source_origin,
        origin_detector,
        corrections
    )

    np.save(
        output_dir / "geometria_cone_vec.npy",
        vectors
    )

    volume = reconstruct_with_geometry(
        projections,
        projection_geometry,
        config
    )

    base.save_volume_results(
        volume,
        output_dir
    )

    effective_config = json.loads(
        json.dumps(config)
    )

    effective_config.setdefault(
        "alinhamento",
        {}
    )

    effective_config["alinhamento"].update({
        "avaliar": True,
        "aplicar_correcao": True,
        "referencia": "moeda",
        "geometria": "cone_vec",
        "criterio": "somente_moeda_completa"
    })

    with (
        output_dir / "parametros_utilizados.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            effective_config,
            file,
            ensure_ascii=False,
            indent=2
        )

    shutil.copy2(
        corrections_csv,
        output_dir / "medidas_alinhamento_utilizadas.csv"
    )

    metadata = {
        "geometria": "cone_vec",
        "correcao_experimental": True,
        "convencao_residuo": "observado_menos_previsto",
        "convencao_correcao": "negativo_do_residuo",
        "criterio_aplicacao": "somente_qualidade_completa",
        "numero_total_projecoes": number,
        "numero_correcoes_aplicadas": int(applied_count),
        "fator_reducao": downsample_factor,
        "dicoms_originais_modificados": False
    }

    with (
        output_dir / "metadados_correcao.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nReconstrução cone_vec concluída.")
    print("Tabela:", correction_table_path.resolve())
    print("Resultados:", output_dir.resolve())


if __name__ == "__main__":
    main()
