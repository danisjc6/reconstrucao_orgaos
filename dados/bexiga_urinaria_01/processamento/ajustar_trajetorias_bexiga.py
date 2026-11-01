
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE = Path(__file__).resolve().parents[1]
RELATORIOS = BASE / "relatorios"

ARQUIVO_CANDIDATOS = (
    RELATORIOS / "candidatos_hibridos_moedas.csv"
)

NUM_PROJECOES = 50
PASSO_ANGULAR = 360.0 / NUM_PROJECOES

# Coordenadas confirmadas visualmente (pixels reduzidos)
REFERENCIAS = {
    "cranial": {
        1: (619.0, 112.0),
        26: (98.0, 141.0),
    },
    "caudal": {
        1: (110.0, 734.0),
        26: (538.0, 709.0),
    },
}

# Projeções inicialmente consideradas mais favoráveis
PROJECOES_CONFIAVEIS = (
    list(range(1, 9))
    + list(range(21, 34))
    + list(range(46, 51))
)

# Limites máximos para associação de candidatos
LIMITE_X_PX = 45.0
LIMITE_Y_PX = 45.0

# Faixas verticais exploratórias
FAIXAS_Y = {
    "cranial": (60, 250),
    "caudal": (600, 820),
}


# ============================================================
# FUNÇÕES
# ============================================================

def angulo_rad(numero):
    return np.deg2rad((numero - 1) * PASSO_ANGULAR)


def matriz_harmonica(angulos):
    """Modelo periódico de primeira ordem."""
    return np.column_stack([
        np.ones_like(angulos),
        np.cos(angulos),
        np.sin(angulos),
    ])


def ajustar_harmonico(angulos, valores):
    A = matriz_harmonica(angulos)
    coeficientes, *_ = np.linalg.lstsq(
        A, valores, rcond=None
    )
    return coeficientes


def prever_harmonico(coeficientes, angulos):
    A = matriz_harmonica(np.asarray(angulos))
    return A @ coeficientes


def obter_candidatos(df, numero, marcador):
    nome = f"{numero}.dcm"

    dados = df[
        (df["arquivo"] == nome)
        & (df["marcador_provavel"] == marcador)
    ].copy()

    ymin, ymax = FAIXAS_Y[marcador]

    dados = dados[
        dados["y_reduzido_px"].between(ymin, ymax)
    ]

    return dados


def selecionar_referencias(df, marcador):
    """
    Seleciona candidatos circulares nas projeções
    favoráveis e acrescenta as referências confirmadas.
    """
    registros = []

    for numero in PROJECOES_CONFIAVEIS:
        candidatos = obter_candidatos(
            df, numero, marcador
        )

        candidatos = candidatos[
            candidatos["metodo"] == "circulo"
        ]

        # Evitar assumir que múltiplos círculos são válidos
        if len(candidatos) != 1:
            continue

        candidato = candidatos.iloc[0]

        registros.append({
            "numero": numero,
            "x": float(candidato["x_reduzido_px"]),
            "y": float(candidato["y_reduzido_px"]),
            "origem": "deteccao",
        })

    # Referências visuais prevalecem
    for numero, (x, y) in REFERENCIAS[marcador].items():
        registros = [
            r for r in registros
            if r["numero"] != numero
        ]

        registros.append({
            "numero": numero,
            "x": x,
            "y": y,
            "origem": "referencia_visual",
        })

    return pd.DataFrame(registros).sort_values("numero")


def ajustar_trajetoria_inicial(referencias):
    angulos = np.array([
        angulo_rad(n)
        for n in referencias["numero"]
    ])

    x = referencias["x"].to_numpy(float)
    y = referencias["y"].to_numpy(float)

    if len(referencias) < 3:
        raise ValueError(
            "São necessárias pelo menos 3 observações "
            "para ajustar a trajetória harmônica."
        )

    coef_x = ajustar_harmonico(angulos, x)
    coef_y = ajustar_harmonico(angulos, y)

    return coef_x, coef_y


def associar_candidatos(df, marcador, coef_x, coef_y):
    registros = []

    for numero in range(1, NUM_PROJECOES + 1):
        theta = np.array([angulo_rad(numero)])

        x_prev = float(
            prever_harmonico(coef_x, theta)[0]
        )
        y_prev = float(
            prever_harmonico(coef_y, theta)[0]
        )

        candidatos = obter_candidatos(
            df, numero, marcador
        )

        melhor = None
        melhor_distancia = np.inf

        for _, candidato in candidatos.iterrows():
            x_obs = float(
                candidato["x_reduzido_px"]
            )
            y_obs = float(
                candidato["y_reduzido_px"]
            )

            dx = x_obs - x_prev
            dy = y_obs - y_prev

            if abs(dx) > LIMITE_X_PX:
                continue

            if abs(dy) > LIMITE_Y_PX:
                continue

            distancia = np.hypot(dx, dy)

            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor = candidato

        # Preservar referências visuais
        if numero in REFERENCIAS[marcador]:
            x_obs, y_obs = REFERENCIAS[marcador][numero]
            metodo = "referencia_visual"
            status = "referencia"
        elif melhor is not None:
            x_obs = float(melhor["x_reduzido_px"])
            y_obs = float(melhor["y_reduzido_px"])
            metodo = melhor["metodo"]
            status = "candidato_associado"
        else:
            x_obs = np.nan
            y_obs = np.nan
            metodo = "nenhum"
            status = "sem_observacao"

        dx = x_obs - x_prev
        dy = y_obs - y_prev

        registros.append({
            "arquivo": f"{numero}.dcm",
            "numero": numero,
            "angulo_graus": (numero - 1) * PASSO_ANGULAR,
            "marcador": marcador,
            "status": status,
            "metodo": metodo,
            "x_observado_px": x_obs,
            "y_observado_px": y_obs,
            "x_previsto_px": x_prev,
            "y_previsto_px": y_prev,
            "residuo_x_px": dx,
            "residuo_y_px": dy,
            "residuo_total_px": np.hypot(dx, dy),
        })

    return pd.DataFrame(registros)


# ============================================================
# PROCESSAMENTO
# ============================================================

df = pd.read_csv(ARQUIVO_CANDIDATOS)

resultados = []
modelos = {}

for marcador in ["cranial", "caudal"]:
    referencias = selecionar_referencias(
        df, marcador
    )

    print(f"\nMarcador: {marcador}")
    print("Observações iniciais:", len(referencias))

    coef_x, coef_y = ajustar_trajetoria_inicial(
        referencias
    )

    modelos[marcador] = {
        "x": coef_x,
        "y": coef_y,
    }

    resultado = associar_candidatos(
        df, marcador, coef_x, coef_y
    )

    resultados.append(resultado)

    print(
        resultado["status"].value_counts().to_string()
    )

tabela = pd.concat(resultados, ignore_index=True)

arquivo_saida = (
    RELATORIOS / "trajetorias_moedas_bexiga.csv"
)

tabela.to_csv(
    arquivo_saida,
    index=False,
    float_format="%.4f"
)

print("\nTabela salva:", arquivo_saida)


# ============================================================
# GRÁFICOS
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 10),
    sharex=True
)

cores = {
    "cranial": "tab:blue",
    "caudal": "tab:orange",
}

for linha, marcador in enumerate(["cranial", "caudal"]):
    dados = tabela[
        tabela["marcador"] == marcador
    ].sort_values("angulo_graus")

    cor = cores[marcador]

    axes[linha, 0].plot(
        dados["angulo_graus"],
        dados["x_previsto_px"],
        color=cor,
        label="Modelo"
    )

    observados = dados[
        dados["status"] != "sem_observacao"
    ]

    axes[linha, 0].scatter(
        observados["angulo_graus"],
        observados["x_observado_px"],
        s=25,
        color="black",
        label="Candidatos"
    )

    axes[linha, 0].set_ylabel(
        f"{marcador.capitalize()}\nX (pixels reduzidos)"
    )
    axes[linha, 0].legend()

    axes[linha, 1].scatter(
        observados["angulo_graus"],
        observados["residuo_total_px"],
        s=25,
        color=cor
    )

    axes[linha, 1].axhline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[linha, 1].set_ylabel(
        "Resíduo total (pixels)"
    )

axes[1, 0].set_xlabel("Ângulo (graus)")
axes[1, 1].set_xlabel("Ângulo (graus)")

axes[0, 0].set_title("Trajetórias horizontais")
axes[0, 1].set_title("Resíduos geométricos")

plt.tight_layout()

figura_saida = (
    RELATORIOS / "trajetorias_moedas_bexiga.png"
)

plt.savefig(
    figura_saida,
    dpi=160,
    bbox_inches="tight"
)

plt.show()

print("Gráfico salvo:", figura_saida)
