import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

CSV = Path("resultados/traqueia_caprina_01/alinhamento/esferas.csv")
ESFERAS = ("S1", "S2", "I1")
SOD, SDD, PIXEL = 653.0, 838.0, 0.086215  # mm
Y_MEIO = (3404 - 1) / 2

with CSV.open(newline="", encoding="utf-8") as f:
    dados = [
        r for r in csv.DictReader(f)
        if r["qualidade"] == "visivel" and r["x_px"] and r["y_px"]
    ]

arquivo = np.array([r["arquivo"] for r in dados])
esfera = np.array([ESFERAS.index(r["esfera"]) for r in dados])
theta = np.deg2rad([float(r["angulo_graus"]) for r in dados])
observado = np.array([[float(r["x_px"]), float(r["y_px"])] for r in dados])
eh_48 = arquivo == "48.dcm"

# Dois parâmetros comuns: coluna do eixo na meia altura e inclinação.
# Mais três parâmetros para a posição 3D de cada esfera.
inicial = np.array([
    1351.5, 0.0,
    80.0, 0.0, -90.0,
    -80.0, 0.0, 95.0,
    70.0, 0.0, 95.0,
])
inferior = [0, -0.2] + [-220, -220, -150] * 3
superior = [2703, 0.2] + [220, 220, 150] * 3


def prever(p):
    coluna, inclinacao = p[:2]
    a, b, altura = p[2:].reshape(3, 3)[esfera].T

    # A profundidade muda com o giro, assim como a magnificação.
    lateral = a * np.cos(theta) + b * np.sin(theta)
    profundidade = a * np.sin(theta) - b * np.cos(theta)
    escala = SDD / (SOD + profundidade) / PIXEL

    u, v = escala * lateral, escala * altura
    c, s = np.cos(inclinacao), np.sin(inclinacao)
    return np.column_stack((coluna + c*u + s*v, Y_MEIO - s*u + c*v))


def ajustar(nome, usar):
    ajuste = least_squares(
        lambda p: (prever(p)[usar] - observado[usar]).ravel(),
        inicial, bounds=(inferior, superior),
        loss="soft_l1", f_scale=15.0, x_scale="jac",
    )
    residuo = observado - prever(ajuste.x)
    rms = np.sqrt(np.mean(np.sum(residuo**2, axis=1)[~eh_48]))
    print(f"\n{nome}: {usar.sum()} observações no ajuste")
    print(f"Convergiu: {ajuste.success}")
    print(f"Eixo na meia altura: x = {ajuste.x[0]:.2f} px")
    print(f"Inclinação em relação à vertical: {np.rad2deg(ajuste.x[1]):+.3f}°")
    print(f"Erro RMS nas demais vistas: {rms:.2f} px")
    for i in np.flatnonzero(eh_48):
        dx, dy = residuo[i]
        print(f"48.dcm {ESFERAS[esfera[i]]}: dx={dx:+.1f}, dy={dy:+.1f} px")


ajustar("Com 48.dcm", np.ones(len(dados), dtype=bool))
ajustar("Sem as marcações de 48.dcm", ~eh_48)