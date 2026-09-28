#!/usr/bin/env python3
"""Genera controladores lineales aleatorios para el estudio de fidelidad por ranking.

Cada controlador es action = clip(tanh(W @ obs + b), -1, 1), con W (4x24) y b (4,) de ruido
gaussiano cuya escala sigma se sortea log-uniforme en [0.01, 2.0]. Sirve de sustituto barato de
una población de neuroevolución: sigma pequeño da controladores casi inmóviles (como la política
"zeros"), sigma grande da controladores erráticos y saturados; entre medio aparecen algunos que
reaccionan al contacto o al lidar y sobreviven más.

Escribe tests/data/bipedal_walker/ranking_controllers.csv: una fila por controlador con
id, sigma, y los 100 pesos (W aplanada fila a fila, 24 por fila, luego b).
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    lines = ["id,sigma," + ",".join(f"w{i}" for i in range(96)) + "," + ",".join(f"b{i}" for i in range(4))]
    for cid in range(args.n):
        sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
        W = rng.normal(0, sigma, size=(4, 24))
        b = rng.normal(0, sigma, size=4)
        vals = list(W.flatten()) + list(b)
        lines.append(f"{cid},{sigma:.9g}," + ",".join(f"{v:.9g}" for v in vals))

    out = Path(args.out) / "ranking_controllers.csv"
    out.write_text("\n".join(lines) + "\n")
    print(f"{args.n} controladores en {out}")


if __name__ == "__main__":
    main()
