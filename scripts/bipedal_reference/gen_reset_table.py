#!/usr/bin/env python3
"""Genera physics/reset_table.h a partir de tests/data/bipedal_walker/reset_sweep.csv.

El estado tras reset() es una función continua y afín por tramos de la fuerza inicial F del
casco, con un quiebre en F = 0 (el signo de F decide qué límite/contacto se activa primero):
    estado(F) = s0 + F * pendiente_neg   si F < 0
    estado(F) = s0 + F * pendiente_pos   si F >= 0
El ajuste reproduce el barrido de Box2D con error máximo ~3e-6.

Uso (desde la raíz del repo):
    .venv/bin/python scripts/bipedal_reference/gen_reset_table.py
"""
from pathlib import Path

import numpy as np

SWEEP = Path("tests/data/bipedal_walker/reset_sweep.csv")
OUT = Path("include/rl/environments/bipedal_walker/physics/reset_table.h")
BODIES = ["hull", "upper0", "lower0", "upper1", "lower1"]
FIELDS = ["x", "y", "angle", "vx", "vy", "w"]  # origen del cuerpo, ángulo, v del centro de masa, w


def main():
    d = np.genfromtxt(SWEEP, delimiter=",", names=True)
    cols = [f"{b}_{f}" for b in BODIES for f in FIELDS]
    X = np.stack([d[c] for c in cols], axis=1)
    F = d["force"]
    s0 = X[(F == 0) & (d["seed"] == d["seed"].min())][0]

    slopes = {}
    for name, mask in (("neg", F < 0), ("pos", F > 0)):
        f = F[mask][:, None]
        slopes[name] = ((f * (X[mask] - s0)).sum(0)) / (f * f).sum()

    def arr(v):
        return ",\n".join("        " + ", ".join(f"{x:.9g}" for x in v[i:i + 6]) for i in range(0, len(v), 6))

    text = f'''// GENERADO por scripts/bipedal_reference/gen_reset_table.py: no editar a mano.
//
// Estado que deja BipedalWalker.reset() (incluido su paso de física con colisión continua) en
// función de la fuerza horizontal inicial F del casco, F en [-INITIAL_RANDOM, INITIAL_RANDOM].
// Por cuerpo (casco, muslo izq., pantorrilla izq., muslo der., pantorrilla der.):
//   origen x, origen y, ángulo, velocidad x, velocidad y (centro de masa), velocidad angular.
//   estado(F) = S0 + F * SLOPE_NEG   si F < 0
//   estado(F) = S0 + F * SLOPE_POS   si F >= 0
#pragma once

namespace bipedal_walker::physics {{

    template <typename T>
    struct ResetTable {{
        static constexpr int VALUES = {len(cols)};

        static constexpr T S0[VALUES] = {{
{arr(s0)}
        }};

        static constexpr T SLOPE_NEG[VALUES] = {{
{arr(slopes["neg"])}
        }};

        static constexpr T SLOPE_POS[VALUES] = {{
{arr(slopes["pos"])}
        }};
    }};

}}
'''
    OUT.write_text(text)
    print(f"escrito {OUT} ({len(cols)} valores x 3)")


if __name__ == "__main__":
    main()
