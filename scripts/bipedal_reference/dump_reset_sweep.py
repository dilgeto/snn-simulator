#!/usr/bin/env python3
"""Estado que deja BipedalWalker.reset() en función de la fuerza inicial y de la semilla.

El reset() oficial ejecuta un paso de física con colisión continua (TOI) que apoya las
pantorrillas en el suelo. Como el terreno bajo el inicio es siempre plano, el estado resultante
depende solo de la fuerza horizontal aplicada al casco; el port lo reproduce con una tabla
(ver gen_reset_table.py) en vez de portar TOI.

Genera tests/data/bipedal_walker/reset_sweep.csv: una fila por (semilla, fuerza) con el estado
de los 5 cuerpos y los flags de contacto tras reset().
"""
import argparse
from pathlib import Path

import numpy as np

import dump_reference as ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--forces", type=int, default=21)
    args = p.parse_args()

    forces = np.linspace(-5.0, 5.0, args.forces)
    header = ["seed", "force"] + ref.STATE_COLS + ["obs8", "obs13"]
    lines = [",".join(header)]
    for seed in args.seeds:
        for f in forces:
            env = ref.RecordedWalker()
            env.force_override = float(f)
            obs, _ = env.reset(seed=seed)
            cells = [seed, f"{f:.17g}"]
            for b in ref.all_bodies(env):
                cells += [f"{v:.9g}" for v in ref.body_state(b)]
            cells += [f"{obs[8]:.0f}", f"{obs[13]:.0f}"]
            lines.append(",".join(str(c) for c in cells))
            env.close()
    (Path(args.out) / "reset_sweep.csv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} filas")


if __name__ == "__main__":
    main()
