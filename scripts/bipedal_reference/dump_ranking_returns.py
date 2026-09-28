#!/usr/bin/env python3
"""Retorno de cada controlador de ranking_controllers.csv en Box2D, por semilla de terreno.

Usa el mismo protocolo "exacto" que dump_reference.py --exact: reset() oficial (con TOI) y luego
warm-starting/TOI/sleeping apagados durante el episodio, que es lo que el port reproduce.

Escribe tests/data/bipedal_walker/ranking_box2d_seed<S>.csv: id, retorno, pasos, terminated.
"""
import argparse
from pathlib import Path

import numpy as np

import dump_reference as ref


class LinearController:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def act(self, obs):
        return np.clip(np.tanh(self.W @ obs + self.b), -1.0, 1.0).astype(np.float32)


def load_controllers(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    out = []
    for r in rows:
        W = np.array([r[f"w{i}"] for i in range(96)]).reshape(4, 24)
        b = np.array([r[f"b{i}"] for i in range(4)])
        out.append((int(r["id"]), LinearController(W, b)))
    return out


def rollout(seed, controller, steps, exact=True):
    env = ref.RecordedWalker()
    if exact:
        ref.disable_unported(env)  # igual que --exact: se apaga todo antes de que reset() dé su paso
    obs, _ = env.reset(seed=seed)
    total = 0.0
    n = 0
    terminated = False
    for _ in range(steps):
        a = controller.act(obs)
        obs, r, term, _, _ = env.step(a)
        total += r
        n += 1
        if term:
            terminated = True
            break
    env.close()
    return total, n, terminated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--controllers", default="tests/data/bipedal_walker/ranking_controllers.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--official", action="store_true",
                   help="deja warm-starting/TOI/sleeping activos (Box2D por defecto, sin las opciones "
                        "que el port no implementa); sirve de techo de referencia: cuánto se aparta "
                        "Box2D de sí mismo por esas opciones, para comparar con cuánto se aparta el port")
    args = p.parse_args()

    controllers = load_controllers(args.controllers)
    suffix = "_official" if args.official else ""
    for seed in args.seeds:
        lines = ["id,return,steps,terminated"]
        for cid, c in controllers:
            ret, n, term = rollout(seed, c, args.steps, exact=not args.official)
            lines.append(f"{cid},{ret:.6f},{n},{int(term)}")
        out = Path(args.out) / f"ranking_box2d{suffix}_seed{seed}.csv"
        out.write_text("\n".join(lines) + "\n")
        print(f"seed {seed}: {len(controllers)} controladores -> {out}")


if __name__ == "__main__":
    main()
