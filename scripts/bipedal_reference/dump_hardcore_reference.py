#!/usr/bin/env python3
"""Terreno hardcore de BipedalWalkerHardcore-v3: alturas de aristas y polígonos obstáculo.

BipedalWalker._generate_terrain(hardcore=True) siempre rellena terrain_y[0..200) (como en modo
normal, con reglas de altura que dependen del estado GRASS/STUMP/STAIRS/PIT) y además crea cuerpos
estáticos con forma de polígono (fosos, tocones, escalones) que se agregan a env.terrain junto con
las aristas. Se distinguen por el tipo de shape: b2EdgeShape (arista de terreno, ya cubierta por
dump_reference.py) vs b2PolygonShape (obstáculo).

Genera:
  hardcore_terrain_seed<S>.csv   igual que terrain_seed<S>.csv (i, x, y)
  hardcore_draws_seed<S>.csv     igual que draws_seed<S>.csv (incluye las llamadas de hardcore)
  hardcore_obstacles_seed<S>.csv un polígono por fila: seed, id, count, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
    (los cuerpos obstáculo son siempre cuadriláteros; el orden de vértices ya sale del gift
    wrapping de Box2D, así que se vuelca tal cual lo expone pybox2d)

Uso:
    .venv/bin/python scripts/bipedal_reference/dump_hardcore_reference.py --seeds 0 1 2 3 4
"""
import argparse
from pathlib import Path

import dump_reference as ref


def dump_seed(seed: int, out: Path):
    env = ref.RecordedWalker(hardcore=True)
    env.reset(seed=seed)

    with open(out / f"hardcore_terrain_seed{seed}.csv", "w") as f:
        f.write("i,x,y\n")
        for i, (x, y) in enumerate(zip(env.terrain_x, env.terrain_y)):
            f.write(f"{i},{x:.17g},{y:.17g}\n")

    with open(out / f"hardcore_draws_seed{seed}.csv", "w") as f:
        f.write("phase,method,arg0,arg1,value\n")
        for ph, name, args, val in env.rec.log:
            a0 = f"{args[0]:.17g}" if len(args) > 0 else ""
            a1 = f"{args[1]:.17g}" if len(args) > 1 and args[1] is not None else ""
            f.write(f"{ph},{name},{a0},{a1},{val:.17g}\n")

    # env.terrain se invierte al final de _generate_terrain (self.terrain.reverse()), así que
    # filtrar directamente da los obstáculos en orden inverso de creación; se revierte para que
    # id 0 sea el primero creado, como los produce el generador de C++.
    obstacles = [b for b in env.terrain if type(b.fixtures[0].shape).__name__ == "b2PolygonShape"]
    obstacles.reverse()
    with open(out / f"hardcore_obstacles_seed{seed}.csv", "w") as f:
        f.write("id,count,v0x,v0y,v1x,v1y,v2x,v2y,v3x,v3y\n")
        for oid, b in enumerate(obstacles):
            verts = b.fixtures[0].shape.vertices
            cells = [str(oid), str(len(verts))]
            for vx, vy in verts:
                cells += [f"{vx:.9g}", f"{vy:.9g}"]
            f.write(",".join(cells) + "\n")

    env.close()
    return len(obstacles)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        n = dump_seed(seed, out)
        print(f"seed {seed}: {n} obstáculos")


if __name__ == "__main__":
    main()
