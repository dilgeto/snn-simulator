#!/usr/bin/env python3
"""Escenario de colisión contra un obstáculo hardcore (caja estática) aislado, sin terreno.

Reutiliza el caminante en el aire de dump_free_walker.py y le añade una única caja estática
(fd_polygon, misma fricción que el terreno) bajo los pies, para validar collide_polygons y el
solver de contactos contra Box2D sin mezclar la colisión de aristas. Warm-starting, colisión
continua y sleeping apagados desde el inicio (protocolo "exacto" de siempre).

Genera tests/data/bipedal_walker/obstacle_<policy>.csv: mismo formato que free_<policy>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import Box2D
from Box2D.b2 import polygonShape, fixtureDef

from gymnasium.envs.box2d import bipedal_walker as bw

import dump_free_walker as free
import dump_reference as ref



def build_box(world, x0, x1, y0, y1):
    fd = fixtureDef(shape=polygonShape(vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), friction=bw.FRICTION)
    return world.CreateStaticBody(fixtures=fd)


def run(policy, steps, box):
    world = Box2D.b2World()
    world.warmStarting = False
    world.continuousPhysics = False
    world.allowSleeping = False
    hull, legs, joints = free.build_walker(world)
    world.ClearForces()  # build_walker aplica su propia FORCE_X=4.75 al casco; aquí no se quiere
    build_box(world, *box)
    env = free.Env(hull, legs)
    act = free.free_policies()[policy](0)

    lines = [",".join(free.HEADER), free.row(-1, np.zeros(4), env)]
    for step in range(steps):
        a = act(step, None)
        free.apply_action(joints, a)
        world.Step(1.0 / bw.FPS, 6 * 30, 2 * 30)
        world.ClearForces()
        lines.append(free.row(step, a, env))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--steps", type=int, default=150)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Caja ancha (cubre las dos piernas) con la superficie 1 m bajo los pies iniciales, más un
    # segundo escenario con la caja bien alta para que además choque el casco (juega a "tocón").
    # Ambas cajas quedan bien por debajo de la pose inicial (casco y~5.6, pies y~4.17), para que
    # el contacto salga de una caída corta y no de un solape inicial (que dispara correcciones de
    # posición enormes y sensibles al orden, no representativas del caso real de juego).
    boxes = {
        "platform": (3.0, 6.3, 1.0, 3.0),   # ancha: la tocan las dos piernas
        "stump": (4.3, 5.1, 1.0, 3.4),      # angosta y algo más alta: solo un pie, más cerca
    }
    for name, box in boxes.items():
        for policy in ["zeros", "sine", "sat_neg"]:
            text = run(policy, args.steps, box)
            (out / f"obstacle_{name}_{policy}.csv").write_text(text)
            print(name, policy, "ok")


if __name__ == "__main__":
    main()
