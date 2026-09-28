#!/usr/bin/env python3
"""Escenario S2: el caminante completo en el aire (sin terreno), contra Box2D.

Reproduce BipedalWalker.reset() con un mundo Box2D propio y opciones para apagar lo que el
port en C++ no implementa (warm-starting, colisión continua, sleeping). Con las tres apagadas
el port debe coincidir con Box2D casi bit a bit; con warm-starting activo se mide cuánto se
aleja (ablation, no es una aserción).

Genera tests/data/bipedal_walker/free_<policy>[_ws].csv:
  fila -1: pose inicial (antes de cualquier paso)
  fila k : estado tras el paso k con la acción k (la fuerza inicial se aplica solo en el paso 0)

Uso:
    .venv/bin/python scripts/bipedal_reference/dump_free_walker.py
"""
import argparse
import math
from pathlib import Path

import numpy as np
import Box2D
from Box2D.b2 import polygonShape, fixtureDef, revoluteJointDef  # noqa: F401

from gymnasium.envs.box2d import bipedal_walker as bw

import dump_reference as ref

FORCE_X = 4.75  # fuerza inicial fija sobre el casco (en el env real es uniform(-5, 5))


def build_walker(world):
    """Copia de la construcción de cuerpos y juntas de BipedalWalker.reset()."""
    init_x = bw.TERRAIN_STEP * bw.TERRAIN_STARTPAD / 2
    init_y = bw.TERRAIN_HEIGHT + 2 * bw.LEG_H
    hull = world.CreateDynamicBody(position=(init_x, init_y), fixtures=bw.HULL_FD)
    hull.ApplyForceToCenter((FORCE_X, 0), True)

    legs, joints = [], []
    for i in [-1, +1]:
        leg = world.CreateDynamicBody(
            position=(init_x, init_y - bw.LEG_H / 2 - bw.LEG_DOWN),
            angle=(i * 0.05),
            fixtures=bw.LEG_FD,
        )
        rjd = revoluteJointDef(
            bodyA=hull, bodyB=leg,
            localAnchorA=(0, bw.LEG_DOWN), localAnchorB=(0, bw.LEG_H / 2),
            enableMotor=True, enableLimit=True,
            maxMotorTorque=bw.MOTORS_TORQUE, motorSpeed=i,
            lowerAngle=-0.8, upperAngle=1.1,
        )
        legs.append(leg)
        joints.append(world.CreateJoint(rjd))

        lower = world.CreateDynamicBody(
            position=(init_x, init_y - bw.LEG_H * 3 / 2 - bw.LEG_DOWN),
            angle=(i * 0.05),
            fixtures=bw.LOWER_FD,
        )
        rjd = revoluteJointDef(
            bodyA=leg, bodyB=lower,
            localAnchorA=(0, -bw.LEG_H / 2), localAnchorB=(0, bw.LEG_H / 2),
            enableMotor=True, enableLimit=True,
            maxMotorTorque=bw.MOTORS_TORQUE, motorSpeed=1,
            lowerAngle=-1.6, upperAngle=-0.1,
        )
        legs.append(lower)
        joints.append(world.CreateJoint(rjd))
    return hull, legs, joints


def apply_action(joints, action):
    """Igual que BipedalWalker.step con control_speed = False."""
    speeds = [bw.SPEED_HIP, bw.SPEED_KNEE, bw.SPEED_HIP, bw.SPEED_KNEE]
    for j, s, a in zip(joints, speeds, action):
        j.motorSpeed = float(s * np.sign(a))
        j.maxMotorTorque = float(bw.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1))


class Env:
    """Contenedor mínimo con la interfaz que usa ref.row()/ref.all_bodies()."""

    def __init__(self, hull, legs):
        self.hull, self.legs, self.game_over = hull, legs, False


def free_policies():
    def saturate(sign):
        return lambda seed: (lambda step, obs: np.full(4, sign, dtype=np.float32))

    def chatter(seed):
        # Cambia de signo en cada paso: fuerza el motor a invertir su objetivo continuamente.
        return lambda step, obs: np.array([1, -1, -1, 1], dtype=np.float32) * (1 if step % 2 == 0 else -1)

    return {"zeros": ref.policy_zeros, "sine": ref.policy_sine, "random": ref.policy_random,
            "sat_pos": saturate(1.0), "sat_neg": saturate(-1.0), "chatter": chatter}


HEADER = ["step", "a0", "a1", "a2", "a3"] + ref.STATE_COLS


def row(step, action, env):
    cells = [step] + [ref.fmt(a) for a in action]
    for b in ref.all_bodies(env):
        cells += [ref.fmt(v) for v in ref.body_state(b)]
    return ",".join(str(c) for c in cells)


def run(policy, steps, warm_starting, seed):
    world = Box2D.b2World()
    world.warmStarting = warm_starting
    world.continuousPhysics = False
    world.allowSleeping = False
    hull, legs, joints = build_walker(world)
    env = Env(hull, legs)
    act = free_policies()[policy](seed)

    lines = [",".join(HEADER), row(-1, np.zeros(4), env)]
    for step in range(steps):
        a = act(step, None)
        apply_action(joints, a)
        world.Step(1.0 / bw.FPS, 6 * 30, 2 * 30)
        world.ClearForces()
        lines.append(row(step, a, env))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--steps", type=int, default=150)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for policy in free_policies():
        (out / f"free_{policy}.csv").write_text(run(policy, args.steps, False, 0))
        (out / f"free_{policy}_ws.csv").write_text(run(policy, args.steps, True, 0))
        print(policy, "ok")


if __name__ == "__main__":
    main()
