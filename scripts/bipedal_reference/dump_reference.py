#!/usr/bin/env python3
"""Volcado de referencia de BipedalWalker-v3 (Gymnasium + Box2D) para validar el port a C++.

Genera, en --out (por defecto tests/data/bipedal_walker/):

  constants.txt              constantes del módulo y propiedades de masa/geometría por cuerpo
  terrain_seed<S>.csv        alturas del terreno (i, x, y)
  draws_seed<S>.csv          llamadas a np_random durante reset() (fase, método, args, resultado)
  trace_<policy>_seed<S>.csv traza paso a paso (obs, reward, terminated, estado de los 5 cuerpos)

Las llamadas a np_random se graban para poder reinyectarlas en el generador de terreno de C++
(numpy usa PCG64 y RLTools otro RNG, así que no se puede reproducir por semilla).

Uso:
    .venv/bin/python scripts/bipedal_reference/dump_reference.py --seeds 0 1 2 --steps 300
"""
import argparse
import math
from pathlib import Path

import numpy as np

from gymnasium.envs.box2d import bipedal_walker as bw

BODY_NAMES = ["hull", "upper0", "lower0", "upper1", "lower1"]


def fmt(x) -> str:
    """float32 ida y vuelta: 9 cifras significativas bastan."""
    return f"{float(x):.9g}"


class RecordingRng:
    """Envuelve un numpy Generator y registra cada llamada, etiquetada por fase."""

    def __init__(self, gen, owner=None):
        self._gen = gen
        self.phase = "init"
        self.log = []
        self._owner = owner

    def _rec(self, name, args, result):
        self.log.append((self.phase, name, args, float(result)))
        return result

    def uniform(self, low=0.0, high=1.0, *a, **k):
        result = self._gen.uniform(low, high, *a, **k)
        # Permite fijar la fuerza inicial del casco (fase "force") para barrer el reset().
        override = getattr(self._owner, "force_override", None)
        if self.phase == "force" and override is not None:
            result = override
        return self._rec("uniform", (low, high), result)

    def integers(self, low, high=None, *a, **k):
        return self._rec("integers", (low, high), self._gen.integers(low, high, *a, **k))

    def random(self, *a, **k):
        return self._rec("random", (), self._gen.random(*a, **k))

    def __getattr__(self, name):
        return getattr(self._gen, name)


class RecordedWalker(bw.BipedalWalker):
    """BipedalWalker que graba las llamadas a np_random hechas durante reset()."""

    force_override = None  # si no es None, sustituye la fuerza inicial aleatoria del casco

    def _destroy(self):
        # reset() llama a super().reset(seed) y enseguida a _destroy(): aquí ya existe
        # el Generator nuevo y aún no se generó el terreno.
        super()._destroy()
        self.rec = RecordingRng(self._np_random, owner=self)
        self._np_random = self.rec

    def _generate_terrain(self, hardcore):
        self.rec.phase = "terrain"
        super()._generate_terrain(hardcore)

    def _generate_clouds(self):
        self.rec.phase = "clouds"
        super()._generate_clouds()
        self.rec.phase = "force"


def body_state(b):
    return [b.position[0], b.position[1], b.angle,
            b.linearVelocity[0], b.linearVelocity[1], b.angularVelocity]


def all_bodies(env):
    return [env.hull] + list(env.legs)


STATE_COLS = [f"{n}_{c}" for n in BODY_NAMES for c in ("x", "y", "angle", "vx", "vy", "w")]
OBS_COLS = [f"obs{i}" for i in range(24)]
HEADER = ["step", "a0", "a1", "a2", "a3"] + OBS_COLS + ["reward", "terminated", "game_over"] + STATE_COLS


def row(step, action, obs, reward, terminated, env):
    cells = [step] + [fmt(a) for a in action] + [fmt(o) for o in obs]
    cells += [fmt(reward), int(terminated), int(env.game_over)]
    for b in all_bodies(env):
        cells += [fmt(v) for v in body_state(b)]
    return ",".join(str(c) for c in cells)


# --- políticas de acciones (todas deterministas dada la semilla) ---------------------------------

def policy_zeros(_seed):
    return lambda step, obs: np.zeros(4, dtype=np.float32)


def policy_sine(_seed):
    # Patrón periódico en anti-fase entre piernas: ejercita motores, límites y contacto.
    def act(step, obs):
        t = step / 50.0
        p = 2 * math.pi * 1.0 * t
        return np.array([math.sin(p), 0.5 * math.sin(p + 1.0),
                         -math.sin(p), -0.5 * math.sin(p + 1.0)], dtype=np.float32)
    return act


def policy_random(seed):
    rng = np.random.default_rng(1000 + seed)
    return lambda step, obs: rng.uniform(-1, 1, 4).astype(np.float32)


def policy_heuristic(_seed):
    h = bw.BipedalWalkerHeuristics()
    # Instancia limpia: los atributos de estado se reasignan en la instancia al primer uso.
    return lambda step, obs: np.array(h.step_heuristic(obs), dtype=np.float32)


POLICIES = {"zeros": policy_zeros, "sine": policy_sine, "random": policy_random,
            "heuristic": policy_heuristic}


# --- volcado -------------------------------------------------------------------------------------

def dump_constants(env, path: Path):
    lines = []
    for name in ["FPS", "SCALE", "MOTORS_TORQUE", "SPEED_HIP", "SPEED_KNEE", "LIDAR_RANGE",
                 "INITIAL_RANDOM", "LEG_DOWN", "LEG_W", "LEG_H", "VIEWPORT_W", "VIEWPORT_H",
                 "TERRAIN_STEP", "TERRAIN_LENGTH", "TERRAIN_HEIGHT", "TERRAIN_GRASS",
                 "TERRAIN_STARTPAD", "FRICTION"]:
        lines.append(f"{name} {fmt(getattr(bw, name))}")
    lines.append("HULL_POLY " + " ".join(f"{x} {y}" for x, y in bw.HULL_POLY))
    lines.append(f"gravity {fmt(env.world.gravity[0])} {fmt(env.world.gravity[1])}")
    lines.append("velocity_iterations 180")
    lines.append("position_iterations 60")
    lines.append("dt " + fmt(1.0 / bw.FPS))

    # Propiedades de masa tras reset(): masa, inercia respecto al origen del cuerpo y centro
    # de masa local. El casco no es simétrico, así que origen != centro de masa.
    # `inertia` (GetInertia) es respecto al origen del cuerpo; `inertia_com` es la que usa el
    # solver (respecto al centro de masa): inertia - mass * |local_center|^2.
    for n, b in zip(BODY_NAMES, all_bodies(env)):
        lc = b.localCenter
        i_com = b.inertia - b.mass * (lc[0] ** 2 + lc[1] ** 2)
        lines.append(f"body {n} mass {fmt(b.mass)} inertia {fmt(b.inertia)} "
                     f"inertia_com {fmt(i_com)} local_center {fmt(lc[0])} {fmt(lc[1])}")
        fx = b.fixtures[0]
        sh = fx.shape
        lines.append(f"shape {n} density {fmt(fx.density)} radius {fmt(sh.radius)} "
                     f"count {sh.vertexCount} vertices "
                     + " ".join(f"{fmt(x)} {fmt(y)}" for x, y in sh.vertices)
                     + " normals "
                     + " ".join(f"{fmt(x)} {fmt(y)}" for x, y in sh.normals))
    for i, j in enumerate(env.joints):
        lines.append(f"joint {i} anchorA {fmt(j.anchorA[0])} {fmt(j.anchorA[1])} "
                     f"anchorB {fmt(j.anchorB[0])} {fmt(j.anchorB[1])} "
                     f"limits {fmt(j.lowerLimit)} {fmt(j.upperLimit)}")
    path.write_text("\n".join(lines) + "\n")


def disable_unported(env):
    """Apaga en el mundo Box2D lo que el port en C++ no implementa: warm-starting, colisión
    continua (TOI) y sleeping."""
    env.world.warmStarting = False
    env.world.continuousPhysics = False
    env.world.allowSleeping = False


def dump_seed(seed: int, steps: int, policies, out: Path, hardcore: bool, exact: bool = False):
    suffix = "_exact" if exact else ""
    for pname in policies:
        env = RecordedWalker(hardcore=hardcore)
        obs, _ = env.reset(seed=seed)
        if exact:
            # El reset() oficial se hace con todo activo: su paso de física depende de TOI (las
            # pantorrillas quedan apoyadas en el suelo en vez de hundirse 0.2 m) y el port lo
            # reproduce aparte. Las opciones se apagan a partir del primer paso del episodio.
            disable_unported(env)
        if pname == policies[0]:
            # El terreno y las llamadas de reset() no dependen de la política: se vuelcan una vez.
            with open(out / f"terrain_seed{seed}.csv", "w") as f:
                f.write("i,x,y\n")
                for i, (x, y) in enumerate(zip(env.terrain_x, env.terrain_y)):
                    f.write(f"{i},{x:.17g},{y:.17g}\n")  # doubles: ver draws_seed*.csv
            with open(out / f"draws_seed{seed}.csv", "w") as f:
                f.write("phase,method,arg0,arg1,value\n")
                for ph, name, args, val in env.rec.log:
                    a0 = fmt(args[0]) if len(args) > 0 else ""
                    a1 = fmt(args[1]) if len(args) > 1 and args[1] is not None else ""
                    # 17 cifras: son doubles y el terreno original se calcula en double.
                    f.write(f"{ph},{name},{a0},{a1},{val:.17g}\n")
            if seed == 0 and not (out / "constants.txt").exists():
                dump_constants(env, out / "constants.txt")

        act = POLICIES[pname](seed)
        lines = [",".join(HEADER)]
        # Fila -1: estado justo tras reset() (que ya ejecutó un paso con acción 0).
        lines.append(row(-1, np.zeros(4), obs, 0.0, False, env))
        for step in range(steps):
            a = act(step, obs)
            obs, r, terminated, _, _ = env.step(a)
            lines.append(row(step, a, obs, r, terminated, env))
            if terminated:
                break
        (out / f"trace_{pname}{suffix}_seed{seed}.csv").write_text("\n".join(lines) + "\n")
        env.close()


def check_determinism(seed: int):
    """Misma semilla y mismas acciones deben dar trazas idénticas (requisito del oráculo)."""
    def run():
        env = RecordedWalker()
        obs, _ = env.reset(seed=seed)
        act = policy_sine(seed)
        out = [obs.copy()]
        for step in range(100):
            obs, *_ = env.step(act(step, obs))
            out.append(obs.copy())
        env.close()
        return np.array(out)
    a, b = run(), run()
    assert np.array_equal(a, b), "Box2D no es determinista entre ejecuciones"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/data/bipedal_walker")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--hardcore", action="store_true")
    p.add_argument("--exact", action="store_true",
                   help="tras el reset oficial, apaga en Box2D lo que el port no implementa (trazas *_exact)")
    p.add_argument("--constants-only", action="store_true",
                   help="solo regenera constants.txt (semilla 0)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.constants_only:
        env = RecordedWalker(hardcore=args.hardcore)
        env.reset(seed=0)
        dump_constants(env, out / "constants.txt")
        return
    check_determinism(args.seeds[0])
    for s in args.seeds:
        dump_seed(s, args.steps, args.policies, out, args.hardcore, args.exact)
        print(f"seed {s}: ok")
    print(f"volcado en {out}")


if __name__ == "__main__":
    main()
