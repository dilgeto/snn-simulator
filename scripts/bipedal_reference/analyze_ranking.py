#!/usr/bin/env python3
"""Correlación de Spearman entre los retornos de Box2D y del port, por configuración de iteraciones.

Lee tests/data/bipedal_walker/ranking_box2d_seed*.csv (generado por dump_ranking_returns.py) y
ranking_port_seed*_v*_p*.csv (generado por el binario bipedal_ranking_eval), y para cada semilla y
configuración calcula:
  - Spearman: correlación de rangos entre los retornos de ambos (no hay scipy en el venv, se
    implementa a mano: es Pearson sobre los rangos).
  - Kendall top-10: cuántos de los 10 mejores según Box2D siguen entre los 10 mejores del port.

Uso: .venv/bin/python scripts/bipedal_reference/analyze_ranking.py
"""
import argparse
from pathlib import Path

import numpy as np


def spearman(a, b):
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def combine(rhos):
    """Promedio de correlaciones vía transformación z de Fisher (más sólido que promediar rho
    directo cuando hay pocas semillas), con su desviación estándar aproximada."""
    z = np.arctanh(np.clip(rhos, -0.999, 0.999))
    return float(np.tanh(z.mean())), float(np.tanh(z.mean() - z.std())), float(np.tanh(z.mean() + z.std()))


def top_k_overlap(a, b, k=10):
    top_a = set(np.argsort(-a)[:k])
    top_b = set(np.argsort(-b)[:k])
    return len(top_a & top_b)


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    order = np.argsort(d["id"])
    return d[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="tests/data/bipedal_walker")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--configs", nargs="+", default=["180_60", "90_30", "60_20", "40_15", "30_10", "20_8", "12_4", "8_3"])
    args = p.parse_args()
    d = Path(args.dir)

    # Techo de referencia: cuánto se aparta Box2D de sí mismo solo por warm-starting, colisión
    # continua y sleeping (lo que el port no implementa), con la misma población y terreno.
    rhos_ceiling = []
    for seed in args.seeds:
        exact = load(d / f"ranking_box2d_seed{seed}.csv")
        official = load(d / f"ranking_box2d_official_seed{seed}.csv")
        assert np.array_equal(exact["id"], official["id"])
        re = exact["return_"] if "return_" in exact.dtype.names else exact["return"]
        ro = official["return_"] if "return_" in official.dtype.names else official["return"]
        rhos_ceiling.append(spearman(re, ro))
    ceil_mid, ceil_lo, ceil_hi = combine(np.array(rhos_ceiling))
    print(f"Techo de referencia (Box2D exacto vs. Box2D oficial, misma población): "
          f"Spearman {ceil_mid:.3f} [{ceil_lo:.3f}, {ceil_hi:.3f}] (combinado por semilla, z de Fisher)\n")

    print(f"{'config':>8} {'Spearman':>22} {'top-10 (media)':>16} {'devuelve NaN/inf':>16}")
    for cfg in args.configs:
        rhos, tops, bads = [], [], 0
        for seed in args.seeds:
            box = load(d / f"ranking_box2d_seed{seed}.csv")
            port = load(d / f"ranking_port_seed{seed}_v{cfg.split('_')[0]}_p{cfg.split('_')[1]}.csv")
            assert np.array_equal(box["id"], port["id"])
            ret_box = box["return_"] if "return_" in box.dtype.names else box["return"]
            ret_port = port["return_"] if "return_" in port.dtype.names else port["return"]
            finite = np.isfinite(ret_port)
            bads += (~finite).sum()
            if finite.sum() < len(ret_port):
                ret_port = np.where(finite, ret_port, -1000.0)  # NaN/inf: peor que cualquier caída real
            rhos.append(spearman(ret_box, ret_port))
            tops.append(top_k_overlap(ret_box, ret_port))
        mid, lo, hi = combine(np.array(rhos))
        flag = " <= techo" if mid <= ceil_hi else ""
        print(f"{cfg:>8} {mid:8.3f} [{lo:5.3f}, {hi:5.3f}] {np.mean(tops):16.1f} {bads:16d}{flag}")


if __name__ == "__main__":
    main()
