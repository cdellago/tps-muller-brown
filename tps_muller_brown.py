#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Path Sampling (TPS) on the Müller-Brown potential.

Algorithm:
  1. Generate an initial reactive path by shooting from the transition state.
  2. Repeat: pick a random frame, perturb momenta, integrate forward and
     backward until each leg terminates in a basin, accept if the two
     endpoints are different basins.

Trajectories have flexible length — they stop as soon as they enter A or B.
The inner integration loop is JIT-compiled with Numba for speed.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ── Müller-Brown potential ────────────────────────────────────────────────────
#
#   V(x,y) = Σ_k A_k exp(a_k(x-x0_k)² + b_k(x-x0_k)(y-y0_k) + c_k(y-y0_k)²)
#
#   Minima:  A ≈ ( 0.623,  0.028),  V ≈ −147
#            B ≈ (−0.558,  1.442),  V ≈ −81
#            C ≈ (−0.050,  0.467),  V ≈ −41  (metastable; A→B paths pass through C)

_A  = np.array([-200., -100., -170.,  15.])
_a  = np.array([ -1.,   -1.,  -6.5,   0.7])
_b  = np.array([  0.,    0.,  11.,    0.6])
_c  = np.array([-10.,  -10.,  -6.5,   0.7])
_x0 = np.array([  1.,    0.,  -0.5,  -1. ])
_y0 = np.array([  0.,    0.5,  1.5,   1. ])


def potential(x, y):
    dx, dy = x - _x0, y - _y0
    return float(np.sum(_A * np.exp(_a*dx**2 + _b*dx*dy + _c*dy**2)))


@njit
def gradient(x, y):
    gx = 0.0
    gy = 0.0
    for k in range(4):
        dx = x - _x0[k]
        dy = y - _y0[k]
        e = _A[k] * np.exp(_a[k]*dx*dx + _b[k]*dx*dy + _c[k]*dy*dy)
        gx += e * (2.0*_a[k]*dx + _b[k]*dy)
        gy += e * (_b[k]*dx + 2.0*_c[k]*dy)
    return gx, gy


# ── Stable-state definitions and transition state ─────────────────────────────

CENTER_A, RADIUS_A = ( 0.623,  0.028), 0.35
CENTER_B, RADIUS_B = (-0.558,  1.442), 0.35

# Saddle point between C and B (rate-limiting TS for A→B, confirmed numerically)
TS = (-0.8220, 0.6243)

# State encoding: 0 = neither, 1 = A, 2 = B
_CXA, _CYA, _R2A = CENTER_A[0], CENTER_A[1], RADIUS_A**2
_CXB, _CYB, _R2B = CENTER_B[0], CENTER_B[1], RADIUS_B**2


def in_A(x, y):
    return (x - CENTER_A[0])**2 + (y - CENTER_A[1])**2 < RADIUS_A**2


def in_B(x, y):
    return (x - CENTER_B[0])**2 + (y - CENTER_B[1])**2 < RADIUS_B**2


@njit
def which_state(x, y):
    """Return 1 (basin A), 2 (basin B), or 0 (neither)."""
    if (x - _CXA)*(x - _CXA) + (y - _CYA)*(y - _CYA) < _R2A:
        return 1
    if (x - _CXB)*(x - _CXB) + (y - _CYB)*(y - _CYB) < _R2B:
        return 2
    return 0


# ── BAOAB Langevin integrator ─────────────────────────────────────────────────

@njit
def baoab_step(x, y, px, py, dt, c1, c2):
    """Single BAOAB step (mass = 1). c1, c2 are precomputed thermostat coefficients."""
    gx, gy = gradient(x, y)
    px -= 0.5 * dt * gx
    py -= 0.5 * dt * gy

    x += 0.5 * dt * px
    y += 0.5 * dt * py

    px = c1 * px + c2 * np.random.randn()
    py = c1 * py + c2 * np.random.randn()

    x += 0.5 * dt * px
    y += 0.5 * dt * py

    gx, gy = gradient(x, y)
    px -= 0.5 * dt * gx
    py -= 0.5 * dt * gy

    return x, y, px, py


@njit
def _integrate_kernel(x, y, px, py, max_steps, dt, kT, gamma, traj):
    """
    Fill traj in-place with Langevin frames, stopping when a basin is reached.
    Returns (n_frames_written, state) where state is 1 (A), 2 (B), or 0 (timeout).
    """
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(kT * (1.0 - c1 * c1))

    traj[0, 0] = x;  traj[0, 1] = y
    traj[0, 2] = px; traj[0, 3] = py

    for i in range(max_steps):
        x, y, px, py = baoab_step(x, y, px, py, dt, c1, c2)
        n = i + 1
        traj[n, 0] = x;  traj[n, 1] = y
        traj[n, 2] = px; traj[n, 3] = py
        s = which_state(x, y)
        if s != 0:
            return n + 1, s

    return max_steps + 1, 0


def integrate_until_state(x, y, px, py, max_steps, dt, kT, gamma):
    """Python wrapper: allocates the buffer, calls the compiled kernel."""
    traj = np.empty((max_steps + 1, 4))
    n, state = _integrate_kernel(x, y, px, py, max_steps, dt, kT, gamma, traj)
    return traj[:n], state


# ── Initial reactive path ─────────────────────────────────────────────────────

def make_initial_path(dt, kT, gamma, max_steps):
    """
    Shoot from the transition state with Maxwell-Boltzmann momenta at kT.
    Retry until the forward and backward legs land in different basins.
    """
    attempt = 0
    while True:
        attempt += 1
        px = np.random.randn() * np.sqrt(kT)
        py = np.random.randn() * np.sqrt(kT)

        fwd, fwd_state = integrate_until_state(*TS, px, py,
                                               max_steps, dt, kT, gamma)
        bwd, bwd_state = integrate_until_state(*TS, -px, -py,
                                               max_steps, dt, kT, gamma)

        if fwd_state != 0 and bwd_state != 0 and fwd_state != bwd_state:
            bwd_rev = bwd[::-1].copy()
            bwd_rev[:, 2:] *= -1
            traj = np.concatenate([bwd_rev, fwd[1:]], axis=0)
            print(f"  Initial path found after {attempt} attempt(s), "
                  f"length = {len(traj)} frames.")
            return traj


# ── TPS shooting move ─────────────────────────────────────────────────────────

def shooting_move(path, dt, kT, gamma, dp_scale, max_steps):
    """
    Pick a random interior frame, perturb its momenta, integrate forward and
    backward until each leg reaches a basin.  Accept if the endpoints differ.
    """
    n = len(path)
    idx = np.random.randint(1, n - 1)

    x, y, px, py = path[idx]
    px_s = px + dp_scale * np.random.randn()
    py_s = py + dp_scale * np.random.randn()

    fwd, fwd_state = integrate_until_state(x, y,  px_s,  py_s,
                                           max_steps, dt, kT, gamma)
    bwd, bwd_state = integrate_until_state(x, y, -px_s, -py_s,
                                           max_steps, dt, kT, gamma)

    if fwd_state == 0 or bwd_state == 0 or fwd_state == bwd_state:
        return None

    bwd_rev = bwd[::-1].copy()
    bwd_rev[:, 2:] *= -1

    return np.concatenate([bwd_rev, fwd[1:]], axis=0)


# ── Main TPS loop ─────────────────────────────────────────────────────────────

def run_tps(n_moves, dt, kT, gamma, dp_scale=0.4, max_steps=10000):
    print("=== Transition Path Sampling — Müller-Brown potential ===")
    print(f"  kT={kT}, gamma={gamma}, dt={dt}, "
          f"max_steps={max_steps}, moves={n_moves}\n")

    print("  Compiling Numba kernels (one-time cost)...")
    _buf = np.empty((2, 4))
    _integrate_kernel(0.0, 0.0, 0.0, 0.0, 1, dt, kT, gamma, _buf)
    print("  Done.\n")

    path = make_initial_path(dt, kT, gamma, max_steps)
    paths = [path]
    n_acc = 0

    for move in range(1, n_moves + 1):
        candidate = shooting_move(path, dt, kT, gamma, dp_scale, max_steps)
        if candidate is not None:
            path = candidate
            n_acc += 1
        paths.append(path)

        if move % 100 == 0:
            lengths = [len(p) for p in paths]
            print(f"  Move {move:4d}/{n_moves}  "
                  f"acceptance = {n_acc/move:.2f}  "
                  f"mean path length = {np.mean(lengths):.0f} frames")

    print(f"\n  Done.  {len(paths)} paths stored, "
          f"acceptance rate = {n_acc/n_moves:.2f}")
    return paths


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(paths, filename="tps_muller_brown.png"):
    xg = np.linspace(-1.8, 1.2, 300)
    yg = np.linspace(-0.5, 2.2, 300)
    X, Y = np.meshgrid(xg, yg)
    Z = np.clip(np.vectorize(potential)(X, Y), -200, 120)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cf = ax.contourf(X, Y, Z, levels=40, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(X, Y, Z, levels=20, colors="k", linewidths=0.2, alpha=0.4)
    plt.colorbar(cf, ax=ax, label="V(x, y)")

    stride = max(1, len(paths) // 30)
    for p in paths[::stride]:
        ax.plot(p[:, 0], p[:, 1], "w-", lw=0.6, alpha=0.35)

    for center, radius, label, color in [
        (CENTER_A, RADIUS_A, "A", "cyan"),
        (CENTER_B, RADIUS_B, "B", "lime"),
    ]:
        ax.add_patch(plt.Circle(center, radius, fill=False, color=color, lw=2))
        ax.text(*center, label, color=color, ha="center", va="center",
                fontsize=13, fontweight="bold")

    ax.set_xlim(-1.8, 1.2); ax.set_ylim(-0.5, 2.2)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Sampled Transition Paths")

    ax = axes[1]
    all_x = np.concatenate([p[:, 0] for p in paths])
    all_y = np.concatenate([p[:, 1] for p in paths])
    h, xe, ye = np.histogram2d(all_x, all_y, bins=100,
                               range=[[-1.8, 1.2], [-0.5, 2.2]])
    ax.pcolormesh(xe, ye, np.log1p(h.T), cmap="inferno")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Path Density  log(1 + count)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  Figure saved to {filename}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    paths = run_tps(
        n_moves   = 5000,     # number of shooting moves
        dt        = 0.002,    # time step
        kT        = 15.0,     # thermal energy (barrier from A ≈ 5 kT)
        gamma     = 1.0,      # Langevin friction
        dp_scale  = 0.4,      # momentum perturbation width
        max_steps = 10000,    # safety cap on trajectory length
    )

    plot_results(paths)
