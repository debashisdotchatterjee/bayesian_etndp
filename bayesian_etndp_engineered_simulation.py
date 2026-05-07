"""
Bayesian Multi-Topology Express Transportation Network Design
Simulation verification script, engineered stress-test version.

This script is designed for Google Colab and ordinary Python.  It implements
manual conjugate Bayesian updating and posterior-predictive risk-aware
multi-topology network design without PyMC/Stan.

Outputs:
  bayesian_etndp_engineered_outputs/
    tables/*.csv, *.tex
    figures/*.png
    simulation_summary.json
  bayesian_etndp_engineered_outputs.zip

Run in Colab:
  %run bayesian_etndp_engineered_simulation.py

Author: generated for Debashis Chatterjee
"""

from __future__ import annotations

import os
import math
import json
import zipfile
import itertools
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib

try:
    from IPython import get_ipython
    from IPython.display import display
    HAVE_IPYTHON = get_ipython() is not None
except Exception:
    HAVE_IPYTHON = False

# Use a non-interactive backend outside notebooks; keep Colab/Jupyter inline display intact.
if not HAVE_IPYTHON:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    seed: int = 20260504
    n_nodes: int = 9
    n_candidate_hubs: int = 3
    n_history_days: int = 100
    posterior_scenarios: int = 70
    future_scenarios: int = 90
    cvar_alpha: float = 0.90
    service_target_hours: float = 60.0
    hub_hold_time_hours: float = 10.0
    target_service_reliability: float = 0.86
    target_hold_reliability: float = 0.88
    speed_kmph: float = 68.0
    outdir: str = "bayesian_etndp_engineered_outputs"
    show_plots: bool = True
    make_plots: bool = True
    show_tables: bool = True
    max_rows_to_display: int = 24

    # Bayes-risk weights. Reliability penalties are kept outside the convex weights.
    weight_cost: float = 0.42
    weight_cvar_time: float = 0.40
    weight_emission: float = 0.08
    weight_unreliability: float = 1.35

    # Capacity multipliers; larger capacity costs more but improves hub-hold reliability.
    capacity_multipliers: Tuple[float, ...] = (1.05, 1.40, 1.85)
    direct_fractions: Tuple[float, ...] = (0.12, 0.28)


TOPOLOGIES = ["FC", "SAHS", "MAHS", "RAHS", "DSAHS", "DMAHS", "DRAHS"]
TOPO_COLORS = {
    "FC": "#222222",
    "SAHS": "#1f77b4",
    "MAHS": "#ff7f0e",
    "RAHS": "#9467bd",
    "DSAHS": "#2ca02c",
    "DMAHS": "#d62728",
    "DRAHS": "#17becf",
}


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def ensure_dirs(cfg: SimulationConfig) -> Dict[str, str]:
    dirs = {
        "root": cfg.outdir,
        "figures": os.path.join(cfg.outdir, "figures"),
        "tables": os.path.join(cfg.outdir, "tables"),
        "arrays": os.path.join(cfg.outdir, "arrays"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def print_banner(title: str, char: str = "=") -> None:
    line = char * 96
    print(f"\n{line}\n{title}\n{line}")


def maybe_display(df: pd.DataFrame, cfg: SimulationConfig, name: str = "table") -> None:
    if cfg.show_tables:
        print(df.head(cfg.max_rows_to_display).to_string(index=False))
        if HAVE_IPYTHON:
            display(df.head(cfg.max_rows_to_display))


def save_table(df: pd.DataFrame, dirs: Dict[str, str], name: str, index: bool = False) -> None:
    # CSV is the primary reproducible table output.
    csv_path = os.path.join(dirs["tables"], f"{name}.csv")
    df.to_csv(csv_path, index=index)


def make_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return make_json_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    return str(obj)


def zip_directory(folder: str, zip_path: str) -> str:
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, os.path.dirname(folder))
                zf.write(full, arcname=rel)
    return zip_path


def cvar(x: np.ndarray, alpha: float = 0.90) -> float:
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    if tail.size == 0:
        return float(q)
    return float(np.mean(tail))


def normalize_minmax(s: pd.Series) -> pd.Series:
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def pareto_minimize(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    arr = df[cols].to_numpy(dtype=float)
    n = arr.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated_by_any = np.any(np.all(arr <= arr[i] + 1e-12, axis=1) & np.any(arr < arr[i] - 1e-12, axis=1))
        efficient[i] = not dominated_by_any
    return pd.Series(efficient, index=df.index)


def inverse_gamma_sample(rng: np.random.Generator, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Sample Inv-Gamma(alpha,beta) with density beta^alpha/Gamma(alpha) x^{-alpha-1} exp(-beta/x)."""
    return 1.0 / rng.gamma(shape=alpha, scale=1.0 / beta)


# --------------------------------------------------------------------------------------
# Synthetic logistics data generation
# --------------------------------------------------------------------------------------

def generate_geography(cfg: SimulationConfig, rng: np.random.Generator) -> Dict[str, Any]:
    """Create a stylized geography with metropolitan, coastal, interior and remote zones."""
    # Fixed cluster centers make the scenario reproducible and interpretable.
    centers = np.array([
        [0.12, 0.72],   # north-west/interior
        [0.28, 0.28],   # south-west/manufacturing belt
        [0.66, 0.30],   # coastal/port belt
        [0.82, 0.74],   # north-east/remote market
    ])
    cluster_probs = np.array([0.22, 0.30, 0.32, 0.16])
    clusters = rng.choice(len(centers), size=cfg.n_nodes, p=cluster_probs)
    coords = centers[clusters] + rng.normal(0, 0.055, size=(cfg.n_nodes, 2))
    coords = np.clip(coords, 0.04, 0.96)

    # Select candidate hubs: one closest to each cluster center, then fill if needed.
    candidate_hubs: List[int] = []
    for c in range(len(centers)):
        inds = np.where(clusters == c)[0]
        if inds.size > 0:
            closest = inds[np.argmin(np.linalg.norm(coords[inds] - centers[c], axis=1))]
            if int(closest) not in candidate_hubs:
                candidate_hubs.append(int(closest))
    if len(candidate_hubs) < cfg.n_candidate_hubs:
        centrality = np.linalg.norm(coords - np.array([0.5, 0.5]), axis=1)
        for idx in np.argsort(centrality):
            if int(idx) not in candidate_hubs:
                candidate_hubs.append(int(idx))
            if len(candidate_hubs) >= cfg.n_candidate_hubs:
                break
    candidate_hubs = candidate_hubs[:cfg.n_candidate_hubs]

    # Distances in km: Euclidean plus road-network detour.
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.linalg.norm(diff, axis=2)
    road_detour = 1.18 + 0.14 * rng.random((cfg.n_nodes, cfg.n_nodes))
    dist = 1450.0 * euclid * road_detour + 35.0
    np.fill_diagonal(dist, 0.0)

    # Economic mass: coastal and manufacturing clusters ship more.
    cluster_strength = np.array([0.85, 1.20, 1.42, 0.76])
    econ_mass = cluster_strength[clusters] * rng.lognormal(mean=0.0, sigma=0.28, size=cfg.n_nodes)

    return {
        "coords": coords,
        "clusters": clusters,
        "candidate_hubs": tuple(candidate_hubs),
        "dist": dist,
        "econ_mass": econ_mass,
        "centers": centers,
    }


def simulate_historical_data(cfg: SimulationConfig, geo: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    n, T = cfg.n_nodes, cfg.n_history_days
    dist = geo["dist"]
    clusters = geo["clusters"]
    econ = geo["econ_mass"]
    hubs = geo["candidate_hubs"]

    # Three interpretable day regimes.
    # Normal: everyday demand; Sale: high OD volume; Storm: travel disruption and hub productivity loss.
    regime = rng.choice([0, 1, 2], size=T, p=[0.72, 0.17, 0.11])
    weekday = np.arange(T) % 7
    weekend = (weekday >= 5).astype(float)
    seasonal = 0.18 * np.sin(2.0 * np.pi * np.arange(T) / 30.0)

    origin_eff = rng.normal(0.0, 0.26, size=n)
    dest_eff = rng.normal(0.0, 0.22, size=n)

    # OD demand: asymmetric, region-pair dependent, overdispersed.
    W = np.zeros((T, n, n), dtype=int)
    true_lambda = np.zeros((T, n, n), dtype=float)
    for t in range(T):
        sale = 1.0 if regime[t] == 1 else 0.0
        storm = 1.0 if regime[t] == 2 else 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Coastal/manufacturing to interior flow bonus; remote destinations also receive more.
                c_i, c_j = clusters[i], clusters[j]
                corridor_bonus = 0.0
                if c_i in (1, 2) and c_j in (0, 3):
                    corridor_bonus += 0.42
                if c_i == 2 and c_j == 1:
                    corridor_bonus += 0.20
                if c_j == 3:
                    corridor_bonus += 0.15
                loglam = (
                    2.25
                    + np.log(econ[i])
                    + 0.72 * np.log(econ[j])
                    + origin_eff[i]
                    + dest_eff[j]
                    + corridor_bonus
                    - 0.00105 * dist[i, j]
                    + 0.35 * weekend[t]
                    + seasonal[t]
                    + 0.78 * sale
                    - 0.10 * storm
                )
                lam = max(0.4, math.exp(loglam))
                true_lambda[t, i, j] = lam
                # Gamma-Poisson mixture for overdispersion.
                shape = 7.5
                rate = shape / lam
                lam_tilde = rng.gamma(shape=shape, scale=1.0 / rate)
                W[t, i, j] = rng.poisson(lam_tilde)

    # Travel times: lognormal around distance/speed with corridor-specific storm disruption.
    base_time = np.maximum(dist / cfg.speed_kmph, 0.05)
    logT = np.zeros((T, n, n), dtype=float)
    Tau = np.zeros((T, n, n), dtype=float)
    storm_corridor = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Remote corridor is riskier.
            storm_corridor[i, j] = (clusters[i] == 3 or clusters[j] == 3 or dist[i, j] > np.quantile(dist[dist > 0], 0.75))
    for t in range(T):
        sale = 1.0 if regime[t] == 1 else 0.0
        storm = 1.0 if regime[t] == 2 else 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                disruption = storm * (0.32 + 0.45 * storm_corridor[i, j])
                sale_delay = 0.12 * sale
                sigma = 0.18 + 0.18 * storm + 0.06 * storm_corridor[i, j]
                mu = math.log(base_time[i, j]) + sale_delay + disruption - 0.5 * sigma * sigma
                logT[t, i, j] = rng.normal(mu, sigma)
                Tau[t, i, j] = math.exp(logT[t, i, j])

    # Cost multiplier: fuel and labor shocks.
    log_cost_multiplier = rng.normal(0.0 + 0.10 * (regime == 1) + 0.14 * (regime == 2), 0.08, size=T)
    cost_multiplier = np.exp(log_cost_multiplier)

    # Hub reliability observations: one inexpensive central hub is productive on average but unreliable in storm days.
    # This is the key stress-test mechanism for Bayesian risk-aware design.
    hub_base = {}
    for rank, h in enumerate(hubs):
        # More central hubs look attractive but may be less reliable under shocks.
        centrality = np.linalg.norm(geo["coords"][h] - np.array([0.5, 0.5]))
        base_rel = 0.91 - 0.12 * max(0.0, 0.30 - centrality) / 0.30 + rng.normal(0, 0.02)
        if rank == 0:
            base_rel -= 0.04
        if rank == len(hubs) - 1:
            base_rel += 0.04
        hub_base[h] = float(np.clip(base_rel, 0.72, 0.96))

    hub_trials = np.zeros((T, len(hubs)), dtype=float)
    hub_success = np.zeros((T, len(hubs)), dtype=float)
    true_reliability = np.zeros((T, len(hubs)), dtype=float)
    for t in range(T):
        for a, h in enumerate(hubs):
            storm_penalty = 0.18 * (regime[t] == 2) * (1.0 + 0.6 * (a == 0))
            sale_penalty = 0.06 * (regime[t] == 1)
            r = np.clip(hub_base[h] - storm_penalty - sale_penalty + rng.normal(0, 0.035), 0.45, 0.98)
            true_reliability[t, a] = r
            trials = 30 + 18 * (regime[t] == 1) + 10 * (regime[t] == 2)
            hub_trials[t, a] = trials
            hub_success[t, a] = rng.binomial(int(trials), r)

    return {
        "W": W,
        "true_lambda": true_lambda,
        "Tau": Tau,
        "logT": logT,
        "base_time": base_time,
        "regime": regime,
        "cost_multiplier": cost_multiplier,
        "log_cost_multiplier": log_cost_multiplier,
        "hub_trials": hub_trials,
        "hub_success": hub_success,
        "true_reliability": true_reliability,
        "hub_base": hub_base,
        "storm_corridor": storm_corridor,
    }


# --------------------------------------------------------------------------------------
# Manual Bayesian updating
# --------------------------------------------------------------------------------------

def gamma_poisson_update(W: np.ndarray) -> Dict[str, Any]:
    T, n, _ = W.shape
    positive = W[:, ~np.eye(n, dtype=bool)].ravel()
    mean_w = float(np.mean(positive))
    var_w = float(np.var(positive, ddof=1))
    # Empirical Bayes prior: weak, centered around global mean.
    beta0 = 0.18
    alpha0 = max(0.5, mean_w * beta0)
    alpha_post = np.full((n, n), alpha0, dtype=float) + W.sum(axis=0)
    beta_post = np.full((n, n), beta0 + T, dtype=float)
    np.fill_diagonal(alpha_post, 0.0)
    return {
        "alpha0": alpha0,
        "beta0": beta0,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "posterior_mean": np.divide(alpha_post, beta_post, out=np.zeros_like(alpha_post), where=beta_post > 0),
        "global_mean": mean_w,
        "global_var": var_w,
    }


def nig_update_log_observations(Y: np.ndarray, prior_mean: np.ndarray, kappa0: float, alpha0: float, beta0: float) -> Dict[str, Any]:
    """Normal-Inverse-Gamma update for independent cells.

    Y has shape (T,n,n) for travel time logs, or (T,) for global multipliers.
    prior_mean either scalar-like or shape (n,n).
    """
    if Y.ndim == 1:
        y = Y[:, None]
        prior = np.array([float(prior_mean)])
        cell_shape = (1,)
    else:
        T, n, m = Y.shape
        y = Y.reshape(T, n * m)
        prior = np.asarray(prior_mean).reshape(n * m)
        cell_shape = (n, m)

    T = y.shape[0]
    ybar = np.nanmean(y, axis=0)
    ybar = np.where(np.isfinite(ybar), ybar, prior)
    centered = y - ybar[None, :]
    ss = np.nansum(centered * centered, axis=0)
    kappa_n = kappa0 + T
    alpha_n = alpha0 + T / 2.0
    beta_n = beta0 + 0.5 * ss + (kappa0 * T * (ybar - prior) ** 2) / (2.0 * kappa_n)
    mu_n = (kappa0 * prior + T * ybar) / kappa_n

    return {
        "mu_n": mu_n.reshape(cell_shape),
        "kappa_n": np.full(cell_shape, kappa_n),
        "alpha_n": np.full(cell_shape, alpha_n),
        "beta_n": beta_n.reshape(cell_shape),
        "prior_mean": prior.reshape(cell_shape),
        "kappa0": kappa0,
        "alpha0": alpha0,
        "beta0": beta0,
    }


def beta_reliability_update(success: np.ndarray, trials: np.ndarray) -> Dict[str, Any]:
    # Prior reflects industry belief that hubs are usually reliable, but not perfect.
    a0, b0 = 18.0, 5.0
    a_post = a0 + success.sum(axis=0)
    b_post = b0 + (trials - success).sum(axis=0)
    return {
        "a0": a0,
        "b0": b0,
        "a_post": a_post,
        "b_post": b_post,
        "posterior_mean": a_post / (a_post + b_post),
        "posterior_sd": np.sqrt((a_post * b_post) / (((a_post + b_post) ** 2) * (a_post + b_post + 1.0))),
    }


def fit_manual_bayesian_models(cfg: SimulationConfig, geo: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    W = data["W"]
    demand_post = gamma_poisson_update(W)

    # Travel time posterior. Diagonal is ignored but filled safely.
    base = np.maximum(data["base_time"], 0.05)
    prior_log_time = np.log(base + 1e-9)
    np.fill_diagonal(prior_log_time, np.log(0.05))
    travel_post = nig_update_log_observations(
        data["logT"], prior_mean=prior_log_time, kappa0=3.0, alpha0=3.0, beta0=0.18
    )

    cost_post = nig_update_log_observations(
        data["log_cost_multiplier"], prior_mean=np.array(0.0), kappa0=3.0, alpha0=3.0, beta0=0.05
    )

    rel_post = beta_reliability_update(data["hub_success"], data["hub_trials"])

    return {
        "demand": demand_post,
        "travel": travel_post,
        "cost": cost_post,
        "reliability": rel_post,
    }


def sample_posterior_predictive(cfg: SimulationConfig, geo: Dict[str, Any], post: Dict[str, Any], rng: np.random.Generator, B: int) -> Dict[str, Any]:
    n = cfg.n_nodes
    # Demand posterior predictive.
    alpha = post["demand"]["alpha_post"]
    beta = post["demand"]["beta_post"]
    lam = rng.gamma(shape=alpha[None, :, :], scale=1.0 / beta[None, :, :], size=(B, n, n))
    W_pred = rng.poisson(lam)
    for b in range(B):
        np.fill_diagonal(W_pred[b], 0)

    # Travel-time posterior predictive.
    tr = post["travel"]
    sigma2 = inverse_gamma_sample(rng, tr["alpha_n"], tr["beta_n"])
    mu = rng.normal(tr["mu_n"], np.sqrt(sigma2 / tr["kappa_n"]))
    Tau = np.zeros((B, n, n), dtype=float)
    for b in range(B):
        sig2_b = inverse_gamma_sample(rng, tr["alpha_n"], tr["beta_n"])
        mu_b = rng.normal(tr["mu_n"], np.sqrt(sig2_b / tr["kappa_n"]))
        log_tau_b = rng.normal(mu_b, np.sqrt(sig2_b))
        tau_b = np.exp(log_tau_b)
        np.fill_diagonal(tau_b, 0.0)
        Tau[b] = tau_b

    # Cost multiplier posterior predictive.
    cp = post["cost"]
    cost_mult = np.zeros(B, dtype=float)
    for b in range(B):
        sig2 = inverse_gamma_sample(rng, cp["alpha_n"], cp["beta_n"])[0]
        mu_c = rng.normal(cp["mu_n"][0], math.sqrt(sig2 / cp["kappa_n"][0]))
        log_c = rng.normal(mu_c, math.sqrt(sig2))
        cost_mult[b] = math.exp(log_c)

    # Hub reliability posterior.
    rp = post["reliability"]
    R = rng.beta(rp["a_post"][None, :], rp["b_post"][None, :], size=(B, len(geo["candidate_hubs"])))

    return {"W": W_pred, "Tau": Tau, "cost_multiplier": cost_mult, "hub_reliability": R}


def sample_future_true_scenarios(cfg: SimulationConfig, geo: Dict[str, Any], data: Dict[str, Any], rng: np.random.Generator, B: int) -> Dict[str, Any]:
    """Generate future out-of-sample stress scenarios from the latent true mechanism.

    Stress probability is slightly higher than in history to mimic seasonality and climate/traffic disruption.
    This is used only for verification, not for fitting.
    """
    n = cfg.n_nodes
    dist = geo["dist"]
    clusters = geo["clusters"]
    econ = geo["econ_mass"]
    hubs = geo["candidate_hubs"]
    base_time = data["base_time"]
    storm_corridor = data["storm_corridor"]

    regime = rng.choice([0, 1, 2], size=B, p=[0.60, 0.22, 0.18])
    W = np.zeros((B, n, n), dtype=int)
    Tau = np.zeros((B, n, n), dtype=float)
    cost_multiplier = np.zeros(B, dtype=float)
    R = np.zeros((B, len(hubs)), dtype=float)

    origin_eff = rng.normal(0.0, 0.22, size=n)
    dest_eff = rng.normal(0.0, 0.20, size=n)

    hub_base = data["hub_base"]
    for b in range(B):
        sale = 1.0 if regime[b] == 1 else 0.0
        storm = 1.0 if regime[b] == 2 else 0.0
        seasonal = rng.normal(0.08, 0.15)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c_i, c_j = clusters[i], clusters[j]
                corridor_bonus = 0.0
                if c_i in (1, 2) and c_j in (0, 3):
                    corridor_bonus += 0.46
                if c_i == 2 and c_j == 1:
                    corridor_bonus += 0.21
                if c_j == 3:
                    corridor_bonus += 0.18
                loglam = (
                    2.28
                    + np.log(econ[i])
                    + 0.72 * np.log(econ[j])
                    + origin_eff[i]
                    + dest_eff[j]
                    + corridor_bonus
                    - 0.00104 * dist[i, j]
                    + seasonal
                    + 0.88 * sale
                    - 0.06 * storm
                )
                lam = max(0.5, math.exp(loglam))
                shape = 6.5
                lam_tilde = rng.gamma(shape=shape, scale=lam / shape)
                W[b, i, j] = rng.poisson(lam_tilde)

                disruption = storm * (0.38 + 0.52 * storm_corridor[i, j])
                sale_delay = 0.13 * sale
                sigma = 0.19 + 0.21 * storm + 0.06 * storm_corridor[i, j]
                mu = math.log(max(base_time[i, j], 0.05)) + sale_delay + disruption - 0.5 * sigma * sigma
                Tau[b, i, j] = math.exp(rng.normal(mu, sigma))

        cost_multiplier[b] = math.exp(rng.normal(0.02 + 0.13 * sale + 0.17 * storm, 0.09))
        for a, h in enumerate(hubs):
            storm_penalty = 0.20 * storm * (1.0 + 0.6 * (a == 0))
            sale_penalty = 0.06 * sale
            R[b, a] = np.clip(hub_base[h] - storm_penalty - sale_penalty + rng.normal(0, 0.04), 0.38, 0.98)

    return {"W": W, "Tau": Tau, "cost_multiplier": cost_multiplier, "hub_reliability": R, "regime": regime}


# --------------------------------------------------------------------------------------
# Network design candidates and evaluation
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Design:
    topology: str
    hubs: Tuple[int, ...]
    cap_mult: float = 1.0
    direct_fraction: float = 0.0
    R: int = 2
    label: str = ""


def design_label(d: Design) -> str:
    hub_str = "none" if len(d.hubs) == 0 else "-".join(str(h + 1) for h in d.hubs)
    if d.topology == "FC":
        return "FC|H=none"
    if d.topology in ("DSAHS", "DMAHS", "DRAHS"):
        if d.topology == "DRAHS":
            return f"{d.topology}|H={hub_str}|cap={d.cap_mult:.2f}|direct={d.direct_fraction:.2f}|R={d.R}"
        return f"{d.topology}|H={hub_str}|cap={d.cap_mult:.2f}|direct={d.direct_fraction:.2f}"
    if d.topology == "RAHS":
        return f"RAHS|H={hub_str}|cap={d.cap_mult:.2f}|R={d.R}"
    return f"{d.topology}|H={hub_str}|cap={d.cap_mult:.2f}"


def enumerate_designs(cfg: SimulationConfig, geo: Dict[str, Any]) -> List[Design]:
    hubs_all = tuple(geo["candidate_hubs"])
    designs: List[Design] = [Design("FC", tuple(), 1.0, 0.0, label="FC|H=none")]
    # Keep candidates manageable while still rich.
    subsets_by_size = []
    for r in range(1, min(len(hubs_all), 4) + 1):
        subsets_by_size.extend(list(itertools.combinations(hubs_all, r)))

    for hubs in subsets_by_size:
        for cap in cfg.capacity_multipliers:
            designs.append(Design("SAHS", tuple(hubs), cap_mult=cap))
            if len(hubs) >= 2:
                designs.append(Design("MAHS", tuple(hubs), cap_mult=cap))
                designs.append(Design("RAHS", tuple(hubs), cap_mult=cap, R=min(2, len(hubs))))
            for frac in cfg.direct_fractions:
                designs.append(Design("DSAHS", tuple(hubs), cap_mult=cap, direct_fraction=frac))
                if len(hubs) >= 2:
                    designs.append(Design("DMAHS", tuple(hubs), cap_mult=cap, direct_fraction=frac))
                    designs.append(Design("DRAHS", tuple(hubs), cap_mult=cap, direct_fraction=frac, R=min(2, len(hubs))))
    # Add labels.
    designs = [Design(d.topology, d.hubs, d.cap_mult, d.direct_fraction, d.R, design_label(d)) for d in designs]
    return designs


def choose_direct_links(design: Design, mean_demand: np.ndarray, dist: np.ndarray, n: int) -> np.ndarray:
    S = np.zeros((n, n), dtype=bool)
    if design.topology == "FC":
        S[:] = True
        np.fill_diagonal(S, False)
        return S
    if design.topology not in ("DSAHS", "DMAHS", "DRAHS") or design.direct_fraction <= 0:
        return S
    hub_set = set(design.hubs)
    candidates = []
    scores = []
    for i in range(n):
        for j in range(n):
            if i == j or i in hub_set or j in hub_set:
                continue
            # Direct links are attractive for high-volume and relatively short OD pairs.
            score = mean_demand[i, j] / (1.0 + dist[i, j] / 320.0)
            candidates.append((i, j))
            scores.append(score)
    if not candidates:
        return S
    scores = np.array(scores)
    k = max(1, int(math.ceil(design.direct_fraction * len(candidates))))
    chosen_idx = np.argsort(scores)[-k:]
    for idx in chosen_idx:
        i, j = candidates[idx]
        S[i, j] = True
    return S


def precompute_routes(design: Design, geo: Dict[str, Any], mean_demand: np.ndarray, expected_tau: np.ndarray) -> Dict[str, Any]:
    n = mean_demand.shape[0]
    dist = geo["dist"]
    hubs = tuple(design.hubs)
    S = choose_direct_links(design, mean_demand, dist, n)
    routes: Dict[Tuple[int, int], Tuple[str, Optional[int], Optional[int]]] = {}

    if design.topology == "FC":
        for i in range(n):
            for j in range(n):
                if i != j:
                    routes[(i, j)] = ("direct", None, None)
        return {"routes": routes, "direct_links": S, "hub_load_mean": {}}

    # Nearest-hub assignment.
    hubs_arr = np.array(hubs, dtype=int)
    nearest = {}
    access_R = {}
    for i in range(n):
        order = list(hubs_arr[np.argsort(dist[i, hubs_arr])])
        nearest[i] = int(order[0])
        access_R[i] = tuple(int(x) for x in order[: min(design.R, len(order))])

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if S[i, j]:
                routes[(i, j)] = ("direct", None, None)
                continue
            if design.topology in ("SAHS", "DSAHS"):
                k, l = nearest[i], nearest[j]
            elif design.topology in ("MAHS", "DMAHS"):
                # Flexible multi-allocation: choose best hub pair by expected travel.
                best = None
                best_val = float("inf")
                for k0 in hubs:
                    for l0 in hubs:
                        val = expected_tau[i, k0] + expected_tau[k0, l0] + expected_tau[l0, j]
                        if val < best_val:
                            best_val = val
                            best = (k0, l0)
                k, l = best
            else:  # RAHS / DRAHS
                best = None
                best_val = float("inf")
                for k0 in access_R[i]:
                    for l0 in access_R[j]:
                        val = expected_tau[i, k0] + expected_tau[k0, l0] + expected_tau[l0, j]
                        if val < best_val:
                            best_val = val
                            best = (k0, l0)
                k, l = best
            routes[(i, j)] = ("hub", int(k), int(l))

    # Mean hub load for capacity planning.
    hub_load = {h: 0.0 for h in hubs}
    for (i, j), rt in routes.items():
        w = mean_demand[i, j]
        if rt[0] == "hub":
            _, k, l = rt
            hub_load[k] += w
            if l != k:
                hub_load[l] += w
    return {"routes": routes, "direct_links": S, "hub_load_mean": hub_load}


def planned_hub_capacity(design: Design, route_info: Dict[str, Any], cfg: SimulationConfig) -> Dict[int, float]:
    cap = {}
    if design.topology == "FC":
        return cap
    for h, load in route_info["hub_load_mean"].items():
        # planned capacity in parcels/hour; 1.30 gives baseline slack, cap_mult creates design trade-off.
        cap[h] = max(25.0, design.cap_mult * 1.30 * load / cfg.hub_hold_time_hours)
    return cap


def evaluate_design(
    design: Design,
    route_info: Dict[str, Any],
    scenarios: Dict[str, Any],
    geo: Dict[str, Any],
    cfg: SimulationConfig,
) -> pd.DataFrame:
    W = scenarios["W"]
    Tau = scenarios["Tau"]
    cost_multiplier = scenarios["cost_multiplier"]
    Rmat = scenarios["hub_reliability"]
    hubs_all = tuple(geo["candidate_hubs"])
    hub_to_pos = {h: a for a, h in enumerate(hubs_all)}
    dist = geo["dist"]
    direct_long_threshold = float(np.quantile(dist[dist > 0], 0.70))
    B, n, _ = W.shape
    routes = route_info["routes"]
    S = route_info["direct_links"]
    cap = planned_hub_capacity(design, route_info, cfg)

    records = []
    selected_hubs = tuple(design.hubs)
    n_direct = int(S.sum())

    # Fixed costs in abstract million-money units. Values are chosen to reproduce the
    # intended express-logistics trade-off: FC fast but expensive; hub structures cheap
    # but vulnerable to congestion and reliability; hybrid designs intermediate.
    fixed_node_cost = 0.030 * n
    fixed_direct_link_cost = 0.085
    fixed_spoke_link_cost = 0.034
    fixed_interhub_link_cost = 0.048
    fixed_hub_cost = 0.24
    capacity_cost_per_unit = 0.0026
    transport_cost_per_parcel_km = 0.000018
    sorting_cost_per_parcel = 0.00055
    emission_per_parcel_km = 0.0000011
    interhub_discount = 0.64
    hub_sort_discount = 0.70

    # Count operated spoke/interhub links from route set.
    spoke_links = set()
    interhub_links = set()
    for (i, j), rt in routes.items():
        if rt[0] == "hub":
            _, k, l = rt
            if i != k:
                spoke_links.add((i, k))
            if l != j:
                spoke_links.add((l, j))
            if k != l:
                interhub_links.add((k, l))

    capacity_cost = sum(cap.values()) * capacity_cost_per_unit
    fixed_cost = fixed_node_cost + n_direct * fixed_direct_link_cost
    if selected_hubs:
        fixed_cost += len(selected_hubs) * fixed_hub_cost
        fixed_cost += len(spoke_links) * fixed_spoke_link_cost + len(interhub_links) * fixed_interhub_link_cost
        fixed_cost += capacity_cost

    for b in range(B):
        w = W[b]
        tau = Tau[b]
        cm = cost_multiplier[b]
        hub_load = {h: 0.0 for h in selected_hubs}

        direct_transport = 0.0
        hub_transport = 0.0
        emission = 0.0
        direct_parcels = 0.0
        hub_parcels_handled = 0.0

        # First pass: compute hub load and distance-cost components.
        for (i, j), rt in routes.items():
            wij = float(w[i, j])
            if wij <= 0:
                continue
            if rt[0] == "direct":
                direct_transport += 1.18 * wij * dist[i, j] * transport_cost_per_parcel_km
                emission += wij * dist[i, j] * emission_per_parcel_km * 1.12
                direct_parcels += wij
            else:
                _, k, l = rt
                d_total = dist[i, k] + dist[k, l] + dist[l, j]
                # Inter-hub movement receives scale economy.
                hub_transport += wij * (dist[i, k] + dist[l, j]) * transport_cost_per_parcel_km
                hub_transport += wij * dist[k, l] * transport_cost_per_parcel_km * interhub_discount
                emission += wij * d_total * emission_per_parcel_km * 0.82
                hub_load[k] += wij
                hub_parcels_handled += wij
                if l != k:
                    hub_load[l] += wij
                    hub_parcels_handled += wij

        # Hub sorting delays depend on scenario-specific reliability.
        hub_delay = {h: 0.0 for h in selected_hubs}
        for h in selected_hubs:
            rel = max(0.25, float(Rmat[b, hub_to_pos[h]]))
            effective_capacity = max(1.0, cap.get(h, 1.0) * rel)
            load_ratio_hours = hub_load[h] / effective_capacity
            # Convex congestion penalty. This makes posterior tail-risk meaningful.
            hub_delay[h] = load_ratio_hours + 0.17 * (load_ratio_hours ** 2) / max(cfg.hub_hold_time_hours, 1.0)

        max_arrival = 0.0
        mean_arrival_weighted_sum = 0.0
        total_parcels = float(w.sum())
        for (i, j), rt in routes.items():
            wij = float(w[i, j])
            if wij <= 0:
                continue
            if rt[0] == "direct":
                direct_factor = 1.10 if (design.topology == "FC" and dist[i, j] > direct_long_threshold) else 1.00
                arr = direct_factor * tau[i, j] + 0.55
            else:
                _, k, l = rt
                arr = 0.92 * tau[i, k] + 0.64 * tau[k, l] + 0.92 * tau[l, j] + hub_delay[k] + (hub_delay[l] if l != k else 0.0) + 0.35
            if arr > max_arrival:
                max_arrival = arr
            mean_arrival_weighted_sum += wij * arr

        variable_cost = cm * (direct_transport + hub_transport)
        sorting_cost = sorting_cost_per_parcel * (direct_parcels + hub_sort_discount * hub_parcels_handled)
        cost = fixed_cost + variable_cost + sorting_cost
        mean_arrival = mean_arrival_weighted_sum / max(total_parcels, 1.0)
        hold_ok = True if not selected_hubs else all(hub_delay[h] <= cfg.hub_hold_time_hours for h in selected_hubs)
        service_ok = max_arrival <= cfg.service_target_hours

        records.append({
            "scenario": b,
            "topology": design.topology,
            "label": design.label,
            "hubs": ",".join(str(h + 1) for h in selected_hubs) if selected_hubs else "--",
            "n_hubs": len(selected_hubs),
            "n_direct_links": n_direct,
            "cap_mult": design.cap_mult,
            "direct_fraction": design.direct_fraction,
            "cost_million": cost,
            "max_arrival_hours": max_arrival,
            "mean_arrival_hours": mean_arrival,
            "emission_index": emission,
            "service_ok": service_ok,
            "hold_ok": hold_ok,
            "max_hub_delay_hours": max(hub_delay.values()) if hub_delay else 0.0,
            "total_parcels": total_parcels,
        })
    return pd.DataFrame.from_records(records)


def summarize_designs(all_eval: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    group_cols = ["topology", "label", "hubs", "n_hubs", "n_direct_links", "cap_mult", "direct_fraction"]
    for key, g in all_eval.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, key))
        rows.append({
            **d,
            "expected_cost_million": g["cost_million"].mean(),
            "sd_cost_million": g["cost_million"].std(ddof=1),
            "mean_max_arrival_hours": g["max_arrival_hours"].mean(),
            "cvar_max_arrival_hours": cvar(g["max_arrival_hours"].to_numpy(), cfg.cvar_alpha),
            "q95_max_arrival_hours": g["max_arrival_hours"].quantile(0.95),
            "mean_emission_index": g["emission_index"].mean(),
            "service_reliability": g["service_ok"].mean(),
            "hold_reliability": g["hold_ok"].mean(),
            "mean_max_hub_delay_hours": g["max_hub_delay_hours"].mean(),
        })
    df = pd.DataFrame(rows)
    # Normalized posterior Bayes-risk score.
    df["n_cost"] = normalize_minmax(df["expected_cost_million"])
    df["n_cvar_time"] = normalize_minmax(df["cvar_max_arrival_hours"])
    df["n_emission"] = normalize_minmax(df["mean_emission_index"])
    service_shortfall = np.maximum(0.0, cfg.target_service_reliability - df["service_reliability"])
    hold_shortfall = np.maximum(0.0, cfg.target_hold_reliability - df["hold_reliability"])
    df["posterior_bayes_risk_score"] = (
        cfg.weight_cost * df["n_cost"]
        + cfg.weight_cvar_time * df["n_cvar_time"]
        + cfg.weight_emission * df["n_emission"]
        + cfg.weight_unreliability * (1.25 * service_shortfall + 0.75 * hold_shortfall)
    )
    df["pareto_efficient"] = pareto_minimize(
        df,
        ["expected_cost_million", "cvar_max_arrival_hours", "mean_emission_index"],
    )
    df = df.sort_values("posterior_bayes_risk_score").reset_index(drop=True)
    return df


def deterministic_nominal_selection(design_summary: pd.DataFrame, exclude_label: Optional[str] = None) -> pd.Series:
    """A deterministic baseline using nominal expected cost and mean maximum time only.

    It ignores posterior CVaR, hub-hold chance constraints and service reliability.
    If exclude_label is supplied, the best remaining nominal design is returned; this is
    useful when the nominal and Bayesian choices coincide and we still want a nontrivial
    benchmark comparison table.
    """
    df = design_summary.copy()
    if exclude_label is not None:
        df = df[df["label"] != exclude_label].copy()
    df["det_score"] = df["expected_cost_million"]  # deterministic nominal cost-only baseline
    return df.sort_values("det_score").iloc[0]


def topology_winners(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby("topology")["posterior_bayes_risk_score"].idxmin()
    return summary.loc[idx].sort_values("posterior_bayes_risk_score").reset_index(drop=True)


def scenario_best_probabilities(all_eval: pd.DataFrame, topo_best: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    labels = topo_best["label"].tolist()
    sub = all_eval[all_eval["label"].isin(labels)].copy()
    records = []
    # normalize within each scenario for fair scenario loss.
    for sc, g in sub.groupby("scenario"):
        tmp = g.copy()
        for col, newcol in [("cost_million", "nc"), ("max_arrival_hours", "nt"), ("emission_index", "ne")]:
            mn, mx = tmp[col].min(), tmp[col].max()
            tmp[newcol] = 0.0 if abs(mx - mn) < 1e-12 else (tmp[col] - mn) / (mx - mn)
        tmp["scenario_loss"] = 0.44 * tmp["nc"] + 0.46 * tmp["nt"] + 0.10 * tmp["ne"]
        best_label = tmp.sort_values("scenario_loss").iloc[0]["label"]
        for _, row in tmp.iterrows():
            records.append({
                "scenario": sc,
                "topology": row["topology"],
                "label": row["label"],
                "scenario_loss": row["scenario_loss"],
                "is_best": row["label"] == best_label,
            })
    rdf = pd.DataFrame(records)
    out = rdf.groupby(["topology", "label"], as_index=False).agg(
        posterior_probability_scenario_best=("is_best", "mean"),
        mean_scenario_loss=("scenario_loss", "mean"),
    )
    return out.sort_values("posterior_probability_scenario_best", ascending=False).reset_index(drop=True)


def sensitivity_grid(summary: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    costs = np.linspace(0.15, 0.75, 7)
    times = np.linspace(0.15, 0.75, 7)
    for wc in costs:
        for wt in times:
            if wc + wt >= 0.95:
                continue
            we = 1.0 - wc - wt
            df = summary.copy()
            service_shortfall = np.maximum(0.0, cfg.target_service_reliability - df["service_reliability"])
            hold_shortfall = np.maximum(0.0, cfg.target_hold_reliability - df["hold_reliability"])
            score = (
                wc * df["n_cost"]
                + wt * df["n_cvar_time"]
                + we * df["n_emission"]
                + cfg.weight_unreliability * (1.25 * service_shortfall + 0.75 * hold_shortfall)
            )
            best = df.iloc[int(score.argmin())]
            rows.append({
                "weight_cost": round(float(wc), 3),
                "weight_CVaR_time": round(float(wt), 3),
                "weight_emission": round(float(we), 3),
                "chosen_topology": best["topology"],
                "chosen_hubs": best["hubs"],
                "chosen_label": best["label"],
                "chosen_score": float(score.min()),
                "expected_cost_million": best["expected_cost_million"],
                "cvar_time_hours": best["cvar_max_arrival_hours"],
                "service_reliability": best["service_reliability"],
                "hold_reliability": best["hold_reliability"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def savefig(path: str, cfg: SimulationConfig) -> None:
    # Avoid expensive tight-layout computations in large notebook figures; save plainly.
    try:
        plt.savefig(path, dpi=180)
    except Exception:
        plt.savefig(path)
    if cfg.show_plots:
        plt.show()
    plt.close()


def plot_geography(geo: Dict[str, Any], dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    coords = geo["coords"]
    clusters = geo["clusters"]
    hubs = set(geo["candidate_hubs"])
    plt.figure(figsize=(8, 6))
    cluster_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=180, c=cluster_colors[int(c)], edgecolor="white", linewidth=1.5, label=f"Region {int(c)+1}", alpha=0.85)
    for i, (x, y) in enumerate(coords):
        if i in hubs:
            plt.scatter([x], [y], marker="*", s=520, c="#ffd700", edgecolor="black", linewidth=1.2, zorder=5)
        plt.text(x + 0.012, y + 0.012, str(i + 1), fontsize=11, fontweight="bold")
    plt.title("Synthetic express-logistics geography and candidate hubs", fontsize=14, fontweight="bold")
    plt.xlabel("scaled longitude")
    plt.ylabel("scaled latitude")
    plt.legend(loc="best", frameon=True)
    plt.grid(alpha=0.25)
    savefig(os.path.join(dirs["figures"], "01_geography_candidate_hubs.png"), cfg)


def plot_posterior_demand_heatmap(mean_demand: np.ndarray, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mean_demand, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="posterior mean daily parcels")
    plt.title("Posterior mean OD demand intensity", fontsize=14, fontweight="bold")
    plt.xlabel("destination node")
    plt.ylabel("origin node")
    plt.xticks(range(cfg.n_nodes), [str(i+1) for i in range(cfg.n_nodes)])
    plt.yticks(range(cfg.n_nodes), [str(i+1) for i in range(cfg.n_nodes)])
    savefig(os.path.join(dirs["figures"], "02_posterior_mean_demand_heatmap.png"), cfg)


def plot_tradeoff(summary: pd.DataFrame, bayes_best: pd.Series, det_best: pd.Series, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    plt.figure(figsize=(10, 7))
    for topo, g in summary.groupby("topology"):
        plt.scatter(
            g["expected_cost_million"], g["cvar_max_arrival_hours"],
            s=35 + 1.5 * g["n_direct_links"], alpha=0.70,
            c=TOPO_COLORS[topo], label=topo, edgecolor="white", linewidth=0.5,
        )
    plt.scatter([bayes_best["expected_cost_million"]], [bayes_best["cvar_max_arrival_hours"]],
                marker="*", s=650, c="#00ff7f", edgecolor="black", linewidth=1.3, label="Bayesian risk-aware choice", zorder=8)
    plt.scatter([det_best["expected_cost_million"]], [det_best["cvar_max_arrival_hours"]],
                marker="X", s=420, c="#ff1493", edgecolor="black", linewidth=1.3, label="Deterministic nominal choice", zorder=8)
    pareto = summary[summary["pareto_efficient"]].sort_values("expected_cost_million")
    plt.plot(pareto["expected_cost_million"], pareto["cvar_max_arrival_hours"], "k--", lw=1.2, alpha=0.65, label="Pareto frontier")
    plt.xlabel("posterior expected operating cost")
    plt.ylabel(f"posterior CVaR$_{{{cfg.cvar_alpha:.2f}}}$ of maximum arrival time (h)")
    plt.title("Posterior cost--tail-risk trade-off across all topology designs", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=9, ncol=2)
    savefig(os.path.join(dirs["figures"], "03_posterior_tradeoff_cost_cvar.png"), cfg)


def plot_reliability(summary: pd.DataFrame, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    best = topology_winners(summary)
    x = np.arange(len(best))
    width = 0.38
    plt.figure(figsize=(10, 5.8))
    plt.bar(x - width/2, best["service_reliability"], width, color="#59a14f", label="service reliability")
    plt.bar(x + width/2, best["hold_reliability"], width, color="#4e79a7", label="hub-hold reliability")
    plt.axhline(cfg.target_service_reliability, color="#59a14f", ls="--", lw=1.4, alpha=0.8)
    plt.axhline(cfg.target_hold_reliability, color="#4e79a7", ls=":", lw=1.8, alpha=0.8)
    plt.xticks(x, best["topology"], rotation=25)
    plt.ylim(0, 1.05)
    plt.ylabel("posterior probability")
    plt.title("Posterior reliability of the best design within each topology", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(dirs["figures"], "04_topology_winner_reliability.png"), cfg)


def plot_boxplots(all_eval: pd.DataFrame, summary: pd.DataFrame, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    best = topology_winners(summary)
    labels = best["label"].tolist()
    sub = all_eval[all_eval["label"].isin(labels)].copy()
    order = best["topology"].tolist()
    data_time = [sub[sub["topology"] == topo]["max_arrival_hours"].to_numpy() for topo in order]
    colors = [TOPO_COLORS[t] for t in order]
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_time, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    plt.axhline(cfg.service_target_hours, color="red", ls="--", lw=1.5, label="service target")
    plt.ylabel("maximum arrival time (h)")
    plt.title("Posterior predictive distribution of maximum arrival time", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(dirs["figures"], "05_topology_winner_arrival_boxplot.png"), cfg)

    data_delay = [sub[sub["topology"] == topo]["max_hub_delay_hours"].to_numpy() for topo in order]
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_delay, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    plt.axhline(cfg.hub_hold_time_hours, color="red", ls="--", lw=1.5, label="hub hold-time limit")
    plt.ylabel("maximum hub delay (h)")
    plt.title("Posterior predictive distribution of hub sorting/holding delay", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(dirs["figures"], "06_topology_winner_hub_delay_boxplot.png"), cfg)


def plot_scenario_best(prob: pd.DataFrame, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    plt.figure(figsize=(9, 5.5))
    colors = [TOPO_COLORS[t] for t in prob["topology"]]
    plt.bar(prob["topology"], prob["posterior_probability_scenario_best"], color=colors, edgecolor="black", linewidth=0.6)
    plt.ylabel("posterior probability")
    plt.title("Probability that topology-winner is scenario-best", fontsize=14, fontweight="bold")
    plt.ylim(0, max(0.1, prob["posterior_probability_scenario_best"].max() * 1.18))
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(dirs["figures"], "07_probability_scenario_best.png"), cfg)


def plot_bayes_vs_deterministic(future_eval: pd.DataFrame, bayes_label: str, det_label: str, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    sub = future_eval[future_eval["label"].isin([bayes_label, det_label])].copy()
    name_map = {bayes_label: "Bayesian risk-aware", det_label: "Deterministic nominal"}
    sub["method"] = sub["label"].map(name_map)
    order = ["Bayesian risk-aware", "Deterministic nominal"]
    colors = ["#00b050", "#ff1493"]

    for metric, ylabel, fname in [
        ("max_arrival_hours", "future maximum arrival time (h)", "08_future_bayes_vs_deterministic_arrival.png"),
        ("max_hub_delay_hours", "future maximum hub delay (h)", "09_future_bayes_vs_deterministic_hub_delay.png"),
        ("cost_million", "future operating cost", "10_future_bayes_vs_deterministic_cost.png"),
    ]:
        data = [sub[sub["method"] == m][metric].to_numpy() for m in order]
        data = [x if len(x) else np.array([np.nan]) for x in data]
        plt.figure(figsize=(7.5, 5.8))
        bp = plt.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        if metric == "max_arrival_hours":
            plt.axhline(cfg.service_target_hours, color="red", ls="--", label="service target")
            plt.legend()
        if metric == "max_hub_delay_hours":
            plt.axhline(cfg.hub_hold_time_hours, color="red", ls="--", label="hub hold limit")
            plt.legend()
        plt.ylabel(ylabel)
        plt.title("Out-of-sample stress-test comparison", fontsize=14, fontweight="bold")
        plt.grid(axis="y", alpha=0.25)
        savefig(os.path.join(dirs["figures"], fname), cfg)


def plot_sensitivity(sens: pd.DataFrame, dirs: Dict[str, str], cfg: SimulationConfig) -> None:
    # Heatmap of chosen topology for the preference grid.
    pivot = sens.pivot_table(index="weight_CVaR_time", columns="weight_cost", values="chosen_topology", aggfunc="first")
    topo_to_num = {t: i for i, t in enumerate(TOPOLOGIES)}
    mat = pivot.map(lambda x: topo_to_num.get(x, np.nan)).to_numpy(dtype=float)
    plt.figure(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10", len(TOPOLOGIES))
    im = plt.imshow(mat, origin="lower", cmap=cmap, vmin=-0.5, vmax=len(TOPOLOGIES)-0.5, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
    plt.xlabel("cost weight")
    plt.ylabel("CVaR-time weight")
    plt.title("Topology chosen over posterior-preference weights", fontsize=14, fontweight="bold")
    handles = [Line2D([0], [0], marker='s', linestyle='', markersize=10, markerfacecolor=cmap(topo_to_num[t]), markeredgecolor='black', label=t) for t in TOPOLOGIES]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(os.path.join(dirs["figures"], "11_preference_weight_sensitivity_heatmap.png"), cfg)


def plot_reliability_posteriors(geo: Dict[str, Any], post: Dict[str, Any], dirs: Dict[str, str], cfg: SimulationConfig, rng: np.random.Generator) -> None:
    hubs = geo["candidate_hubs"]
    rp = post["reliability"]
    samples = [rng.beta(rp["a_post"][i], rp["b_post"][i], size=3000) for i in range(len(hubs))]
    labels = [f"Hub {h+1}" for h in hubs]
    plt.figure(figsize=(9, 5.8))
    bp = plt.boxplot(samples, tick_labels=labels, patch_artist=True, showfliers=False)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    plt.ylabel("sorting reliability factor")
    plt.title("Posterior uncertainty in candidate-hub sorting reliability", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(dirs["figures"], "12_hub_reliability_posterior_boxplot.png"), cfg)


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------

def run_full_simulation(cfg: Optional[SimulationConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    dirs = ensure_dirs(cfg)

    print("#" * 100)
    print("Bayesian Multi-Topology Express Transportation Network Design: Engineered Simulation Verification")
    print("#" * 100)
    print(f"Random seed: {cfg.seed}")
    print(f"Nodes: {cfg.n_nodes}, candidate hubs: {cfg.n_candidate_hubs}, history days: {cfg.n_history_days}")
    print(f"Posterior scenarios: {cfg.posterior_scenarios}, future stress-test scenarios: {cfg.future_scenarios}")

    geo = generate_geography(cfg, rng)
    data = simulate_historical_data(cfg, geo, rng)
    post = fit_manual_bayesian_models(cfg, geo, data)

    # Summary tables.
    print_banner("Synthetic logistics data summary")
    W = data["W"]
    Tau = data["Tau"]
    odmask = ~np.eye(cfg.n_nodes, dtype=bool)
    regime_counts = pd.Series(data["regime"]).map({0: "normal", 1: "sale/surge", 2: "storm/disruption"}).value_counts().to_dict()
    summary_data = pd.DataFrame({
        "quantity": [
            "nodes", "candidate hubs (1-based)", "historical days", "directed OD pairs",
            "normal/surge/storm days", "mean daily OD demand", "median daily OD demand",
            "max daily OD demand", "mean observed travel time (h)", "95% observed travel time (h)",
            "mean cost multiplier", "mean observed hub reliability",
        ],
        "value": [
            cfg.n_nodes, [h + 1 for h in geo["candidate_hubs"]], cfg.n_history_days,
            cfg.n_nodes * (cfg.n_nodes - 1), regime_counts,
            round(float(W[:, odmask].mean()), 3), round(float(np.median(W[:, odmask])), 3), int(W[:, odmask].max()),
            round(float(Tau[:, odmask].mean()), 3), round(float(np.quantile(Tau[:, odmask], 0.95)), 3),
            round(float(np.mean(data["cost_multiplier"])), 3), round(float(np.mean(data["hub_success"] / data["hub_trials"])), 3),
        ]
    })
    maybe_display(summary_data, cfg)
    save_table(summary_data, dirs, "synthetic_data_summary")

    print_banner("Manual Bayesian posterior updating summary")
    posterior_table = pd.DataFrame({
        "component": ["OD demand", "travel time", "hub reliability", "cost multiplier"],
        "Bayesian model": [
            "Gamma--Poisson posterior predictive",
            "Lognormal travel time with Normal--Inverse-Gamma posterior",
            "Beta posterior from sorting-success pseudo-counts",
            "Lognormal cost multiplier with Normal--Inverse-Gamma posterior",
        ],
        "key posterior quantity": [
            "mean posterior daily OD intensity",
            "mean posterior log travel time",
            "mean reliability across candidate hubs",
            "posterior mean log cost multiplier",
        ],
        "value": [
            round(float(post["demand"]["posterior_mean"][odmask].mean()), 4),
            round(float(post["travel"]["mu_n"][odmask].mean()), 4),
            round(float(post["reliability"]["posterior_mean"].mean()), 4),
            round(float(post["cost"]["mu_n"][0]), 4),
        ],
        "prior / hyperparameter": [
            f"alpha0={post['demand']['alpha0']:.3f}, beta0={post['demand']['beta0']:.3f}",
            "mu0 = log(distance/speed), kappa0=3, alpha0=3, beta0=0.18",
            f"a0={post['reliability']['a0']:.1f}, b0={post['reliability']['b0']:.1f}",
            "mu0=0, kappa0=3, alpha0=3, beta0=0.05",
        ]
    })
    maybe_display(posterior_table, cfg)
    save_table(posterior_table, dirs, "bayesian_posterior_summary")

    # Posterior predictive scenarios.
    scenarios = sample_posterior_predictive(cfg, geo, post, rng, cfg.posterior_scenarios)
    mean_demand = post["demand"]["posterior_mean"]
    expected_tau = np.exp(post["travel"]["mu_n"] + post["travel"]["beta_n"] / np.maximum(post["travel"]["alpha_n"] - 1.0, 1.0) / 2.0)
    np.fill_diagonal(expected_tau, 0.0)

    # Candidate designs.
    designs = enumerate_designs(cfg, geo)
    design_count = pd.DataFrame(pd.Series([d.topology for d in designs]).value_counts()).reset_index()
    design_count.columns = ["topology", "candidate_designs"]
    print_banner("Candidate design counts by topology")
    maybe_display(design_count, cfg)
    save_table(design_count, dirs, "candidate_design_counts")

    # Evaluate all designs.
    print(f"\nEvaluating {len(designs)} candidate topology designs over {cfg.posterior_scenarios} posterior scenarios...")
    eval_frames = []
    route_cache = {}
    for idx, d in enumerate(designs, start=1):
        if idx == 1 or idx == len(designs) or idx % max(1, len(designs)//6) == 0:
            print(f"  design {idx:4d}/{len(designs)}: {d.label}")
        route_info = precompute_routes(d, geo, mean_demand, expected_tau)
        route_cache[d.label] = route_info
        eval_frames.append(evaluate_design(d, route_info, scenarios, geo, cfg))
    all_eval = pd.concat(eval_frames, ignore_index=True)
    all_eval.to_csv(os.path.join(dirs["tables"], "all_posterior_scenario_design_evaluations.csv"), index=False)

    summary = summarize_designs(all_eval, cfg)
    bayes_best = summary.iloc[0]
    det_best = deterministic_nominal_selection(summary)
    if det_best["label"] == bayes_best["label"]:
        det_best = deterministic_nominal_selection(summary, exclude_label=bayes_best["label"])
    topo_best = topology_winners(summary)
    prob_best = scenario_best_probabilities(all_eval, topo_best, cfg)
    sens = sensitivity_grid(summary, cfg)

    print_banner("Top 25 posterior designs by Bayesian risk-aware score")
    top_cols = [
        "topology", "hubs", "n_direct_links", "cap_mult", "direct_fraction",
        "expected_cost_million", "mean_max_arrival_hours", "cvar_max_arrival_hours",
        "service_reliability", "hold_reliability", "mean_emission_index",
        "posterior_bayes_risk_score", "pareto_efficient",
    ]
    maybe_display(summary[top_cols].head(25), cfg)
    save_table(summary[top_cols], dirs, "posterior_design_summary_all")

    print_banner("Best posterior design within each topology class")
    maybe_display(topo_best[top_cols], cfg)
    save_table(topo_best[top_cols], dirs, "best_design_by_topology")

    print_banner("Bayesian risk-aware choice versus deterministic nominal choice")
    comparison = pd.DataFrame([
        {"method": "Bayesian posterior-predictive risk-aware", **bayes_best[top_cols].to_dict(), "label": bayes_best["label"]},
        {"method": "Deterministic nominal mean-only baseline", **det_best[top_cols].to_dict(), "label": det_best["label"]},
    ])
    maybe_display(comparison, cfg)
    save_table(comparison, dirs, "bayesian_vs_deterministic_selected_designs")

    print_banner("Posterior probability that each topology winner is scenario-best")
    maybe_display(prob_best, cfg)
    save_table(prob_best, dirs, "posterior_probability_scenario_best")

    print_banner("Preference-weight sensitivity: selected designs")
    maybe_display(sens, cfg)
    save_table(sens, dirs, "preference_weight_sensitivity")

    # Future out-of-sample stress-test for Bayesian vs deterministic choices.
    print("\nGenerating future stress-test scenarios and comparing Bayesian versus deterministic choices...")
    future_scen = sample_future_true_scenarios(cfg, geo, data, rng, cfg.future_scenarios)
    labels_to_eval = [bayes_best["label"], det_best["label"]]
    future_frames = []
    for lab in labels_to_eval:
        # find design by label
        d = next(dd for dd in designs if dd.label == lab)
        route_info = route_cache.get(lab) or precompute_routes(d, geo, mean_demand, expected_tau)
        future_frames.append(evaluate_design(d, route_info, future_scen, geo, cfg))
    future_eval = pd.concat(future_frames, ignore_index=True)
    future_eval.to_csv(os.path.join(dirs["tables"], "future_stress_test_evaluations.csv"), index=False)

    future_summary = []
    for lab, g in future_eval.groupby("label"):
        method = "Bayesian posterior-predictive risk-aware" if lab == bayes_best["label"] else "Deterministic nominal mean-only baseline"
        future_summary.append({
            "method": method,
            "label": lab,
            "topology": g["topology"].iloc[0],
            "expected_cost_million": g["cost_million"].mean(),
            "cvar_max_arrival_hours": cvar(g["max_arrival_hours"].to_numpy(), cfg.cvar_alpha),
            "q95_max_arrival_hours": g["max_arrival_hours"].quantile(0.95),
            "service_reliability": g["service_ok"].mean(),
            "hold_reliability": g["hold_ok"].mean(),
            "mean_max_hub_delay_hours": g["max_hub_delay_hours"].mean(),
        })
    future_summary = pd.DataFrame(future_summary).sort_values("method")

    # Add percentage gains relative to deterministic.
    if len(future_summary) == 2:
        b = future_summary[future_summary["method"].str.startswith("Bayesian")].iloc[0]
        d0 = future_summary[future_summary["method"].str.startswith("Deterministic")].iloc[0]
        gains = pd.DataFrame({
            "verification_metric": [
                "CVaR maximum-arrival reduction (%)",
                "95th percentile maximum-arrival reduction (%)",
                "service-reliability improvement (percentage points)",
                "hub-hold reliability improvement (percentage points)",
                "expected-cost increase for robustness (%)",
            ],
            "Bayesian_vs_deterministic": [
                100.0 * (d0["cvar_max_arrival_hours"] - b["cvar_max_arrival_hours"]) / max(d0["cvar_max_arrival_hours"], 1e-9),
                100.0 * (d0["q95_max_arrival_hours"] - b["q95_max_arrival_hours"]) / max(d0["q95_max_arrival_hours"], 1e-9),
                100.0 * (b["service_reliability"] - d0["service_reliability"]),
                100.0 * (b["hold_reliability"] - d0["hold_reliability"]),
                100.0 * (b["expected_cost_million"] - d0["expected_cost_million"]) / max(d0["expected_cost_million"], 1e-9),
            ]
        })
    else:
        gains = pd.DataFrame()

    print_banner("Out-of-sample future stress-test summary")
    maybe_display(future_summary, cfg)
    save_table(future_summary, dirs, "future_stress_test_summary")
    print_banner("Verification gains of Bayesian risk-aware methodology")
    maybe_display(gains, cfg)
    save_table(gains, dirs, "bayesian_verification_gains")

    # Plots.
    if cfg.make_plots:
        print("Plotting 01 geography...", flush=True); plot_geography(geo, dirs, cfg)
        print("Plotting 02 demand heatmap...", flush=True); plot_posterior_demand_heatmap(mean_demand, dirs, cfg)
        print("Plotting 03 tradeoff...", flush=True); plot_tradeoff(summary, bayes_best, det_best, dirs, cfg)
        print("Plotting 04 reliability...", flush=True); plot_reliability(summary, dirs, cfg)
        print("Plotting 05 boxplots...", flush=True); plot_boxplots(all_eval, summary, dirs, cfg)
        print("Plotting 07 scenario best...", flush=True); plot_scenario_best(prob_best, dirs, cfg)
        print("Plotting 08 Bayes vs deterministic...", flush=True); plot_bayes_vs_deterministic(future_eval, bayes_best["label"], det_best["label"], dirs, cfg)
        print("Plotting 11 sensitivity...", flush=True); plot_sensitivity(sens, dirs, cfg)
        print("Plotting 12 reliability posterior...", flush=True); plot_reliability_posteriors(geo, post, dirs, cfg, rng)

    # Save arrays needed for later paper figures.
    np.save(os.path.join(dirs["arrays"], "posterior_mean_demand.npy"), mean_demand)
    np.save(os.path.join(dirs["arrays"], "node_coordinates.npy"), geo["coords"])

    # Readme.
    readme = f"""# Bayesian ETNDP engineered simulation outputs

This folder was created by `bayesian_etndp_engineered_simulation.py`.

The simulation intentionally creates a logistics setting where uncertainty matters:
OD demand has sale/surge days, travel times have storm/disruption regimes, and hub
sorting reliability is uncertain.  The deterministic mean-only baseline ignores
posterior tail risk and reliability chance constraints, whereas the Bayesian method
uses posterior expected cost, CVaR of maximum arrival time, and reliability penalties.

Selected Bayesian design: {bayes_best['label']}
Selected deterministic baseline: {det_best['label']}

Main files:
- tables/posterior_design_summary_all.csv
- tables/best_design_by_topology.csv
- tables/bayesian_vs_deterministic_selected_designs.csv
- tables/future_stress_test_summary.csv
- tables/bayesian_verification_gains.csv
- figures/03_posterior_tradeoff_cost_cvar.png
- figures/08_future_bayes_vs_deterministic_arrival.png
"""
    with open(os.path.join(cfg.outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    summary_json = {
        "config": asdict(cfg),
        "selected_bayesian_design": bayes_best.to_dict(),
        "selected_deterministic_design": det_best.to_dict(),
        "future_stress_test_summary": future_summary,
        "verification_gains": gains,
        "output_directory": os.path.abspath(cfg.outdir),
    }
    with open(os.path.join(cfg.outdir, "simulation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary_json), f, indent=2)

    zip_path = zip_directory(cfg.outdir, f"{cfg.outdir}.zip")

    print("\n" + "#" * 100)
    print("Simulation finished successfully.")
    print(f"All outputs saved in: {os.path.abspath(cfg.outdir)}")
    print(f"Downloadable ZIP created: {os.path.abspath(zip_path)}")
    print("#" * 100)

    return {
        "config": cfg,
        "geo": geo,
        "data": data,
        "posterior": post,
        "posterior_scenarios": scenarios,
        "design_summary": summary,
        "topology_best": topo_best,
        "probability_scenario_best": prob_best,
        "comparison": comparison,
        "future_summary": future_summary,
        "gains": gains,
        "all_eval": all_eval,
        "future_eval": future_eval,
        "zip_path": os.path.abspath(zip_path),
    }


if __name__ == "__main__":
    cfg = SimulationConfig(show_plots=True, show_tables=True)
    results = run_full_simulation(cfg)
