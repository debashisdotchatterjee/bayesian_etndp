# -*- coding: utf-8 -*-
"""
Bayesian Multi-Topology Express Transportation Network Design
Real-data case study + posterior scenario experiments + sensitivity analysis
using the classical CAB25 hub-location benchmark data.

Author-ready Colab script.

What this script does
---------------------
1. Downloads the CAB25 hub-location data from a public GitHub mirror of the
   standard CAB data set used in hub-location studies.
2. Uses the real CAB OD-flow matrix and distance matrix as the empirical network.
3. Constructs a posterior-predictive Bayesian case-study experiment:
   - Gamma--Poisson OD-demand posterior;
   - lognormal travel-time model with manual Normal--Inverse-Gamma updating;
   - Beta posterior for hub reliability / sorting productivity;
   - lognormal cost-multiplier posterior.
4. Enumerates seven topology classes inspired by the deterministic BO-ETNDP paper:
   FC, SAHS, MAHS, RAHS, DSAHS, DMAHS, DRAHS.
5. Evaluates candidate designs by posterior expected cost, posterior CVaR of
   maximum arrival time, service reliability, hub-hold reliability, and emissions.
6. Compares the Bayesian posterior-risk-aware design with a deterministic
   cost-priority baseline under out-of-sample stress scenarios.
7. Creates CSV tables, LaTeX tables, high-resolution plots, and a downloadable ZIP.

Important note for the manuscript
---------------------------------
The CAB25 data set is a real benchmark OD-flow/distance network. It is static;
therefore, this script constructs a pseudo-historical daily panel around the real
CAB mean structure to fit posterior distributions and then evaluates posterior
predictive and future-stress scenarios. This is a real-network posterior scenario
experiment, not a proprietary courier-company field deployment.

Run in Colab
------------
%run case_study_cab_bayesian_etndp.py

Or:
from case_study_cab_bayesian_etndp import CaseStudyConfig, run_case_study
cfg = CaseStudyConfig(n_nodes=12, posterior_scenarios=120, future_scenarios=180)
results = run_case_study(cfg)
"""

from __future__ import annotations

import os
import io
import re
import json
import math
import time
import zipfile
import shutil
import warnings
import urllib.request
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class CaseStudyConfig:
    random_seed: int = 20260507

    # CAB real network subset. For fast Colab execution use 10--14. For richer
    # case study use 15 or 20; enumeration grows with candidate hubs.
    n_nodes: int = 12
    n_candidate_hubs: int = 4

    # Pseudo-history and posterior predictive scenario sizes.
    history_days: int = 120
    posterior_scenarios: int = 120
    future_scenarios: int = 180

    # Scaling from CAB annual/aggregate passenger OD flows to daily shipment-like
    # units. Spatial pattern remains real; magnitude is normalized for case-study
    # interpretability.
    target_mean_daily_od_demand: float = 24.0

    # Transport / service parameters.
    average_speed_distance_units_per_hour: float = 58.0
    service_time_node_hours: float = 0.35
    hold_time_hours: float = 8.0
    service_target_hours: float = 34.0
    cvar_alpha: float = 0.90

    # Vehicle and operation-cost parameters in arbitrary monetary units.
    transport_cost_per_unit_distance: float = 0.018
    fixed_node_cost: float = 0.030e6
    fixed_hub_cost: float = 0.180e6
    fixed_direct_link_cost: float = 0.006e6
    fixed_hub_link_cost: float = 0.014e6
    sorting_capacity_unit_cost: float = 0.0018e6
    sorting_unit_cost: float = 0.020
    emission_cost_per_unit_distance_flow: float = 0.00016

    # Hub sorting capacity construction.
    base_capacity_per_hour: float = 150.0
    cap_multipliers: Tuple[float, ...] = (0.90, 1.10, 1.35, 1.65)
    congestion_power: float = 1.28
    congestion_scale: float = 0.030

    # Candidate design generation.
    min_hub_subset_size: int = 1
    max_hub_subset_size: int = 3
    direct_threshold_grid: Tuple[float, ...] = (0.65, 0.75, 0.85, 0.95)
    r_allocation_value: int = 2

    # Bayesian risk preferences. These are used to choose the posterior design.
    weight_cost: float = 0.47
    weight_cvar_time: float = 0.33
    weight_emission: float = 0.05
    weight_service_penalty: float = 0.08
    weight_hold_penalty: float = 0.07

    # Future-stress scenario severity, used for out-of-sample validation.
    future_stress_multiplier: float = 1.22
    future_disruption_probability: float = 0.26

    # Output behavior.
    output_dir: str = "cab_bayesian_etndp_outputs"
    show_plots: bool = True
    show_tables: bool = True
    save_dpi: int = 260

    # CAB data source.
    cab_raw_url: str = "https://raw.githubusercontent.com/mcroboredo/Hub-Location-Instances/main/CAB25.txt"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

TOPO_COLORS = {
    "FC": "#1f77b4",
    "SAHS": "#ff7f0e",
    "MAHS": "#2ca02c",
    "RAHS": "#9467bd",
    "DSAHS": "#d62728",
    "DMAHS": "#17becf",
    "DRAHS": "#8c564b",
}


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cvar(values: np.ndarray, alpha: float = 0.90) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    q = np.quantile(values, alpha)
    tail = values[values >= q]
    if tail.size == 0:
        return float(q)
    return float(np.mean(tail))


def normalize_minmax(x: np.ndarray, larger_is_worse: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if abs(mx - mn) < 1e-12:
        y = np.zeros_like(x, dtype=float)
    else:
        y = (x - mn) / (mx - mn)
    if not larger_is_worse:
        y = 1.0 - y
    return y


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) < 1e-12 else a / b


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, np.ndarray)) else False:
        return None
    return obj


def show_df(df: pd.DataFrame, name: str, cfg: CaseStudyConfig, max_rows: int = 40) -> None:
    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 30, "display.width", 180):
        print(df.head(max_rows).to_string(index=False))
    if cfg.show_tables and display is not None:
        display(df.head(max_rows))


def save_table(df: pd.DataFrame, out_dir: str, stem: str, index: bool = False) -> None:
    tables_dir = os.path.join(out_dir, "tables")
    ensure_dir(tables_dir)
    df.to_csv(os.path.join(tables_dir, stem + ".csv"), index=index)
    try:
        with open(os.path.join(tables_dir, stem + ".tex"), "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=index, escape=False, float_format="%.4f"))
    except Exception:
        pass


def savefig(path: str, cfg: CaseStudyConfig) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.save_dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# CAB data loading and parsing
# -----------------------------------------------------------------------------


def download_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_cab25_raw(raw_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse CAB25 file: n followed by n*n flow entries and n*n distance entries."""
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_text)]
    if len(nums) < 1:
        raise ValueError("No numeric content found in CAB25 file.")
    n = int(nums[0])
    expected = 1 + 2 * n * n
    if len(nums) < expected:
        raise ValueError(f"CAB25 parse failed: found {len(nums)} numbers; expected at least {expected}.")
    vals = np.array(nums[1:expected], dtype=float)
    flow = vals[: n * n].reshape(n, n)
    dist = vals[n * n : 2 * n * n].reshape(n, n)
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(dist, 0.0)
    return flow, dist


def classical_mds(distance: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS coordinates from a distance matrix, no sklearn needed."""
    D = np.asarray(distance, dtype=float)
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals_pos = np.maximum(eigvals[:n_components], 0.0)
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals_pos)
    if np.allclose(coords, 0):
        coords = np.column_stack([np.cos(np.linspace(0, 2*np.pi, n, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, n, endpoint=False))])
    return coords


def select_nodes_by_flow(flow25: np.ndarray, dist25: np.ndarray, n_nodes: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    total = flow25.sum(axis=0) + flow25.sum(axis=1)
    # Keep the most active nodes, then sort by original index for reproducibility.
    selected = np.argsort(total)[::-1][:n_nodes]
    selected = np.sort(selected)
    return flow25[np.ix_(selected, selected)], dist25[np.ix_(selected, selected)], [int(x + 1) for x in selected]


def choose_candidate_hubs(flow: np.ndarray, dist: np.ndarray, k: int) -> List[int]:
    """Choose candidate hubs by flow-weighted closeness centrality."""
    n = flow.shape[0]
    total_flow_node = flow.sum(axis=0) + flow.sum(axis=1)
    score = []
    for h in range(n):
        weighted_distance = np.sum((flow + flow.T)[:, h] * dist[:, h]) + 1e-9
        closeness = total_flow_node[h] / weighted_distance
        # boost nodes serving large total demand and central distance
        total_weighted_dist = np.sum(total_flow_node * dist[:, h]) + 1e-9
        score.append(0.65 * closeness + 0.35 * total_flow_node[h] / total_weighted_dist)
    return list(np.argsort(score)[::-1][:k])


def load_real_cab_data(cfg: CaseStudyConfig) -> Dict[str, Any]:
    print("Downloading CAB25 hub-location data from public mirror...")
    try:
        raw = download_text(cfg.cab_raw_url)
    except Exception as e:
        raise RuntimeError(
            "Could not download CAB25 data. In Colab, please ensure internet is enabled. "
            f"Source attempted: {cfg.cab_raw_url}. Error: {e}"
        )
    flow25, dist25 = parse_cab25_raw(raw)
    flow, dist, original_labels = select_nodes_by_flow(flow25, dist25, cfg.n_nodes)

    # Distances in the raw file are large. Normalize to a clean transportation scale while
    # preserving all pairwise relative distances.
    positive_dist = dist[dist > 0]
    dist_scaled = dist / np.median(positive_dist) * 520.0
    np.fill_diagonal(dist_scaled, 0.0)

    # Convert real CAB OD-flow magnitudes to daily shipment-equivalent rates.
    pos_flow = flow[flow > 0]
    flow_scaled = flow / np.mean(pos_flow) * cfg.target_mean_daily_od_demand
    flow_scaled = np.maximum(flow_scaled, 0.0)
    np.fill_diagonal(flow_scaled, 0.0)

    coords = classical_mds(dist_scaled)
    candidate_hubs = choose_candidate_hubs(flow_scaled, dist_scaled, cfg.n_candidate_hubs)

    node_names = [f"CAB-{lab}" for lab in original_labels]
    return {
        "raw_flow": flow,
        "distance": dist_scaled,
        "base_daily_flow": flow_scaled,
        "coords": coords,
        "candidate_hubs": candidate_hubs,
        "node_names": node_names,
        "original_labels": original_labels,
        "source_url": cfg.cab_raw_url,
    }


# -----------------------------------------------------------------------------
# Construct pseudo-history around real static CAB network
# -----------------------------------------------------------------------------


def generate_real_network_history(data: Dict[str, Any], cfg: CaseStudyConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    W0 = data["base_daily_flow"]
    D = data["distance"]
    hubs = data["candidate_hubs"]
    n = W0.shape[0]
    T = cfg.history_days

    demand_hist = np.zeros((T, n, n), dtype=float)
    travel_hist = np.zeros((T, n, n), dtype=float)
    cost_mult_hist = np.zeros(T, dtype=float)

    # Risk corridors: long and high-flow pairs have worse and more uncertain travel.
    pair_pressure = W0 * D
    risk_threshold = np.quantile(pair_pressure[pair_pressure > 0], 0.78)
    risky_pair = pair_pressure >= risk_threshold
    np.fill_diagonal(risky_pair, False)

    base_time = np.where(D > 0, D / cfg.average_speed_distance_units_per_hour, 0.0)
    weekday = np.arange(T) % 7
    weekly = 1.0 + 0.08 * np.sin(2 * np.pi * weekday / 7.0)

    for t in range(T):
        # Normal/peak/disruption mixture. The historical period contains rare stress,
        # enough for the posterior to learn but not enough to make robust planning trivial.
        u = rng.uniform()
        if u < 0.07:
            regime_mult = rng.lognormal(np.log(1.45), 0.12)
            shock_mult = 1.32
        elif u < 0.18:
            regime_mult = rng.lognormal(np.log(1.20), 0.08)
            shock_mult = 1.12
        else:
            regime_mult = rng.lognormal(np.log(1.00), 0.06)
            shock_mult = 1.00

        lam = W0 * weekly[t] * regime_mult
        lam = np.maximum(lam, 1e-9)
        # Overdispersed demand through Gamma-Poisson mixture.
        overdisp_shape = 11.0
        gamma_noise = rng.gamma(shape=overdisp_shape, scale=1.0 / overdisp_shape, size=(n, n))
        lam2 = lam * gamma_noise
        demand = rng.poisson(lam2)
        np.fill_diagonal(demand, 0.0)
        demand_hist[t] = demand

        # Travel time: lognormal; risky corridors get extra heavy-tail stress.
        log_noise = rng.normal(0.0, 0.13, size=(n, n))
        risk_noise = rng.normal(0.0, 0.22, size=(n, n)) * risky_pair
        ttime = base_time * np.exp(log_noise + risk_noise) * shock_mult
        ttime = np.where(D > 0, np.maximum(ttime, 0.10), 0.0)
        np.fill_diagonal(ttime, 0.0)
        travel_hist[t] = ttime

        cost_mult_hist[t] = float(rng.lognormal(mean=np.log(1.0 + 0.04 * (shock_mult - 1.0)), sigma=0.055))

    # Hub reliability: busier and riskier candidate hubs have slightly lower reliability.
    reliability_trials = 28
    hub_success = np.zeros((T, len(hubs)), dtype=float)
    hub_trials = np.full((T, len(hubs)), reliability_trials, dtype=float)
    inbound_base = W0.sum(axis=0) + W0.sum(axis=1)
    max_inbound = max(np.max(inbound_base[hubs]), 1.0)
    for hi, h in enumerate(hubs):
        base_rel = 0.91 - 0.11 * (inbound_base[h] / max_inbound) + rng.normal(0.0, 0.015)
        base_rel = float(np.clip(base_rel, 0.72, 0.95))
        for t in range(T):
            stress = demand_hist[t].sum() / max(np.mean(demand_hist.sum(axis=(1, 2))), 1.0)
            p = np.clip(base_rel - 0.055 * max(stress - 1.0, 0), 0.55, 0.97)
            hub_success[t, hi] = rng.binomial(reliability_trials, p)

    return {
        "demand_hist": demand_hist,
        "travel_hist": travel_hist,
        "cost_mult_hist": cost_mult_hist,
        "hub_success": hub_success,
        "hub_trials": hub_trials,
        "risky_pair": risky_pair,
    }


# -----------------------------------------------------------------------------
# Manual Bayesian posterior updating
# -----------------------------------------------------------------------------


def posterior_gamma_poisson(demand_hist: np.ndarray, base_mean: np.ndarray) -> Dict[str, np.ndarray]:
    # Prior equivalent exposure: 8 days centered at the real CAB scaled mean.
    prior_days = 8.0
    alpha0 = np.maximum(base_mean * prior_days, 0.15)
    beta0 = np.full_like(base_mean, prior_days, dtype=float)
    post_alpha = alpha0 + demand_hist.sum(axis=0)
    post_beta = beta0 + demand_hist.shape[0]
    np.fill_diagonal(post_alpha, 0.0)
    np.fill_diagonal(post_beta, 1.0)
    return {"alpha": post_alpha, "beta": post_beta, "mean": post_alpha / post_beta}


def nig_update_1d(y: np.ndarray, mu0: float, kappa0: float, alpha0: float, beta0: float) -> Tuple[float, float, float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return mu0, kappa0, alpha0, beta0
    ybar = float(np.mean(y))
    ss = float(np.sum((y - ybar) ** 2))
    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2.0
    mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
    beta_n = beta0 + 0.5 * ss + (kappa0 * n * (ybar - mu0) ** 2) / (2.0 * kappa_n)
    return mu_n, kappa_n, alpha_n, beta_n


def posterior_travel_time(travel_hist: np.ndarray, distance: np.ndarray, cfg: CaseStudyConfig) -> Dict[str, np.ndarray]:
    T, n, _ = travel_hist.shape
    mu_n = np.zeros((n, n))
    kappa_n = np.zeros((n, n))
    alpha_n = np.zeros((n, n))
    beta_n = np.zeros((n, n))
    base = np.where(distance > 0, distance / cfg.average_speed_distance_units_per_hour, 0.1)
    for i in range(n):
        for j in range(n):
            if i == j:
                mu_n[i, j] = np.log(0.1)
                kappa_n[i, j] = 1
                alpha_n[i, j] = 2.5
                beta_n[i, j] = 0.05
                continue
            y = np.log(np.maximum(travel_hist[:, i, j], 1e-6))
            mu0 = float(np.log(max(base[i, j], 0.1)))
            mu, kappa, alpha, beta = nig_update_1d(y, mu0=mu0, kappa0=3.0, alpha0=3.0, beta0=0.10)
            mu_n[i, j] = mu
            kappa_n[i, j] = kappa
            alpha_n[i, j] = alpha
            beta_n[i, j] = beta
    return {"mu": mu_n, "kappa": kappa_n, "alpha": alpha_n, "beta": beta_n}


def posterior_beta_reliability(hub_success: np.ndarray, hub_trials: np.ndarray) -> Dict[str, np.ndarray]:
    # Prior says hubs are usually good but not perfect.
    a0, b0 = 18.0, 4.5
    a = a0 + hub_success.sum(axis=0)
    b = b0 + (hub_trials - hub_success).sum(axis=0)
    return {"a": a, "b": b, "mean": a / (a + b)}


def posterior_cost_multiplier(cost_hist: np.ndarray) -> Dict[str, float]:
    y = np.log(np.maximum(cost_hist, 1e-6))
    mu, kappa, alpha, beta = nig_update_1d(y, mu0=0.0, kappa0=3.0, alpha0=3.0, beta0=0.08)
    return {"mu": mu, "kappa": kappa, "alpha": alpha, "beta": beta}


def invgamma_sample(rng: np.random.Generator, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # If X~Gamma(alpha, rate=beta), then 1/X~InvGamma(alpha,beta)
    return 1.0 / rng.gamma(shape=alpha, scale=1.0 / beta)


def sample_posterior_scenarios(
    data: Dict[str, Any],
    post: Dict[str, Any],
    cfg: CaseStudyConfig,
    rng: np.random.Generator,
    B: int,
    future_stress: bool = False,
) -> Dict[str, np.ndarray]:
    n = data["base_daily_flow"].shape[0]
    hubs = data["candidate_hubs"]
    risky_pair = post["risky_pair"]

    demand_s = np.zeros((B, n, n), dtype=float)
    travel_s = np.zeros((B, n, n), dtype=float)
    cost_s = np.zeros(B, dtype=float)
    rel_s = np.zeros((B, len(hubs)), dtype=float)

    # Posterior parameter arrays.
    dem_alpha = post["demand"]["alpha"]
    dem_beta = post["demand"]["beta"]
    tt = post["travel"]
    rel = post["reliability"]
    cp = post["cost"]

    for b in range(B):
        # In future stress experiments, demand/travel regimes are harsher.
        u = rng.uniform()
        if future_stress:
            if u < cfg.future_disruption_probability:
                demand_mult = rng.lognormal(np.log(1.55 * cfg.future_stress_multiplier), 0.15)
                travel_mult = rng.lognormal(np.log(1.35), 0.12)
            elif u < 0.55:
                demand_mult = rng.lognormal(np.log(1.22), 0.10)
                travel_mult = rng.lognormal(np.log(1.12), 0.08)
            else:
                demand_mult = rng.lognormal(np.log(1.02), 0.07)
                travel_mult = rng.lognormal(np.log(1.00), 0.06)
        else:
            if u < 0.11:
                demand_mult = rng.lognormal(np.log(1.38), 0.13)
                travel_mult = rng.lognormal(np.log(1.22), 0.10)
            elif u < 0.28:
                demand_mult = rng.lognormal(np.log(1.15), 0.08)
                travel_mult = rng.lognormal(np.log(1.08), 0.07)
            else:
                demand_mult = rng.lognormal(np.log(1.00), 0.06)
                travel_mult = rng.lognormal(np.log(1.00), 0.05)

        lam = rng.gamma(shape=dem_alpha, scale=1.0 / dem_beta) * demand_mult
        W = rng.poisson(np.maximum(lam, 1e-9)).astype(float)
        np.fill_diagonal(W, 0.0)
        demand_s[b] = W

        sigma2 = invgamma_sample(rng, tt["alpha"], tt["beta"])
        mu = rng.normal(tt["mu"], np.sqrt(sigma2 / tt["kappa"]))
        extra_tail = np.zeros((n, n))
        if future_stress:
            extra_tail[risky_pair] = rng.lognormal(mean=0.02, sigma=0.22, size=int(risky_pair.sum())) - 1.0
        else:
            extra_tail[risky_pair] = rng.lognormal(mean=0.00, sigma=0.13, size=int(risky_pair.sum())) - 1.0
        Tmat = np.exp(mu + rng.normal(0.0, np.sqrt(sigma2))) * travel_mult * (1.0 + np.maximum(extra_tail, 0.0))
        np.fill_diagonal(Tmat, 0.0)
        travel_s[b] = np.maximum(Tmat, 0.0)

        # Cost multiplier.
        sig2_c = float(invgamma_sample(rng, np.array(cp["alpha"]), np.array(cp["beta"])))
        mu_c = float(rng.normal(cp["mu"], np.sqrt(sig2_c / cp["kappa"])))
        cost_stress = 1.0 + (0.10 if future_stress and u < cfg.future_disruption_probability else 0.0)
        cost_s[b] = float(rng.lognormal(mu_c, np.sqrt(sig2_c)) * cost_stress)

        # Hub reliability/productivity.
        R = rng.beta(rel["a"], rel["b"])
        if future_stress:
            R = R * rng.beta(32, 4, size=len(hubs))  # mild degradation in the future-stress world
        rel_s[b] = np.clip(R, 0.45, 0.99)

    return {"demand": demand_s, "travel_time": travel_s, "cost_multiplier": cost_s, "hub_reliability": rel_s}


# -----------------------------------------------------------------------------
# Topology design enumeration
# -----------------------------------------------------------------------------

class Design:
    def __init__(self, topology: str, hubs: Tuple[int, ...], cap_mult: float = 1.0, direct_threshold: Optional[float] = None, r_value: Optional[int] = None):
        self.topology = topology
        self.hubs = tuple(int(x) for x in hubs)
        self.cap_mult = float(cap_mult)
        self.direct_threshold = None if direct_threshold is None else float(direct_threshold)
        self.r_value = r_value
        hubs_str = "none" if len(hubs) == 0 else "-".join(str(h + 1) for h in hubs)
        parts = [f"{topology}|H={hubs_str}", f"cap={cap_mult:.2f}"]
        if direct_threshold is not None:
            parts.append(f"dirq={direct_threshold:.2f}")
        if r_value is not None:
            parts.append(f"R={r_value}")
        self.label = "|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "hubs": ",".join(str(h + 1) for h in self.hubs) if self.hubs else "--",
            "cap_mult": self.cap_mult,
            "direct_threshold": self.direct_threshold,
            "r_value": self.r_value,
            "design_label": self.label,
        }


def enumerate_designs(data: Dict[str, Any], cfg: CaseStudyConfig) -> List[Design]:
    cand = data["candidate_hubs"]
    designs: List[Design] = []
    designs.append(Design("FC", tuple(), 1.0))

    max_size = min(cfg.max_hub_subset_size, len(cand))
    for r in range(cfg.min_hub_subset_size, max_size + 1):
        for hubs in combinations(cand, r):
            for cap in cfg.cap_multipliers:
                designs.append(Design("SAHS", hubs, cap))
                designs.append(Design("MAHS", hubs, cap))
                designs.append(Design("RAHS", hubs, cap, r_value=min(cfg.r_allocation_value, len(hubs))))
                for th in cfg.direct_threshold_grid:
                    designs.append(Design("DSAHS", hubs, cap, direct_threshold=th))
                    designs.append(Design("DMAHS", hubs, cap, direct_threshold=th))
                    designs.append(Design("DRAHS", hubs, cap, direct_threshold=th, r_value=min(cfg.r_allocation_value, len(hubs))))
    # Remove duplicates where RAHS/MAHS identical for one hub? Keep them; useful for topology table.
    return designs


def nearest_hub_assignments(distance: np.ndarray, hubs: Tuple[int, ...]) -> np.ndarray:
    n = distance.shape[0]
    if len(hubs) == 0:
        return np.full(n, -1, dtype=int)
    hubs_arr = np.array(hubs, dtype=int)
    nearest_idx = np.argmin(distance[:, hubs_arr], axis=1)
    return hubs_arr[nearest_idx]


def choose_route(i: int, j: int, design: Design, distance: np.ndarray, demand_mean: np.ndarray) -> Tuple[str, int, int]:
    """Return route_type, origin_hub, dest_hub. direct route uses (-1,-1)."""
    if i == j:
        return "none", -1, -1
    if design.topology == "FC":
        return "direct", -1, -1

    hubs = design.hubs
    if len(hubs) == 0:
        return "direct", -1, -1

    assign = nearest_hub_assignments(distance, hubs)
    k0, l0 = int(assign[i]), int(assign[j])

    # Multi-allocation variants choose best origin/destination hub pair by distance and expected load.
    if design.topology in {"MAHS", "DMAHS"}:
        best_val, best_pair = np.inf, (k0, l0)
        for k in hubs:
            for l in hubs:
                val = distance[i, k] + 0.62 * distance[k, l] + distance[l, j]
                if val < best_val:
                    best_val, best_pair = val, (k, l)
        k0, l0 = best_pair
    elif design.topology in {"RAHS", "DRAHS"}:
        # Restrict to R closest hubs for origin and destination.
        R = max(1, min(design.r_value or 1, len(hubs)))
        hubs_arr = np.array(hubs, dtype=int)
        oset = hubs_arr[np.argsort(distance[i, hubs_arr])[:R]]
        dset = hubs_arr[np.argsort(distance[j, hubs_arr])[:R]]
        best_val, best_pair = np.inf, (k0, l0)
        for k in oset:
            for l in dset:
                val = distance[i, k] + 0.66 * distance[k, l] + distance[l, j]
                if val < best_val:
                    best_val, best_pair = val, (int(k), int(l))
        k0, l0 = best_pair

    # Hybrid direct link decision. Heavy or geographically shortcut OD pairs are allowed direct service.
    if design.topology in {"DSAHS", "DMAHS", "DRAHS"} and design.direct_threshold is not None:
        direct_dist = distance[i, j]
        hub_dist = distance[i, k0] + 0.70 * distance[k0, l0] + distance[l0, j]
        demand_q = np.quantile(demand_mean[demand_mean > 0], design.direct_threshold)
        high_demand = demand_mean[i, j] >= demand_q
        useful_shortcut = direct_dist <= 0.86 * hub_dist
        if high_demand or useful_shortcut:
            return "direct", -1, -1

    return "hub", int(k0), int(l0)


def build_design_static(design: Design, data: Dict[str, Any], demand_mean: np.ndarray) -> Dict[str, Any]:
    D = data["distance"]
    n = D.shape[0]
    route_type = np.full((n, n), "none", dtype=object)
    origin_hub = np.full((n, n), -1, dtype=int)
    dest_hub = np.full((n, n), -1, dtype=int)
    direct_links = set()
    hub_links = set()
    node_hub_links = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rt, k, l = choose_route(i, j, design, D, demand_mean)
            route_type[i, j] = rt
            origin_hub[i, j] = k
            dest_hub[i, j] = l
            if rt == "direct":
                direct_links.add((i, j))
            elif rt == "hub":
                node_hub_links.add((i, k))
                node_hub_links.add((j, l))
                if k != l:
                    hub_links.add((k, l))

    return {
        "route_type": route_type,
        "origin_hub": origin_hub,
        "dest_hub": dest_hub,
        "direct_links": direct_links,
        "hub_links": hub_links,
        "node_hub_links": node_hub_links,
        "n_direct_links": len(direct_links),
        "n_hub_links": len(hub_links),
        "n_node_hub_links": len(node_hub_links),
    }


# -----------------------------------------------------------------------------
# Evaluation of designs under posterior/future scenarios
# -----------------------------------------------------------------------------


def evaluate_design_scenario(
    design: Design,
    static: Dict[str, Any],
    W: np.ndarray,
    TT: np.ndarray,
    cost_multiplier: float,
    hub_reliability_vec: np.ndarray,
    data: Dict[str, Any],
    cfg: CaseStudyConfig,
) -> Dict[str, float]:
    D = data["distance"]
    candidate_hubs = data["candidate_hubs"]
    hub_index_map = {h: idx for idx, h in enumerate(candidate_hubs)}
    n = W.shape[0]

    route_type = static["route_type"]
    K = static["origin_hub"]
    L = static["dest_hub"]

    hub_load = {h: 0.0 for h in design.hubs}
    transport_distance_flow = 0.0
    max_arrival_base = 0.0

    # First pass: route loads and transport distance.
    for i in range(n):
        for j in range(n):
            if i == j or W[i, j] <= 0:
                continue
            wij = W[i, j]
            if route_type[i, j] == "direct":
                transport_distance_flow += wij * D[i, j]
            else:
                k, l = int(K[i, j]), int(L[i, j])
                if k < 0 or l < 0:
                    continue
                transport_distance_flow += wij * (D[i, k] + 0.70 * D[k, l] + D[l, j])
                hub_load[k] = hub_load.get(k, 0.0) + wij
                if l != k:
                    hub_load[l] = hub_load.get(l, 0.0) + wij

    # Effective capacity and sorting times at hubs.
    hub_sort_time: Dict[int, float] = {}
    hub_effective_cap: Dict[int, float] = {}
    for h in design.hubs:
        hi = hub_index_map.get(h, 0)
        reliability = float(hub_reliability_vec[hi]) if hi < len(hub_reliability_vec) else 0.80
        cap = cfg.base_capacity_per_hour * design.cap_mult * reliability
        cap = max(cap, 1e-6)
        load = hub_load.get(h, 0.0)
        utilization = load / max(cap * cfg.hold_time_hours, 1e-6)
        sort_time = load / cap + cfg.congestion_scale * (max(utilization, 0.0) ** cfg.congestion_power) * cfg.hold_time_hours
        hub_sort_time[h] = float(sort_time)
        hub_effective_cap[h] = float(cap)

    # Arrival times.
    for i in range(n):
        for j in range(n):
            if i == j or W[i, j] <= 0:
                continue
            if route_type[i, j] == "direct":
                arrival = TT[i, j] + cfg.service_time_node_hours
            else:
                k, l = int(K[i, j]), int(L[i, j])
                if k < 0 or l < 0:
                    arrival = TT[i, j] + cfg.service_time_node_hours
                else:
                    arrival = TT[i, k] + hub_sort_time.get(k, 0.0)
                    if k != l:
                        arrival += TT[k, l] + hub_sort_time.get(l, 0.0)
                    arrival += TT[l, j] + 2 * cfg.service_time_node_hours
            if arrival > max_arrival_base:
                max_arrival_base = float(arrival)

    # Costs.
    fixed_cost = cfg.fixed_node_cost * n
    fixed_cost += cfg.fixed_hub_cost * len(design.hubs)
    fixed_cost += cfg.fixed_direct_link_cost * static["n_direct_links"]
    fixed_cost += cfg.fixed_hub_link_cost * static["n_hub_links"]
    fixed_cost += 0.20 * cfg.fixed_hub_link_cost * static["n_node_hub_links"]
    sorting_capacity_cost = cfg.sorting_capacity_unit_cost * sum(hub_effective_cap.values())
    sorting_variable_cost = cfg.sorting_unit_cost * sum(hub_load.values())
    transport_cost = cfg.transport_cost_per_unit_distance * transport_distance_flow * cost_multiplier
    emission_cost = cfg.emission_cost_per_unit_distance_flow * transport_distance_flow
    total_cost = fixed_cost + sorting_capacity_cost + sorting_variable_cost + transport_cost + emission_cost

    hold_ok = all(st <= cfg.hold_time_hours for st in hub_sort_time.values()) if len(design.hubs) > 0 else True
    service_ok = max_arrival_base <= cfg.service_target_hours

    return {
        "cost": float(total_cost),
        "transport_cost": float(transport_cost),
        "fixed_cost": float(fixed_cost),
        "sorting_capacity_cost": float(sorting_capacity_cost),
        "sorting_variable_cost": float(sorting_variable_cost),
        "emission_cost": float(emission_cost),
        "max_arrival": float(max_arrival_base),
        "hold_ok": float(hold_ok),
        "service_ok": float(service_ok),
        "max_hub_sort_time": float(max(hub_sort_time.values()) if hub_sort_time else 0.0),
        "total_hub_load": float(sum(hub_load.values())),
        "transport_distance_flow": float(transport_distance_flow),
    }


def summarize_design_evaluation(design: Design, scenario_metrics: List[Dict[str, float]], cfg: CaseStudyConfig) -> Dict[str, Any]:
    df = pd.DataFrame(scenario_metrics)
    out = design.to_dict()
    out.update({
        "expected_cost_million": df["cost"].mean() / 1e6,
        "sd_cost_million": df["cost"].std(ddof=1) / 1e6,
        "mean_max_arrival_hours": df["max_arrival"].mean(),
        "p95_max_arrival_hours": df["max_arrival"].quantile(0.95),
        "cvar_max_arrival_hours": cvar(df["max_arrival"].values, cfg.cvar_alpha),
        "service_reliability": df["service_ok"].mean(),
        "hold_reliability": df["hold_ok"].mean(),
        "mean_emission_cost_million": df["emission_cost"].mean() / 1e6,
        "mean_max_hub_sort_hours": df["max_hub_sort_time"].mean(),
        "p95_max_hub_sort_hours": df["max_hub_sort_time"].quantile(0.95),
        "mean_transport_distance_flow": df["transport_distance_flow"].mean(),
        "n_direct_links": int(getattr(design, "n_direct_links", 0)),
        "n_hub_links": int(getattr(design, "n_hub_links", 0)),
        "n_node_hub_links": int(getattr(design, "n_node_hub_links", 0)),
    })
    return out


def evaluate_all_designs(
    designs: List[Design],
    scenarios: Dict[str, np.ndarray],
    data: Dict[str, Any],
    cfg: CaseStudyConfig,
    demand_mean_for_routes: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    B = scenarios["demand"].shape[0]
    rows = []
    detail: Dict[str, pd.DataFrame] = {}
    statics: Dict[str, Dict[str, Any]] = {}

    print(f"Evaluating {len(designs)} candidate real-data topology designs over {B} posterior scenarios...")
    for idx, d in enumerate(designs):
        static = build_design_static(d, data, demand_mean_for_routes)
        d.n_direct_links = static["n_direct_links"]
        d.n_hub_links = static["n_hub_links"]
        d.n_node_hub_links = static["n_node_hub_links"]
        statics[d.label] = static
        metrics = []
        if idx % max(1, len(designs) // 8) == 0 or idx == len(designs) - 1:
            print(f"  design {idx + 1:4d}/{len(designs)}: {d.label}")
        for b in range(B):
            metrics.append(evaluate_design_scenario(
                d, static,
                scenarios["demand"][b], scenarios["travel_time"][b], scenarios["cost_multiplier"][b], scenarios["hub_reliability"][b],
                data, cfg,
            ))
        detail[d.label] = pd.DataFrame(metrics)
        rows.append(summarize_design_evaluation(d, metrics, cfg))
    df = pd.DataFrame(rows)
    return df, detail, statics


def add_pareto_and_scores(df: pd.DataFrame, cfg: CaseStudyConfig) -> pd.DataFrame:
    df = df.copy()
    df["norm_cost"] = normalize_minmax(df["expected_cost_million"].values)
    df["norm_cvar"] = normalize_minmax(df["cvar_max_arrival_hours"].values)
    df["norm_emission"] = normalize_minmax(df["mean_emission_cost_million"].values)
    df["service_penalty"] = 1.0 - df["service_reliability"]
    df["hold_penalty"] = 1.0 - df["hold_reliability"]
    df["posterior_bayes_risk_score"] = (
        cfg.weight_cost * df["norm_cost"]
        + cfg.weight_cvar_time * df["norm_cvar"]
        + cfg.weight_emission * df["norm_emission"]
        + cfg.weight_service_penalty * df["service_penalty"]
        + cfg.weight_hold_penalty * df["hold_penalty"]
    )

    # Pareto: lower cost, lower cvar, higher service, higher hold.
    pareto = []
    vals = df[["expected_cost_million", "cvar_max_arrival_hours", "service_reliability", "hold_reliability"]].values
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            better_or_equal = (
                vals[j, 0] <= vals[i, 0] + 1e-12 and
                vals[j, 1] <= vals[i, 1] + 1e-12 and
                vals[j, 2] >= vals[i, 2] - 1e-12 and
                vals[j, 3] >= vals[i, 3] - 1e-12
            )
            strictly_better = (
                vals[j, 0] < vals[i, 0] - 1e-12 or
                vals[j, 1] < vals[i, 1] - 1e-12 or
                vals[j, 2] > vals[i, 2] + 1e-12 or
                vals[j, 3] > vals[i, 3] + 1e-12
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    df["pareto_efficient"] = pareto
    return df.sort_values("posterior_bayes_risk_score").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Baselines and validation
# -----------------------------------------------------------------------------


def deterministic_baseline_choice(df: pd.DataFrame) -> pd.Series:
    """Deterministic analogue: prioritize expected/nominal cost with mild mean-time tie breaker."""
    temp = df.copy()
    temp["det_score"] = 0.82 * normalize_minmax(temp["expected_cost_million"].values) + 0.18 * normalize_minmax(temp["mean_max_arrival_hours"].values)
    return temp.sort_values("det_score").iloc[0]


def future_stress_compare(
    selected_rows: Dict[str, pd.Series],
    designs_by_label: Dict[str, Design],
    statics: Dict[str, Dict[str, Any]],
    future_scenarios: Dict[str, np.ndarray],
    data: Dict[str, Any],
    cfg: CaseStudyConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    detail = {}
    B = future_scenarios["demand"].shape[0]
    for method_name, row in selected_rows.items():
        label = row["design_label"]
        design = designs_by_label[label]
        static = statics[label]
        metrics = []
        for b in range(B):
            metrics.append(evaluate_design_scenario(
                design, static,
                future_scenarios["demand"][b], future_scenarios["travel_time"][b], future_scenarios["cost_multiplier"][b], future_scenarios["hub_reliability"][b],
                data, cfg,
            ))
        ddf = pd.DataFrame(metrics)
        detail[method_name] = ddf
        rows.append({
            "method": method_name,
            "topology": row["topology"],
            "hubs": row["hubs"],
            "design_label": label,
            "future_expected_cost_million": ddf["cost"].mean() / 1e6,
            "future_p95_cost_million": ddf["cost"].quantile(0.95) / 1e6,
            "future_mean_max_arrival_hours": ddf["max_arrival"].mean(),
            "future_p95_max_arrival_hours": ddf["max_arrival"].quantile(0.95),
            "future_cvar_max_arrival_hours": cvar(ddf["max_arrival"].values, cfg.cvar_alpha),
            "future_service_reliability": ddf["service_ok"].mean(),
            "future_hold_reliability": ddf["hold_ok"].mean(),
            "future_mean_max_hub_sort_hours": ddf["max_hub_sort_time"].mean(),
        })
    comp = pd.DataFrame(rows)
    return comp, detail


def posterior_probability_scenario_best(best_by_topology: pd.DataFrame, detail: Dict[str, pd.DataFrame], cfg: CaseStudyConfig) -> pd.DataFrame:
    labels = list(best_by_topology["design_label"])
    if not labels:
        return pd.DataFrame()
    B = len(detail[labels[0]])
    loss_mat = np.zeros((B, len(labels)))
    # Scenario-wise loss normalized across the topology winners, not all designs.
    costs = np.column_stack([detail[l]["cost"].values / 1e6 for l in labels])
    times = np.column_stack([detail[l]["max_arrival"].values for l in labels])
    emissions = np.column_stack([detail[l]["emission_cost"].values / 1e6 for l in labels])
    for b in range(B):
        cost_norm = normalize_minmax(costs[b])
        time_norm = normalize_minmax(times[b])
        emi_norm = normalize_minmax(emissions[b])
        loss_mat[b] = cfg.weight_cost * cost_norm + cfg.weight_cvar_time * time_norm + cfg.weight_emission * emi_norm
    winners = np.argmin(loss_mat, axis=1)
    rows = []
    for j, label in enumerate(labels):
        row = best_by_topology.iloc[j]
        rows.append({
            "topology": row["topology"],
            "hubs": row["hubs"],
            "posterior_probability_scenario_best": float(np.mean(winners == j)),
            "mean_scenario_loss": float(np.mean(loss_mat[:, j])),
            "design_label": label,
        })
    return pd.DataFrame(rows).sort_values("posterior_probability_scenario_best", ascending=False)


def sensitivity_to_preferences(df: pd.DataFrame) -> pd.DataFrame:
    prefs = [
        (0.70, 0.20, 0.04, 0.03, 0.03, "cost-dominant"),
        (0.55, 0.30, 0.05, 0.05, 0.05, "balanced-cost-risk"),
        (0.42, 0.38, 0.05, 0.08, 0.07, "tail-risk-aware"),
        (0.30, 0.42, 0.05, 0.13, 0.10, "reliability-dominant"),
        (0.25, 0.50, 0.05, 0.10, 0.10, "time-critical"),
    ]
    rows = []
    for wc, wt, we, ws, wh, name in prefs:
        score = (
            wc * df["norm_cost"] + wt * df["norm_cvar"] + we * df["norm_emission"]
            + ws * df["service_penalty"] + wh * df["hold_penalty"]
        )
        idx = int(np.argmin(score.values))
        row = df.iloc[idx]
        rows.append({
            "preference_profile": name,
            "weight_cost": wc,
            "weight_CVaR_time": wt,
            "weight_emission": we,
            "weight_service_penalty": ws,
            "weight_hold_penalty": wh,
            "chosen_topology": row["topology"],
            "chosen_hubs": row["hubs"],
            "chosen_score": float(score.iloc[idx]),
            "expected_cost_million": row["expected_cost_million"],
            "cvar_time_hours": row["cvar_max_arrival_hours"],
            "service_reliability": row["service_reliability"],
            "hold_reliability": row["hold_reliability"],
            "design_label": row["design_label"],
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_real_network(data: Dict[str, Any], cfg: CaseStudyConfig, out_dir: str) -> None:
    coords = data["coords"]
    W = data["base_daily_flow"]
    hubs = set(data["candidate_hubs"])
    n = W.shape[0]
    strength = W.sum(axis=0) + W.sum(axis=1)
    sizes = 90 + 420 * strength / max(strength.max(), 1.0)

    plt.figure(figsize=(8.4, 6.4))
    # top OD flows as faint edges
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 0:
                edges.append((W[i, j], i, j))
    edges = sorted(edges, reverse=True)[:min(45, len(edges))]
    maxw = max([e[0] for e in edges]) if edges else 1
    for w, i, j in edges:
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#9ecae1", alpha=0.18 + 0.40 * w / maxw, linewidth=0.5 + 2.0 * w / maxw)
    for i in range(n):
        if i in hubs:
            plt.scatter(coords[i, 0], coords[i, 1], s=sizes[i] * 1.15, c="#d62728", marker="*", edgecolor="black", linewidth=0.8, zorder=4)
        else:
            plt.scatter(coords[i, 0], coords[i, 1], s=sizes[i], c="#2ca25f", edgecolor="black", linewidth=0.55, zorder=3)
        plt.text(coords[i, 0], coords[i, 1], str(i + 1), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
    plt.title("Real CAB OD network: MDS layout, flow intensity and candidate hubs")
    plt.xlabel("MDS coordinate 1")
    plt.ylabel("MDS coordinate 2")
    legend_items = [
        Line2D([0], [0], marker="*", color="w", label="Candidate hub", markerfacecolor="#d62728", markeredgecolor="black", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Demand node", markerfacecolor="#2ca25f", markeredgecolor="black", markersize=10),
        Line2D([0], [0], color="#9ecae1", lw=2, label="High CAB OD flow"),
    ]
    plt.legend(handles=legend_items, loc="best", frameon=True)
    plt.grid(alpha=0.22)
    savefig(os.path.join(out_dir, "figures", "01_real_cab_network_mds.png"), cfg)


def plot_heatmap(matrix: np.ndarray, title: str, path: str, cfg: CaseStudyConfig, cmap: str = "magma") -> None:
    plt.figure(figsize=(8.0, 6.8))
    im = plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Destination node")
    plt.ylabel("Origin node")
    plt.xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
    plt.yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
    savefig(path, cfg)


def plot_tradeoff(df: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str, bayes_label: str, det_label: str) -> None:
    plt.figure(figsize=(8.8, 6.7))
    for topo, sub in df.groupby("topology"):
        plt.scatter(sub["expected_cost_million"], sub["cvar_max_arrival_hours"], s=55, alpha=0.72, c=TOPO_COLORS.get(topo, "gray"), label=topo, edgecolor="white", linewidth=0.4)
    pareto = df[df["pareto_efficient"]]
    plt.scatter(pareto["expected_cost_million"], pareto["cvar_max_arrival_hours"], s=130, facecolors="none", edgecolors="black", linewidths=1.3, label="Pareto-efficient")
    b = df[df["design_label"] == bayes_label].iloc[0]
    d = df[df["design_label"] == det_label].iloc[0]
    plt.scatter([b["expected_cost_million"]], [b["cvar_max_arrival_hours"]], marker="*", s=360, c="#ffdf00", edgecolor="black", linewidth=1.0, label="Bayesian selected")
    plt.scatter([d["expected_cost_million"]], [d["cvar_max_arrival_hours"]], marker="X", s=190, c="#000000", edgecolor="white", linewidth=0.8, label="Deterministic baseline")
    plt.xlabel("Posterior expected cost (million units)")
    plt.ylabel(f"Posterior CVaR$_{{{cfg.cvar_alpha:.2f}}}$ of maximum arrival time (hours)")
    plt.title("Posterior cost--tail-risk trade-off across real-data topology designs")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    savefig(os.path.join(out_dir, "figures", "03_posterior_tradeoff_cost_cvar.png"), cfg)


def plot_topology_box(detail: Dict[str, pd.DataFrame], best_by_topology: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str) -> None:
    labels = list(best_by_topology["topology"])
    data = [detail[l]["max_arrival"].values for l in best_by_topology["design_label"]]
    plt.figure(figsize=(9.0, 6.2))
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(TOPO_COLORS.get(lab, "gray"))
        patch.set_alpha(0.72)
    plt.ylabel("Posterior maximum arrival time (hours)")
    plt.title("Posterior distribution of maximum arrival time: best design within each topology")
    plt.grid(axis="y", alpha=0.24)
    savefig(os.path.join(out_dir, "figures", "04_topology_winner_arrival_boxplot.png"), cfg)


def plot_probability_best(prob_df: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str) -> None:
    plt.figure(figsize=(8.4, 5.6))
    colors = [TOPO_COLORS.get(t, "gray") for t in prob_df["topology"]]
    plt.bar(prob_df["topology"], prob_df["posterior_probability_scenario_best"], color=colors, edgecolor="black", alpha=0.85)
    plt.ylim(0, 1.02)
    plt.ylabel("Posterior probability of being scenario-best")
    plt.title("Scenario-wise posterior probability that a topology winner is best")
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, "figures", "05_posterior_probability_scenario_best.png"), cfg)


def plot_reliability_bubble(df: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str, bayes_label: str) -> None:
    plt.figure(figsize=(8.8, 6.5))
    sizes = 50 + 500 * normalize_minmax(df["expected_cost_million"].values, larger_is_worse=True)
    for topo, sub in df.groupby("topology"):
        idx = sub.index
        plt.scatter(sub["service_reliability"], sub["hold_reliability"], s=sizes[idx], c=TOPO_COLORS.get(topo, "gray"), alpha=0.65, edgecolor="white", linewidth=0.5, label=topo)
    b = df[df["design_label"] == bayes_label].iloc[0]
    plt.scatter([b["service_reliability"]], [b["hold_reliability"]], marker="*", s=420, c="#ffdf00", edgecolor="black", linewidth=1.0, label="Bayesian selected")
    plt.xlabel("Posterior service reliability")
    plt.ylabel("Posterior hub-hold reliability")
    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.title("Reliability surface: service guarantee versus hub-hold feasibility")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    savefig(os.path.join(out_dir, "figures", "06_reliability_bubble.png"), cfg)


def plot_future_comparison(future_detail: Dict[str, pd.DataFrame], cfg: CaseStudyConfig, out_dir: str) -> None:
    methods = list(future_detail.keys())
    data = [future_detail[m]["max_arrival"].values for m in methods]
    plt.figure(figsize=(8.5, 6.0))
    bp = plt.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False)
    cols = ["#66c2a5", "#fc8d62", "#8da0cb"]
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.80)
    plt.ylabel("Out-of-sample maximum arrival time under stress (hours)")
    plt.title("Stress validation: Bayesian risk-aware design versus deterministic baseline")
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, "figures", "07_future_stress_arrival_comparison.png"), cfg)

    plt.figure(figsize=(8.5, 6.0))
    positions = np.arange(len(methods))
    service = [future_detail[m]["service_ok"].mean() for m in methods]
    hold = [future_detail[m]["hold_ok"].mean() for m in methods]
    width = 0.35
    plt.bar(positions - width/2, service, width, label="Service reliability", color="#4daf4a", edgecolor="black", alpha=0.85)
    plt.bar(positions + width/2, hold, width, label="Hub-hold reliability", color="#377eb8", edgecolor="black", alpha=0.85)
    plt.xticks(positions, methods)
    plt.ylim(0, 1.02)
    plt.ylabel("Reliability under stress")
    plt.title("Out-of-sample reliability comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, "figures", "08_future_stress_reliability_comparison.png"), cfg)


def plot_sensitivity(sens: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str) -> None:
    plt.figure(figsize=(10.5, 5.8))
    x = np.arange(len(sens))
    colors = [TOPO_COLORS.get(t, "gray") for t in sens["chosen_topology"]]
    plt.bar(x, sens["cvar_time_hours"], color=colors, edgecolor="black", alpha=0.85)
    plt.xticks(x, sens["preference_profile"], rotation=25, ha="right")
    plt.ylabel("CVaR maximum arrival time (hours)")
    plt.title("Preference sensitivity: selected topology and tail-risk performance")
    for xi, topo in zip(x, sens["chosen_topology"]):
        plt.text(xi, sens["cvar_time_hours"].iloc[xi] + 0.4, topo, ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.grid(axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, "figures", "09_preference_sensitivity.png"), cfg)


def plot_design_network(design_row: pd.Series, designs_by_label: Dict[str, Design], statics: Dict[str, Dict[str, Any]], data: Dict[str, Any], cfg: CaseStudyConfig, out_dir: str, stem: str, title: str) -> None:
    label = design_row["design_label"]
    design = designs_by_label[label]
    static = statics[label]
    coords = data["coords"]
    W = data["base_daily_flow"]
    n = W.shape[0]

    plt.figure(figsize=(8.5, 6.5))
    # direct links
    for (i, j) in list(static["direct_links"])[:150]:
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#3182bd", alpha=0.23, linewidth=0.8)
    for (k, l) in static["hub_links"]:
        plt.plot([coords[k, 0], coords[l, 0]], [coords[k, 1], coords[l, 1]], color="#e6550d", alpha=0.88, linewidth=2.6)
    for (i, k) in list(static["node_hub_links"])[:160]:
        plt.plot([coords[i, 0], coords[k, 0]], [coords[i, 1], coords[k, 1]], color="#31a354", alpha=0.16, linewidth=0.8)

    strengths = W.sum(axis=0) + W.sum(axis=1)
    sizes = 80 + 360 * strengths / max(strengths.max(), 1.0)
    for i in range(n):
        if i in design.hubs:
            plt.scatter(coords[i, 0], coords[i, 1], s=sizes[i] * 1.25, c="#d62728", marker="*", edgecolor="black", linewidth=0.9, zorder=5)
        else:
            plt.scatter(coords[i, 0], coords[i, 1], s=sizes[i], c="#756bb1", edgecolor="black", linewidth=0.55, zorder=4)
        plt.text(coords[i, 0], coords[i, 1], str(i + 1), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
    plt.title(title + "\n" + label)
    plt.xlabel("MDS coordinate 1")
    plt.ylabel("MDS coordinate 2")
    legend_items = [
        Line2D([0], [0], color="#3182bd", lw=2, label="Direct line"),
        Line2D([0], [0], color="#31a354", lw=2, label="Node--hub line"),
        Line2D([0], [0], color="#e6550d", lw=3, label="Inter-hub line"),
        Line2D([0], [0], marker="*", color="w", label="Selected hub", markerfacecolor="#d62728", markeredgecolor="black", markersize=14),
    ]
    plt.legend(handles=legend_items, loc="best", frameon=True)
    plt.grid(alpha=0.22)
    savefig(os.path.join(out_dir, "figures", stem + ".png"), cfg)


def plot_risk_components(df: pd.DataFrame, cfg: CaseStudyConfig, out_dir: str) -> None:
    best = df.groupby("topology", as_index=False).first().sort_values("posterior_bayes_risk_score")
    x = np.arange(len(best))
    comps = ["norm_cost", "norm_cvar", "norm_emission", "service_penalty", "hold_penalty"]
    labels = ["Cost", "CVaR", "Emission", "Service penalty", "Hold penalty"]
    colors = ["#4daf4a", "#e41a1c", "#984ea3", "#ff7f00", "#377eb8"]
    bottom = np.zeros(len(best))
    plt.figure(figsize=(9.5, 6.0))
    for comp, lab, col in zip(comps, labels, colors):
        vals = best[comp].values
        plt.bar(x, vals, bottom=bottom, label=lab, color=col, edgecolor="white", alpha=0.82)
        bottom += vals
    plt.xticks(x, best["topology"])
    plt.ylabel("Normalized risk-component height")
    plt.title("Risk-component decomposition for best design in each topology")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(axis="y", alpha=0.23)
    savefig(os.path.join(out_dir, "figures", "12_risk_component_decomposition.png"), cfg)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def run_case_study(cfg: Optional[CaseStudyConfig] = None) -> Dict[str, Any]:
    cfg = cfg or CaseStudyConfig()
    t0 = time.time()
    rng = set_seed(cfg.random_seed)

    out_dir = cfg.output_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "tables"))

    print("#" * 100)
    print("Bayesian Multi-Topology ETNDP: Real-data CAB case study")
    print("#" * 100)
    print(f"Random seed: {cfg.random_seed}")
    print(f"CAB subset nodes: {cfg.n_nodes}; candidate hubs: {cfg.n_candidate_hubs}")
    print(f"Posterior scenarios: {cfg.posterior_scenarios}; future stress scenarios: {cfg.future_scenarios}")

    data = load_real_cab_data(cfg)
    hist = generate_real_network_history(data, cfg, rng)

    # Posterior updating.
    post = {
        "demand": posterior_gamma_poisson(hist["demand_hist"], data["base_daily_flow"]),
        "travel": posterior_travel_time(hist["travel_hist"], data["distance"], cfg),
        "reliability": posterior_beta_reliability(hist["hub_success"], hist["hub_trials"]),
        "cost": posterior_cost_multiplier(hist["cost_mult_hist"]),
        "risky_pair": hist["risky_pair"],
    }

    # Descriptive tables.
    n = cfg.n_nodes
    summary = pd.DataFrame([
        {"quantity": "data source", "value": "CAB25 hub-location benchmark"},
        {"quantity": "public mirror", "value": cfg.cab_raw_url},
        {"quantity": "selected CAB node labels", "value": ", ".join(map(str, data["original_labels"]))},
        {"quantity": "nodes in case study", "value": n},
        {"quantity": "candidate hubs (case-study indices)", "value": ", ".join(str(h + 1) for h in data["candidate_hubs"])},
        {"quantity": "directed OD pairs", "value": int(n * (n - 1))},
        {"quantity": "mean scaled daily OD demand", "value": round(float(data["base_daily_flow"][data["base_daily_flow"] > 0].mean()), 3)},
        {"quantity": "median scaled daily OD demand", "value": round(float(np.median(data["base_daily_flow"][data["base_daily_flow"] > 0])), 3)},
        {"quantity": "mean pairwise distance", "value": round(float(data["distance"][data["distance"] > 0].mean()), 3)},
        {"quantity": "95th percentile pairwise distance", "value": round(float(np.quantile(data["distance"][data["distance"] > 0], 0.95)), 3)},
        {"quantity": "historical pseudo-panel days", "value": cfg.history_days},
    ])
    show_df(summary, "Real-data case-study summary", cfg)
    save_table(summary, out_dir, "01_real_data_summary")

    post_summary = pd.DataFrame([
        {"component": "OD demand", "Bayesian model": "Gamma--Poisson", "posterior quantity": "mean daily OD intensity", "value": round(float(post["demand"]["mean"][post["demand"]["mean"] > 0].mean()), 4)},
        {"component": "travel time", "Bayesian model": "Lognormal NIG", "posterior quantity": "mean log travel time", "value": round(float(np.mean(post["travel"]["mu"][data["distance"] > 0])), 4)},
        {"component": "hub reliability", "Bayesian model": "Beta--Binomial", "posterior quantity": "mean candidate-hub reliability", "value": round(float(np.mean(post["reliability"]["mean"])), 4)},
        {"component": "cost multiplier", "Bayesian model": "Lognormal NIG", "posterior quantity": "posterior mean log multiplier", "value": round(float(post["cost"]["mu"]), 4)},
    ])
    show_df(post_summary, "Manual Bayesian posterior updating summary", cfg)
    save_table(post_summary, out_dir, "02_posterior_summary")

    posterior_scenarios = sample_posterior_scenarios(data, post, cfg, rng, cfg.posterior_scenarios, future_stress=False)
    future_scenarios = sample_posterior_scenarios(data, post, cfg, rng, cfg.future_scenarios, future_stress=True)

    designs = enumerate_designs(data, cfg)
    design_counts = pd.Series([d.topology for d in designs]).value_counts().rename_axis("topology").reset_index(name="candidate_designs")
    show_df(design_counts, "Candidate design counts by topology", cfg)
    save_table(design_counts, out_dir, "03_candidate_design_counts")

    df_raw, detail, statics = evaluate_all_designs(designs, posterior_scenarios, data, cfg, post["demand"]["mean"])
    df = add_pareto_and_scores(df_raw, cfg)

    designs_by_label = {d.label: d for d in designs}
    bayes_row = df.iloc[0]
    det_row = deterministic_baseline_choice(df)
    best_by_topology = df.sort_values("posterior_bayes_risk_score").groupby("topology", as_index=False).first().sort_values("posterior_bayes_risk_score")
    prob_best = posterior_probability_scenario_best(best_by_topology, detail, cfg)

    selected_rows = {
        "Bayesian posterior-risk design": bayes_row,
        "Deterministic cost-priority baseline": det_row,
    }
    if bayes_row["design_label"] != det_row["design_label"]:
        # Also include FC as speed benchmark if not selected.
        fc_best = df[df["topology"] == "FC"].iloc[0]
        selected_rows["Fully-connected speed benchmark"] = fc_best
    future_comp, future_detail = future_stress_compare(selected_rows, designs_by_label, statics, future_scenarios, data, cfg)
    sens = sensitivity_to_preferences(df)

    # Improvement summary.
    bfc = future_comp[future_comp["method"] == "Bayesian posterior-risk design"].iloc[0]
    dfc = future_comp[future_comp["method"] == "Deterministic cost-priority baseline"].iloc[0]
    improvement = pd.DataFrame([
        {"metric": "Expected cost premium of Bayesian design (%)", "value": 100 * (bfc["future_expected_cost_million"] - dfc["future_expected_cost_million"]) / max(dfc["future_expected_cost_million"], 1e-9)},
        {"metric": "CVaR max-arrival reduction (%)", "value": 100 * (dfc["future_cvar_max_arrival_hours"] - bfc["future_cvar_max_arrival_hours"]) / max(dfc["future_cvar_max_arrival_hours"], 1e-9)},
        {"metric": "95th-percentile max-arrival reduction (%)", "value": 100 * (dfc["future_p95_max_arrival_hours"] - bfc["future_p95_max_arrival_hours"]) / max(dfc["future_p95_max_arrival_hours"], 1e-9)},
        {"metric": "Service reliability gain (percentage points)", "value": 100 * (bfc["future_service_reliability"] - dfc["future_service_reliability"])},
        {"metric": "Hub-hold reliability gain (percentage points)", "value": 100 * (bfc["future_hold_reliability"] - dfc["future_hold_reliability"])},
    ])
    improvement["value"] = improvement["value"].round(3)

    # Output tables.
    top_cols = [
        "topology", "hubs", "cap_mult", "n_direct_links", "expected_cost_million", "mean_max_arrival_hours",
        "p95_max_arrival_hours", "cvar_max_arrival_hours", "service_reliability", "hold_reliability",
        "mean_emission_cost_million", "posterior_bayes_risk_score", "pareto_efficient", "design_label"
    ]
    show_df(df[top_cols].head(25), "Top 25 posterior designs by Bayes-risk score", cfg, max_rows=25)
    save_table(df[top_cols].head(40), out_dir, "04_top_posterior_designs")
    show_df(best_by_topology[top_cols], "Best posterior design within each topology", cfg)
    save_table(best_by_topology[top_cols], out_dir, "05_best_by_topology")
    show_df(prob_best, "Posterior probability that each topology winner is scenario-best", cfg)
    save_table(prob_best, out_dir, "06_posterior_probability_scenario_best")
    show_df(future_comp, "Out-of-sample future-stress validation", cfg)
    save_table(future_comp, out_dir, "07_future_stress_validation")
    show_df(improvement, "Bayesian design improvement over deterministic baseline under stress", cfg)
    save_table(improvement, out_dir, "08_bayesian_improvement")
    show_df(sens, "Sensitivity analysis under posterior uncertainty", cfg)
    save_table(sens, out_dir, "09_preference_sensitivity")

    # Plots.
    plot_real_network(data, cfg, out_dir)
    plot_heatmap(data["base_daily_flow"], "Scaled real CAB OD-demand matrix used as posterior prior center", os.path.join(out_dir, "figures", "02_real_cab_demand_heatmap.png"), cfg, cmap="inferno")
    plot_tradeoff(df, cfg, out_dir, bayes_row["design_label"], det_row["design_label"])
    plot_topology_box(detail, best_by_topology, cfg, out_dir)
    plot_probability_best(prob_best, cfg, out_dir)
    plot_reliability_bubble(df, cfg, out_dir, bayes_row["design_label"])
    plot_future_comparison(future_detail, cfg, out_dir)
    plot_sensitivity(sens, cfg, out_dir)
    plot_design_network(bayes_row, designs_by_label, statics, data, cfg, out_dir, "10_selected_bayesian_network_design", "Selected Bayesian posterior-risk network design")
    plot_design_network(det_row, designs_by_label, statics, data, cfg, out_dir, "11_deterministic_baseline_network_design", "Deterministic cost-priority baseline network design")
    plot_risk_components(df, cfg, out_dir)

    # Save scenario-level data for selected methods.
    for method, ddf in future_detail.items():
        stem = method.lower().replace(" ", "_").replace("-", "_")[:60]
        save_table(ddf, out_dir, f"future_scenario_metrics_{stem}")

    # README and summary JSON.
    readme = f"""# CAB Bayesian ETNDP real-data case study outputs

This folder was generated by `case_study_cab_bayesian_etndp.py`.

Data source: CAB25 hub-location benchmark downloaded from:
{cfg.cab_raw_url}

The CAB data set supplies a real OD-flow matrix and pairwise distance matrix. Because the public benchmark is static rather than a repeated daily courier panel, the script constructs a pseudo-historical daily panel around the real CAB network structure, fits manual Bayesian posterior models, and evaluates posterior predictive plus future-stress scenarios.

Selected Bayesian design: {bayes_row['design_label']}
Deterministic baseline: {det_row['design_label']}

Main output tables are in `tables/` and figures are in `figures/`.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    summary_json = {
        "config": asdict(cfg),
        "selected_bayesian_design": bayes_row.to_dict(),
        "deterministic_baseline_design": det_row.to_dict(),
        "future_stress_validation": future_comp.to_dict(orient="records"),
        "improvement_summary": improvement.to_dict(orient="records"),
        "runtime_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary_json), f, indent=2)

    # Zip everything.
    zip_path = out_dir + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(out_dir, "zip", out_dir)

    print("\nSelected Bayesian posterior-risk design:")
    print(bayes_row["design_label"])
    print("\nDeterministic baseline:")
    print(det_row["design_label"])
    print("\n" + "#" * 100)
    print("Case study finished successfully.")
    print(f"Outputs saved in: {os.path.abspath(out_dir)}")
    print(f"Downloadable ZIP created: {os.path.abspath(zip_path)}")
    print("#" * 100)

    return {
        "config": cfg,
        "data": data,
        "posterior_summary": post_summary,
        "all_designs": df,
        "best_by_topology": best_by_topology,
        "posterior_probability_best": prob_best,
        "future_stress_validation": future_comp,
        "bayesian_improvement": improvement,
        "sensitivity": sens,
        "selected_bayesian_design": bayes_row,
        "deterministic_baseline": det_row,
        "output_dir": out_dir,
        "zip_path": zip_path,
    }


if __name__ == "__main__":
    cfg = CaseStudyConfig(
        n_nodes=12,
        n_candidate_hubs=4,
        history_days=120,
        posterior_scenarios=120,
        future_scenarios=180,
        show_plots=True,
        show_tables=True,
    )
    results = run_case_study(cfg)
