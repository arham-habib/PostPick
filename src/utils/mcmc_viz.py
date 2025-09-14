# mcmc_viz.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Iterable, Optional, Dict, Any

sns.set_context("talk")
sns.set_style("whitegrid")

# ---------- Helpers ----------

def _to_idata_from_numpyro(mcmc, *, model=None, model_kwargs: Optional[Dict[str, Any]] = None):
    """
    Convert a NumPyro MCMC object into an ArviZ InferenceData.
    You’ll get posterior, sample_stats, and optionally posterior_predictive if model is provided.
    """
    return az.from_numpyro(mcmc, model=model, model_kwargs=model_kwargs or {})

def _flatten_param(samples: Dict[str, np.ndarray], name: str) -> np.ndarray:
    """
    Returns draws for a parameter as a 2D array (n_draws, dim).
    Works for scalar (dim=1) and vector parameters (dim=k).
    Assumes samples are already combined across chains (NumPyro default).
    If you used group_by_chain=True, reshape first: samples[name] -> (chains*samples, ...)
    """
    x = np.asarray(samples[name])
    if x.ndim == 1:
        return x[:, None]
    return x  # (n_draws, k)

def _hdi(x: np.ndarray, hdi_prob: float = 0.94) -> tuple[float, float]:
    hdi = az.hdi(x, hdi_prob=hdi_prob)
    # az.hdi returns array for vectors; for 1D it returns shape (2,)
    return (float(hdi[0]), float(hdi[1]))

# ---------- Population pairplot ----------

def pairplot_population(samples: Dict[str, np.ndarray], corner: bool = True):
    """
    Pairplot over population-level params: alpha, sigma_off, sigma_def, tau_h, h_mu
    """
    wanted = ["alpha", "sigma_off", "sigma_def", "tau_h", "h_mu"]
    data = {}
    for k in wanted:
        if k in samples:
            v = np.asarray(samples[k]).reshape(-1)
            data[k] = v
    df = pd.DataFrame(data)
    g = sns.pairplot(df, corner=corner, diag_kind="kde", plot_kws=dict(s=8, alpha=0.4))
    g.figure.suptitle("Population parameter posterior pairplot", y=1.02)
    return g

# ---------- Per-team pairplots ----------

def pairplot_team_triplet(samples: Dict[str, np.ndarray], team_idx: int, team_name: Optional[str] = None, corner: bool = False):
    """
    Pairplot over (offense_i, defense_i, h_i) for a single team i.
    """
    off = _flatten_param(samples, "offense")[:, team_idx]
    dff = _flatten_param(samples, "defense")[:, team_idx]
    hh  = _flatten_param(samples, "h")[:, team_idx]
    df = pd.DataFrame({"offense": off, "defense": dff, "h": hh})
    label = f"{team_name or f'Team {team_idx}'}"
    g = sns.pairplot(df, corner=corner, diag_kind="kde", plot_kws=dict(s=8, alpha=0.35))
    g.figure.suptitle(f"Team triplet pairplot: {label}", y=1.02)
    return g

def pairplot_many_teams(samples: Dict[str, np.ndarray], team_indices: Iterable[int], team_names: Optional[Iterable[str]] = None):
    """
    Faceted KDEs for (offense, defense, h) across a handful of teams.
    Use this when pairplots for every team would be too heavy.
    """
    team_indices = list(team_indices)
    names = list(team_names) if team_names is not None else [f"Team {i}" for i in team_indices]
    off = _flatten_param(samples, "offense")
    dff = _flatten_param(samples, "defense")
    hh  = _flatten_param(samples, "h")

    rows = []
    for i, name in zip(team_indices, names):
        rows.append(pd.DataFrame({
            "value": off[:, i], "param": "offense", "team": name
        }))
        rows.append(pd.DataFrame({
            "value": dff[:, i], "param": "defense", "team": name
        }))
        rows.append(pd.DataFrame({
            "value": hh[:, i], "param": "h", "team": name
        }))
    df = pd.concat(rows, ignore_index=True)

    g = sns.F
