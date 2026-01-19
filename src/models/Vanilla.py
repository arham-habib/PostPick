import os
# Force CPU backend to avoid Metal backend issues with NumPyro HMC
# Metal doesn't support all operations (e.g., bitwise_count/popcnt) needed by NumPyro
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from typing import Dict, Tuple
from src.utils.enums import EncodedSeason


"""
Vanilla model with home advantage and team-level effects
"""
def hierarchal_model(
    home_idx: jnp.ndarray,
    away_idx: jnp.ndarray,
    y_home: jnp.ndarray,
    y_away: jnp.ndarray,
    n_teams: int
):
    # Intercept: weakly-informative
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))

    # Hierarchy scales (positive)
    sigma_off = numpyro.sample("sigma_off", dist.HalfNormal(1.0))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(1.0))
    tau_h     = numpyro.sample("tau_h",     dist.HalfNormal(1.0))

    # League-level home advantage mean
    h_mu = numpyro.sample("h_mu", dist.Normal(0.0, 1.0))

    # team-level effects
    with numpyro.plate("team", n_teams):
        h       = numpyro.sample("h",       dist.Normal(h_mu, tau_h))
        offense_uncentered: jnp.ndarray = numpyro.sample("offense", dist.Normal(0.0, sigma_off)) # type: ignore
        defense_uncentered: jnp.ndarray = numpyro.sample("defense", dist.Normal(0.0, sigma_def)) # type: ignore

    offense = offense_uncentered - offense_uncentered.mean()
    defense = defense_uncentered - defense_uncentered.mean()

    # Linear predictors
    eta_home = alpha + offense[home_idx] - defense[away_idx] + h[home_idx]      # type: ignore
    eta_away = alpha + offense[away_idx] - defense[home_idx]                    # type: ignore

    # Likelihood
    numpyro.sample("y_home", dist.Poisson(jnp.exp(eta_home)), obs=y_home)
    numpyro.sample("y_away", dist.Poisson(jnp.exp(eta_away)), obs=y_away)

def fit_hierarchal_model(encoded: EncodedSeason, seed: int = 0, num_chains: int = 2, num_warmup: int = 100, num_samples: int = 300):
    """
    Fit the model_hier_offdef_home model on an EncodedSeason.
    Returns the MCMC object and posterior samples.
    """
    home_idx = jnp.array(encoded.home_idx)
    away_idx = jnp.array(encoded.away_idx)
    y_home = jnp.array(encoded.y_home)
    y_away = jnp.array(encoded.y_away)
    n_teams = encoded.n_teams

    kernel = numpyro.infer.NUTS(hierarchal_model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        chain_method="sequential"
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        home_idx=home_idx,
        away_idx=away_idx,
        y_home=y_home,
        y_away=y_away,
        n_teams=n_teams
    )
    samples = mcmc.get_samples()
    return mcmc, samples


def simulate_games_batch(
    home_indices: jnp.ndarray,
    away_indices: jnp.ndarray,
    alpha: float,
    offense: jnp.ndarray,
    defense: jnp.ndarray,
    h: jnp.ndarray,
    rng_keys: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized batch simulation for Vanilla model.
    
    Args:
        home_indices: Array of home team indices (shape: [n_games])
        away_indices: Array of away team indices (shape: [n_games])
        alpha: Intercept parameter
        offense: Array of offense ratings (shape: [n_teams])
        defense: Array of defense ratings (shape: [n_teams])
        h: Array of home advantages (shape: [n_teams])
        rng_keys: Array of random keys (shape: [n_games, 2])
    
    Returns:
        Tuple of (home_scores, away_scores) arrays (shape: [n_games])
    """
    # Vectorized computation
    home_offenses = offense[home_indices]
    home_defenses = defense[home_indices]
    away_offenses = offense[away_indices]
    away_defenses = defense[away_indices]
    home_hs = h[home_indices]
    
    # Compute rates for all games
    eta_home = alpha + home_offenses - away_defenses + home_hs
    eta_away = alpha + away_offenses - home_defenses
    lambda_home = jnp.exp(eta_home)
    lambda_away = jnp.exp(eta_away)
    
    # Sample from Poisson for all games
    # Vectorize poisson over keys using vmap
    poisson_vmap = jax.vmap(random.poisson, in_axes=(0, 0))
    home_scores = poisson_vmap(rng_keys[:, 0], lambda_home)
    away_scores = poisson_vmap(rng_keys[:, 1], lambda_away)
    
    return home_scores, away_scores


def simulate_games(
    home_indices: jnp.ndarray,
    away_indices: jnp.ndarray,
    samples: Dict[str, jnp.ndarray],
    n_sims: int,
    rng_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate games with Vanilla model across all parameter draws.
    
    Args:
        home_indices: Array of home team indices (shape: [n_games])
        away_indices: Array of away team indices (shape: [n_games])
        samples: Dict of posterior samples with keys: 'alpha', 'offense', 'defense', 'h'
        n_sims: Number of simulations per parameter draw
        rng_key: Random key
    
    Returns:
        Tuple of (home_scores, away_scores) arrays (shape: [n_param_draws * n_games * n_sims])
    """
    n_param_draws = samples['alpha'].shape[0]
    n_games = len(home_indices)
    
    # Generate random keys for all simulations
    keys = random.split(rng_key, n_param_draws * n_games * n_sims * 2)
    keys = keys.reshape(n_param_draws, n_games, n_sims, 2)  # [param_draw, game, sim, home/away]
    
    all_home_scores = []
    all_away_scores = []
    
    for param_draw_idx in range(n_param_draws):
        alpha = float(samples['alpha'][param_draw_idx])
        offense = samples['offense'][param_draw_idx]
        defense = samples['defense'][param_draw_idx]
        h = samples['h'][param_draw_idx]
        
        param_home_scores = []
        param_away_scores = []
        
        for sim_idx in range(n_sims):
            game_keys = keys[param_draw_idx, :, sim_idx, :]  # [n_games, 2]
            
            home_scores, away_scores = simulate_games_batch(
                home_indices, away_indices, alpha, offense, defense, h, game_keys
            )
            param_home_scores.append(home_scores)
            param_away_scores.append(away_scores)
        
        all_home_scores.append(jnp.stack(param_home_scores))
        all_away_scores.append(jnp.stack(param_away_scores))
    
    # Concatenate all results: [n_param_draws, n_sims, n_games] -> flatten
    all_home_scores = jnp.concatenate([s.flatten() for s in all_home_scores])
    all_away_scores = jnp.concatenate([s.flatten() for s in all_away_scores])
    
    return all_home_scores, all_away_scores