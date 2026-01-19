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
Team volatility model with team-specific offensive and defensive std devs
Each team's points arrival rate is a function of their team-specific volatility
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

    # Global hierarchy scales for team effects (positive)
    sigma_off = numpyro.sample("sigma_off", dist.HalfNormal(1.0))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(1.0))
    tau_h     = numpyro.sample("tau_h",     dist.HalfNormal(1.0))

    # Global priors for team-specific volatility std devs
    sigma_off_team_std = numpyro.sample("sigma_off_team_std", dist.HalfNormal(0.5))
    sigma_def_team_std = numpyro.sample("sigma_def_team_std", dist.HalfNormal(0.5))

    # League-level home advantage mean
    h_mu = numpyro.sample("h_mu", dist.Normal(0.0, 1.0))

    # Team-level effects
    with numpyro.plate("team", n_teams):
        h       = numpyro.sample("h",       dist.Normal(h_mu, tau_h))
        offense_uncentered: jnp.ndarray = numpyro.sample("offense", dist.Normal(0.0, sigma_off)) # type: ignore
        defense_uncentered: jnp.ndarray = numpyro.sample("defense", dist.Normal(0.0, sigma_def)) # type: ignore
        
        # Team-specific offensive and defensive volatility std devs
        team_off_std: jnp.ndarray = numpyro.sample("team_off_std", dist.HalfNormal(sigma_off_team_std)) # type: ignore
        team_def_std: jnp.ndarray = numpyro.sample("team_def_std", dist.HalfNormal(sigma_def_team_std)) # type: ignore

    offense = offense_uncentered - offense_uncentered.mean()
    defense = defense_uncentered - defense_uncentered.mean()

    # Game-level random effects incorporating team volatility
    # Home team's offensive volatility and away team's defensive volatility affect home scoring
    epsilon_home: jnp.ndarray = numpyro.sample( # type: ignore
        "epsilon_home", 
        dist.Normal(0.0, team_off_std[home_idx] + team_def_std[away_idx])
    )
    # Away team's offensive volatility and home team's defensive volatility affect away scoring
    epsilon_away: jnp.ndarray = numpyro.sample( # type: ignore
        "epsilon_away", 
        dist.Normal(0.0, team_off_std[away_idx] + team_def_std[home_idx])
    )

    # Linear predictors with team volatility effects
    eta_home = alpha + offense[home_idx] - defense[away_idx] + h[home_idx] + epsilon_home      # type: ignore
    eta_away = alpha + offense[away_idx] - defense[home_idx] + epsilon_away                    # type: ignore

    # Likelihood
    numpyro.sample("y_home", dist.Poisson(jnp.exp(eta_home)), obs=y_home)
    numpyro.sample("y_away", dist.Poisson(jnp.exp(eta_away)), obs=y_away)

def fit_hierarchal_model(encoded: EncodedSeason, seed: int = 0, num_chains: int = 2, num_warmup: int = 100, num_samples: int = 300):
    """
    Fit the team volatility model on an EncodedSeason.
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
    team_off_std: jnp.ndarray,
    team_def_std: jnp.ndarray,
    rng_keys: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized batch simulation for TeamVol model.
    
    Args:
        home_indices: Array of home team indices (shape: [n_games])
        away_indices: Array of away team indices (shape: [n_games])
        alpha: Intercept parameter
        offense: Array of offense ratings (shape: [n_teams])
        defense: Array of defense ratings (shape: [n_teams])
        h: Array of home advantages (shape: [n_teams])
        team_off_std: Array of offensive volatilities (shape: [n_teams])
        team_def_std: Array of defensive volatilities (shape: [n_teams])
        rng_keys: Array of random keys (shape: [n_games, 4]) for epsilon and poisson
    
    Returns:
        Tuple of (home_scores, away_scores) arrays (shape: [n_games])
    """
    # Vectorized computation
    home_offenses = offense[home_indices]
    home_defenses = defense[home_indices]
    away_offenses = offense[away_indices]
    away_defenses = defense[away_indices]
    home_hs = h[home_indices]
    
    # Get team volatilities
    home_off_stds = team_off_std[home_indices]
    away_def_stds = team_def_std[away_indices]
    away_off_stds = team_off_std[away_indices]
    home_def_stds = team_def_std[home_indices]
    
    # Sample epsilon for all games
    epsilon_home_stds = home_off_stds + away_def_stds
    epsilon_away_stds = away_off_stds + home_def_stds
    
    # Vectorize random operations over keys using vmap
    normal_vmap = jax.vmap(random.normal, in_axes=(0,))
    epsilon_home = normal_vmap(rng_keys[:, 0]) * epsilon_home_stds
    epsilon_away = normal_vmap(rng_keys[:, 1]) * epsilon_away_stds
    
    # Compute rates for all games
    eta_home = alpha + home_offenses - away_defenses + home_hs + epsilon_home
    eta_away = alpha + away_offenses - home_defenses + epsilon_away
    lambda_home = jnp.exp(eta_home)
    lambda_away = jnp.exp(eta_away)
    
    # Sample from Poisson for all games
    poisson_vmap = jax.vmap(random.poisson, in_axes=(0, 0))
    home_scores = poisson_vmap(rng_keys[:, 2], lambda_home)
    away_scores = poisson_vmap(rng_keys[:, 3], lambda_away)
    
    return home_scores, away_scores


def simulate_games(
    home_indices: jnp.ndarray,
    away_indices: jnp.ndarray,
    samples: Dict[str, jnp.ndarray],
    n_sims: int,
    rng_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate games with TeamVol model across all parameter draws.
    
    Args:
        home_indices: Array of home team indices (shape: [n_games])
        away_indices: Array of away team indices (shape: [n_games])
        samples: Dict of posterior samples with keys: 'alpha', 'offense', 'defense', 'h', 'team_off_std', 'team_def_std'
        n_sims: Number of simulations per parameter draw
        rng_key: Random key
    
    Returns:
        Tuple of (home_scores, away_scores) arrays (shape: [n_param_draws * n_games * n_sims])
    """
    n_param_draws = samples['alpha'].shape[0]
    n_games = len(home_indices)
    
    # Generate random keys for all simulations (4 keys per game: 2 for epsilon, 2 for poisson)
    keys = random.split(rng_key, n_param_draws * n_games * n_sims * 4)
    keys = keys.reshape(n_param_draws, n_games, n_sims, 4)  # [param_draw, game, sim, 4 keys]
    
    all_home_scores = []
    all_away_scores = []
    
    for param_draw_idx in range(n_param_draws):
        alpha = float(samples['alpha'][param_draw_idx])
        offense = samples['offense'][param_draw_idx]
        defense = samples['defense'][param_draw_idx]
        h = samples['h'][param_draw_idx]
        team_off_std = samples['team_off_std'][param_draw_idx]
        team_def_std = samples['team_def_std'][param_draw_idx]
        
        param_home_scores = []
        param_away_scores = []
        
        for sim_idx in range(n_sims):
            game_keys = keys[param_draw_idx, :, sim_idx, :]  # [n_games, 4]
            
            home_scores, away_scores = simulate_games_batch(
                home_indices, away_indices, alpha, offense, defense, h,
                team_off_std, team_def_std, game_keys
            )
            param_home_scores.append(home_scores)
            param_away_scores.append(away_scores)
        
        all_home_scores.append(jnp.stack(param_home_scores))
        all_away_scores.append(jnp.stack(param_away_scores))
    
    # Concatenate all results: [n_param_draws, n_sims, n_games] -> flatten
    all_home_scores = jnp.concatenate([s.flatten() for s in all_home_scores])
    all_away_scores = jnp.concatenate([s.flatten() for s in all_away_scores])
    
    return all_home_scores, all_away_scores
