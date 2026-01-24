import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from typing import Dict, Tuple
from src.utils.enums import EncodedSeason
from functools import partial


def hierarchal_model(
    home_idx: jnp.ndarray,
    away_idx: jnp.ndarray,
    y_home: jnp.ndarray,
    y_away: jnp.ndarray,
    n_teams: int
):
    n_games = home_idx.shape[0]
    alpha = numpyro.sample("alpha", dist.Normal(4.0, 1.0)) # Centered for log-points

    # Hierarchy scales
    sigma_off = numpyro.sample("sigma_off", dist.HalfNormal(.15))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(.15))
    tau_h     = numpyro.sample("tau_h",     dist.HalfNormal(.03))
    
    # Shared Game-Level Scale (The pace/rhythm factor)
    sigma_game = numpyro.sample("sigma_game", dist.HalfNormal(0.05))

    # Volatility scales
    sigma_off_team_std = numpyro.sample("sigma_off_team_std", dist.HalfNormal(0.05))
    sigma_def_team_std = numpyro.sample("sigma_def_team_std", dist.HalfNormal(0.05))

    h_mu = numpyro.sample("h_mu", dist.Normal(0.0, .05))

    with numpyro.plate("team", n_teams):
        h = numpyro.sample("h", dist.Normal(h_mu, tau_h))
        off_un = numpyro.sample("offense", dist.Normal(0.0, sigma_off))
        def_un = numpyro.sample("defense", dist.Normal(0.0, sigma_def))
        t_off_std = numpyro.sample("team_off_std", dist.HalfNormal(sigma_off_team_std))
        t_def_std = numpyro.sample("team_def_std", dist.HalfNormal(sigma_def_team_std))

    offense = off_un - off_un.mean() # type: ignore
    defense = def_un - def_un.mean() # type: ignore

    # The Shared Game Effect: One draw per game, applied to both teams
    with numpyro.plate("games", n_games):
        gamma = numpyro.sample("gamma", dist.Normal(0.0, sigma_game))

    # Independent Volatility (Epsilon)
    eps_h = numpyro.sample("eps_h", dist.Normal(0.0, t_off_std[home_idx] + t_def_std[away_idx])) # type: ignore
    eps_a = numpyro.sample("eps_a", dist.Normal(0.0, t_off_std[away_idx] + t_def_std[home_idx])) # type: ignore

    eta_home = alpha + offense[home_idx] - defense[away_idx] + h[home_idx] + gamma + eps_h # type: ignore
    eta_away = alpha + offense[away_idx] - defense[home_idx] + gamma + eps_a # type: ignore

    numpyro.sample("y_home", dist.Poisson(jnp.exp(eta_home)), obs=y_home)
    numpyro.sample("y_away", dist.Poisson(jnp.exp(eta_away)), obs=y_away)

def fit_hierarchal_model(encoded: EncodedSeason, seed: int = 0, num_chains: int = 2, num_warmup: int = 100, num_samples: int = 300):
    """
    Fit the TeamVolGameRandomness model on an EncodedSeason.
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


@partial(jax.jit, static_argnums=(10,))
def _simulate_kernel(
    alpha_b: jnp.ndarray,
    offense_b: jnp.ndarray,
    defense_b: jnp.ndarray,
    h_b: jnp.ndarray,
    team_off_std_b: jnp.ndarray,
    team_def_std_b: jnp.ndarray,
    sigma_game_b: jnp.ndarray,
    home_idx: jnp.ndarray,
    away_idx: jnp.ndarray,
    key: jnp.ndarray,
    n_sims: int
):
    Db = alpha_b.shape[0]
    G = home_idx.shape[0]

    # Extract team parameters for games: [Db, G]
    home_off = jnp.take(offense_b, home_idx, axis=1)
    away_off = jnp.take(offense_b, away_idx, axis=1)
    home_def = jnp.take(defense_b, home_idx, axis=1)
    away_def = jnp.take(defense_b, away_idx, axis=1)
    home_h = jnp.take(h_b, home_idx, axis=1)
    home_off_std = jnp.take(team_off_std_b, home_idx, axis=1)
    away_off_std = jnp.take(team_off_std_b, away_idx, axis=1)
    home_def_std = jnp.take(team_def_std_b, home_idx, axis=1)
    away_def_std = jnp.take(team_def_std_b, away_idx, axis=1)

    # Base eta without gamma and epsilon: [Db, G] -> [Db, 1, G] for broadcasting
    eta_home_base = (alpha_b[:, None] + home_off - away_def + home_h)[:, None, :]  # [Db, 1, G]
    eta_away_base = (alpha_b[:, None] + away_off - home_def)[:, None, :]  # [Db, 1, G]

    # Epsilon stds: [Db, G] -> [Db, 1, G] for broadcasting
    eps_home_std = (home_off_std + away_def_std)[:, None, :]  # [Db, 1, G]
    eps_away_std = (away_off_std + home_def_std)[:, None, :]  # [Db, 1, G]

    # Sigma_game: [Db] -> [Db, 1, 1] for broadcasting
    sigma_game_expanded = sigma_game_b[:, None, None]  # [Db, 1, 1]

    # Split only 4 times: one for gamma, one for eps_home, one for eps_away, one for poisson
    k1, k2, k3, k4 = random.split(key, 4)

    # Sample ALL noise at once in a single massive block [Db, n_sims, G]
    # Gamma (shared game effect): [Db, n_sims, G]
    gamma = random.normal(k1, shape=(Db, n_sims, G)) * sigma_game_expanded
    
    # Epsilon noise: [Db, n_sims, G]
    eps_home = random.normal(k2, shape=(Db, n_sims, G)) * eps_home_std
    eps_away = random.normal(k3, shape=(Db, n_sims, G)) * eps_away_std

    # Compute lambda: [Db, n_sims, G]
    lam_home = jnp.exp(eta_home_base + gamma + eps_home)
    lam_away = jnp.exp(eta_away_base + gamma + eps_away)

    # Sample Poisson likelihoods
    home_scores = random.poisson(k4, lam_home)
    # Use fold_in to ensure home and away aren't using identical Poisson seeds
    away_scores = random.poisson(random.fold_in(k4, 1), lam_away)

    return home_scores.astype(jnp.int32), away_scores.astype(jnp.int32)

def simulate_scores_block(
    home_indices: jnp.ndarray,
    away_indices: jnp.ndarray,
    samples: Dict[str, jnp.ndarray],
    *,
    draw_start: int,
    draw_end: int,
    n_sims: int,
    rng_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate an exploded block of scores for the TeamVolGameRandomness model.

    This is the GPU-friendly path used by the simulation pipeline:
    - Vectorized over posterior draws, simulations, and games
    - Returns integer score tensors shaped [Db, S, G]

    Args:
        home_indices: [G] home team indices
        away_indices: [G] away team indices
        samples: posterior samples (expects keys: 'alpha', 'offense', 'defense', 'h', 'team_off_std', 'team_def_std', 'sigma_game')
                 Shapes: alpha [D], offense/defense/h/team_off_std/team_def_std [D, T], sigma_game [D]
        draw_start: starting posterior draw index (inclusive)
        draw_end: ending posterior draw index (exclusive)
        n_sims: simulations per posterior draw per game
        rng_key: base RNG key; block is deterministically derived from this + draw_start

    Returns:
        (home_scores, away_scores): both shaped [Db, S, G], dtype int32
    """
    if draw_end <= draw_start:
        raise ValueError(f"draw_end must be > draw_start (got {draw_start=}, {draw_end=})")
    if n_sims <= 0:
        raise ValueError(f"n_sims must be > 0 (got {n_sims})")

    alpha = samples["alpha"][draw_start:draw_end]  # [Db]
    offense = samples["offense"][draw_start:draw_end]  # [Db, T]
    defense = samples["defense"][draw_start:draw_end]  # [Db, T]
    h = samples["h"][draw_start:draw_end]  # [Db, T]
    team_off_std = samples["team_off_std"][draw_start:draw_end]  # [Db, T]
    team_def_std = samples["team_def_std"][draw_start:draw_end]  # [Db, T]
    sigma_game = samples["sigma_game"][draw_start:draw_end]  # [Db]
    block_key = random.fold_in(rng_key, draw_start)

    return _simulate_kernel(alpha, offense, defense, h, team_off_std, team_def_std, sigma_game, home_indices, away_indices, block_key, n_sims)