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
    """Simulate an exploded block of scores for the TeamVol model.

    GPU-friendly path used by the simulation pipeline:
    - Vectorized over posterior draws, simulations, and games
    - Includes team-specific volatility (epsilon) terms
    - Returns integer score tensors shaped [Db, S, G]

    Args:
        home_indices: [G] home team indices
        away_indices: [G] away team indices
        samples: posterior samples (expects keys: 'alpha', 'offense', 'defense', 'h',
                 'team_off_std', 'team_def_std'). Shapes:
                 alpha [D], offense/defense/h/team_* [D, T]
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

    block_key = random.fold_in(rng_key, draw_start)

    @jax.jit
    def _simulate(
        alpha_b: jnp.ndarray,
        offense_b: jnp.ndarray,
        defense_b: jnp.ndarray,
        h_b: jnp.ndarray,
        team_off_std_b: jnp.ndarray,
        team_def_std_b: jnp.ndarray,
        home_idx: jnp.ndarray,
        away_idx: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        Db = alpha_b.shape[0]

        # Gather team parameters: all [Db, G]
        home_off = jnp.take(offense_b, home_idx, axis=1)
        away_off = jnp.take(offense_b, away_idx, axis=1)
        home_def = jnp.take(defense_b, home_idx, axis=1)
        away_def = jnp.take(defense_b, away_idx, axis=1)
        home_h = jnp.take(h_b, home_idx, axis=1)

        home_off_std = jnp.take(team_off_std_b, home_idx, axis=1)
        away_off_std = jnp.take(team_off_std_b, away_idx, axis=1)
        home_def_std = jnp.take(team_def_std_b, home_idx, axis=1)
        away_def_std = jnp.take(team_def_std_b, away_idx, axis=1)

        eps_home_std = home_off_std + away_def_std
        eps_away_std = away_off_std + home_def_std

        # Base etas: [Db, G]
        eta_home_base = alpha_b[:, None] + home_off - away_def + home_h
        eta_away_base = alpha_b[:, None] + away_off - home_def

        # Keys: [Db, S, 4, 2]
        # (eps_home_key, eps_away_key, poisson_home_key, poisson_away_key)
        keys = random.split(key, Db * n_sims * 4).reshape(Db, n_sims, 4, 2)

        def _sim_one_draw(keys_draw: jnp.ndarray, eta_h: jnp.ndarray, eta_a: jnp.ndarray, eh_std: jnp.ndarray, ea_std: jnp.ndarray):
            # keys_draw: [S, 4, 2]; eta_*: [G]; e*_std: [G]
            eps_home = jax.vmap(random.normal, in_axes=(0, None))(keys_draw[:, 0], (eta_h.shape[0],)) * eh_std
            eps_away = jax.vmap(random.normal, in_axes=(0, None))(keys_draw[:, 1], (eta_a.shape[0],)) * ea_std

            lam_home = jnp.exp(eta_h + eps_home)
            lam_away = jnp.exp(eta_a + eps_away)

            home_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 2], lam_home)
            away_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 3], lam_away)
            return home_scores, away_scores  # [S, G], [S, G]

        home_s, away_s = jax.vmap(_sim_one_draw, in_axes=(0, 0, 0, 0, 0))(keys, eta_home_base, eta_away_base, eps_home_std, eps_away_std)
        return home_s.astype(jnp.int32), away_s.astype(jnp.int32)

    return _simulate(alpha, offense, defense, h, team_off_std, team_def_std, home_indices, away_indices, block_key)
