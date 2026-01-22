import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from typing import Dict, Tuple
from src.utils.enums import EncodedSeason
from functools import partial

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


@partial(jax.jit, static_argnums=(7,))
def _simulate_kernel(
    alpha_b: jnp.ndarray,
    offense_b: jnp.ndarray,
    defense_b: jnp.ndarray,
    h_b: jnp.ndarray,
    home_idx: jnp.ndarray,
    away_idx: jnp.ndarray,
    key: jnp.ndarray,
    n_sims: int
):
    Db = alpha_b.shape[0]

    home_off = jnp.take(offense_b, home_idx, axis=1)
    away_off = jnp.take(offense_b, away_idx, axis=1)
    home_def = jnp.take(defense_b, home_idx, axis=1)
    away_def = jnp.take(defense_b, away_idx, axis=1)
    home_h = jnp.take(h_b, home_idx, axis=1)

    eta_home = alpha_b[:, None] + home_off - away_def + home_h
    eta_away = alpha_b[:, None] + away_off - home_def
    lam_home = jnp.exp(eta_home)
    lam_away = jnp.exp(eta_away)

    keys = random.split(key, Db * n_sims * 2).reshape(Db, n_sims, 2, 2)

    def _sim_one_draw(keys_draw: jnp.ndarray, lam_h: jnp.ndarray, lam_a: jnp.ndarray):
        home_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 0], lam_h)
        away_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 1], lam_a)
        return home_scores, away_scores

    home_s, away_s = jax.vmap(_sim_one_draw, in_axes=(0, 0, 0))(keys, lam_home, lam_away)
    return home_s.astype(jnp.int32), away_s.astype(jnp.int32)

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
    """Simulate an exploded block of scores for the Vanilla model.

    This is the GPU-friendly path used by the simulation pipeline:
    - Vectorized over posterior draws, simulations, and games
    - Returns integer score tensors shaped [Db, S, G]

    Args:
        home_indices: [G] home team indices
        away_indices: [G] away team indices
        samples: posterior samples (expects keys: 'alpha', 'offense', 'defense', 'h')
                 Shapes: alpha [D], offense/defense/h [D, T]
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
    block_key = random.fold_in(rng_key, draw_start)

    return _simulate_kernel(alpha, offense, defense, h, home_indices, away_indices, block_key, n_sims)

# def simulate_scores_block(
#     home_indices: jnp.ndarray,
#     away_indices: jnp.ndarray,
#     samples: Dict[str, jnp.ndarray],
#     *,
#     draw_start: int,
#     draw_end: int,
#     n_sims: int,
#     rng_key: jnp.ndarray,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """Simulate an exploded block of scores for the Vanilla model.

#     This is the GPU-friendly path used by the simulation pipeline:
#     - Vectorized over posterior draws, simulations, and games
#     - Returns integer score tensors shaped [Db, S, G]

#     Args:
#         home_indices: [G] home team indices
#         away_indices: [G] away team indices
#         samples: posterior samples (expects keys: 'alpha', 'offense', 'defense', 'h')
#                  Shapes: alpha [D], offense/defense/h [D, T]
#         draw_start: starting posterior draw index (inclusive)
#         draw_end: ending posterior draw index (exclusive)
#         n_sims: simulations per posterior draw per game
#         rng_key: base RNG key; block is deterministically derived from this + draw_start

#     Returns:
#         (home_scores, away_scores): both shaped [Db, S, G], dtype int32
#     """
#     # Defensive checks (cheap; keeps weird silent shape bugs from propagating)
#     if draw_end <= draw_start:
#         raise ValueError(f"draw_end must be > draw_start (got {draw_start=}, {draw_end=})")
#     if n_sims <= 0:
#         raise ValueError(f"n_sims must be > 0 (got {n_sims})")

#     # Slice samples for this block on host side (keeps JIT signature simpler)
#     alpha = samples["alpha"][draw_start:draw_end]  # [Db]
#     offense = samples["offense"][draw_start:draw_end]  # [Db, T]
#     defense = samples["defense"][draw_start:draw_end]  # [Db, T]
#     h = samples["h"][draw_start:draw_end]  # [Db, T]

#     # Make the block RNG deterministic and independent across blocks
#     block_key = random.fold_in(rng_key, draw_start)

#     @jax.jit
#     def _simulate(
#         alpha_b: jnp.ndarray,
#         offense_b: jnp.ndarray,
#         defense_b: jnp.ndarray,
#         h_b: jnp.ndarray,
#         home_idx: jnp.ndarray,
#         away_idx: jnp.ndarray,
#         key: jnp.ndarray,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         Db = alpha_b.shape[0]
#         G = home_idx.shape[0]

#         # Gather team parameters: result shapes [Db, G]
#         home_off = jnp.take(offense_b, home_idx, axis=1)
#         away_off = jnp.take(offense_b, away_idx, axis=1)
#         home_def = jnp.take(defense_b, home_idx, axis=1)
#         away_def = jnp.take(defense_b, away_idx, axis=1)
#         home_h = jnp.take(h_b, home_idx, axis=1)

#         # Compute rates: [Db, G]
#         eta_home = alpha_b[:, None] + home_off - away_def + home_h
#         eta_away = alpha_b[:, None] + away_off - home_def
#         lam_home = jnp.exp(eta_home)
#         lam_away = jnp.exp(eta_away)

#         # Generate keys: [Db, S, 2, 2] => (home_key, away_key)
#         keys = random.split(key, Db * n_sims * 2).reshape(Db, n_sims, 2, 2)

#         def _sim_one_draw(keys_draw: jnp.ndarray, lam_h: jnp.ndarray, lam_a: jnp.ndarray):
#             # keys_draw: [S, 2, 2], lam_*: [G]
#             home_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 0], lam_h)
#             away_scores = jax.vmap(random.poisson, in_axes=(0, None))(keys_draw[:, 1], lam_a)
#             return home_scores, away_scores  # [S, G], [S, G]

#         home_s, away_s = jax.vmap(_sim_one_draw, in_axes=(0, 0, 0))(keys, lam_home, lam_away)
#         # home_s/away_s: [Db, S, G]
#         return home_s.astype(jnp.int32), away_s.astype(jnp.int32)

#     return _simulate(alpha, offense, defense, h, home_indices, away_indices, block_key)