import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
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
