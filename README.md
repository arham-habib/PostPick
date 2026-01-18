# PostPick

PostPick is a sports prediction platform that uses hierarchical Bayesian models to predict NCAA basketball game outcomes. The models learn team strengths (offense and defense) from historical game data and use these estimates to simulate future games.

## Overview

The core idea is to model each team's offensive and defensive capabilities using a hierarchical Bayesian framework. Once we've observed about half a season's worth of games, we can fit the model to estimate these team parameters. These estimates are then used to simulate upcoming games and generate predictions for spreads, totals, and win probabilities.

## Vanilla Model

The "Vanilla" model is a hierarchical Bayesian model that captures team-level effects and home-court advantage. It's designed to be fit after approximately half the season has been played, when we have enough data to reliably estimate team strengths.

### Model Structure

The model assumes that each team has:
- An **offensive strength** parameter ($o_i$)
- A **defensive strength** parameter ($d_i$)
- A **home-court advantage** parameter ($h_i$)

For a game where team $i$ plays at home against team $j$, the expected scores are modeled as:

$$
\begin{align}
\eta_{\text{home}} &= \alpha + o_i - d_j + h_i \\
\eta_{\text{away}} &= \alpha + o_j - d_i \\
\lambda_{\text{home}} &= \exp(\eta_{\text{home}}) \\
\lambda_{\text{away}} &= \exp(\eta_{\text{away}})
\end{align}
$$

where $\alpha$ is a global intercept and the scores follow a Poisson distribution:

$$
\begin{align}
\text{score}_{\text{home}} &\sim \text{Poisson}(\lambda_{\text{home}}) \\
\text{score}_{\text{away}} &\sim \text{Poisson}(\lambda_{\text{away}})
\end{align}
$$

### Hierarchical Priors

The model uses hierarchical priors to share information across teams:

$$
\begin{align}
\alpha &\sim \mathcal{N}(0, 5) \\
\sigma_{\text{off}}, \sigma_{\text{def}}, \tau_h &\sim \text{HalfNormal}(1) \\
h_\mu &\sim \mathcal{N}(0, 1) \\
o_i^{\text{uncentered}} &\sim \mathcal{N}(0, \sigma_{\text{off}}) \\
d_i^{\text{uncentered}} &\sim \mathcal{N}(0, \sigma_{\text{def}}) \\
h_i &\sim \mathcal{N}(h_\mu, \tau_h) \\
o_i &= o_i^{\text{uncentered}} - \bar{o}^{\text{uncentered}} \\
d_i &= d_i^{\text{uncentered}} - \bar{d}^{\text{uncentered}}
\end{align}
$$

The offense and defense parameters are centered (sum to zero) to ensure identifiability, while the home advantage parameters share a common mean $h_\mu$ across all teams.

### Fitting

The model is fit using MCMC (Markov Chain Monte Carlo) via NumPyro, which provides posterior distributions for all parameters. These posterior samples are then used to simulate future games by sampling from the posterior predictive distribution.
