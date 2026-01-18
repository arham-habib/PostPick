"""
Game Simulation Module

This module implements simulation of upcoming NCAA basketball games using
a fitted hierarchical Bayesian model. It simulates game outcomes and computes
statistics on spread, total, and moneyline probabilities.
"""
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import json

# Import existing modules
from src.ncaa.data_cleaning import (
    build_team_indexer,
    encode_season,
    drop_teams_with_few_games,
    _normalize_team_name,
)
from src.models.Vanilla import fit_hierarchal_model
from src.utils.enums import EncodedSeason

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "logs"
SCHEDULE_DIR = DATA_DIR / "ncaa" / "schedule"
SIMULATIONS_DIR = DATA_DIR / "simulations"
MODEL_CACHE_DIR = DATA_DIR / "models"

# Ensure directories exist
SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data(sport: str, division: str, year: int) -> pd.DataFrame:
    """
    Load training data from existing file.
    
    Args:
        sport: "men" or "women"
        division: "d1", "d2", or "d3"
        year: Year of the season (e.g., 2024 for 2024-2025 season)
    
    Returns:
        DataFrame with completed game data
    """
    file_str = f"ncaab_{year}_{sport}_{division}.csv"
    file_path = DATA_DIR / file_str
    
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    logging.info(f"Loading training data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = {"home_team", "away_team", "home_score", "away_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")
    
    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    logging.info(f"Loaded {len(df)} games from {file_path}")
    
    return df


def load_schedule(sport: str, division: str, year: int) -> pd.DataFrame:
    """
    Load scheduled games from existing file.
    
    Args:
        sport: "men" or "women"
        division: "d1", "d2", or "d3"
        year: Starting year of the season
    
    Returns:
        DataFrame with scheduled games
    """
    file_str = f"schedule_{year}_{sport}_{division}.csv"
    file_path = SCHEDULE_DIR / file_str
    
    if not file_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {file_path}")
    
    logging.info(f"Loading scheduled games from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} scheduled games")
    
    return df


def filter_unplayed_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for games that haven't been played yet (future dates).
    
    Args:
        schedule_df: DataFrame with scheduled games
    
    Returns:
        Filtered DataFrame with only future games
    """
    if schedule_df.empty:
        return schedule_df
    
    # Convert date column to datetime
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    
    # Filter for future games
    today = pd.Timestamp.now().normalize()
    future_games = schedule_df[schedule_df["date"] >= today].copy()  # type: ignore
    
    logging.info(f"Found {len(future_games)} unplayed games out of {len(schedule_df)} total scheduled")
    
    return future_games  # type: ignore


def get_latest_monday(date: pd.Timestamp) -> str:
    """Get the latest Monday on or before the given date."""
    if pd.isna(date):
        raise ValueError("Cannot get Monday from NaT/NaN date")
    # Ensure it's a Timestamp
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    days_since_monday = date.weekday()
    monday = date - pd.Timedelta(days=days_since_monday)
    monday_str = monday.strftime("%Y-%m-%d")  # type: ignore
    return monday_str


def get_week_from_date(date_str: str) -> str:
    """Get the Monday of the week for a given date string (YYYY-MM-DD format)."""
    try:
        date = pd.to_datetime(date_str)
        return get_latest_monday(date)
    except (ValueError, TypeError):
        return date_str  # Return original if parsing fails


def get_model_cache_name(df: pd.DataFrame) -> str:
    """
    Generate model cache name based on latest Monday of last game in training data.
    
    Args:
        df: Training DataFrame with 'date' column
    
    Returns:
        Model cache name string (e.g., "model_2025-01-12")
    """
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    max_date_val = df_copy["date"].max()
    
    # Handle Series result (shouldn't happen, but be safe)
    if isinstance(max_date_val, pd.Series):
        max_date_val = max_date_val.iloc[0]
    
    # Check for NaT/NaN
    if pd.isna(max_date_val).any() if hasattr(pd.isna(max_date_val), 'any') else pd.isna(max_date_val):
        raise ValueError("No valid dates found in training data")
    
    # Convert to Timestamp scalar
    if not isinstance(max_date_val, pd.Timestamp):
        last_game_date = pd.Timestamp(max_date_val)
    else:
        last_game_date = max_date_val
    
    # Final check
    if not isinstance(last_game_date, pd.Timestamp) or pd.isna(last_game_date):
        raise ValueError("Invalid date value in training data")
    
    monday = get_latest_monday(last_game_date)
    return f"model_{monday}"


def find_latest_model() -> Optional[Path]:
    """Find the latest cached model file."""
    model_files = list(MODEL_CACHE_DIR.glob("model_*.npz"))
    if not model_files:
        return None
    
    # Sort by filename (which includes date) to get latest
    model_files.sort(reverse=True)
    return model_files[0]


def load_cached_model(model_path: Path) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Load cached model samples and metadata.
    
    Args:
        model_path: Path to cached model file
    
    Returns:
        Tuple of (samples dict, metadata dict)
    """
    logging.info(f"Loading cached model from {model_path}")
    data = np.load(model_path, allow_pickle=True)
    
    # Load samples
    samples = {key: jnp.array(data[key]) for key in data.files if key != "metadata"}
    
    # Load metadata
    import pickle
    if "metadata" in data:
        metadata_bytes = data["metadata"][0]
        metadata = pickle.loads(metadata_bytes)
    else:
        metadata = {}
    
    logging.info(f"Loaded model with {samples['alpha'].shape[0]} posterior samples")
    return samples, metadata


def save_model_cache(samples: Dict[str, jnp.ndarray], metadata: Dict[str, Any], 
                    cache_name: str) -> Path:
    """
    Save model samples and metadata to cache.
    
    Args:
        samples: Posterior samples dictionary
        metadata: Metadata dictionary
        cache_name: Cache name (e.g., "model_2025-01-12")
    
    Returns:
        Path to saved cache file
    """
    cache_path = MODEL_CACHE_DIR / f"{cache_name}.npz"
    logging.info(f"Saving model cache to {cache_path}")
    
    # Convert JAX arrays to numpy for saving
    save_dict = {k: np.array(v) for k, v in samples.items()}
    # Save metadata separately using pickle
    import pickle
    metadata_bytes = pickle.dumps(metadata)
    save_dict["metadata"] = np.array([metadata_bytes], dtype=object)
    
    np.savez_compressed(cache_path, **save_dict)
    logging.info(f"Model cache saved")
    
    return cache_path


def prepare_training_data(year: int, sport: str = "men", division: str = "d1", 
                         min_games: int = 30) -> Tuple[pd.DataFrame, EncodedSeason, Dict[str, int], List[str]]:
    """
    Prepare training data: load, clean, and encode.
    
    Args:
        year: Season year
        sport: "men" or "women"
        division: "d1", "d2", or "d3"
        min_games: Minimum number of games per team to include
    
    Returns:
        Tuple of (cleaned DataFrame, EncodedSeason, team_to_id dict, id_to_team list)
    """
    # Load training data
    df = load_training_data(sport, division, year)
    
    # Drop teams with few games
    df_cleaned = drop_teams_with_few_games(df, n=min_games)
    if not isinstance(df_cleaned, pd.DataFrame):
        df_cleaned = pd.DataFrame(df_cleaned)
    
    logging.info(f"After filtering: {len(df_cleaned)} games, {df_cleaned['home_team'].nunique()} unique teams")
    
    # Build team indexer
    team_to_id, id_to_team = build_team_indexer(df_cleaned)
    
    # Encode season
    encoded = encode_season(df_cleaned, team_to_id)
    
    return df_cleaned, encoded, team_to_id, id_to_team


def handle_unknown_team(team_name: str, model_team_to_id: Dict[str, int], 
                       samples: Dict[str, jnp.ndarray]) -> Tuple[int, bool]:
    """
    Handle teams not in the training data.
    
    Args:
        team_name: Name of the team
        model_team_to_id: Team name to index mapping from model
        samples: Posterior samples dictionary
    
    Returns:
        Tuple of (team_index, is_unknown)
        For unknown teams, returns a placeholder index (will use average parameters)
    """
    normalized_name = _normalize_team_name(team_name)
    
    if normalized_name in model_team_to_id:
        return model_team_to_id[normalized_name], False
    else:
        logging.warning(f"Unknown team '{team_name}' (normalized: '{normalized_name}') not in training data. Using average parameters.")
        # Return -1 as a marker for unknown teams
        return -1, True


def map_schedule_teams_to_model(schedule_df: pd.DataFrame, 
                               model_team_to_id: Dict[str, int],
                               samples: Dict[str, jnp.ndarray]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Map schedule team names to model team indices.
    
    Args:
        schedule_df: DataFrame with scheduled games
        model_team_to_id: Team name to index mapping from model
        samples: Posterior samples (for getting n_teams)
    
    Returns:
        Tuple of (DataFrame with mapped indices, metadata dict with unknown teams info)
    """
    df = schedule_df.copy()
    
    # Get number of teams in model
    n_teams = samples['offense'].shape[1]
    
    # Map home and away teams
    home_indices = []
    away_indices = []
    unknown_teams = set()
    
    for _, row in df.iterrows():
        home_team_name = str(row['home_team'])
        away_team_name = str(row['away_team'])
        home_idx, home_unknown = handle_unknown_team(home_team_name, model_team_to_id, samples)
        away_idx, away_unknown = handle_unknown_team(away_team_name, model_team_to_id, samples)
        
        home_indices.append(home_idx)
        away_indices.append(away_idx)
        
        if home_unknown:
            unknown_teams.add(row['home_team'])
        if away_unknown:
            unknown_teams.add(row['away_team'])
    
    df['home_idx'] = home_indices
    df['away_idx'] = away_indices
    
    metadata = {
        'unknown_teams': list(unknown_teams),
        'n_unknown_teams': len(unknown_teams),
        'n_teams_in_model': n_teams,
    }
    
    return df, metadata


def fit_or_load_model(df: pd.DataFrame, encoded: EncodedSeason, 
                     seed: int = 0, num_chains: int = 2, 
                     num_warmup: int = 100, num_samples: int = 300,
                     force_refit: bool = False) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Fit the hierarchical model or load from cache.
    
    Args:
        df: Training DataFrame (for determining cache name)
        encoded: EncodedSeason object with training data
        seed: Random seed for MCMC
        num_chains: Number of MCMC chains
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples per chain
        force_refit: If True, refit even if cache exists
    
    Returns:
        Tuple of (samples dictionary, metadata dictionary)
    """
    cache_name = get_model_cache_name(df)
    cache_path = MODEL_CACHE_DIR / f"{cache_name}.npz"
    
    # Try to load from cache
    if not force_refit and cache_path.exists():
        logging.info(f"Loading cached model: {cache_name}")
        samples, metadata = load_cached_model(cache_path)
        return samples, metadata
    
    # Fit model
    logging.info(f"Fitting model: {cache_name}")
    logging.info(f"  {num_chains} chains, {num_warmup} warmup, {num_samples} samples")
    mcmc, samples = fit_hierarchal_model(
        encoded,
        seed=seed,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples
    )
    
    # Create metadata
    metadata = {
        "cache_name": cache_name,
        "n_teams": encoded.n_teams,
        "n_games": len(encoded.home_idx),
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "fitted_at": datetime.now().isoformat(),
    }
    
    # Save to cache
    save_model_cache(samples, metadata, cache_name)
    
    logging.info(f"Model fitting complete. Posterior samples shape: {samples['alpha'].shape}")
    
    return samples, metadata


def simulate_single_game(home_idx: int, away_idx: int, samples: Dict[str, jnp.ndarray],
                        n_draws: int = 1_000_000, rng_key: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single game with n_draws from the posterior predictive distribution.
    
    For unknown teams (idx == -1), use league-average parameters (offense=0, defense=0, h=h_mu).
    
    Args:
        home_idx: Index of home team (or -1 for unknown)
        away_idx: Index of away team (or -1 for unknown)
        samples: Dictionary of posterior samples with keys: 'alpha', 'offense', 'defense', 'h', 'h_mu'
        n_draws: Number of draws per posterior sample
        rng_key: Optional JAX random key
    
    Returns:
        Tuple of (home_scores, away_scores) arrays of shape (total_draws,)
        where total_draws = n_draws * n_posterior_samples
    """
    n_samples = samples['alpha'].shape[0]
    total_draws = n_samples * n_draws
    
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Convert to numpy for easier handling of unknown teams
    alpha_samples = np.array(samples['alpha'])
    offense_samples = np.array(samples['offense'])  # shape: (n_samples, n_teams)
    defense_samples = np.array(samples['defense'])  # shape: (n_samples, n_teams)
    h_samples = np.array(samples['h'])  # shape: (n_samples, n_teams)
    h_mu_samples = np.array(samples['h_mu'])  # shape: (n_samples,)
    
    # Handle unknown teams (use average parameters)
    if home_idx == -1:
        # Use league average: offense=0, defense=0, h=h_mu
        home_offense = np.zeros(n_samples)
        home_defense = np.zeros(n_samples)
        home_h = h_mu_samples
    else:
        home_offense = offense_samples[:, home_idx]
        home_defense = defense_samples[:, home_idx]
        home_h = h_samples[:, home_idx]
    
    if away_idx == -1:
        away_offense = np.zeros(n_samples)
        away_defense = np.zeros(n_samples)
    else:
        away_offense = offense_samples[:, away_idx]
        away_defense = defense_samples[:, away_idx]
    
    # Compute linear predictors for all samples
    # eta_home = alpha + offense[home] - defense[away] + h[home]
    # eta_away = alpha + offense[away] - defense[home]
    eta_home = alpha_samples + home_offense - away_defense + home_h
    eta_away = alpha_samples + away_offense - home_defense
    
    # Convert to rates (lambda)
    lambda_home = np.exp(eta_home)  # shape: (n_samples,)
    lambda_away = np.exp(eta_away)  # shape: (n_samples,)
    
    # Sample from Poisson for each posterior sample
    # Strategy: For each posterior sample, generate n_draws samples
    # We can use JAX's vectorized poisson which accepts array of rates
    home_scores_list = []
    away_scores_list = []
    
    # Generate random keys for each sample
    rng_key1, rng_key2 = random.split(rng_key)
    
    # Sample for each posterior sample
    for i in range(n_samples):
        # Get keys for this sample's draws
        rng_key1, subkey1 = random.split(rng_key1)
        rng_key2, subkey2 = random.split(rng_key2)
        
        # Sample n_draws from Poisson for this posterior sample
        # JAX's poisson can sample multiple times with same rate
        # We'll sample each draw individually (could optimize further with vmap)
        lambda_home_val = float(lambda_home[i])
        lambda_away_val = float(lambda_away[i])
        
        # Use numpy's random for faster sampling (converted from JAX)
        # Generate keys for all draws from this sample
        subkeys1 = random.split(subkey1, n_draws)
        subkeys2 = random.split(subkey2, n_draws)
        
        # Sample all draws for this posterior sample
        # Convert to numpy array for efficiency
        home_scores_sample = np.array([
            random.poisson(subkeys1[j], lambda_home_val).item()
            for j in range(n_draws)
        ])
        away_scores_sample = np.array([
            random.poisson(subkeys2[j], lambda_away_val).item()
            for j in range(n_draws)
        ])
        
        home_scores_list.append(home_scores_sample)
        away_scores_list.append(away_scores_sample)
    
    # Concatenate all samples
    home_scores = np.concatenate(home_scores_list)
    away_scores = np.concatenate(away_scores_list)
    
    return home_scores, away_scores


def simulate_games(games_df: pd.DataFrame, samples: Dict[str, jnp.ndarray],
                  n_draws: int = 1_000_000) -> pd.DataFrame:
    """
    Simulate multiple games.
    
    Args:
        games_df: DataFrame with games to simulate (must have 'home_idx' and 'away_idx' columns)
        samples: Posterior samples dictionary
        n_draws: Number of draws per posterior sample
    
    Returns:
        DataFrame with simulation results (one row per game with arrays of scores)
    """
    results = []
    
    logging.info(f"Simulating {len(games_df)} games with {n_draws} draws per posterior sample")
    
    for i, (idx, row) in enumerate(games_df.iterrows()):
        if i % 10 == 0:
            logging.info(f"Simulating game {i + 1}/{len(games_df)}: {row['home_team']} vs {row['away_team']}")
        
        home_idx = int(row['home_idx'])
        away_idx = int(row['away_idx'])
        
        # Use a different random key for each game (use index as seed)
        game_seed = int(idx) if isinstance(idx, (int, np.integer)) else i
        rng_key = random.PRNGKey(game_seed)
        home_scores, away_scores = simulate_single_game(
            home_idx, away_idx, samples, n_draws=n_draws, rng_key=rng_key
        )
        
        results.append({
            'gameID': row.get('gameID', idx),
            'date': row.get('date', ''),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_idx': home_idx,
            'away_idx': away_idx,
            'home_scores': home_scores,
            'away_scores': away_scores,
        })
    
    return pd.DataFrame(results)


def compute_game_statistics(home_scores: np.ndarray, away_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a single game simulation.
    
    Args:
        home_scores: Array of home scores
        away_scores: Array of away scores
    
    Returns:
        Dictionary with statistics
    """
    spread = home_scores - away_scores
    total = home_scores + away_scores
    
    # Spread statistics
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread))
    spread_p5 = float(np.percentile(spread, 5))
    spread_p25 = float(np.percentile(spread, 25))
    spread_p50 = float(np.percentile(spread, 50))
    spread_p75 = float(np.percentile(spread, 75))
    spread_p95 = float(np.percentile(spread, 95))
    
    # Total statistics
    total_mean = float(np.mean(total))
    total_std = float(np.std(total))
    total_p5 = float(np.percentile(total, 5))
    total_p25 = float(np.percentile(total, 25))
    total_p50 = float(np.percentile(total, 50))
    total_p75 = float(np.percentile(total, 75))
    total_p95 = float(np.percentile(total, 95))
    
    # Moneyline probabilities
    home_wins = np.sum(home_scores > away_scores)
    away_wins = np.sum(away_scores > home_scores)
    ties = np.sum(home_scores == away_scores)
    n_sims = len(home_scores)
    
    moneyline_home_win = float(home_wins / n_sims)
    moneyline_away_win = float(away_wins / n_sims)
    moneyline_tie = float(ties / n_sims)
    
    # Correlation between spread and total
    spread_total_correlation = float(np.corrcoef(spread, total)[0, 1])
    
    # Covariance (for multivariate analysis)
    spread_total_cov = float(np.cov(spread, total)[0, 1])
    
    # Build result dict with explicit float conversions
    result: Dict[str, float] = {
        'spread_mean': float(spread_mean),
        'spread_std': float(spread_std),
        'spread_p5': float(spread_p5),
        'spread_p25': float(spread_p25),
        'spread_p50': float(spread_p50),
        'spread_p75': float(spread_p75),
        'spread_p95': float(spread_p95),
        'total_mean': float(total_mean),
        'total_std': float(total_std),
        'total_p5': float(total_p5),
        'total_p25': float(total_p25),
        'total_p50': float(total_p50),
        'total_p75': float(total_p75),
        'total_p95': float(total_p95),
        'moneyline_home_win': moneyline_home_win,
        'moneyline_away_win': moneyline_away_win,
        'moneyline_tie': moneyline_tie,
        'spread_total_correlation': spread_total_correlation,
        'spread_total_covariance': spread_total_cov,
    }
    return result


def compute_multivariate_stats(all_spreads: np.ndarray, all_totals: np.ndarray) -> Dict[str, Any]:
    """
    Compute overall multivariate statistics across all games.
    
    Args:
        all_spreads: Concatenated spread values from all games
        all_totals: Concatenated total values from all games
    
    Returns:
        Dictionary with multivariate statistics
    """
    # Empirical covariance matrix
    cov_matrix = np.cov(all_spreads, all_totals)
    
    # Correlation
    corr = float(np.corrcoef(all_spreads, all_totals)[0, 1])
    
    # Marginal statistics
    spread_stats = {
        'mean': float(np.mean(all_spreads)),
        'std': float(np.std(all_spreads)),
        'p5': float(np.percentile(all_spreads, 5)),
        'p25': float(np.percentile(all_spreads, 25)),
        'p50': float(np.percentile(all_spreads, 50)),
        'p75': float(np.percentile(all_spreads, 75)),
        'p95': float(np.percentile(all_spreads, 95)),
    }
    
    total_stats = {
        'mean': float(np.mean(all_totals)),
        'std': float(np.std(all_totals)),
        'p5': float(np.percentile(all_totals, 5)),
        'p25': float(np.percentile(all_totals, 25)),
        'p50': float(np.percentile(all_totals, 50)),
        'p75': float(np.percentile(all_totals, 75)),
        'p95': float(np.percentile(all_totals, 95)),
    }
    
    return {
        'covariance_matrix': cov_matrix.tolist(),
        'correlation': corr,
        'spread': spread_stats,
        'total': total_stats,
        'n_observations': len(all_spreads),
    }


def generate_report(simulation_results: pd.DataFrame, summary_stats: Dict[str, Any],
                   output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Generate and save summary report.
    
    Args:
        simulation_results: DataFrame with per-game statistics
        summary_stats: Dictionary with aggregate multivariate statistics
        output_path: Base path for output files
        metadata: Optional metadata to include
    """
    # Compute statistics for each game
    game_stats = []
    all_spreads = []
    all_totals = []
    
    for _, row in simulation_results.iterrows():
        home_scores_arr = np.array(row['home_scores'])
        away_scores_arr = np.array(row['away_scores'])
        stats = compute_game_statistics(home_scores_arr, away_scores_arr)
        
        # Calculate week (Monday of the week)
        date_str = str(row['date'])
        week = get_week_from_date(date_str)
        
        game_stats.append({
            'gameID': row['gameID'],
            'date': row['date'],
            'week': week,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            **stats
        })
        
        all_spreads.extend(row['home_scores'] - row['away_scores'])
        all_totals.extend(row['home_scores'] + row['away_scores'])
    
    # Create results DataFrame
    results_df = pd.DataFrame(game_stats)
    
    # Compute multivariate stats if not provided
    if summary_stats is None:
        summary_stats = compute_multivariate_stats(
            np.array(all_spreads), np.array(all_totals)
        )
    
    # Save per-game results
    csv_path = output_path.with_suffix('.csv')
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved per-game results to {csv_path}")
    
    # Save summary statistics
    json_path = output_path.with_suffix('.json')
    summary_dict = {
        'metadata': metadata or {},
        'summary_statistics': summary_stats,
        'generated_at': datetime.now().isoformat(),
        'n_games': len(results_df),
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    logging.info(f"Saved summary statistics to {json_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Games simulated: {len(results_df)}")
    print(f"\nAggregate Statistics:")
    print(f"  Spread: mean={summary_stats['spread']['mean']:.2f}, std={summary_stats['spread']['std']:.2f}")
    print(f"  Total: mean={summary_stats['total']['mean']:.2f}, std={summary_stats['total']['std']:.2f}")
    print(f"  Correlation (spread, total): {summary_stats['correlation']:.4f}")
    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"{'='*60}\n")


def save_results(results_df: pd.DataFrame, summary_stats: Dict[str, Any],
                output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save results to files (alias for generate_report for backward compatibility).
    """
    generate_report(results_df, summary_stats, output_path, metadata)


def main():
    """Main function to orchestrate the simulation pipeline."""
    parser = argparse.ArgumentParser(
        description="Simulate upcoming NCAA basketball games using hierarchical Bayesian model"
    )
    parser.add_argument(
        "--sport", type=str, choices=["men", "women"], default="men",
        help="Sport category: men or women (default: men)"
    )
    parser.add_argument(
        "--division", type=str, choices=["d1", "d2", "d3"], default="d1",
        help="NCAA division (default: d1)"
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Starting year of the season (e.g., 2024 for 2024-2025 season)"
    )
    parser.add_argument(
        "--training-year", type=int, default=None,
        help="Year of training data (default: same as --year)"
    )
    parser.add_argument(
        "--min-games", type=int, default=30,
        help="Minimum number of games per team to include in training (default: 30)"
    )
    parser.add_argument(
        "--n-draws", type=int, default=1_000_000,
        help="Number of draws per posterior sample (default: 1,000,000)"
    )
    parser.add_argument(
        "--num-chains", type=int, default=2,
        help="Number of MCMC chains (default: 2)"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=100,
        help="Number of warmup samples (default: 100)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=300,
        help="Number of posterior samples per chain (default: 300)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/simulations/game_simulations_{year}_{sport}_{division}.csv)"
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Force model refit even if cache exists"
    )
    parser.add_argument(
        "--use-latest-model", action="store_true",
        help="Use latest cached model instead of training data-based model"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = LOG_DIR / f"simulation_{args.year}_{args.sport}_{args.division}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("Starting game simulation pipeline")
    logging.info("="*60)
    logging.info(f"Sport: {args.sport}, Division: {args.division}, Year: {args.year}")
    logging.info(f"Training data year: {args.training_year or args.year}")
    logging.info(f"Number of draws per sample: {args.n_draws:,}")
    
    try:
        # Step 1: Load or fit model
        if args.use_latest_model:
            # Use latest cached model
            latest_model = find_latest_model()
            if latest_model is None:
                raise FileNotFoundError("No cached models found. Run without --use-latest-model to fit a model.")
            logging.info(f"\nStep 1: Loading latest model: {latest_model.name}")
            samples, model_metadata = load_cached_model(latest_model)
            
            # Load training data to get team mapping
            training_year = args.training_year or args.year
            df_training = load_training_data(args.sport, args.division, training_year)
            df_training = drop_teams_with_few_games(df_training, n=args.min_games)
            if not isinstance(df_training, pd.DataFrame):
                df_training = pd.DataFrame(df_training)
            team_to_id, id_to_team = build_team_indexer(df_training)
        else:
            # Prepare training data and fit/load model
            training_year = args.training_year or args.year
            logging.info(f"\nStep 1: Preparing training data from {training_year} season...")
            df_training, encoded, team_to_id, id_to_team = prepare_training_data(
                training_year, args.sport, args.division, min_games=args.min_games
            )
            
            logging.info(f"\nStep 2: Fitting or loading model...")
            samples, model_metadata = fit_or_load_model(
                df_training,
                encoded,
                seed=0,
                num_chains=args.num_chains,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                force_refit=args.force_refit
            )
        
        # Step 2/3: Load schedule and filter for unplayed games
        logging.info(f"\nStep {2 if args.use_latest_model else 3}: Loading schedule...")
        schedule_df = load_schedule(args.sport, args.division, args.year)
        upcoming_games = filter_unplayed_games(schedule_df)
        
        if upcoming_games.empty:
            logging.warning("No unplayed games found. Exiting.")
            return
        
        logging.info(f"Found {len(upcoming_games)} unplayed games to simulate")
        
        # Step 3/4: Map teams to model indices
        logging.info(f"\nStep {3 if args.use_latest_model else 4}: Mapping teams to model indices...")
        games_with_indices, mapping_metadata = map_schedule_teams_to_model(
            upcoming_games, team_to_id, samples
        )
        
        if mapping_metadata['n_unknown_teams'] > 0:
            logging.warning(f"Found {mapping_metadata['n_unknown_teams']} unknown teams: {mapping_metadata['unknown_teams']}")
        
        # Step 4/5: Simulate games
        logging.info(f"\nStep {4 if args.use_latest_model else 5}: Simulating games...")
        simulation_results = simulate_games(games_with_indices, samples, n_draws=args.n_draws)
        
        # Step 5/6: Compute statistics
        logging.info(f"\nStep {5 if args.use_latest_model else 6}: Computing statistics...")
        all_spreads = []
        all_totals = []
        
        for _, row in simulation_results.iterrows():
            spreads = row['home_scores'] - row['away_scores']
            totals = row['home_scores'] + row['away_scores']
            all_spreads.extend(spreads)
            all_totals.extend(totals)
        
        summary_stats = compute_multivariate_stats(
            np.array(all_spreads), np.array(all_totals)
        )
        
        # Combine metadata
        full_metadata = {
            **mapping_metadata,
            **model_metadata,
            'sport': args.sport,
            'division': args.division,
            'season_year': args.year,
            'training_year': training_year,
            'n_draws_per_sample': args.n_draws,
            'n_teams_in_training': len(team_to_id),
            'min_games_per_team': args.min_games,
        }
        
        # Step 6/7: Generate report
        logging.info(f"\nStep {6 if args.use_latest_model else 7}: Generating report...")
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = SIMULATIONS_DIR / f"game_simulations_{args.year}_{args.sport}_{args.division}"
        
        generate_report(simulation_results, summary_stats, output_path, full_metadata)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
