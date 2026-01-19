"""
Game Simulation Module

This module implements simulation of upcoming NCAA basketball games using
a fitted hierarchical Bayesian model. Supports multiple model flavors with
sport-specific storage organization and vectorized simulation.
"""
import os
# Force CPU backend to avoid Metal backend issues with NumPyro HMC
# Metal doesn't support all operations (e.g., bitwise_count/popcnt) needed by NumPyro
# This must be set BEFORE importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import importlib

from src.ncaa.data_cleaning import (
    build_team_indexer,
    encode_season,
    drop_teams_with_few_games
)
from src.utils.logging_utils import (
    setup_fitting_logger,
    setup_simulation_logger,
    setup_logger,
    get_log_base_dir,
    get_timestamp
)
from src.utils.data_utils import (
    get_model_data_path,
    get_simulation_data_path,
    get_game_data_path,
    get_schedule_data_path
)
from src.utils.enums import EncodedSeason


def get_model_path(sport: str, model_name: str, monday_date: str) -> Path:
    """Get path to model file."""
    return get_model_data_path("ncaab", sport, model_name, monday_date)


def get_simulation_path(sport: str, model_name: str, monday_date: str) -> Path:
    """Get path to simulation file."""
    return get_simulation_data_path("ncaab", sport, model_name, monday_date)


def load_training_data(sport: str, year: int) -> pd.DataFrame:
    """Load training data (assumes d1 division)."""
    file_path = get_game_data_path("ncaab", year, sport, "d1")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    logging.info(f"Loading training data from {file_path}")
    df = pd.read_csv(file_path)
    
    required_cols = {"home_team", "away_team", "home_score", "away_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")
    
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    logging.info(f"Loaded {len(df)} games from {file_path}")
    return df


def load_schedule(sport: str, year: int) -> pd.DataFrame:
    """Load scheduled games (assumes d1 division)."""
    file_path = get_schedule_data_path("ncaab", year, sport, "d1")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {file_path}")
    
    logging.info(f"Loading scheduled games from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} scheduled games")
    return df


def get_latest_monday(date: pd.Timestamp) -> str:
    """Get the latest Monday on or before the given date."""
    if pd.isna(date):
        raise ValueError("Cannot get Monday from NaT/NaN date")
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    days_since_monday = date.weekday()
    monday = date - pd.Timedelta(days=days_since_monday)
    return monday.strftime("%Y-%m-%d")  # type: ignore


def get_monday_cutoff(df: pd.DataFrame) -> str:
    """Get Monday cutoff date for training (Monday at midnight)."""
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    max_date = df_copy["date"].max()
    
    if isinstance(max_date, pd.Series):
        max_date = max_date.iloc[0]
    
    if not isinstance(max_date, pd.Timestamp):
        max_date = pd.Timestamp(max_date)
    
    if not isinstance(max_date, pd.Timestamp) or pd.isna(max_date):
        raise ValueError("No valid dates found in training data")
    
    # Get Monday of the week containing the last game
    monday = get_latest_monday(max_date)
    return monday


def filter_next_week_games(schedule_df: pd.DataFrame, cutoff_monday: str) -> pd.DataFrame:
    """Filter for games in the next week (Monday through Sunday after cutoff)."""
    if schedule_df.empty:
        return schedule_df
    
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    
    cutoff_date = pd.to_datetime(cutoff_monday)
    week_end = cutoff_date + pd.Timedelta(days=6)  # Sunday
    
    # Filter: games on cutoff Monday through following Sunday
    mask = (schedule_df["date"] >= cutoff_date) & (schedule_df["date"] <= week_end)  # type: ignore
    next_week = schedule_df[mask].copy()  # type: ignore
    
    logging.info(f"Found {len(next_week)} games in next week (Monday-Sunday) out of {len(schedule_df)} total scheduled")
    return next_week  # type: ignore


def prepare_training_data(year: int, sport: str = "men", min_games: int = 10) -> Tuple[pd.DataFrame, EncodedSeason, Dict[str, int], List[str]]:
    """Prepare training data: load, clean, and encode."""
    df = load_training_data(sport, year)
    df_cleaned = drop_teams_with_few_games(df, n=min_games)
    if not isinstance(df_cleaned, pd.DataFrame):
        df_cleaned = pd.DataFrame(df_cleaned)
    
    logging.info(f"After filtering: {len(df_cleaned)} games, {df_cleaned['home_team'].nunique()} unique teams")
    
    team_to_id, id_to_team = build_team_indexer(df_cleaned)
    encoded = encode_season(df_cleaned, team_to_id)
    
    return df_cleaned, encoded, team_to_id, id_to_team


def map_schedule_teams_to_model(schedule_df: pd.DataFrame, team_to_id: Dict[str, int]) -> pd.DataFrame:
    """Map schedule team names to model team indices."""
    df = schedule_df.copy()
    home_indices = []
    away_indices = []
    
    for _, row in df.iterrows():
        home_idx = team_to_id[str(row['home_team'])]
        away_idx = team_to_id[str(row['away_team'])]
        home_indices.append(home_idx)
        away_indices.append(away_idx)
    
    df['home_idx'] = home_indices
    df['away_idx'] = away_indices
    
    return df


def load_model(sport: str, model_name: str, monday_date: str) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any], Dict[str, int], List[str]]:
    """Load model from pickle file."""
    model_path = get_model_path(sport, model_name, monday_date)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Convert numpy arrays back to JAX arrays
    samples = {k: jnp.array(v) for k, v in model_data['samples'].items()}
    metadata = model_data['metadata']
    team_to_id = model_data['team_to_id']
    id_to_team = model_data['id_to_team']
    
    logging.info(f"Loaded model with {samples['alpha'].shape[0]} posterior samples")
    return samples, metadata, team_to_id, id_to_team


def save_model(sport: str, model_name: str, monday_date: str, samples: Dict[str, jnp.ndarray],
               metadata: Dict[str, Any], team_to_id: Dict[str, int], id_to_team: List[str]) -> Path:
    """Save model to pickle file."""
    model_path = get_model_path(sport, model_name, monday_date)
    logging.info(f"Saving model to {model_path}")
    
    # Convert JAX arrays to numpy for pickling
    model_data = {
        'samples': {k: np.array(v) for k, v in samples.items()},
        'metadata': metadata,
        'team_to_id': team_to_id,
        'id_to_team': id_to_team,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logging.info(f"Model saved")
    return model_path


def fit_or_load_model(df: pd.DataFrame, encoded: EncodedSeason, model_name: str, sport: str,
                     monday_date: str, seed: int = 0, num_chains: int = 2,
                     num_warmup: int = 100, num_samples: int = 300,
                     force_refit: bool = False, fitting_logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any], Dict[str, int], List[str]]:
    """Fit model or load from cache."""
    model_path = get_model_path(sport, model_name, monday_date)
    
    # Try to load from cache
    if not force_refit and model_path.exists():
        logging.info(f"Loading cached model: {model_name}_{monday_date}")
        return load_model(sport, model_name, monday_date)
    
    # Use fitting logger if provided, otherwise use main logger
    fit_log = fitting_logger if fitting_logger else logging
    
    # Dynamically import model
    model_module = importlib.import_module(f"src.models.{model_name}")
    fit_hierarchal_model = getattr(model_module, 'fit_hierarchal_model')
    
    # Fit model
    fit_log.info(f"Fitting model: {model_name}_{monday_date}")
    fit_log.info(f"  {num_chains} chains, {num_warmup} warmup, {num_samples} samples")
    mcmc, samples = fit_hierarchal_model(
        encoded,
        seed=seed,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples
    )
    
    # Get team mapping
    team_to_id = encoded.team_to_id  # type: ignore
    id_to_team = encoded.id_to_team  # type: ignore
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "monday_date": monday_date,
        "n_teams": encoded.n_teams,
        "n_games": len(encoded.home_idx),
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "fitted_at": datetime.now().isoformat(),
    }
    
    # Save to cache
    save_model(sport, model_name, monday_date, samples, metadata, team_to_id, id_to_team)
    
    fit_log.info(f"Model fitting complete. Posterior samples shape: {samples['alpha'].shape}")
    return samples, metadata, team_to_id, id_to_team


def simulate_games(games_df: pd.DataFrame, samples: Dict[str, jnp.ndarray],
                  model_name: str, n_sims: int = 250, rng_key: Optional[jnp.ndarray] = None,
                  sim_logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Simulate games using model-specific vectorized simulation.
    
    Args:
        games_df: DataFrame with games to simulate (must have 'home_idx' and 'away_idx' columns)
        samples: Posterior samples dictionary
        model_name: Name of the model ('Vanilla' or 'TeamVol')
        n_sims: Number of non-tie simulations per parameter draw per game
        rng_key: Optional random key
    
    Returns:
        DataFrame with simulation results
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Use simulation logger if provided, otherwise use main logger
    sim_log = sim_logger if sim_logger else logging
    
    if games_df.empty:
        sim_log.warning("No games to simulate")
        return pd.DataFrame()
    
    # Check that all teams are known (no -1 indices)
    if (games_df['home_idx'] == -1).any() or (games_df['away_idx'] == -1).any():
        raise ValueError("Games with unknown teams (idx == -1) should be filtered out before simulation")
    
    # Dynamically import model-specific simulator
    model_module = importlib.import_module(f"src.models.{model_name}")
    simulate_batch_func = getattr(model_module, 'simulate_games_batch')
    
    n_param_draws = samples['alpha'].shape[0]
    n_games = len(games_df)
    
    sim_log.info(f"Simulating {n_games} games with {n_param_draws} parameter draws, {n_sims} non-tie simulations per draw using {model_name} model")
    
    # Get indices as JAX arrays
    home_indices = jnp.array(games_df['home_idx'].values, dtype=jnp.int32)
    away_indices = jnp.array(games_df['away_idx'].values, dtype=jnp.int32)
    
    # Ensure rng_key is not None
    current_rng_key = rng_key if rng_key is not None else random.PRNGKey(42)
    
    # Determine number of rng keys needed per game based on model
    # Vanilla needs 2 keys per game, TeamVol needs 4
    n_keys_per_game = 4 if model_name == "TeamVol" else 2
    
    # Vectorized simulation - generates n_sims per param draw per game
    # We need to simulate more to account for ties, then filter
    over_sim_factor = 1.05  # Simulate 5% more to account for ties
    n_sims_with_ties = int(n_sims * over_sim_factor)
    
    all_home_scores_list = []
    all_away_scores_list = []
    
    # Process parameter draws
    for param_draw_idx in range(n_param_draws):
        if param_draw_idx % 50 == 0:
            sim_log.info(f"Processing parameter draw {param_draw_idx + 1}/{n_param_draws}")
        
        # Get single parameter draw
        param_draw_samples = {k: v[param_draw_idx] for k, v in samples.items()}
        
        # Extract parameters
        alpha = float(param_draw_samples['alpha'])
        offense = param_draw_samples['offense']
        defense = param_draw_samples['defense']
        h = param_draw_samples['h']
        
        # Simulate a large batch at once to account for ties, then filter vectorized
        # Simulate ~30% more to account for ties (typically ties are rare in basketball)
        batch_size = int(n_sims * 1.3)
        
        # Generate all random keys upfront for the batch
        keys_split = random.split(current_rng_key, batch_size + 1)
        current_rng_key = keys_split[0]
        batch_keys = keys_split[1:]
        
        # For each simulation in the batch, generate keys for all games
        all_home_batch = []
        all_away_batch = []
        
        for batch_idx in range(batch_size):
            batch_rng_key = batch_keys[batch_idx]
            
            # Generate rng keys for all games: [n_games, n_keys_per_game, 2]
            game_base_keys = random.split(batch_rng_key, n_games)  # Shape: (n_games, 2)
            
            # For each game, split the base key into n_keys_per_game subkeys
            game_keys_list = []
            for i in range(n_games):
                subkeys = random.split(game_base_keys[i], n_keys_per_game)  # Shape: (n_keys_per_game, 2)
                game_keys_list.append(subkeys)
            game_keys = jnp.stack(game_keys_list)  # Shape: (n_games, n_keys_per_game, 2)
            
            # Vectorized simulation for all games
            if model_name == "TeamVol":
                batch_home, batch_away = simulate_batch_func(
                    home_indices, away_indices, alpha, offense, defense, h,
                    param_draw_samples['team_off_std'], param_draw_samples['team_def_std'],
                    game_keys
                )
            else:  # Vanilla
                batch_home, batch_away = simulate_batch_func(
                    home_indices, away_indices, alpha, offense, defense, h, game_keys
                )
            
            all_home_batch.append(batch_home)
            all_away_batch.append(batch_away)
        
        # Stack all simulations: [batch_size, n_games]
        home_scores_all = jnp.stack(all_home_batch)  # Shape: (batch_size, n_games)
        away_scores_all = jnp.stack(all_away_batch)  # Shape: (batch_size, n_games)
        
        # Filter ties vectorized: [batch_size, n_games] boolean mask
        non_ties = home_scores_all != away_scores_all  # Shape: (batch_size, n_games)
        
        # For each game, take first n_sims non-ties
        # Transpose for easier per-game processing: [n_games, batch_size]
        home_scores_per_game = []
        away_scores_per_game = []
        
        for game_idx in range(n_games):
            game_home = home_scores_all[:, game_idx]  # Shape: (batch_size,)
            game_away = away_scores_all[:, game_idx]  # Shape: (batch_size,)
            game_non_ties = non_ties[:, game_idx]  # Shape: (batch_size,)
            
            # Find indices where non-tie
            non_tie_mask = game_non_ties
            n_non_ties = jnp.sum(non_tie_mask)
            
            if n_non_ties >= n_sims:
                # Take first n_sims non-ties
                non_tie_idx = 0
                selected_home = []
                selected_away = []
                for i in range(batch_size):
                    if non_tie_mask[i] and non_tie_idx < n_sims:
                        selected_home.append(game_home[i])
                        selected_away.append(game_away[i])
                        non_tie_idx += 1
                    if non_tie_idx >= n_sims:
                        break
                
                home_scores_per_game.append(jnp.array(selected_home[:n_sims]))
                away_scores_per_game.append(jnp.array(selected_away[:n_sims]))
            else:
                # Not enough non-ties - take what we have and pad
                selected_home = game_home[non_tie_mask]
                selected_away = game_away[non_tie_mask]
                
                if len(selected_home) > 0:
                    # Pad with last value
                    last_home = selected_home[-1]
                    last_away = selected_away[-1]
                    padding_size = n_sims - len(selected_home)
                    padding_home = jnp.full((padding_size,), last_home, dtype=jnp.int32)
                    padding_away = jnp.full((padding_size,), last_away, dtype=jnp.int32)
                    selected_home = jnp.concatenate([selected_home, padding_home])
                    selected_away = jnp.concatenate([selected_away, padding_away])
                else:
                    # All ties - use zeros (shouldn't happen in basketball)
                    selected_home = jnp.zeros(n_sims, dtype=jnp.int32)
                    selected_away = jnp.zeros(n_sims, dtype=jnp.int32)
                
                home_scores_per_game.append(selected_home)
                away_scores_per_game.append(selected_away)
        
        all_home_scores_list.append(jnp.stack(home_scores_per_game))  # Shape: (n_games, n_sims)
        all_away_scores_list.append(jnp.stack(away_scores_per_game))  # Shape: (n_games, n_sims)
    
    # Stack all parameter draws: [n_param_draws, n_games, n_sims]
    sim_log.info("Stacking all simulation results...")
    all_home_scores = jnp.stack(all_home_scores_list)
    all_away_scores = jnp.stack(all_away_scores_list)
    
    # Build results DataFrame more efficiently
    sim_log.info(f"Building results DataFrame from {n_param_draws * n_games * n_sims:,} simulations...")
    
    # Convert to numpy for faster iteration
    home_scores_np = np.array(all_home_scores)
    away_scores_np = np.array(all_away_scores)
    
    # Pre-allocate arrays for better performance
    n_total = n_param_draws * n_games * n_sims
    results_dict = {
        'game_id': np.empty(n_total, dtype=object),
        'date': np.empty(n_total, dtype=object),
        'home_team': np.empty(n_total, dtype=object),
        'away_team': np.empty(n_total, dtype=object),
        'param_draw_idx': np.empty(n_total, dtype=np.int32),
        'sim_idx': np.empty(n_total, dtype=np.int32),
        'home_score': np.empty(n_total, dtype=np.int32),
        'away_score': np.empty(n_total, dtype=np.int32),
        'spread': np.empty(n_total, dtype=np.int32),
        'total': np.empty(n_total, dtype=np.int32),
    }
    
    idx = 0
    for param_draw_idx in range(n_param_draws):
        for game_idx, (_, game) in enumerate(games_df.iterrows()):
            game_id = game.get('gameID', game_idx)
            date = game.get('date', '')
            home_team = game['home_team']
            away_team = game['away_team']
            
            for sim_idx in range(n_sims):
                home_score = int(home_scores_np[param_draw_idx, game_idx, sim_idx])
                away_score = int(away_scores_np[param_draw_idx, game_idx, sim_idx])
                
                results_dict['game_id'][idx] = game_id
                results_dict['date'][idx] = date
                results_dict['home_team'][idx] = home_team
                results_dict['away_team'][idx] = away_team
                results_dict['param_draw_idx'][idx] = param_draw_idx
                results_dict['sim_idx'][idx] = sim_idx
                results_dict['home_score'][idx] = home_score
                results_dict['away_score'][idx] = away_score
                results_dict['spread'][idx] = home_score - away_score
                results_dict['total'][idx] = home_score + away_score
                idx += 1
            
            if game_idx % 50 == 0 and param_draw_idx == 0:
                sim_log.info(f"  Processed {game_idx}/{n_games} games for first param draw...")
    
    sim_log.info("Creating DataFrame from results...")
    return pd.DataFrame(results_dict)


def save_simulations(sport: str, model_name: str, monday_date: str, simulation_df: pd.DataFrame) -> Path:
    """Save simulations to parquet file."""
    sim_path = get_simulation_path(sport, model_name, monday_date)
    logging.info(f"Saving {len(simulation_df)} simulations to {sim_path}")
    simulation_df.to_parquet(sim_path, index=False)
    logging.info(f"Simulations saved")
    return sim_path


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
        "--model", type=str, required=True, choices=["Vanilla", "TeamVol"],
        help="Model to use: Vanilla or TeamVol"
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
        "--min-games", type=int, default=10,
        help="Minimum number of games per team to include in training (default: 10)"
    )
    parser.add_argument(
        "--n-sims", type=int, default=250,
        help="Number of non-tie simulations per parameter draw per game (default: 250)"
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
        "--force-refit", action="store_true",
        help="Force model refit even if cache exists"
    )
    
    args = parser.parse_args()
    
    # Setup main pipeline logger (for general pipeline progress)
    # This logs to a general location, but fitting and simulation have their own logs
    log_base = get_log_base_dir()
    pipeline_log_file = log_base / "ncaab" / "simulation" / f"pipeline_{args.model}_{get_timestamp()}.log"
    pipeline_log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logger(pipeline_log_file)
    
    # Setup fitting logger (will be used when model is actually fitted)
    fitting_logger = setup_fitting_logger("ncaab", args.model)
    
    # Setup simulation logger (for simulation-specific messages)
    simulation_logger = setup_simulation_logger("ncaab", args.model)
    
    logging.info("="*60)
    logging.info("Starting game simulation pipeline")
    logging.info("="*60)
    logging.info(f"Sport: {args.sport}, Model: {args.model}, Year: {args.year}")
    logging.info(f"Training data year: {args.training_year or args.year}")
    logging.info(f"Simulations per draw per game: {args.n_sims}")
    
    try:
        # Step 1: Prepare training data
        training_year = args.training_year or args.year
        logging.info(f"\nStep 1: Preparing training data from {training_year} season...")
        df_training, encoded, team_to_id, id_to_team = prepare_training_data(
            training_year, args.sport, min_games=args.min_games
        )
        
        # Step 2: Get Monday cutoff
        monday_date = get_monday_cutoff(df_training)
        logging.info(f"Monday cutoff date: {monday_date}")
        
        # Step 3: Fit or load model
        logging.info(f"\nStep 2: Fitting or loading model...")
        samples, model_metadata, team_to_id, id_to_team = fit_or_load_model(
            df_training,
            encoded,
            args.model,
            args.sport,
            monday_date,
            seed=0,
            num_chains=args.num_chains,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            force_refit=args.force_refit,
            fitting_logger=fitting_logger
        )
        
        # Step 4: Load schedule and filter for next week
        logging.info(f"\nStep 3: Loading schedule and filtering for next week...")
        schedule_df = load_schedule(args.sport, args.year)
        next_week_games = filter_next_week_games(schedule_df, monday_date)
        
        if next_week_games.empty:
            logging.warning("No games found in next week. Exiting.")
            return
        
        logging.info(f"Found {len(next_week_games)} games in next week")
        
        # Step 5: Map teams to model indices and filter unknown teams
        logging.info(f"\nStep 4: Mapping teams to model indices...")
        unknown_teams = []
        valid_games = []
        
        for _, row in next_week_games.iterrows():
            home_team = str(row['home_team'])
            away_team = str(row['away_team'])
            if home_team not in team_to_id:
                unknown_teams.append(home_team)
                continue
            if away_team not in team_to_id:
                unknown_teams.append(away_team)
                continue
            valid_games.append(row)
        
        if unknown_teams:
            logging.warning(f"Found {len(set(unknown_teams))} unknown teams: {list(set(unknown_teams))[:10]}")
            logging.warning("Filtering out games with unknown teams")
        
        if not valid_games:
            logging.warning("No games with known teams found. Exiting.")
            return
        
        games_with_indices = pd.DataFrame(valid_games).reset_index(drop=True)
        games_with_indices = map_schedule_teams_to_model(games_with_indices, team_to_id)
        
        logging.info(f"Simulating {len(games_with_indices)} games with known teams")
        
        logging.info(f"\nStep 5: Simulating games...")
        simulation_logger.info("="*60)
        simulation_logger.info("Starting game simulation")
        simulation_logger.info("="*60)
        simulation_logger.info(f"Model: {args.model}, Games: {len(games_with_indices)}, Sims per draw per game: {args.n_sims}")
        simulation_results = simulate_games(games_with_indices, samples, args.model, n_sims=args.n_sims, sim_logger=simulation_logger)
        
        logging.info(f"\nStep 6: Saving simulations...")
        sim_path = save_simulations(args.sport, args.model, monday_date, simulation_results)
        
        simulation_logger.info(f"Simulation complete. Saved {len(simulation_results)} simulations to {sim_path}")
        logging.info("Pipeline completed successfully!")
        logging.info(f"Saved {len(simulation_results)} simulations to {sim_path}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
