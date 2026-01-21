"""
Game Simulation Module

This module implements simulation of upcoming NCAA basketball games using
a fitted hierarchical Bayesian model. Supports multiple model flavors with
sport-specific storage organization and vectorized simulation.
"""
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
import pyarrow as pa
import pyarrow.parquet as pq

from src.ncaa.data_cleaning import (
    build_team_indexer,
    encode_season,
    drop_teams_with_few_games
)
from src.utils.logging_utils import (
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
    """Filter for games in the week starting the day after cutoff Monday.
    
    If cutoff is Monday 2026-01-19, looks for games from 2026-01-20 to 2026-01-26.
    """
    if schedule_df.empty:
        return schedule_df
    
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    
    cutoff_date = pd.to_datetime(cutoff_monday)
    # Start from the day AFTER the cutoff Monday (Tuesday)
    week_start = cutoff_date + pd.Timedelta(days=1)
    # End 6 days later (Sunday, or Monday if we want 7 days total)
    week_end = cutoff_date + pd.Timedelta(days=7)  # Next Monday (inclusive)
    
    # Filter: games from Tuesday through next Monday
    mask = (schedule_df["date"] >= week_start) & (schedule_df["date"] <= week_end)  # type: ignore
    next_week = schedule_df[mask].copy()  # type: ignore
    
    logging.info(f"Found {len(next_week)} games in next week ({week_start.date()} to {week_end.date()}) out of {len(schedule_df)} total scheduled")
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


def get_latest_model_monday_date(sport: str, model_name: str) -> str:
    """Find the latest cached model file for a model and return its monday_date string."""
    # data/ncaab/models/{Model}_{YYYY-MM-DD}.pkl
    model_dir = get_model_data_path("ncaab", sport, model_name, "DUMMY").parent
    pattern = f"{model_name}_*.pkl"
    candidates = sorted(model_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No cached model files found in {model_dir} matching {pattern}")
    # Sort lexicographically; YYYY-MM-DD sorts correctly
    latest = candidates[-1].stem  # e.g. Vanilla_2026-01-19
    return latest.split("_", 1)[1]


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


def simulate_games_to_parquet(
    *,
    games_df: pd.DataFrame,
    samples: Dict[str, jnp.ndarray],
    model_name: str,
    n_sims: int,
    draw_block_size: int,
    output_path: Path,
    rng_seed: int = 42,
    sim_logger: Optional[logging.Logger] = None,
) -> Path:
    """Simulate games and stream exploded results to parquet (same schema as before)."""
    sim_log = sim_logger if sim_logger else logging
    if games_df.empty:
        sim_log.warning("No games to simulate")
        # Write empty parquet with schema? For now return without writing.
        return output_path

    # Check that all teams are known (no -1 indices)
    if (games_df["home_idx"] == -1).any() or (games_df["away_idx"] == -1).any():
        raise ValueError("Games with unknown teams (idx == -1) should be filtered out before simulation")

    model_module = importlib.import_module(f"src.models.{model_name}")
    simulate_scores_block = getattr(model_module, "simulate_scores_block")

    n_param_draws = int(samples["alpha"].shape[0])
    n_games = int(len(games_df))

    sim_log.info(
        f"Simulating {n_games} games with {n_param_draws} posterior draws, {n_sims} sims/draw using {model_name} (streaming parquet)"
    )

    # Indices to device
    home_indices = jnp.array(games_df["home_idx"].values, dtype=jnp.int32)
    away_indices = jnp.array(games_df["away_idx"].values, dtype=jnp.int32)

    # Static metadata columns (host)
    if "gameID" in games_df.columns:
        game_ids = games_df["gameID"].astype(object).to_numpy()
    else:
        game_ids = np.arange(n_games, dtype=np.int64).astype(object)

    if "date" in games_df.columns:
        dates = games_df["date"].astype(object).to_numpy()
    else:
        dates = pd.Series([pd.NaT] * n_games).astype(object).to_numpy()
    home_teams = games_df["home_team"].astype(object).to_numpy()
    away_teams = games_df["away_team"].astype(object).to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None

    rng_key = random.PRNGKey(rng_seed)
    total_rows = n_param_draws * n_sims * n_games
    sim_log.info(f"Writing ~{total_rows:,} rows to {output_path}")

    try:
        for draw_start in range(0, n_param_draws, draw_block_size):
            draw_end = min(n_param_draws, draw_start + draw_block_size)
            Db = draw_end - draw_start

            sim_log.info(f"Simulating draw block {draw_start}:{draw_end} ...")
            block_home, block_away = simulate_scores_block(
                home_indices,
                away_indices,
                samples,
                draw_start=draw_start,
                draw_end=draw_end,
                n_sims=n_sims,
                rng_key=rng_key,
            )
            # Transfer to host: [Db, S, G]
            home_np = np.array(block_home)
            away_np = np.array(block_away)

            # Flatten order: draw -> sim -> game
            home_flat = home_np.reshape(Db * n_sims * n_games).astype(np.int32, copy=False)
            away_flat = away_np.reshape(Db * n_sims * n_games).astype(np.int32, copy=False)
            spread_flat = (home_flat - away_flat).astype(np.int32, copy=False)
            total_flat = (home_flat + away_flat).astype(np.int32, copy=False)

            # Metadata arrays
            # Repeat games for each sim, and repeat that for each draw in block
            game_id_flat = np.tile(np.tile(game_ids, n_sims), Db)
            date_flat = np.tile(np.tile(dates, n_sims), Db)
            home_team_flat = np.tile(np.tile(home_teams, n_sims), Db)
            away_team_flat = np.tile(np.tile(away_teams, n_sims), Db)

            # Indices
            draw_idx_flat = np.repeat(np.arange(draw_start, draw_end, dtype=np.int32), n_sims * n_games)
            sim_idx_flat = np.tile(np.repeat(np.arange(n_sims, dtype=np.int32), n_games), Db)

            table = pa.table(
                {
                    "game_id": game_id_flat,
                    "date": date_flat,
                    "home_team": home_team_flat,
                    "away_team": away_team_flat,
                    "param_draw_idx": draw_idx_flat,
                    "sim_idx": sim_idx_flat,
                    "home_score": home_flat,
                    "away_score": away_flat,
                    "spread": spread_flat,
                    "total": total_flat,
                }
            )

            # Initialize writer from first produced schema to match inferred dtypes
            if writer is None:
                writer = pq.ParquetWriter(output_path.as_posix(), schema=table.schema, compression="snappy")
            writer.write_table(table)

        sim_log.info("Parquet write complete.")
        return output_path
    finally:
        if writer is not None:
            writer.close()


def save_simulations(sport: str, model_name: str, monday_date: str, simulation_df: pd.DataFrame) -> Path:
    """Save simulations to parquet file."""
    sim_path = get_simulation_path(sport, model_name, monday_date)
    logging.info(f"Saving {len(simulation_df)} simulations to {sim_path}")
    simulation_df.to_parquet(sim_path, index=False)
    logging.info(f"Simulations saved")
    return sim_path


def main():
    """Main function to orchestrate the simulation pipeline.

    Note: This entrypoint is intended to *simulate from cached posterior samples*.
    Fitting/NumPyro MCMC is intentionally not part of this pipeline (GPU runs on Modal).
    """
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
        "--n-sims", type=int, default=1000,
        help="Number of simulations per posterior draw per game (default: 1000)"
    )
    parser.add_argument(
        "--draw-block-size", type=int, default=25,
        help="Posterior draw block size for chunked simulation/writing (default: 25)"
    )
    parser.add_argument(
        "--monday-date", type=str, default=None,
        help="Optional cached model monday date (YYYY-MM-DD). Defaults to latest available."
    )
    parser.add_argument(
        "--backend", type=str, choices=["modal", "local"], default="modal",
        help="Execution backend: modal (GPU) or local (CPU/Metal) (default: modal)"
    )
    
    args = parser.parse_args()
    
    # Setup main pipeline logger (for general pipeline progress)
    # This logs to a general location, but fitting and simulation have their own logs
    log_base = get_log_base_dir()
    pipeline_log_file = log_base / "ncaab" / "simulation" / f"pipeline_{args.model}_{get_timestamp()}.log"
    pipeline_log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logger(pipeline_log_file)
    
    # Setup simulation logger (for simulation-specific messages)
    simulation_logger = setup_simulation_logger("ncaab", args.model)
    
    logging.info("="*60)
    logging.info("Starting game simulation pipeline")
    logging.info("="*60)
    logging.info(f"Sport: {args.sport}, Model: {args.model}, Year: {args.year}")
    logging.info(f"Simulations per draw per game: {args.n_sims}")
    logging.info(f"Backend: {args.backend}")
    
    try:
        # Step 1: Select cached model file
        monday_date = args.monday_date or get_latest_model_monday_date(args.sport, args.model)
        logging.info(f"Using cached model monday date: {monday_date}")

        if args.backend == "modal":
            # Defer execution to Modal GPU runtime (see src/ncaa/modal_simulate.py)
            from src.ncaa.modal_simulate import run_modal_simulation  # type: ignore

            run_modal_simulation(
                sport=args.sport,
                model_name=args.model,
                year=args.year,
                monday_date=monday_date,
                n_sims=args.n_sims,
                draw_block_size=args.draw_block_size,
            )
            logging.info("Modal job submitted/completed (see Modal logs).")
            return

        # Local backend: load cached model samples (no fitting)
        logging.info("\nStep 2: Loading cached model...")
        samples, model_metadata, team_to_id, id_to_team = load_model(args.sport, args.model, monday_date)
        
        # Step 3: Load schedule and filter for next week relative to monday_date
        logging.info("\nStep 3: Loading schedule and filtering for next week...")
        schedule_df = load_schedule(args.sport, args.year)
        next_week_games = filter_next_week_games(schedule_df, monday_date)
        
        if next_week_games.empty:
            logging.warning("No games found in next week. Exiting.")
            return
        
        logging.info(f"Found {len(next_week_games)} games in next week")
        
        # Step 4: Map teams to model indices and filter unknown teams
        logging.info("\nStep 4: Mapping teams to model indices...")
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
        
        logging.info("\nStep 5: Simulating games (streaming parquet)...")
        simulation_logger.info("="*60)
        simulation_logger.info("Starting game simulation")
        simulation_logger.info("="*60)
        simulation_logger.info(
            f"Model: {args.model}, Games: {len(games_with_indices)}, Sims per draw per game: {args.n_sims}, Draw block size: {args.draw_block_size}"
        )

        sim_path = get_simulation_path(args.sport, args.model, monday_date)
        simulate_games_to_parquet(
            games_df=games_with_indices,
            samples=samples,
            model_name=args.model,
            n_sims=args.n_sims,
            draw_block_size=args.draw_block_size,
            output_path=sim_path,
            rng_seed=42,
            sim_logger=simulation_logger,
        )

        simulation_logger.info(f"Simulation complete. Saved parquet to {sim_path}")
        logging.info("Pipeline completed successfully!")
        logging.info(f"Saved simulations to {sim_path}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
