"""
Game Simulation Module

This module implements simulation of upcoming NCAA basketball games using
a fitted hierarchical Bayesian model. Supports multiple model flavors with
sport-specific storage organization and vectorized simulation.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import importlib
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.data_utils import (
    get_model_data_path,
    get_simulation_data_path,
    get_game_data_path
)


def get_model_path(sport: str, model_name: str, monday_date: str) -> Path:
    """Get path to model file."""
    return get_model_data_path("ncaab", sport, model_name, monday_date)


def get_simulation_path(sport: str, model_name: str, monday_date: str) -> Path:
    """Get path to simulation file."""
    return get_simulation_data_path("ncaab", sport, model_name, monday_date)


def load_schedule(sport: str, year: int) -> pd.DataFrame:
    """Load scheduled games (unfinished games) from game data (assumes d1 division)."""
    file_path = get_game_data_path("ncaab", year, sport, "d1")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Game data file not found: {file_path}")
    
    logging.info(f"Loading unfinished games from {file_path}")
    df = pd.read_csv(file_path)
    
    # Filter for unfinished games (not FINAL)
    # Games are unfinished if finalMessage is not "FINAL" or if scores are missing
    if "finalMessage" in df.columns:
        unfinished_mask = df["finalMessage"] != "FINAL"
        unfinished = df[unfinished_mask].copy()
    else:
        # Fallback: check if scores are missing
        unfinished_mask = (df["home_score"].isna()) | (df["away_score"].isna())
        unfinished = df[unfinished_mask].copy()
    
    # Ensure we return a DataFrame
    if not isinstance(unfinished, pd.DataFrame):
        unfinished = pd.DataFrame(unfinished)
    
    logging.info(f"Loaded {len(unfinished)} unfinished games out of {len(df)} total games")
    return unfinished


def filter_next_week_games(schedule_df: pd.DataFrame, cutoff_monday: str) -> pd.DataFrame:
    """Filter for games in the week starting the day after cutoff Monday.
    
    If cutoff is Monday 2026-01-19, looks for games from 2026-01-20 to 2026-01-25 (Tuesday through Sunday).
    """
    if schedule_df.empty:
        return schedule_df
    
    schedule_df = schedule_df.copy()
    schedule_df["date"] = pd.to_datetime(schedule_df["date"])
    
    cutoff_date = pd.to_datetime(cutoff_monday)
    # Start from the day AFTER the cutoff Monday (Tuesday)
    week_start = cutoff_date + pd.Timedelta(days=1)
    # End 6 days later (Sunday, not including the following Monday)
    week_end = cutoff_date + pd.Timedelta(days=6)  # Sunday (inclusive)
    
    # Filter: games from Tuesday through Sunday
    mask = (schedule_df["date"] >= week_start) & (schedule_df["date"] <= week_end)  # type: ignore
    next_week = schedule_df[mask].copy()  # type: ignore
    
    logging.info(f"Found {len(next_week)} games in next week ({week_start.date()} to {week_end.date()}) out of {len(schedule_df)} total scheduled")
    return next_week  # type: ignore


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


