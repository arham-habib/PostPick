#!/usr/bin/env python3
"""
Unified pipeline for NCAA basketball data processing.

This script orchestrates the complete pipeline:
1. Scrape game data
2. Fit/train the model
3. Run simulations (local or Modal)

Usage:
    python scripts/ncaab_pipeline.py --sport men --year 2025 --model Vanilla --monday-date 2026-01-19 [--skip-scrape] [--skip-fit] [--backend modal] [--n-sims 1000]
"""

# Set JAX to use CPU backend before any JAX imports
# This avoids Metal backend issues on Apple Silicon
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import jax.numpy as jnp

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ncaa.game_scraper import scrape_games
from src.ncaa.data_cleaning import build_team_indexer, encode_season, drop_teams_with_few_games
from src.ncaa.simulate_games import (
    load_schedule,
    filter_next_week_games,
    map_schedule_teams_to_model,
    simulate_games_to_parquet,
    get_model_path,
    get_simulation_path,
)
from src.ncaa.modal_simulate import run_modal_simulation
from src.utils.data_utils import get_game_data_path, get_model_data_path
from src.utils.logging_utils import setup_scraping_logger, setup_simulation_logger, setup_fitting_logger


def get_monday_date(date_str: Optional[str] = None) -> str:
    """Get Monday date string for the current week or specified date."""
    if date_str:
        date = pd.to_datetime(date_str)
        # Ensure we have a valid Timestamp
        if pd.isna(date):
            raise ValueError(f"Invalid date: {date_str}")
    else:
        date = pd.Timestamp.now()
    
    # Get the Monday of the week containing this date
    days_since_monday = date.weekday()  # type: ignore
    monday = date - pd.Timedelta(days=days_since_monday)
    return monday.strftime("%Y-%m-%d")  # type: ignore


def fit_model(
    sport: str,
    year: int,
    model_name: str,
    monday_date: str,
    min_games_per_team: int = 10,
    num_chains: int = 2,
    num_warmup: int = 100,
    num_samples: int = 300,
) -> None:
    """Fit the model on completed games and save to pickle file."""
    logging.info(f"Fitting {model_name} model for {sport} {year} (cutoff: {monday_date})")
    
    # Load game data
    game_path = get_game_data_path("ncaab", year, sport, "d1")
    if not game_path.exists():
        raise FileNotFoundError(f"Game data file not found: {game_path}. Run scraping first.")
    
    df = pd.read_csv(game_path)
    logging.info(f"Loaded {len(df)} total games from {game_path}")
    
    # Filter for completed games up to the cutoff Monday
    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = pd.to_datetime(monday_date)
    
    # Only use FINAL games before or on the cutoff Monday
    completed = df[
        (df["finalMessage"] == "FINAL") &
        (df["date"] <= cutoff_date) &
        df["home_score"].notna() &
        df["away_score"].notna()
    ].copy()
    
    if completed.empty:
        raise ValueError(f"No completed games found before {monday_date}")
    
    logging.info(f"Using {len(completed)} completed games for model fitting")
    
    # Drop teams with too few games
    if min_games_per_team > 0:
        completed = drop_teams_with_few_games(completed, n=min_games_per_team)
        logging.info(f"After filtering teams with <{min_games_per_team} games: {len(completed)} games")
    
    # Ensure we have a DataFrame (type check)
    assert isinstance(completed, pd.DataFrame), "Expected DataFrame after filtering"
    
    # Build team indexer and encode season
    team_to_id, id_to_team = build_team_indexer(completed)
    encoded = encode_season(completed, team_to_id)
    
    logging.info(f"Encoded season: {encoded.n_teams} teams, {len(encoded.home_idx)} games")
    
    # Import model module
    import importlib
    model_module = importlib.import_module(f"src.models.{model_name}")
    fit_hierarchal_model = getattr(model_module, "fit_hierarchal_model")
    
    # Fit model
    logging.info(f"Fitting model with {num_chains} chains, {num_warmup} warmup, {num_samples} samples...")
    mcmc, samples = fit_hierarchal_model(
        encoded,
        seed=42,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    
    logging.info(f"Model fitting complete. Posterior samples shape: {samples['alpha'].shape}")
    
    # Convert JAX arrays to numpy for serialization
    samples_np = {k: np.array(v) for k, v in samples.items()}
    
    # Prepare metadata
    metadata = {
        "sport": sport,
        "year": year,
        "model_name": model_name,
        "monday_date": monday_date,
        "n_teams": encoded.n_teams,
        "n_games": len(encoded.home_idx),
        "fit_date": datetime.now().isoformat(),
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
    }
    
    # Save model
    model_path = get_model_data_path("ncaab", sport, model_name, monday_date)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "samples": samples_np,
        "metadata": metadata,
        "team_to_id": team_to_id,
        "id_to_team": id_to_team,
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logging.info(f"Model saved to {model_path}")


def run_simulation_local(
    sport: str,
    year: int,
    model_name: str,
    monday_date: str,
    n_sims: int = 1000,
    draw_block_size: int = 50,
) -> None:
    """Run simulation locally."""
    from src.ncaa.simulate_games import load_model
    
    logging.info(f"Running local simulation for {model_name} {monday_date}")
    
    # Load model
    samples, metadata, team_to_id, id_to_team = load_model(sport, model_name, monday_date)
    
    # Load schedule and filter for next week
    schedule_df = load_schedule(sport, year)
    next_week_games = filter_next_week_games(schedule_df, monday_date)
    
    if next_week_games.empty:
        logging.warning("No games in next week; nothing to simulate.")
        return
    
    # Filter unknown teams
    valid_games = []
    for _, row in next_week_games.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        if home_team in team_to_id and away_team in team_to_id:
            valid_games.append(row)
    
    if not valid_games:
        logging.warning("No games with known teams; nothing to simulate.")
        return
    
    logging.info(f"Found {len(valid_games)} games with known teams")
    
    # Map teams to model indices
    games_with_indices = map_schedule_teams_to_model(
        next_week_games.loc[[r.name for r in valid_games]].reset_index(drop=True),
        team_to_id,
    )
    
    # Run simulation
    output_path = get_simulation_path(sport, model_name, monday_date)
    simulate_games_to_parquet(
        games_df=games_with_indices,
        samples=samples,
        model_name=model_name,
        n_sims=n_sims,
        draw_block_size=draw_block_size,
        output_path=output_path,
        rng_seed=42,
    )
    
    logging.info(f"Simulation complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for NCAA basketball: scrape, fit, simulate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (scrape, fit, simulate locally)
  python scripts/ncaab_pipeline.py --sport men --year 2025 --model Vanilla --monday-date 2026-01-19

  # Skip scraping (use existing data)
  python scripts/ncaab_pipeline.py --sport men --year 2025 --model Vanilla --monday-date 2026-01-19 --skip-scrape

  # Only fit model (skip scrape and simulation)
  python scripts/ncaab_pipeline.py --sport men --year 2025 --model Vanilla --monday-date 2026-01-19 --skip-scrape --skip-simulate

  # Use Modal for simulation
  python scripts/ncaab_pipeline.py --sport men --year 2025 --model Vanilla --monday-date 2026-01-19 --backend modal
        """
    )
    
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men",
                       help="Sport category: men or women (default: men)")
    parser.add_argument("--year", type=int, required=True,
                       help="Year to scrape data for")
    parser.add_argument("--model", type=str, choices=["Vanilla", "TeamVol"], default="Vanilla",
                       help="Model name (default: Vanilla)")
    parser.add_argument("--monday-date", type=str, required=True,
                       help="Monday date string (YYYY-MM-DD) for model cutoff and simulation week")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1",
                       help="NCAA division (default: d1)")
    
    # Pipeline control
    parser.add_argument("--skip-scrape", action="store_true",
                       help="Skip scraping step (use existing game data)")
    parser.add_argument("--skip-fit", action="store_true",
                       help="Skip model fitting step (use existing model)")
    parser.add_argument("--skip-simulate", action="store_true",
                       help="Skip simulation step")
    
    # Simulation options
    parser.add_argument("--backend", type=str, choices=["local", "modal"], default="local",
                       help="Simulation backend: local or modal (default: local)")
    parser.add_argument("--n-sims", type=int, default=1000,
                       help="Number of simulations per posterior draw (default: 1000)")
    parser.add_argument("--draw-block-size", type=int, default=50,
                       help="Number of posterior draws to process in each block (default: 50)")
    
    # Model fitting options
    parser.add_argument("--min-games-per-team", type=int, default=10,
                       help="Minimum games per team to include in model (default: 10)")
    parser.add_argument("--num-chains", type=int, default=2,
                       help="Number of MCMC chains (default: 2)")
    parser.add_argument("--num-warmup", type=int, default=100,
                       help="Number of warmup samples per chain (default: 100)")
    parser.add_argument("--num-samples", type=int, default=300,
                       help="Number of posterior samples per chain (default: 300)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_scraping_logger("ncaab", "game")
    setup_fitting_logger("ncaab", args.model)
    setup_simulation_logger("ncaab", args.model)
    
    logging.info("=" * 80)
    logging.info("NCAA Basketball Pipeline")
    logging.info("=" * 80)
    logging.info(f"Sport: {args.sport}, Year: {args.year}, Model: {args.model}")
    logging.info(f"Monday Date: {args.monday_date}")
    logging.info(f"Backend: {args.backend}")
    logging.info("=" * 80)
    
    # Step 1: Scrape game data
    if not args.skip_scrape:
        logging.info("\n[STEP 1] Scraping game data...")
        scrape_games(args.sport, args.division, args.year)
        logging.info("✓ Scraping complete")
    else:
        logging.info("\n[STEP 1] Skipping scraping (using existing data)")
    
    # Step 2: Fit model
    if not args.skip_fit:
        logging.info("\n[STEP 2] Fitting model...")
        fit_model(
            sport=args.sport,
            year=args.year,
            model_name=args.model,
            monday_date=args.monday_date,
            min_games_per_team=args.min_games_per_team,
            num_chains=args.num_chains,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
        )
        logging.info("✓ Model fitting complete")
    else:
        logging.info("\n[STEP 2] Skipping model fitting (using existing model)")
        # Verify model exists
        model_path = get_model_data_path("ncaab", args.sport, args.model, args.monday_date)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Cannot skip fitting without existing model.")
    
    # Step 3: Run simulation
    if not args.skip_simulate:
        logging.info("\n[STEP 3] Running simulation...")
        if args.backend == "modal":
            run_modal_simulation(
                sport=args.sport,
                model_name=args.model,
                year=args.year,
                monday_date=args.monday_date,
                n_sims=args.n_sims,
                draw_block_size=args.draw_block_size,
            )
        else:
            run_simulation_local(
                sport=args.sport,
                year=args.year,
                model_name=args.model,
                monday_date=args.monday_date,
                n_sims=args.n_sims,
                draw_block_size=args.draw_block_size,
            )
        logging.info("✓ Simulation complete")
    else:
        logging.info("\n[STEP 3] Skipping simulation")
    
    logging.info("\n" + "=" * 80)
    logging.info("Pipeline complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
