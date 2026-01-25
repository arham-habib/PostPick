"""
Data loading utilities for the game visualizer.

Handles loading of simulation data and model parameters with caching.
Uses Polars for efficient lazy aggregation of large simulation files.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import re
import polars as pl

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Set JAX to use CPU to avoid backend issues
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from src.ncaa.simulate_games import load_model

# Get project root
DATA_DIR = SCRIPT_DIR / "data"
SIMULATIONS_DIR = DATA_DIR / "ncaab" / "simulations"
MODELS_DIR = DATA_DIR / "ncaab" / "models"

# Default gender for ncaab (can be made configurable)
DEFAULT_GENDER = "men"


def get_available_combinations() -> List[Dict[str, str]]:
    """
    Scan simulations directory for available simulation files.
    
    Returns:
        List of dicts with 'model_name', 'monday_date', 'file_path' keys
    """
    combinations = []
    pattern = re.compile(r"([A-Za-z]+)_(\d{4}-\d{2}-\d{2})\.parquet")
    
    if not SIMULATIONS_DIR.exists():
        return combinations
    
    for file_path in SIMULATIONS_DIR.glob("*.parquet"):
        match = pattern.match(file_path.name)
        if match:
            model_name, monday_date = match.groups()
            combinations.append({
                "model_name": model_name,
                "monday_date": monday_date,
                "file_path": file_path
            })
    
    return sorted(combinations, key=lambda x: (x["monday_date"], x["model_name"]), reverse=True)


def load_aggregated_simulation_data(model_name: str, monday_date: str) -> Optional[pd.DataFrame]:
    """
    Load and aggregate simulation data from parquet file using Polars lazy evaluation.
    Only loads aggregated summary statistics, not raw simulation data.
    Uses lazy evaluation to minimize memory usage and improve performance.
    
    Args:
        model_name: Name of the model (e.g., "Vanilla", "TeamVol")
        monday_date: Monday date string (e.g., "2026-01-15")
    
    Returns:
        DataFrame with aggregated simulation results or None if file not found
    """
    # Check cache first
    import streamlit as st
    cache_key = f"agg_sim_{model_name}_{monday_date}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        if cached is not None:
            return cached.copy()
    
    file_path = SIMULATIONS_DIR / f"{model_name}_{monday_date}.parquet"
    
    if not file_path.exists():
        return None
    
    try:
        # Use Polars lazy evaluation for efficient aggregation
        # This only reads and processes data when collect() is called
        # Select only needed columns to reduce memory usage
        lazy_df = (
            pl.scan_parquet(str(file_path))
            .select(['game_id', 'date', 'home_team', 'away_team', 
                    'home_score', 'away_score', 'spread', 'total'])
        )
        
        # Build aggregation query lazily - no data loaded yet
        # Check schema to see if date column exists
        group_cols = ['game_id', 'home_team', 'away_team']
        if 'date' in lazy_df.collect_schema().keys():
            group_cols.append('date')
        
        # Perform all aggregations in a single lazy query
        result = (
            lazy_df
            .filter(pl.col('home_score') != pl.col('away_score'))  # Filter out ties before aggregation
            .with_columns([
                # Compute win indicators
                (pl.col('home_score') > pl.col('away_score')).cast(pl.Int64).alias('home_wins'),
                (pl.col('away_score') > pl.col('home_score')).cast(pl.Int64).alias('away_wins'),
            ])
            .group_by(group_cols)
            .agg([
                # Spread statistics
                pl.col('spread').mean().alias('spread_mean'),
                pl.col('spread').std().alias('spread_std'),
                pl.col('spread').quantile(0.05).alias('spread_p5'),
                pl.col('spread').median().alias('spread_p50'),
                pl.col('spread').quantile(0.95).alias('spread_p95'),
                # Total statistics
                pl.col('total').mean().alias('total_mean'),
                pl.col('total').std().alias('total_std'),
                pl.col('total').quantile(0.05).alias('total_p5'),
                pl.col('total').median().alias('total_p50'),
                pl.col('total').quantile(0.95).alias('total_p95'),
                # Score statistics
                pl.col('home_score').mean().alias('expected_home_score'),
                pl.col('away_score').mean().alias('expected_away_score'),
                # Win statistics
                pl.count().alias('total_sims'),
                pl.col('home_wins').sum().alias('home_wins'),
                pl.col('away_wins').sum().alias('away_wins'),
            ])
            .with_columns([
                # Calculate win probabilities
                (pl.col('home_wins') / pl.col('total_sims')).fill_null(0).alias('moneyline_home_win'),
                (pl.col('away_wins') / pl.col('total_sims')).fill_null(0).alias('moneyline_away_win'),
            ])
            .drop(['home_wins', 'away_wins', 'total_sims'])
        )
        
        # Execute the lazy query and convert to pandas
        result_df = result.collect()
        aggregated_df = result_df.to_pandas()
        
        # Convert date column to datetime if present
        if 'date' in aggregated_df.columns:
            aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
        
        # Cache the result
        st.session_state[cache_key] = aggregated_df
        
        return aggregated_df
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading data: {e}")
        return None


def load_raw_simulation_data_for_game(
    model_name: str, 
    monday_date: str, 
    game_id: Any
) -> Optional[pd.DataFrame]:
    """
    Load raw simulation data for a SINGLE game only using Polars lazy filtering.
    This is called lazily when a game detail view is expanded.
    
    Args:
        model_name: Name of the model (e.g., "Vanilla", "TeamVol")
        monday_date: Monday date string (e.g., "2026-01-15")
        game_id: Game identifier to filter by
    
    Returns:
        DataFrame with raw simulation results for the specified game or None if not found
    """
    file_path = SIMULATIONS_DIR / f"{model_name}_{monday_date}.parquet"
    
    if not file_path.exists():
        return None
    
    try:
        # Use Polars lazy evaluation to filter by game_id efficiently
        # This only reads the matching rows from the parquet file
        lazy_df = pl.scan_parquet(str(file_path))
        
        # Filter by game_id lazily - Polars will push this filter down to the parquet reader
        if isinstance(game_id, (int, float)):
            filtered_df = lazy_df.filter(pl.col('game_id') == game_id)
        else:
            filtered_df = lazy_df.filter(pl.col('game_id') == str(game_id))
        
        # Execute the query and convert to pandas
        result_df = filtered_df.collect()
        
        if result_df.is_empty():
            return None
        
        df = result_df.to_pandas()
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        return df
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading raw simulation data for game: {e}")
        return None


def load_model_parameters(
    model_name: str, 
    monday_date: str, 
    gender: str = DEFAULT_GENDER
) -> Optional[Tuple[Dict[str, Any], Dict[str, int], List[str]]]:
    """
    Load model parameters from pickle file with caching.
    Only loads when needed (when viewing game details).
    
    Args:
        model_name: Name of the model (e.g., "Vanilla", "TeamVol")
        monday_date: Monday date string (e.g., "2026-01-15")
        gender: Gender specification (default: "men")
    
    Returns:
        Tuple of (samples, team_to_id, id_to_team) or None if file not found
    """
    # Check cache first
    import streamlit as st
    cache_key = f"model_{model_name}_{monday_date}_{gender}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        if cached is not None:
            return cached
    
    try:
        # Convert JAX arrays to numpy for easier handling in Streamlit
        samples_jax, metadata, team_to_id, id_to_team = load_model(gender, model_name, monday_date)
        
        # Convert JAX arrays to numpy arrays for easier manipulation
        samples = {k: np.array(v) for k, v in samples_jax.items()}
        
        result = (samples, team_to_id, id_to_team)
        
        # Cache the result
        st.session_state[cache_key] = result
        
        return result
    except FileNotFoundError:
        import streamlit as st
        st.warning(f"Model file not found for {model_name} {monday_date}")
        return None
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading model: {e}")
        return None
