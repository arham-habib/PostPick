"""
Data loading utilities for the game visualizer.

Handles loading of simulation data and model parameters with caching.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import re

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
    Load and aggregate simulation data from parquet file.
    Only loads aggregated summary statistics, not raw simulation data.
    
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
        # Use pyarrow for more efficient reading
        import pyarrow.parquet as pq
        
        # Read only necessary columns to reduce memory
        table = pq.read_table(
            file_path,
            columns=['game_id', 'date', 'home_team', 'away_team', 
                    'home_score', 'away_score', 'spread', 'total']
        )
        df = table.to_pandas()
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Aggregate simulation data
        # Import here to avoid circular import
        from src.gui.game_summary import aggregate_simulation_data
        aggregated_df = aggregate_simulation_data(df)
        
        # Convert date back to datetime if needed
        if "date" in aggregated_df.columns:
            aggregated_df["date"] = pd.to_datetime(aggregated_df["date"])
        
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
    Load raw simulation data for a SINGLE game only.
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
        # Load only the specific game using pyarrow for efficient filtering
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        
        # Read parquet file
        table = pq.read_table(file_path)
        
        # Filter by game_id using pyarrow compute
        # Convert game_id to match the column type if needed
        game_id_col = table['game_id']
        if isinstance(game_id, (int, float)):
            # For numeric game_ids
            mask = pc.equal(game_id_col, game_id)
        else:
            # For string/object game_ids
            mask = pc.equal(game_id_col, str(game_id))
        
        filtered_table = table.filter(mask)
        
        # Convert to pandas
        df = filtered_table.to_pandas()
        
        if df.empty:
            return None
        
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
