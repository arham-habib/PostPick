"""
Game Simulation Visualizer

A Streamlit-based web application to visualize predicted game results
with spread, total, and moneyline metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from typing import List, Tuple, Optional, Dict

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
SIMULATIONS_DIR = DATA_DIR / "ncaab" / "simulations"

# Page configuration
st.set_page_config(
    page_title="Game Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)


def aggregate_simulation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw simulation data to compute statistics per game.
    
    Args:
        df: DataFrame with columns: game_id, date, home_team, away_team, 
            param_draw_idx, sim_idx, home_score, away_score, spread, total
    
    Returns:
        DataFrame with aggregated statistics per game
    """
    if df.empty:
        return df
    
    # Group by game - use game_id as primary key, include other fields for reference
    # If game_id is not unique, fall back to grouping by all identifying fields
    group_cols = ['game_id', 'home_team', 'away_team']
    if 'date' in df.columns:
        group_cols.append('date')
    
    grouped = df.groupby(group_cols)
    
    results = []
    for group_key, group in grouped:
        # Unpack group key based on number of columns
        if len(group_cols) == 4:
            game_id, home_team, away_team, date = group_key
        else:
            game_id, home_team, away_team = group_key
            date = group['date'].iloc[0] if 'date' in group.columns else None
        
        spreads = group['spread'].values
        totals = group['total'].values
        home_scores = group['home_score'].values
        away_scores = group['away_score'].values
        
        # Compute spread percentiles
        spread_p5 = float(np.percentile(spreads, 5))
        spread_p50 = float(np.percentile(spreads, 50))
        spread_p95 = float(np.percentile(spreads, 95))
        
        # Compute total percentiles
        total_p5 = float(np.percentile(totals, 5))
        total_p50 = float(np.percentile(totals, 50))
        total_p95 = float(np.percentile(totals, 95))
        
        # Compute moneyline probabilities
        home_wins = (home_scores > away_scores).sum()
        away_wins = (away_scores > home_scores).sum()
        total_sims = len(group)
        
        moneyline_home_win = float(home_wins / total_sims) if total_sims > 0 else 0.0
        moneyline_away_win = float(away_wins / total_sims) if total_sims > 0 else 0.0
        
        result = {
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'spread_p5': spread_p5,
            'spread_p50': spread_p50,
            'spread_p95': spread_p95,
            'total_p5': total_p5,
            'total_p50': total_p50,
            'total_p95': total_p95,
            'moneyline_home_win': moneyline_home_win,
            'moneyline_away_win': moneyline_away_win,
        }
        
        if date is not None:
            result['date'] = date
        
        results.append(result)
    
    return pd.DataFrame(results)


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


def load_simulation_data(model_name: str, monday_date: str) -> Optional[pd.DataFrame]:
    """
    Load and aggregate simulation data from parquet file.
    
    Args:
        model_name: Name of the model (e.g., "Vanilla", "TeamVol")
        monday_date: Monday date string (e.g., "2026-01-15")
    
    Returns:
        DataFrame with aggregated simulation results or None if file not found
    """
    file_path = SIMULATIONS_DIR / f"{model_name}_{monday_date}.parquet"
    
    if not file_path.exists():
        return None
    
    try:
        # Load raw simulation data
        df = pd.read_parquet(file_path)
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Aggregate simulation data
        aggregated_df = aggregate_simulation_data(df)
        
        # Convert date back to datetime if needed
        if "date" in aggregated_df.columns:
            aggregated_df["date"] = pd.to_datetime(aggregated_df["date"])
        
        return aggregated_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for display with key columns."""
    if df.empty:
        return df
    
    display_df = df.copy()
    
    # Format date
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    
    # Format spread
    if "spread_p50" in display_df.columns:
        spread_sign = display_df["spread_p50"].apply(lambda x: "+" if x >= 0 else "")
        display_df["spread"] = spread_sign + display_df["spread_p50"].round(1).astype(str)
        if "spread_p5" in display_df.columns and "spread_p95" in display_df.columns:
            display_df["spread_range"] = (
                display_df["spread_p5"].round(0).astype(int).astype(str) + " to " +
                display_df["spread_p95"].round(0).astype(int).astype(str)
            )
    
    # Format total
    if "total_p50" in display_df.columns:
        display_df["total"] = display_df["total_p50"].round(1).astype(str)
        if "total_p5" in display_df.columns and "total_p95" in display_df.columns:
            display_df["total_range"] = (
                display_df["total_p5"].round(0).astype(int).astype(str) + " to " +
                display_df["total_p95"].round(0).astype(int).astype(str)
            )
    
    # Format moneyline percentages
    if "moneyline_home_win" in display_df.columns:
        display_df["home_win_pct"] = (display_df["moneyline_home_win"] * 100).round(1).astype(str) + "%"
    if "moneyline_away_win" in display_df.columns:
        display_df["away_win_pct"] = (display_df["moneyline_away_win"] * 100).round(1).astype(str) + "%"
    
    # Select display columns
    display_cols = []
    col_order = ["date", "away_team", "home_team", "spread", "spread_range", 
                 "total", "total_range", "home_win_pct", "away_win_pct"]
    
    for col in col_order:
        if col in display_df.columns:
            display_cols.append(col)
    
    # Add any remaining columns not in the ordered list
    for col in display_df.columns:
        if col not in display_cols and col not in ["spread_p50", "spread_p5", "spread_p95",
                                                     "total_p50", "total_p5", "total_p95",
                                                     "moneyline_home_win", "moneyline_away_win"]:
            display_cols.append(col)
    
    # Ensure we return a DataFrame (not a Series if only one column)
    result = display_df[display_cols]
    if isinstance(result, pd.Series):
        return result.to_frame()
    return result  # type: ignore


def main():
    """Main Streamlit application."""
    # Title
    st.title("Game Prediction Visualizer")
    
    # Get available combinations
    combinations = get_available_combinations()
    
    if not combinations:
        st.error("No simulation files found in the simulations directory.")
        st.info(f"Expected location: {SIMULATIONS_DIR}")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Extract unique values
        models = sorted(set(c["model_name"] for c in combinations))
        monday_dates = sorted(set(c["monday_date"] for c in combinations), reverse=True)
        
        # Model selector
        selected_model = st.selectbox(
            "Model",
            options=models,
            index=0
        )
        
        # Monday date selector
        selected_monday_date = st.selectbox(
            "Monday Date",
            options=monday_dates,
            index=0
        )
        
        st.markdown("---")
        
        # Load data
        df = load_simulation_data(selected_model, selected_monday_date)
        
        if df is None or df.empty:
            st.error(f"No data found for {selected_model} {selected_monday_date}")
            return
        
        # Extract year from dates for display
        if "date" in df.columns and not df["date"].isna().all():
            df_with_dates = df[df["date"].notna()].copy()
            if not df_with_dates.empty:
                years = sorted(df_with_dates["date"].dt.year.unique(), reverse=True)
                if years:
                    selected_year = years[0]  # Use most recent year
                else:
                    selected_year = None
            else:
                selected_year = None
        else:
            selected_year = None
        
        # Date range filter
        has_dates = "date" in df.columns and bool(df["date"].notna().any())
        if has_dates:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
                df = df.loc[mask]  # type: ignore
        
        st.info(f"Total games: {len(df)}")
    
    # Main content area
    if df is None or df.empty:
        st.warning("No games match the selected filters.")
        return
    
    # Ensure df is a DataFrame (type guard)
    assert isinstance(df, pd.DataFrame), "df must be a DataFrame"
    
    # Sort by date (default)
    if "date" in df.columns:
        df = df.sort_values(by="date", ascending=True)
    
    # Search and filter controls in main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        team_search = st.text_input(
            "Search Team",
            placeholder="Enter team name...",
            key="team_search"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Date", "Spread (Median)", "Total (Median)", "Home Win %", "Away Win %"],
            index=0,
            key="sort_by"
        )
    
    # Apply team search filter
    if team_search and isinstance(df, pd.DataFrame):
        team_search_lower = team_search.lower()
        mask = (
            df["home_team"].astype(str).str.lower().str.contains(team_search_lower, na=False) |
            df["away_team"].astype(str).str.lower().str.contains(team_search_lower, na=False)
        )
        df = df.loc[mask]  # type: ignore
    
    # Apply sorting
    if isinstance(df, pd.DataFrame):
        if sort_by == "Date" and "date" in df.columns:
            df = df.sort_values(by="date", ascending=True)
        elif sort_by == "Spread (Median)" and "spread_p50" in df.columns:
            df = df.sort_values(by="spread_p50", ascending=False)
        elif sort_by == "Total (Median)" and "total_p50" in df.columns:
            df = df.sort_values(by="total_p50", ascending=False)
        elif sort_by == "Home Win %" and "moneyline_home_win" in df.columns:
            df = df.sort_values(by="moneyline_home_win", ascending=False)
        elif sort_by == "Away Win %" and "moneyline_away_win" in df.columns:
            df = df.sort_values(by="moneyline_away_win", ascending=False)
    
    # Display header
    year_str = f" {selected_year}" if selected_year else ""
    st.header(f"{selected_model} Model - {selected_monday_date}{year_str}")
    
    # Format and display data as table
    if isinstance(df, pd.DataFrame) and not df.empty:
        display_df = format_display_df(df)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} games")


if __name__ == "__main__":
    main()
