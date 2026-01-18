"""
Game Simulation Visualizer

A Streamlit-based web application to visualize predicted game results
with spread, total, and moneyline metrics.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from typing import List, Tuple, Optional, Dict

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
SIMULATIONS_DIR = DATA_DIR / "simulations"

# Page configuration
st.set_page_config(
    page_title="Game Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_available_combinations() -> List[Dict[str, str]]:
    """
    Scan simulations directory for available simulation files.
    
    Returns:
        List of dicts with 'year', 'sport', 'division' keys
    """
    combinations = []
    pattern = re.compile(r"game_simulations_(\d{4})_(men|women)_(d[123])\.csv")
    
    if not SIMULATIONS_DIR.exists():
        return combinations
    
    for file_path in SIMULATIONS_DIR.glob("game_simulations_*.csv"):
        match = pattern.match(file_path.name)
        if match:
            year, sport, division = match.groups()
            combinations.append({
                "year": year,
                "sport": sport,
                "division": division,
                "file_path": file_path
            })
    
    return sorted(combinations, key=lambda x: (x["year"], x["sport"], x["division"]))


def load_simulation_data(sport: str, division: str, year: str) -> Optional[pd.DataFrame]:
    """
    Load simulation data from CSV file.
    
    Args:
        sport: "men" or "women"
        division: "d1", "d2", or "d3"
        year: Year as string (e.g., "2025")
    
    Returns:
        DataFrame with simulation results or None if file not found
    """
    file_path = SIMULATIONS_DIR / f"game_simulations_{year}_{sport}_{division}.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
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
        years = sorted(set(c["year"] for c in combinations), reverse=True)
        sports = sorted(set(c["sport"] for c in combinations))
        divisions = sorted(set(c["division"] for c in combinations))
        
        # Year selector
        selected_year = st.selectbox(
            "Year",
            options=years,
            index=0
        )
        
        # Filter combinations by year
        year_combinations = [c for c in combinations if c["year"] == selected_year]
        
        if not year_combinations:
            st.warning(f"No data available for year {selected_year}")
            return
        
        # Sport selector
        available_sports = sorted(set(c["sport"] for c in year_combinations))
        selected_sport = st.selectbox(
            "Sport",
            options=available_sports,
            index=0 if "men" in available_sports else 0
        )
        
        # Division selector
        available_divisions = sorted(set(
            c["division"] for c in year_combinations 
            if c["sport"] == selected_sport
        ))
        selected_division = st.selectbox(
            "Division",
            options=available_divisions,
            index=0
        )
        
        st.markdown("---")
        
        # Load data
        df = load_simulation_data(selected_sport, selected_division, selected_year)
        
        if df is None or df.empty:
            st.error(f"No data found for {selected_sport} {selected_division} {selected_year}")
            return
        
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
    st.header(f"{selected_sport.title()} {selected_division.upper()} {selected_year}")
    
    # Format and display data as table
    if isinstance(df, pd.DataFrame) and not df.empty:
        display_df = format_display_df(df)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} games")


if __name__ == "__main__":
    main()
