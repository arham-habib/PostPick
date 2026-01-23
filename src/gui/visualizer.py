"""
Game Simulation Visualizer

A Streamlit-based web application to visualize predicted game results
with spread, total, and moneyline metrics.
"""
import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Set JAX to use CPU to avoid backend issues
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import streamlit as st
import pandas as pd

from src.gui.data_loader import (
    get_available_combinations,
    load_aggregated_simulation_data,
    DEFAULT_GENDER
)
from src.gui.game_summary import format_display_df
from src.gui.game_details import render_game_detail_view

# Page configuration
st.set_page_config(
    page_title="Game Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application."""
    # Title
    st.title("Game Prediction Visualizer")
    
    # Get available combinations
    combinations = get_available_combinations()
    
    if not combinations:
        st.error("No simulation files found in the simulations directory.")
        st.info(f"Expected location: {SCRIPT_DIR / 'data' / 'ncaab' / 'simulations'}")
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
        
        # Load aggregated data (summary only, not raw simulations)
        with st.spinner("Loading game summaries..."):
            df = load_aggregated_simulation_data(selected_model, selected_monday_date)
        
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
    
    # Format and display data with expandable rows
    if isinstance(df, pd.DataFrame) and not df.empty:
        display_df = format_display_df(df)
        
        # Show summary table
        st.dataframe(display_df, width='stretch', hide_index=True)
        st.caption(f"Showing {len(df)} games. Click on a game below to see detailed analysis.")
        
        # Display games with expandable detail views
        st.markdown("---")
        st.subheader("Game Details")
        st.info("💡 Expand a game below to load its detailed simulation data and visualizations. Data is loaded on-demand to save memory.")
        
        # Add pagination to limit number of games shown
        games_per_page = st.sidebar.number_input("Games per page", min_value=10, max_value=100, value=20, step=10)
        
        total_games = len(df)
        num_pages = (total_games + games_per_page - 1) // games_per_page
        
        if num_pages > 1:
            page = st.sidebar.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1)
            start_idx = (page - 1) * games_per_page
            end_idx = start_idx + games_per_page
            df_page = df.iloc[start_idx:end_idx]
            st.caption(f"Showing games {start_idx + 1}-{min(end_idx, total_games)} of {total_games}")
        else:
            df_page = df
            page = 1
        
        # Use expanders with buttons inside - only load when button is clicked
        # This prevents loading data when expanders are just created
        for idx, row in df_page.iterrows():
            game_id = row['game_id']
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Create label for each game
            game_label = f"{away_team} @ {home_team}"
            if "date" in row and pd.notna(row["date"]):
                date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                game_label = f"{date_str}: {game_label}"
            
            # Add summary stats to label
            if "spread_p50" in row and "moneyline_home_win" in row:
                spread_str = f"{row['spread_p50']:+.1f}"
                home_win_pct = f"{row['moneyline_home_win']*100:.1f}%"
                game_label = f"{game_label} | Spread: {spread_str} | Home Win: {home_win_pct}"
            
            # Create unique key for this game's loaded state
            load_key = f"load_{game_id}_{selected_model}_{selected_monday_date}"
            
            # Use expander but only load content when button is clicked
            with st.expander(game_label, expanded=False):
                if not st.session_state.get(load_key, False):
                    st.info("Click the button below to load detailed game analysis")
                    if st.button("Load Game Details", key=f"btn_{load_key}"):
                        st.session_state[load_key] = True
                        st.rerun()
                else:
                    # Only render details if button was clicked
                    render_game_detail_view(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        model_name=selected_model,
                        monday_date=selected_monday_date,
                        gender=DEFAULT_GENDER
                    )


if __name__ == "__main__":
    main()
