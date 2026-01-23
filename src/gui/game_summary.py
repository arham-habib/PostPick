"""
Game summary aggregation and formatting utilities.

Handles aggregation of raw simulation data into summary statistics per game.
"""
import pandas as pd
import numpy as np
from typing import Optional


def aggregate_simulation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw simulation data to compute statistics per game.
    Uses vectorized pandas operations for speed.
    
    Args:
        df: DataFrame with columns: game_id, date, home_team, away_team, 
            param_draw_idx, sim_idx, home_score, away_score, spread, total
    
    Returns:
        DataFrame with aggregated statistics per game
    """
    if df.empty:
        return df
    
    # Group by game - use game_id as primary key
    group_cols = ['game_id', 'home_team', 'away_team']
    if 'date' in df.columns:
        group_cols.append('date')
    
    # Compute win indicators first (before grouping)
    df = df.copy()
    df['home_wins'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_wins'] = (df['away_score'] > df['home_score']).astype(int)
    df['ties'] = (df['home_score'] == df['away_score']).astype(int)
    
    # Use vectorized aggregation - much faster than iterating
    grouped = df.groupby(group_cols, observed=True)
    
    # Aggregate using named functions to avoid lambda issues
    def q05(x):
        return x.quantile(0.05)
    q05.__name__ = 'q05'
    
    def q95(x):
        return x.quantile(0.95)
    q95.__name__ = 'q95'
    
    # Use apply with a function to compute all stats at once (more reliable)
    def compute_game_stats(group):
        return pd.Series({
            'spread_mean': group['spread'].mean(),
            'spread_std': group['spread'].std(),
            'spread_p5': group['spread'].quantile(0.05),
            'spread_p50': group['spread'].median(),
            'spread_p95': group['spread'].quantile(0.95),
            'total_mean': group['total'].mean(),
            'total_std': group['total'].std(),
            'total_p5': group['total'].quantile(0.05),
            'total_p50': group['total'].median(),
            'total_p95': group['total'].quantile(0.95),
            'expected_home_score': group['home_score'].mean(),
            'expected_away_score': group['away_score'].mean(),
            'total_sims': len(group),
            'home_wins': group['home_wins'].sum(),
            'away_wins': group['away_wins'].sum(),
            'ties': group['ties'].sum(),
        })
    
    result = grouped.apply(compute_game_stats).reset_index()
    
    # Calculate win probabilities
    if 'total_sims' in result.columns and result['total_sims'].sum() > 0:
        result['moneyline_home_win'] = (result['home_wins'] / result['total_sims']).fillna(0)
        result['moneyline_away_win'] = (result['away_wins'] / result['total_sims']).fillna(0)
        result['moneyline_tie'] = (result['ties'] / result['total_sims']).fillna(0)
        result = result.drop(columns=['home_wins', 'away_wins', 'ties', 'total_sims'], errors='ignore')
    
    # Calculate win probabilities
    if 'total_sims' in result.columns and result['total_sims'].sum() > 0:
        result['moneyline_home_win'] = (result['home_wins'] / result['total_sims']).fillna(0)
        result['moneyline_away_win'] = (result['away_wins'] / result['total_sims']).fillna(0)
        result['moneyline_tie'] = (result['ties'] / result['total_sims']).fillna(0)
        result = result.drop(columns=['home_wins', 'away_wins', 'ties', 'total_sims'], errors='ignore')
    
    return result


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
