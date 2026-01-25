"""
Game detail visualization module.

Handles rendering of detailed views for individual games including
team parameters, outcome distributions, and Monte Carlo statistics.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

from src.gui.data_loader import load_raw_simulation_data_for_game, load_model_parameters, DEFAULT_GENDER


def _plot_kde_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    xlabel: str,
    label1: str,
    label2: str
) -> None:
    """
    Plot KDE comparison of two distributions using plotly.
    
    Args:
        data1: First dataset (will be plotted in red)
        data2: Second dataset (will be plotted in blue)
        xlabel: Label for x-axis
        label1: Label for first dataset
        label2: Label for second dataset
    """
    # Compute KDE for both datasets
    # Handle edge case where data might have very low variance
    try:
        kde1 = gaussian_kde(data1)
        kde2 = gaussian_kde(data2)
    except (ValueError, np.linalg.LinAlgError):
        # Fallback: if KDE fails (e.g., constant data), use histogram instead
        st.warning("KDE computation failed, using histogram instead")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data1, name=label1, marker_color='red', opacity=0.7, nbinsx=30))
        fig.add_trace(go.Histogram(x=data2, name=label2, marker_color='blue', opacity=0.7, nbinsx=30))
        fig.update_layout(
            title=f"Distribution: {xlabel}",
            xaxis_title=xlabel,
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
        return
    
    # Create evaluation range
    min_val = min(data1.min(), data2.min())
    max_val = max(data1.max(), data2.max())
    range_size = max_val - min_val
    if range_size == 0:
        # Handle case where all values are the same
        x_range = np.linspace(min_val - 1, max_val + 1, 200)
    else:
        x_range = np.linspace(min_val - 0.1 * range_size, 
                             max_val + 0.1 * range_size, 
                             200)
    
    # Evaluate KDEs
    y1 = kde1(x_range)
    y2 = kde2(x_range)
    
    fig = go.Figure()
    
    # Add first distribution (red)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y1,
        mode='lines',
        name=label1,
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    # Add second distribution (blue)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y2,
        mode='lines',
        name=label2,
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Kernel Density Estimate: {xlabel}",
        xaxis_title=xlabel,
        yaxis_title="Density",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, width='stretch')


def get_team_parameters(
    samples: Dict[str, np.ndarray],
    team_to_id: Dict[str, int],
    home_team: str,
    away_team: str,
    model_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Extract and summarize team parameters from model samples.
    
    Args:
        samples: Dictionary of model parameter samples
        team_to_id: Mapping from team name to team index
        home_team: Home team name
        away_team: Away team name
        model_name: Name of the model (to determine which parameters exist)
    
    Returns:
        Dictionary with parameter summaries for home and away teams
    """
    result = {}
    
    # Get team indices
    home_idx = team_to_id.get(home_team, -1)
    away_idx = team_to_id.get(away_team, -1)
    
    if home_idx == -1 or away_idx == -1:
        return {"home": {}, "away": {}}
    
    def compute_stats(param_array: np.ndarray) -> Dict[str, float]:
        """Compute mean, std, and percentiles for a parameter."""
        return {
            "mean": float(np.mean(param_array)),
            "std": float(np.std(param_array)),
            "p5": float(np.percentile(param_array, 5)),
            "p50": float(np.percentile(param_array, 50)),
            "p95": float(np.percentile(param_array, 95)),
        }
    
    # Extract parameters for home team
    home_params = {}
    if "offense" in samples:
        home_params["offense"] = compute_stats(samples["offense"][:, home_idx])
    if "defense" in samples:
        home_params["defense"] = compute_stats(samples["defense"][:, home_idx])
    if "h" in samples:
        home_params["home_advantage"] = compute_stats(samples["h"][:, home_idx])
    
    # TeamVol-specific parameters
    if model_name == "TeamVol":
        if "team_off_std" in samples:
            home_params["team_off_std"] = compute_stats(samples["team_off_std"][:, home_idx])
        if "team_def_std" in samples:
            home_params["team_def_std"] = compute_stats(samples["team_def_std"][:, home_idx])
    
    # Extract parameters for away team
    away_params = {}
    if "offense" in samples:
        away_params["offense"] = compute_stats(samples["offense"][:, away_idx])
    if "defense" in samples:
        away_params["defense"] = compute_stats(samples["defense"][:, away_idx])
    
    # TeamVol-specific parameters
    if model_name == "TeamVol":
        if "team_off_std" in samples:
            away_params["team_off_std"] = compute_stats(samples["team_off_std"][:, away_idx])
        if "team_def_std" in samples:
            away_params["team_def_std"] = compute_stats(samples["team_def_std"][:, away_idx])
    
    result["home"] = home_params
    result["away"] = away_params
    
    return result


def render_game_detail_view(
    game_id: Any,
    home_team: str,
    away_team: str,
    model_name: str,
    monday_date: str,
    gender: str = DEFAULT_GENDER
) -> None:
    """
    Render detailed view for a specific game.
    Only loads data when this function is called (lazy loading).
    
    Args:
        game_id: Game identifier
        home_team: Home team name
        away_team: Away team name
        model_name: Name of the model
        monday_date: Monday date string
        gender: Gender specification (default: "men")
    """
    st.markdown("---")
    st.subheader(f"Game Details: {away_team} @ {home_team}")
    
    # Show loading indicator
    with st.spinner("Loading game simulation data..."):
        # Load raw simulation data for this game ONLY
        raw_df = load_raw_simulation_data_for_game(model_name, monday_date, game_id)
    
    if raw_df is None or raw_df.empty:
        st.error("No simulation data available for this game.")
        return
    
    st.info(f"Loaded {len(raw_df):,} simulations for this game")
    
    # Load model parameters (only when needed)
    with st.spinner("Loading model parameters..."):
        model_data = load_model_parameters(model_name, monday_date, gender)
    
    if model_data is None:
        st.warning("Model parameters not available. Visualizations will be limited.")
        team_params = None
    else:
        samples, team_to_id, id_to_team = model_data
        team_params = get_team_parameters(samples, team_to_id, home_team, away_team, model_name)
    
    # Section 1: Team Parameters
    if team_params:
        st.markdown("### Team Parameters")
        
        # Create parameter display table
        param_data = []
        for team_type in ["home", "away"]:
            team_name = home_team if team_type == "home" else away_team
            params = team_params[team_type]
            
            for param_name, stats in params.items():
                param_display = param_name.replace("_", " ").title()
                param_data.append({
                    "Team": team_name,
                    "Parameter": param_display,
                    "Mean": f"{stats['mean']:.3f}",
                    "Std Dev": f"{stats['std']:.3f}",
                    "5th %ile": f"{stats['p5']:.3f}",
                    "Median": f"{stats['p50']:.3f}",
                    "95th %ile": f"{stats['p95']:.3f}",
                })
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, width='stretch', hide_index=True)
    
    # Section 2: Parameter Distribution Plots (KDE)
    if model_data is None:
        st.warning("Model parameters not available. Cannot display parameter distributions.")
    else:
        samples, team_to_id, id_to_team = model_data
        home_idx = team_to_id.get(home_team, -1)
        away_idx = team_to_id.get(away_team, -1)
        
        if home_idx == -1 or away_idx == -1:
            st.warning("Team indices not found. Cannot display parameter distributions.")
        else:
            # Extract parameter samples for the two teams (excluding warmup - samples already exclude warmup)
            # Samples shape: [num_draws, num_teams]
            home_offense = samples["offense"][:, home_idx]
            away_offense = samples["offense"][:, away_idx]
            home_defense = samples["defense"][:, home_idx]
            away_defense = samples["defense"][:, away_idx]
            
            # Get home advantage if available
            if "h" in samples:
                home_h = samples["h"][:, home_idx]
            else:
                home_h = np.zeros_like(home_offense)
            
            # Plot 1: Offensive Rating KDE
            st.markdown("### Offensive Rating Distribution")
            _plot_kde_comparison(
                home_offense, 
                away_offense, 
                "Offensive Rating",
                "Home (Red)", 
                "Away (Blue)"
            )
            
            # Plot 2: Defensive Rating KDE
            st.markdown("### Defensive Rating Distribution")
            _plot_kde_comparison(
                home_defense, 
                away_defense, 
                "Defensive Rating",
                "Home (Red)", 
                "Away (Blue)"
            )
            
            # Plot 3: Net Rating Comparison
            # Home: offense_home + h_home - defense_away
            # Away: offense_away - defense_home
            home_net = home_offense + home_h - away_defense
            away_net = away_offense - home_defense
            
            st.markdown("### Net Rating Comparison")
            _plot_kde_comparison(
                home_net,
                away_net,
                "Net Rating (Offense - Opponent Defense)",
                "Home Net (Red): offense + home_effect - opponent_defense",
                "Away Net (Blue): offense - opponent_defense"
            )
    
    # Section 4: Additional Monte Carlo Statistics
    st.markdown("### Monte Carlo Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Spread Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_df['spread'],
            nbinsx=50,
            name='Spread',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Distribution of Spread",
            xaxis_title="Spread",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, width='stretch')
        
        # Spread statistics
        spread_stats = {
            "Mean": f"{raw_df['spread'].mean():.2f}",
            "Std Dev": f"{raw_df['spread'].std():.2f}",
            "5th %ile": f"{raw_df['spread'].quantile(0.05):.2f}",
            "Median": f"{raw_df['spread'].median():.2f}",
            "95th %ile": f"{raw_df['spread'].quantile(0.95):.2f}",
        }
        st.json(spread_stats)
    
    with col2:
        st.markdown("#### Total Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_df['total'],
            nbinsx=50,
            name='Total',
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Distribution of Total",
            xaxis_title="Total Points",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, width='stretch')
        
        # Total statistics
        total_stats = {
            "Mean": f"{raw_df['total'].mean():.2f}",
            "Std Dev": f"{raw_df['total'].std():.2f}",
            "5th %ile": f"{raw_df['total'].quantile(0.05):.2f}",
            "Median": f"{raw_df['total'].median():.2f}",
            "95th %ile": f"{raw_df['total'].quantile(0.95):.2f}",
        }
        st.json(total_stats)
    
    with col3:
        st.markdown("#### Score Distributions")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_df['home_score'],
            nbinsx=30,
            name='Home',
            marker_color='blue',
            opacity=0.7
        ))
        fig.add_trace(go.Histogram(
            x=raw_df['away_score'],
            nbinsx=30,
            name='Away',
            marker_color='red',
            opacity=0.7
        ))
        fig.update_layout(
            title="Home vs Away Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=300
        )
        st.plotly_chart(fig, width='stretch')
        
        # Score statistics
        score_stats = {
            "Home Mean": f"{raw_df['home_score'].mean():.2f}",
            "Home Std": f"{raw_df['home_score'].std():.2f}",
            "Away Mean": f"{raw_df['away_score'].mean():.2f}",
            "Away Std": f"{raw_df['away_score'].std():.2f}",
        }
        st.json(score_stats)
    
    # Win probability by score ranges
    st.markdown("#### Win Probability by Score Ranges")
    
    # Define score ranges
    home_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 200)]
    away_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 200)]
    
    range_data = []
    for h_min, h_max in home_ranges:
        for a_min, a_max in away_ranges:
            mask = (
                (raw_df['home_score'] >= h_min) & (raw_df['home_score'] < h_max) &
                (raw_df['away_score'] >= a_min) & (raw_df['away_score'] < a_max)
            )
            range_df = raw_df[mask]
            if len(range_df) > 0:
                home_wins = (range_df['home_score'] > range_df['away_score']).sum()
                away_wins = (range_df['away_score'] > range_df['home_score']).sum()
                total = len(range_df)
                range_data.append({
                    "Home Score": f"{h_min}-{h_max}",
                    "Away Score": f"{a_min}-{a_max}",
                    "Sims": total,
                    "Home Win %": f"{(home_wins / total * 100):.1f}%",
                    "Away Win %": f"{(away_wins / total * 100):.1f}%",
                })
    
    if range_data:
        range_df = pd.DataFrame(range_data)
        st.dataframe(range_df, width='stretch', hide_index=True)
