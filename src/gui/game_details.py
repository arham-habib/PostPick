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

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

from src.gui.data_loader import load_raw_simulation_data_for_game, load_model_parameters, DEFAULT_GENDER


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
    
    # Prepare data for visualizations
    raw_df = raw_df.copy()
    raw_df['winner'] = raw_df.apply(
        lambda row: 'Home' if row['home_score'] > row['away_score'] 
        else ('Away' if row['away_score'] > row['home_score'] else 'Tie'),
        axis=1
    )
    
    # Section 2: Outcome Distribution (Spread vs Total)
    st.markdown("### Outcome Distribution: Spread vs Total")
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        for winner in ['Home', 'Away', 'Tie']:
            winner_data = raw_df[raw_df['winner'] == winner]
            if len(winner_data) > 0:
                color_map = {'Home': 'blue', 'Away': 'red', 'Tie': 'gray'}
                fig.add_trace(go.Scatter(
                    x=winner_data['spread'],
                    y=winner_data['total'],
                    mode='markers',
                    name=winner,
                    marker=dict(
                        color=color_map.get(winner, 'gray'),
                        size=3,
                        opacity=0.6
                    ),
                    hovertemplate=f'<b>{winner} Win</b><br>' +
                                  'Spread: %{x}<br>' +
                                  'Total: %{y}<br>' +
                                  '<extra></extra>'
                ))
        
        fig.update_layout(
            title="Spread vs Total (colored by winner)",
            xaxis_title="Spread (Home - Away)",
            yaxis_title="Total Points",
            hovermode='closest',
            height=500
        )
        st.plotly_chart(fig, width='stretch')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for winner, color in [('Home', 'blue'), ('Away', 'red'), ('Tie', 'gray')]:
            winner_data = raw_df[raw_df['winner'] == winner]
            if len(winner_data) > 0:
                ax.scatter(winner_data['spread'], winner_data['total'], 
                          c=color, label=winner, alpha=0.6, s=10)
        ax.set_xlabel("Spread (Home - Away)")
        ax.set_ylabel("Total Points")
        ax.set_title("Spread vs Total (colored by winner)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    # Section 3: Score Correlation Plot
    st.markdown("### Score Correlation: Home vs Away")
    
    # Calculate correlation
    correlation = float(raw_df['home_score'].corr(raw_df['away_score']))
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        for winner in ['Home', 'Away', 'Tie']:
            winner_data = raw_df[raw_df['winner'] == winner]
            if len(winner_data) > 0:
                color_map = {'Home': 'blue', 'Away': 'red', 'Tie': 'gray'}
                fig.add_trace(go.Scatter(
                    x=winner_data['home_score'],
                    y=winner_data['away_score'],
                    mode='markers',
                    name=winner,
                    marker=dict(
                        color=color_map.get(winner, 'gray'),
                        size=3,
                        opacity=0.6
                    ),
                    hovertemplate=f'<b>{winner} Win</b><br>' +
                                  'Home: %{x}<br>' +
                                  'Away: %{y}<br>' +
                                  '<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"Home Score vs Away Score (Correlation: {correlation:.3f})",
            xaxis_title="Home Score",
            yaxis_title="Away Score",
            hovermode='closest',
            height=500
        )
        st.plotly_chart(fig, width='stretch')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for winner, color in [('Home', 'blue'), ('Away', 'red'), ('Tie', 'gray')]:
            winner_data = raw_df[raw_df['winner'] == winner]
            if len(winner_data) > 0:
                ax.scatter(winner_data['home_score'], winner_data['away_score'],
                          c=color, label=winner, alpha=0.6, s=10)
        ax.set_xlabel("Home Score")
        ax.set_ylabel("Away Score")
        ax.set_title(f"Home Score vs Away Score (Correlation: {correlation:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    # Section 4: Additional Monte Carlo Statistics
    st.markdown("### Monte Carlo Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Spread Distribution")
        if PLOTLY_AVAILABLE:
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
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(raw_df['spread'], bins=50, color='lightblue', edgecolor='black')
            ax.set_xlabel("Spread")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Spread")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
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
        if PLOTLY_AVAILABLE:
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
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(raw_df['total'], bins=50, color='lightgreen', edgecolor='black')
            ax.set_xlabel("Total Points")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Total")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
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
        if PLOTLY_AVAILABLE:
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
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(raw_df['home_score'], bins=30, color='blue', alpha=0.7, 
                   label='Home', edgecolor='black')
            ax.hist(raw_df['away_score'], bins=30, color='red', alpha=0.7,
                   label='Away', edgecolor='black')
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Home vs Away Score Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # Score statistics
        score_stats = {
            "Home Mean": f"{raw_df['home_score'].mean():.2f}",
            "Home Std": f"{raw_df['home_score'].std():.2f}",
            "Away Mean": f"{raw_df['away_score'].mean():.2f}",
            "Away Std": f"{raw_df['away_score'].std():.2f}",
        }
        st.json(score_stats)
    
    # Extreme outcomes
    st.markdown("#### Extreme Outcomes")
    col1, col2 = st.columns(2)
    
    with col1:
        # Blowout probability (win by 20+)
        home_blowouts = ((raw_df['home_score'] - raw_df['away_score']) >= 20).sum()
        away_blowouts = ((raw_df['away_score'] - raw_df['home_score']) >= 20).sum()
        total_sims = len(raw_df)
        
        st.metric("Home Blowout (20+ pts)", 
                 f"{(home_blowouts / total_sims * 100):.1f}%",
                 f"{home_blowouts:,} sims")
        st.metric("Away Blowout (20+ pts)",
                 f"{(away_blowouts / total_sims * 100):.1f}%",
                 f"{away_blowouts:,} sims")
    
    with col2:
        # Close game probability (within 3 points)
        close_games = (abs(raw_df['spread']) <= 3).sum()
        st.metric("Close Game (≤3 pts)", 
                 f"{(close_games / total_sims * 100):.1f}%",
                 f"{close_games:,} sims")
        
        # High scoring (total > 160)
        high_scoring = (raw_df['total'] > 160).sum()
        st.metric("High Scoring (>160 pts)",
                 f"{(high_scoring / total_sims * 100):.1f}%",
                 f"{high_scoring:,} sims")
    
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
