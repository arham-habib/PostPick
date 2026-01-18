"""
Unit tests for game simulation module.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ncaa.simulate_games import (
    handle_unknown_team,
    map_schedule_teams_to_model,
    simulate_single_game,
    compute_game_statistics,
    compute_multivariate_stats,
)
from ncaa.data_cleaning import _normalize_team_name


class TestTeamMapping:
    """Test team name mapping and normalization."""
    
    def test_normalize_team_name(self):
        """Test team name normalization."""
        assert _normalize_team_name("  Duke  ") == "Duke"
        assert _normalize_team_name("North   Carolina") == "North Carolina"
        assert _normalize_team_name("Kentucky\n") == "Kentucky"
    
    def test_handle_unknown_team(self):
        """Test handling of unknown teams."""
        team_to_id = {"Duke": 0, "UNC": 1}
        samples = {
            'offense': jnp.array([[0.1, 0.2], [0.15, 0.25]]),  # (n_samples, n_teams)
            'defense': jnp.array([[0.05, 0.1], [0.08, 0.12]]),
            'h': jnp.array([[0.02, 0.03], [0.025, 0.035]]),
            'h_mu': jnp.array([0.02, 0.025]),
        }
        
        # Known team
        idx, is_unknown = handle_unknown_team("Duke", team_to_id, samples)
        assert idx == 0
        assert not is_unknown
        
        # Unknown team
        idx, is_unknown = handle_unknown_team("Unknown Team", team_to_id, samples)
        assert idx == -1
        assert is_unknown
    
    def test_map_schedule_teams_to_model(self):
        """Test mapping schedule teams to model indices."""
        schedule_df = pd.DataFrame({
            'home_team': ['Duke', 'UNC', 'Unknown'],
            'away_team': ['UNC', 'Duke', 'Duke'],
            'gameID': [1, 2, 3],
        })
        
        team_to_id = {"Duke": 0, "UNC": 1}
        samples = {
            'offense': jnp.array([[0.1, 0.2], [0.15, 0.25]]),
            'defense': jnp.array([[0.05, 0.1], [0.08, 0.12]]),
            'h': jnp.array([[0.02, 0.03], [0.025, 0.035]]),
            'h_mu': jnp.array([0.02, 0.025]),
        }
        
        df_mapped, metadata = map_schedule_teams_to_model(schedule_df, team_to_id, samples)
        
        assert 'home_idx' in df_mapped.columns
        assert 'away_idx' in df_mapped.columns
        assert df_mapped.loc[0, 'home_idx'] == 0  # Duke
        assert df_mapped.loc[0, 'away_idx'] == 1  # UNC
        assert df_mapped.loc[2, 'home_idx'] == -1  # Unknown
        assert metadata['n_unknown_teams'] == 1
        assert 'Unknown' in metadata['unknown_teams']


class TestSimulation:
    """Test game simulation logic."""
    
    def test_simulate_single_game_basic(self):
        """Test basic game simulation with known teams."""
        # Create simple posterior samples
        n_samples = 10
        n_teams = 5
        
        samples = {
            'alpha': jnp.array([4.5] * n_samples),  # log mean around 90 points
            'offense': jnp.array([[0.1, 0.2, -0.1, 0.0, 0.15] for _ in range(n_samples)]),
            'defense': jnp.array([[0.05, -0.05, 0.1, 0.0, -0.1] for _ in range(n_samples)]),
            'h': jnp.array([[0.03] * n_teams for _ in range(n_samples)]),
            'h_mu': jnp.array([0.03] * n_samples),
        }
        
        home_idx = 0
        away_idx = 1
        n_draws = 100  # Small for testing
        
        home_scores, away_scores = simulate_single_game(
            home_idx, away_idx, samples, n_draws=n_draws
        )
        
        assert len(home_scores) == n_samples * n_draws
        assert len(away_scores) == n_samples * n_draws
        assert all(home_scores >= 0)
        assert all(away_scores >= 0)
        # Scores should be reasonable (not negative, typically in range 0-200)
        assert all(home_scores < 300)
        assert all(away_scores < 300)
    
    def test_simulate_single_game_unknown_team(self):
        """Test simulation with unknown team (using average parameters)."""
        n_samples = 5
        n_teams = 3
        
        samples = {
            'alpha': jnp.array([4.5] * n_samples),
            'offense': jnp.array([[0.1, 0.2, -0.1] for _ in range(n_samples)]),
            'defense': jnp.array([[0.05, -0.05, 0.1] for _ in range(n_samples)]),
            'h': jnp.array([[0.03] * n_teams for _ in range(n_samples)]),
            'h_mu': jnp.array([0.03] * n_samples),
        }
        
        # Unknown home team (idx = -1)
        home_idx = -1
        away_idx = 0
        n_draws = 50
        
        home_scores, away_scores = simulate_single_game(
            home_idx, away_idx, samples, n_draws=n_draws
        )
        
        # Should still produce valid results
        assert len(home_scores) == n_samples * n_draws
        assert len(away_scores) == n_samples * n_draws
        assert all(home_scores >= 0)
        assert all(away_scores >= 0)


class TestStatistics:
    """Test statistics computation."""
    
    def test_compute_game_statistics(self):
        """Test per-game statistics computation."""
        # Create simple test data
        np.random.seed(42)
        n_sims = 10000
        home_scores = np.random.poisson(75, n_sims)
        away_scores = np.random.poisson(70, n_sims)
        
        stats = compute_game_statistics(home_scores, away_scores)
        
        # Check that all required keys are present
        required_keys = [
            'spread_mean', 'spread_std', 'spread_p5', 'spread_p25', 'spread_p50',
            'spread_p75', 'spread_p95',
            'total_mean', 'total_std', 'total_p5', 'total_p25', 'total_p50',
            'total_p75', 'total_p95',
            'moneyline_home_win', 'moneyline_away_win', 'moneyline_tie',
            'spread_total_correlation', 'spread_total_covariance',
        ]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # Check that percentiles are ordered correctly
        assert stats['spread_p5'] <= stats['spread_p25'] <= stats['spread_p50'] <= stats['spread_p75'] <= stats['spread_p95']
        assert stats['total_p5'] <= stats['total_p25'] <= stats['total_p50'] <= stats['total_p75'] <= stats['total_p95']
        
        # Moneyline probabilities should sum to approximately 1
        total_prob = stats['moneyline_home_win'] + stats['moneyline_away_win'] + stats['moneyline_tie']
        assert abs(total_prob - 1.0) < 0.01
        
        # Since home_mean > away_mean, home should win more often
        assert stats['moneyline_home_win'] > stats['moneyline_away_win']
    
    def test_compute_multivariate_stats(self):
        """Test multivariate statistics computation."""
        np.random.seed(42)
        n_obs = 5000
        
        # Create correlated data
        spreads = np.random.normal(5, 10, n_obs)
        totals = 145 + 0.5 * spreads + np.random.normal(0, 5, n_obs)  # Some correlation
        
        stats = compute_multivariate_stats(spreads, totals)
        
        # Check required keys
        assert 'covariance_matrix' in stats
        assert 'correlation' in stats
        assert 'spread' in stats
        assert 'total' in stats
        assert 'n_observations' in stats
        
        # Check covariance matrix shape
        cov_matrix = np.array(stats['covariance_matrix'])
        assert cov_matrix.shape == (2, 2)
        
        # Check correlation is reasonable (should be positive)
        assert -1 <= stats['correlation'] <= 1
        
        # Check marginal statistics
        assert 'mean' in stats['spread']
        assert 'std' in stats['spread']
        assert 'p50' in stats['spread']
        
        # Check observation count
        assert stats['n_observations'] == n_obs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
