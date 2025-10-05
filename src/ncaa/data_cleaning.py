from typing import TypedDict, Dict, List, Tuple
from dataclasses import dataclass
import jax.numpy as jnp
import pandas as pd
import numpy as np
from utils.enums import EncodedSeason, SeasonDF

def _normalize_team_name(name: str) -> str:
    # light normalization to avoid accidental dupes
    return " ".join(name.strip().split())

def build_team_indexer(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    assert {"home_team", "away_team"}.issubset(df.columns), "Missing required columns."
    teams: List[str] = sorted({
        _normalize_team_name(t) for t in
        df["home_team"].astype(str).tolist() + df["away_team"].astype(str).tolist()
    })
    team_to_id = {t: i for i, t in enumerate(teams)}
    id_to_team = teams
    return team_to_id, id_to_team

def encode_season(df: pd.DataFrame, team_to_id: Dict[str, int]) -> EncodedSeason:

    req = {"home_team","away_team","home_score","away_score"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    home_idx = df["home_team"].astype(str).map(lambda x: team_to_id[_normalize_team_name(x)]).astype(np.int32).to_numpy()
    away_idx = df["away_team"].astype(str).map(lambda x: team_to_id[_normalize_team_name(x)]).astype(np.int32).to_numpy()

    y_home = df["home_score"].astype(np.int32).to_numpy()
    y_away = df["away_score"].astype(np.int32).to_numpy()

    n_teams = len(team_to_id)
    id_to_team: List[str] = [""] * n_teams
    for t, i in team_to_id.items():
        id_to_team[i] = t

    return EncodedSeason(
        home_idx=jnp.array(home_idx),
        away_idx=jnp.array(away_idx),
        y_home=jnp.array(y_home),
        y_away=jnp.array(y_away),
        n_teams=n_teams,
        id_to_team=id_to_team,  # type: ignore
        team_to_id=team_to_id,
    )

def drop_teams_with_few_games(df, n=30):
    """
    Drops teams from the dataframe that have fewer than n unique games (home or away).
    Returns a filtered dataframe.
    """
    # Count number of unique games for each team (home or away)
    team_games = {}

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        game_id = row['gameID']
        # Add game to home team
        if home not in team_games:
            team_games[home] = set()
        team_games[home].add(game_id)
        # Add game to away team
        if away not in team_games:
            team_games[away] = set()
        team_games[away].add(game_id)

    # Find teams with at least n games
    teams_to_keep = {team for team, games in team_games.items() if len(games) >= n}

    # Filter dataframe to only keep games where both teams are in teams_to_keep
    filtered_df = df[
        df['home_team'].isin(teams_to_keep) & df['away_team'].isin(teams_to_keep)
    ].copy()

    return filtered_df
