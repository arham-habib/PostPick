from enum import Enum
from typing import Dict, List, Tuple, Iterable, TypedDict, Optional
import jax.numpy as jnp
from dataclasses import dataclass

class League(Enum):
    NCAAMB = "NCAAMB"
    NCAAWB = "NCAAWB"
    NBA = "NBA"
    WNBA = "WNBA"

class SeasonType(Enum):
    PRE = "Pre"
    REGULAR = "Regular"
    POST = "Post"

class SeasonDF(TypedDict):
    gameID: int
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    finalMessage: str
    start_time: str
    url: str
    conference_home: str
    conference_away: str

@dataclass(frozen=True)
class EncodedSeason:
    home_idx: jnp.ndarray     # shape (n_games,), int32
    away_idx: jnp.ndarray     # shape (n_games,), int32
    y_home: jnp.ndarray       # shape (n_games,), int32
    y_away: jnp.ndarray       # shape (n_games,), int32
    n_teams: int
    id_to_team: List[str]
    team_to_id: Dict[str, int]

