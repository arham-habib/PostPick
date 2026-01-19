"""
Data path utilities for sports data processing.

Provides structured data paths with organized directory structure:
- data/{sport}/game/
- data/{sport}/pbp/
- data/{sport}/player/
- data/{sport}/schedule/
- data/{sport}/models/
- data/{sport}/simulations/
"""
from pathlib import Path
from typing import Optional


def get_data_base_dir() -> Path:
    """Get base data directory (project root / data)."""
    script_dir = Path(__file__).resolve().parent.parent.parent
    return script_dir / "data"


def get_sport_folder_name(sport_type: str, gender: Optional[str] = None) -> str:
    """
    Get sport folder name for data organization.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab', 'nba', 'wnba')
        gender: Optional gender specification (e.g., 'men', 'women')
                For ncaab, this is typically included in the sport_type
    
    Returns:
        Sport folder name (e.g., 'ncaab')
    """
    # For now, we standardize on 'ncaab' for NCAA basketball
    # In the future, this could handle other sports
    if sport_type.startswith('ncaab'):
        return 'ncaab'
    return sport_type


def get_game_data_path(sport_type: str, year: int, gender: str, division: str) -> Path:
    """
    Get path for game data CSV files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        year: Year of the season
        gender: Gender ('men' or 'women')
        division: Division ('d1', 'd2', 'd3')
    
    Returns:
        Path to game data file: data/{sport}/game/ncaab_{year}_{gender}_{division}.csv
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    game_dir = base_dir / sport_folder / "game"
    game_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"ncaab_{year}_{gender}_{division}.csv"
    return game_dir / filename


def get_pbp_data_path(sport_type: str, year: int, gender: str, division: str) -> Path:
    """
    Get path for play-by-play data CSV files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        year: Year of the season
        gender: Gender ('men' or 'women')
        division: Division ('d1', 'd2', 'd3')
    
    Returns:
        Path to PBP data file: data/{sport}/pbp/play_by_play_{gender}_{year}_{division}.csv
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    pbp_dir = base_dir / sport_folder / "pbp"
    pbp_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"play_by_play_{gender}_{year}_{division}.csv"
    return pbp_dir / filename


def get_player_data_path(sport_type: str, year: int, gender: str, division: str) -> Path:
    """
    Get path for player box score data CSV files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        year: Year of the season
        gender: Gender ('men' or 'women')
        division: Division ('d1', 'd2', 'd3')
    
    Returns:
        Path to player data file: data/{sport}/player/box_score_{gender}_{year}_{division}.csv
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    player_dir = base_dir / sport_folder / "player"
    player_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"box_score_{gender}_{year}_{division}.csv"
    return player_dir / filename


def get_schedule_data_path(sport_type: str, year: int, gender: str, division: str) -> Path:
    """
    Get path for schedule data CSV files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        year: Year of the season
        gender: Gender ('men' or 'women')
        division: Division ('d1', 'd2', 'd3')
    
    Returns:
        Path to schedule data file: data/{sport}/schedule/schedule_{year}_{gender}_{division}.csv
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    schedule_dir = base_dir / sport_folder / "schedule"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"schedule_{year}_{gender}_{division}.csv"
    return schedule_dir / filename


def get_model_data_path(sport_type: str, gender: str, model_name: str, monday_date: str) -> Path:
    """
    Get path for model pickle files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        gender: Gender ('men' or 'women')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
        monday_date: Monday date string (e.g., '2026-01-15')
    
    Returns:
        Path to model file: data/{sport}/models/{model_name}_{monday_date}.pkl
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    model_dir = base_dir / sport_folder / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_{monday_date}.pkl"
    return model_dir / filename


def get_simulation_data_path(sport_type: str, gender: str, model_name: str, monday_date: str) -> Path:
    """
    Get path for simulation parquet files.
    
    Args:
        sport_type: Sport type (e.g., 'ncaab')
        gender: Gender ('men' or 'women')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
        monday_date: Monday date string (e.g., '2026-01-15')
    
    Returns:
        Path to simulation file: data/{sport}/simulations/{model_name}_{monday_date}.parquet
    """
    base_dir = get_data_base_dir()
    sport_folder = get_sport_folder_name(sport_type)
    sim_dir = base_dir / sport_folder / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_{monday_date}.parquet"
    return sim_dir / filename
