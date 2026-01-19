import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.logging_utils import setup_scraping_logger
from src.utils.data_utils import get_game_data_path

BASE_URL = "https://ncaa-api.henrygd.me/scoreboard/basketball-{}/{}/{}/{}/{}/all-conf"


def fetch_game_data(sport: str, division: str, date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch game data for a specific date."""
    year_str, month_str, day_str = date_str.split("-")
    url = BASE_URL.format(sport, division, year_str, month_str, day_str)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if not data or "games" not in data:
            return None
        return data
    except Exception as e:
        logging.warning(f"Failed to fetch {date_str}: {e}")
        return None

def parse_games(data: Dict[str, Any], date_str: str) -> pd.DataFrame:
    """Extract relevant game information and return as DataFrame, including division, home_id, and away_id."""
    if not data or "games" not in data:
        return pd.DataFrame()
    
    game_list = []
    for game in data.get("games", []):
        g = game.get("game", {})
        if not g:
            continue

        # Check if scores exist, skip if missing
        home_score = g.get("home", {}).get("score")
        away_score = g.get("away", {}).get("score")
        if home_score is None or away_score is None:
            continue

        # Only keep games that are FINAL (exclude games in progress)
        final_message = g.get("finalMessage", "")
        if final_message != "FINAL":
            continue

        # Extract division, home_id, away_id
        division = g.get("division", "")
        home_id = g.get("home", {}).get("id", "")
        away_id = g.get("away", {}).get("id", "")

        game_list.append({
            "gameID": g.get("gameID", ""),
            "date": date_str,
            "division": division,
            "home_id": home_id,
            "away_id": away_id,
            "home_team": g.get("home", {}).get("names", {}).get("short", "Unknown"),
            "away_team": g.get("away", {}).get("names", {}).get("short", "Unknown"),
            "home_score": home_score,
            "away_score": away_score,
            "finalMessage": g.get("finalMessage", "Unknown"),
            "start_time": g.get("startTime", "Unknown"),
            "url": g.get("url", "game/Unknown").split("/")[-1],
            "conference_home": g.get("home", {}).get("conferences", [{}])[0].get("conferenceName", ""),
            "conference_away": g.get("away", {}).get("conferences", [{}])[0].get("conferenceName", ""),
        })
    return pd.DataFrame(game_list)

def scrape_games(sport: str, division: str, year: int):
    """Scrape game data for the given year, with incremental updates."""
    start_date = datetime(year, 11, 1)
    end_date = datetime(year + 1, 4, 10)
    output_path = get_game_data_path("ncaab", year, sport, division)
    
    # Check for existing data to enable incremental updates
    existing_df = pd.DataFrame()
    scrape_start_date = start_date
    
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            # Find latest date with FINAL games
            if not existing_df.empty and "date" in existing_df.columns and "finalMessage" in existing_df.columns:
                final_games = existing_df[existing_df["finalMessage"] == "FINAL"]
                if not final_games.empty:
                    latest_date = pd.to_datetime(final_games["date"]).max()
                    scrape_start_date = latest_date + pd.Timedelta(days=1)
                    logging.info(f"Found existing data. Latest FINAL game date: {latest_date.date()}. Starting from {scrape_start_date.date()}")
        except Exception as e:
            logging.warning(f"Could not read existing data: {e}. Starting fresh scrape.")
    
    # Only scrape dates from scrape_start_date onwards
    if scrape_start_date > end_date:
        logging.info("No new dates to scrape. All games are up to date.")
        return
    
    date_range = pd.date_range(scrape_start_date, end_date)
    logging.info(f"Scraping {len(date_range)} dates for {sport} {division} {year}")
    
    all_data = []
    for day in tqdm(date_range, desc="Scraping games"):
        date_str = day.strftime("%Y-%m-%d")
        data = fetch_game_data(sport, division, date_str)
        df = parse_games(data, date_str)
        
        if not df.empty:
            all_data.append(df)
            logging.info(f"Scraped {len(df)} games for {date_str}")
        
        # Be respectful to the API
        time.sleep(1)
    
    # Combine with existing data and deduplicate
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        
        if not existing_df.empty:
            # Remove old non-FINAL games that might now be FINAL
            existing_df = existing_df[existing_df["finalMessage"] == "FINAL"]
            # Combine and deduplicate by gameID
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["gameID"], keep="last")
            combined_df = combined_df.sort_values("date").reset_index(drop=True)
            full_df = combined_df
            logging.info(f"Added {len(new_df)} new games. Total: {len(full_df)} games")
        else:
            full_df = new_df
        
        full_df.to_csv(output_path, index=False)
        logging.info(f"Saved {len(full_df)} games to {output_path}")
    else:
        if not existing_df.empty:
            # Clean up existing data: remove non-FINAL games
            existing_df = existing_df[existing_df["finalMessage"] == "FINAL"]
            existing_df.to_csv(output_path, index=False)
            logging.info(f"No new games found. Cleaned existing data: {len(existing_df)} FINAL games")
        else:
            logging.warning(f"No data scraped for {sport} {division} {year}")

def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA game data.")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape data for")

    args = parser.parse_args()

    # Setup structured logging
    setup_scraping_logger("ncaab", "game")
    
    scrape_games(args.sport, args.division, args.year)

if __name__ == "__main__":
    main()