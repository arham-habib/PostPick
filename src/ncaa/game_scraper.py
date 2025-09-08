import os
import json
import time
import logging
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "logs"
SCRAPING_DIR = SCRIPT_DIR / "src/ncaa"

# Constants
BASE_URL = "https://ncaa-api.henrygd.me/scoreboard/basketball-{}/{}/{}/{}/{}/all-conf"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRAPING_DIR.mkdir(parents=True, exist_ok=True)


def fetch_game_data(sport: str, division: str, year: int, date_str: str) -> Optional[Dict[str, Any]]:
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
    """Extract relevant game information and return as DataFrame."""
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
            
        game_list.append({
            "gameID": g.get("gameID", ""),
            "date": date_str,
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
    """Scrape all game data for the given year."""
    file_str = f"ncaab_{year}_{sport}_{division}.csv"
    start_date = datetime(year, 11, 1)
    end_date = datetime(year + 1, 4, 10)
    all_data = []
    
    date_range = pd.date_range(start_date, end_date)
    logging.info(f"Starting to scrape {len(date_range)} dates for {sport} {division} {year}")
    
    for day in tqdm(date_range, desc="Scraping games"):
        date_str = day.strftime("%Y-%m-%d")
        data = fetch_game_data(sport, division, year, date_str)
        df = parse_games(data, date_str)
        
        if not df.empty:
            all_data.append(df)
            logging.info(f"Scraped {len(df)} games for {date_str}")
        
        # Be respectful to the API
        time.sleep(1)
    
    # Save results
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(DATA_DIR / file_str, index=False)
        logging.info(f"Saved {len(full_df)} games to {file_str}")
    else:
        logging.warning(f"No data scraped for {sport} {division} {year}")

def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA game data.")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape data for")

    args = parser.parse_args()

    logging.basicConfig(
        filename=LOG_DIR / f"ncaab_{args.year}_{args.sport}_{args.division}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    scrape_games(args.sport, args.division, args.year)

if __name__ == "__main__":
    main()