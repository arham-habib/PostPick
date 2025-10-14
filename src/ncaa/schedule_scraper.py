import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data" / "ncaa" / "schedule"
LOG_DIR = SCRIPT_DIR / "logs"
BASE_URL = "https://ncaa-api.henrygd.me/scoreboard/basketball-{}/{}/{}/{}/{}/all-conf"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY = 0.25  # 250ms between requests


def fetch_schedule_data(sport: str, division: str, date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch scheduled games (future games) for a specific date."""
    year_str, month_str, day_str = date_str.split("-")
    url = BASE_URL.format(sport, division, year_str, month_str, day_str)
    
    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if not data or "games" not in data:
            return None
        return data
    except Exception as e:
        logging.warning(f"Failed to fetch schedule for {date_str}: {e}")
        return None


def parse_scheduled_games(data: Dict[str, Any], date_str: str) -> pd.DataFrame:
    """
    Extract scheduled games (games without scores) and return as DataFrame.
    This is the opposite of game_scraper which only keeps completed games.
    """
    if not data or "games" not in data:
        return pd.DataFrame()
    
    game_list = []
    for game in data.get("games", []):
        g = game.get("game", {})
        if not g:
            continue

        # Keep games WITHOUT scores (scheduled/future games)
        home_score = g.get("home", {}).get("score")
        away_score = g.get("away", {}).get("score")
        
        # Skip completed games (ones with scores)
        if home_score is not None or away_score is not None:
            continue

        # Extract game information for scheduled games
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
            "start_time": g.get("startTime", "Unknown"),
            "start_date": g.get("startDate", ""),
            "start_time_epoch": g.get("startTimeEpoch", ""),
            "game_state": g.get("gameState", ""),
            "venue": g.get("venue", ""),
            "url": g.get("url", "game/Unknown").split("/")[-1],
            "conference_home": g.get("home", {}).get("conferences", [{}])[0].get("conferenceName", ""),
            "conference_away": g.get("away", {}).get("conferences", [{}])[0].get("conferenceName", ""),
            "network": g.get("network", ""),
        })
    return pd.DataFrame(game_list)


def scrape_schedule(sport: str, division: str, year: int, start_month: int = 11, end_month: int = 4):
    """
    Scrape scheduled games (future matchups) for the given year.
    
    This scraper captures games that haven't been played yet (no scores).
    It uses the scoreboard endpoint but filters for games WITHOUT scores.
    
    For basketball, the season typically runs from November to April (next year).
    start_month: First month to scrape (default: 11 for November)
    end_month: Last month to scrape (default: 4 for April, in the next year)
    """
    file_str = f"schedule_{year}_{sport}_{division}.csv"
    
    # Generate date range for the season
    start_date = datetime(year, start_month, 1)
    end_date = datetime(year + 1, end_month, 30) if end_month < start_month else datetime(year, end_month, 30)
    
    all_data = []
    date_range = pd.date_range(start_date, end_date)
    
    logging.info(f"Starting to scrape {len(date_range)} dates for scheduled games: {sport} {division} {year}")
    
    for day in tqdm(date_range, desc="Scraping scheduled games"):
        date_str = day.strftime("%Y-%m-%d")
        data = fetch_schedule_data(sport, division, date_str)
        
        if data is None:
            continue
            
        df = parse_scheduled_games(data, date_str)
        
        if not df.empty:
            all_data.append(df)
            logging.info(f"Scraped {len(df)} scheduled games for {date_str}")
    
    # Save results
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        # Sort by date for better organization
        if "date" in full_df.columns:
            full_df = full_df.sort_values("date").reset_index(drop=True)
        
        output_file = DATA_DIR / file_str
        full_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(full_df)} scheduled games to {file_str}")
        print(f"✓ Saved {len(full_df)} scheduled games to {output_file}")
    else:
        logging.warning(f"No scheduled games found for {sport} {division} {year}")
        print(f"⚠ No scheduled games found for {sport} {division} {year}")


def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA basketball scheduled games (future matchups without scores).")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", 
                       help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", 
                       help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, 
                       help="Starting year of the season to scrape (e.g., 2024 for 2024-2025 season)")
    parser.add_argument("--start-month", type=int, default=11, 
                       help="Starting month (default: 11 for November)")
    parser.add_argument("--end-month", type=int, default=4, 
                       help="Ending month in the next year (default: 4 for April)")

    args = parser.parse_args()

    logging.basicConfig(
        filename=LOG_DIR / f"schedule_{args.year}_{args.sport}_{args.division}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    scrape_schedule(args.sport, args.division, args.year, args.start_month, args.end_month)


if __name__ == "__main__":
    main()

