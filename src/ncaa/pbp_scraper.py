import os
import json
import time
import logging
import aiohttp
import asyncio
import pandas as pd
import argparse
from tqdm.asyncio import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiolimiter import AsyncLimiter

BASE_URL = "https://ncaa-api.henrygd.me/game/{}/play-by-play"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
LOG_DIR = SCRIPT_DIR.parent / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiter (4 requests per second)
RATE_LIMITER = AsyncLimiter(4, 1)

async def fetch_play_by_play(session: aiohttp.ClientSession, game_id: int) -> Optional[Dict[str, Any]]:
    """Fetch play-by-play data for a given game asynchronously."""
    url = BASE_URL.format(game_id)
    retries = 3
    backoff = 1

    for attempt in range(retries):
        async with RATE_LIMITER:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 404:
                        logging.warning(f"No play-by-play data available for {game_id}.")
                        return None
                    response.raise_for_status()
                    data = await response.json()
                    if not data or "periods" not in data:
                        logging.warning(f"No play-by-play data available for game {game_id}.")
                        return None
                    return data
            except asyncio.TimeoutError:
                logging.warning(f"Timeout on attempt {attempt + 1} for game {game_id}. Retrying...")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except aiohttp.ClientError as e:
                logging.warning(f"Attempt {attempt + 1}: Failed to fetch {game_id}: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2

    logging.error(f"Failed to fetch play-by-play data for game {game_id} after {retries} attempts.")
    return None

async def parse_play_by_play(data: Dict[str, Any], game_id: int, date: str) -> pd.DataFrame:
    """Extract play-by-play data from the API response."""
    if not data or "periods" not in data:
        return pd.DataFrame()

    play_list = []
    meta = data.get("meta", {})
    
    # Get team information
    teams = {team["id"]: team for team in meta.get("teams", [])}
    
    # Process each period
    for period in data["periods"]:
        period_number = period.get("periodNumber", "Unknown")
        period_display = period.get("periodDisplay", period_number)
        
        # Process each play in the period
        for play in period.get("playStats", []):
            play_list.append({
                "gameID": game_id,
                "date": date,
                "periodNumber": period_number,
                "periodDisplay": period_display,
                "time": play.get("time", ""),
                "score": play.get("score", ""),
                "visitorText": play.get("visitorText", ""),
                "homeText": play.get("homeText", ""),
                "homeTeam": meta.get("teams", [{}])[0].get("shortName", "Unknown") if meta.get("teams") else "Unknown",
                "visitorTeam": meta.get("teams", [{}])[1].get("shortName", "Unknown") if len(meta.get("teams", [])) > 1 else "Unknown",
                "gameStatus": meta.get("status", "Unknown"),
                "division": meta.get("division", "Unknown"),
                "title": meta.get("title", ""),
                "description": meta.get("description", "")
            })

    return pd.DataFrame(play_list)

async def scrape_play_by_play(sport: str, division: str, year: int, game_id: Optional[int] = None):
    """Scrape play-by-play data asynchronously."""
    game_file = DATA_DIR / f"ncaab_{year}_{sport}_{division}.csv"
    if not game_file.exists():
        logging.error(f"Game data file not found: {game_file}")
        raise FileNotFoundError(f"Missing game data file: {game_file}")

    game_data = pd.read_csv(game_file)
    game_ids = game_data[game_data["url"] == str(game_id)] if game_id else game_data

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_play_by_play(session, row["url"]) for _, row in game_ids.iterrows()]
        responses = await tqdm.gather(*tasks)

    all_data = []
    for response, (_, row) in zip(responses, game_ids.iterrows()):
        if response:
            df = await parse_play_by_play(response, row["url"], row["date"])
            if not df.empty:
                all_data.append(df)
        else:
            logging.warning(f"No play-by-play data for game {row['url']}.")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        output_file = DATA_DIR / f"play_by_play_{sport}_{year}_{division}.csv"
        full_df.to_csv(output_file, index=False)
        logging.info(f"Saved all play-by-play data ({len(full_df)} rows) to {output_file}.")
    else:
        logging.warning("No play-by-play data was collected.")


async def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA play-by-play data asynchronously.")
    parser.add_argument("--sport", type=str, choices=["men", "women"], default="men", help="Sport category: men or women (default: men)")
    parser.add_argument("--division", type=str, choices=["d1", "d2", "d3"], default="d1", help="NCAA division (default: d1)")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape data for")
    parser.add_argument("--game_id", type=int, help="Optional game ID for testing a single game")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        filename=LOG_DIR / f"play_by_play_{args.year}_{args.sport}_{args.division}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    await scrape_play_by_play(args.sport, args.division, args.year, args.game_id)

if __name__ == "__main__":
    asyncio.run(main())