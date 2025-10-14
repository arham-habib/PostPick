# NCAA Basketball Scrapers

This directory contains scrapers for NCAA basketball data using the [henrygd/ncaa-api](https://github.com/henrygd/ncaa-api).

## Scrapers Overview

### 1. `schedule_scraper.py` - Future Games (Scheduled Matchups)
**Purpose**: Capture upcoming games that haven't been played yet (no scores).

**What it does**:
- Fetches games from the scoreboard endpoint
- Keeps games **WITHOUT** scores (scheduled/future games)
- Captures matchup information: teams, date, time, venue, network

**When to use**:
- During the active season to capture upcoming matchups
- Before games are played to see the schedule ahead
- To build a future game calendar

**Output**: CSV with scheduled games including:
- Game ID, teams (home/away), date, start time
- Venue, network, conference info
- No scores (games haven't been played)

**Example**:
```bash
# Scrape scheduled games for current season
python src/ncaa/schedule_scraper.py --sport men --division d1 --year 2024
```

---

### 2. `game_scraper.py` - Completed Games (Results)
**Purpose**: Capture game results after games have been played.

**What it does**:
- Fetches games from the scoreboard endpoint
- Keeps games **WITH** scores (completed games)
- Captures final scores and game results

**When to use**:
- After games are played to collect results
- To build historical game databases
- For analysis of completed games

**Output**: CSV with completed games including:
- Game ID, teams (home/away), scores
- Date, start time, final message
- Conference information

**Example**:
```bash
# Scrape completed games for 2023 season
python src/ncaa/game_scraper.py --sport men --division d1 --year 2023
```

---

### 3. `pbp_scraper.py` - Play-by-Play Data
**Purpose**: Capture detailed play-by-play data for completed games.

**What it does**:
- Takes game IDs from `game_scraper.py` output
- Fetches detailed play-by-play for each game
- Captures every action in the game by period

**When to use**:
- After running `game_scraper.py` to get game IDs
- For deep analysis of game flow
- To study specific plays and game dynamics

**Output**: CSV with play-by-play data including:
- Game ID, period, time, score
- Home and visitor actions
- Team stats and game status

**Example**:
```bash
# Scrape play-by-play for all games in 2023 season
python src/ncaa/pbp_scraper.py --sport men --division d1 --year 2023
```

---

## Typical Workflow

### During the Season (Real-time)
1. **Week ahead**: Run `schedule_scraper.py` to see upcoming matchups
2. **After games**: Run `game_scraper.py` to collect results
3. **Deep dive**: Run `pbp_scraper.py` for detailed analysis

### Historical Analysis
1. Run `game_scraper.py` for past seasons to get completed games
2. Run `pbp_scraper.py` on those games for detailed data
3. Note: `schedule_scraper.py` won't find data for past dates (all games completed)

## Batch Scripts

### `scripts/scrape_schedule_data.sh`
Scrapes scheduled games for multiple years and both men's and women's divisions.

### `scripts/scrape_game_data.sh`
Scrapes completed game results for multiple years and both men's and women's divisions.

### `scripts/scrape_player_data.sh`
Scrapes player data (if applicable).

## Output Directories

- **Schedule data**: `data/ncaa/schedule/`
- **Game data**: `data/`
- **Play-by-play data**: `data/`
- **Logs**: `logs/`

## API Reference

Base API: `https://ncaa-api.henrygd.me`

All scrapers use the same scoreboard endpoint with different filters:
```
GET /scoreboard/basketball-{sport}/{division}/{year}/{month}/{day}/all-conf
```

Where:
- `sport`: `men` or `women`
- `division`: `d1`, `d2`, or `d3`
- `date`: `YYYY/MM/DD`

## Rate Limiting

All scrapers implement rate limiting (250ms between requests) to be respectful to the API.

