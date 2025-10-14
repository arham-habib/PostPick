#!/bin/bash

for sport in men women; do
    for year in 2021 2022 2023 2024; do
        echo "Scraping $sport D1 schedule data for year $year..."
        uv run python src/ncaa/schedule_scraper.py --sport $sport --division d1 --year $year
    done
done

