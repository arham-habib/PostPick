for sport in men women; do
    for year in 2021 2022 2023 2024; do
        python src/scraping/async_player_box_score_scraper.py --sport $sport --division d1 --year $year
    done
done