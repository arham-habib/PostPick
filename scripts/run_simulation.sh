#!/bin/bash
# Convenience script to run game simulations

# Default values
SPORT="men"
DIVISION="d1"
YEAR=$(date +%Y)
N_DRAWS=1000000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sport)
            SPORT="$2"
            shift 2
            ;;
        --division)
            DIVISION="$2"
            shift 2
            ;;
        --year)
            YEAR="$2"
            shift 2
            ;;
        --n-draws)
            N_DRAWS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--sport men|women] [--division d1|d2|d3] [--year YEAR] [--n-draws N]"
            echo "  --sport: Sport category (default: men)"
            echo "  --division: NCAA division (default: d1)"
            echo "  --year: Season year (default: current year)"
            echo "  --n-draws: Number of draws per sample (default: 1000000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -f "activate.sh" ]; then
    source activate.sh
fi

# Run simulation
python -m src.ncaa.simulate_games \
    --sport "$SPORT" \
    --division "$DIVISION" \
    --year "$YEAR" \
    --n-draws "$N_DRAWS"
