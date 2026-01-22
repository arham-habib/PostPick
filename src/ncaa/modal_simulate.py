"""
Modal GPU entrypoint for NCAA game simulation.

This module runs the simulation pipeline on a CUDA GPU (A100 recommended),
loading cached posterior samples from `data/ncaab/models/` and writing the
exploded simulation parquet to `data/ncaab/simulations/` (same schema as local).
"""

from __future__ import annotations

from typing import Any, Dict

import modal

# Enable output to see build logs
modal.enable_output()

PROJECT_MOUNT_PATH = "/root/project"


def _build_image() -> modal.Image:
    """
    Build a CUDA-capable image for JAX simulation.

    Notes:
    - We deliberately do NOT install `jax-metal` in the Modal image.
    - We install CUDA-enabled `jaxlib` via JAX's CUDA wheels index.
    - We copy the source code into the image so it's available at runtime.
    """
    from pathlib import Path
    
    # Get absolute path to src directory
    src_path = Path(__file__).parent.parent.parent / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"src directory not found at {src_path}")
    
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        # Install JAX with CUDA support
        # Note: pip_install doesn't support -f flag directly, so we use run_commands for JAX
        .run_commands(
            "python -m pip install --upgrade pip",
            "python -m pip install 'jax[cuda12]==0.4.28' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        )
        # Use pip_install for standard packages (new API)
        .pip_install("pandas", "pyarrow", "numpy", "numpyro")
        .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        .env({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"})
        # Add local files with copy=True to bake into image (still valid in new API)
        .add_local_dir(str(src_path), remote_path=f"{PROJECT_MOUNT_PATH}/src", copy=True)
    )


app = modal.App("postpick-ncaab-sim")
image = _build_image()

# Persistent volume for `data/` (user must hydrate it once from local)
data_volume = modal.Volume.from_name("postpick-data", create_if_missing=True)

# Code is now baked into the image via .copy_local_dir() above
# No need for a separate mount - the code is already in the image at PROJECT_MOUNT_PATH


@app.function(
    gpu="A100",
    image=image,
    volumes={f"{PROJECT_MOUNT_PATH}/data": data_volume},
    timeout=300,  # 5min
)
def _run_simulation_on_gpu(args: Dict[str, Any]) -> str:
    """Run simulation on GPU with error handling and logging."""
    import os
    import sys
    import traceback
    from pathlib import Path
    
    # Write to stderr so it appears in Modal logs
    def log(msg: str) -> None:
        print(f"[GPU-FUNC] {msg}", file=sys.stderr, flush=True)
    
    try:
        log("Starting GPU simulation function...")
        
        # Ensure project path is importable
        # src/ is at /root/project/src, so add /root/project to path
        if PROJECT_MOUNT_PATH not in sys.path:
            sys.path.insert(0, PROJECT_MOUNT_PATH)
        log(f"Python path: {sys.path[:3]}")
        
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        
        # Initialize JAX
        log("Importing JAX...")
        import jax
        import jax.numpy as jnp
        log("JAX imported successfully")
        
        # Check devices
        try:
            devices = jax.devices()
            log(f"JAX devices: {[str(d) for d in devices]}")
            # Do a small test computation
            test = jnp.array([1.0])
            _ = test + 1.0
            log("JAX test computation successful")
        except Exception as e:
            log(f"JAX initialization error: {e}")
            raise

        # Check if src directory exists
        src_dir = Path(f"{PROJECT_MOUNT_PATH}/src")
        log(f"Checking src directory: {src_dir} (exists: {src_dir.exists()})")
        if src_dir.exists():
            log(f"Contents of src: {list(src_dir.iterdir())[:5]}")
        
        log("Importing simulation modules...")
        from src.ncaa.simulate_games import (
            load_model,
            load_schedule,
            filter_next_week_games,
            map_schedule_teams_to_model,
            get_simulation_path,
            simulate_games_to_parquet,
        )
        log("Imports successful")

        log(f"Received args: {list(args.keys())}")
        
        # Initialize JAX before loading model (this might help with segfault)
        log("Initializing JAX...")
        import jax
        import jax.numpy as jnp
        # Force JAX to initialize and see available devices
        try:
            devices = jax.devices()
            log(f"JAX devices: {[str(d) for d in devices]}")
            # Do a simple computation to ensure JAX is working
            test_arr = jnp.array([1.0, 2.0, 3.0])
            log(f"JAX test computation successful: {float(test_arr.sum())}")
        except Exception as e:
            log(f"JAX initialization warning: {e}")
        
        args = args  # appease type checkers; Modal passes a plain dict
        log("Loading model pickle file...")
        # Load model with explicit error handling and careful JAX array conversion
        try:
            from src.ncaa.simulate_games import get_model_path
            import pickle
            
            model_path = get_model_path(str(args["sport"]), str(args["model_name"]), str(args["monday_date"]))
            log(f"Model path: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            log("Reading pickle file...")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            log("Pickle file loaded")
            
            log("Converting numpy arrays to JAX arrays (this may take a moment)...")
            # Convert arrays one at a time to avoid memory issues
            samples = {}
            for key, value in model_data['samples'].items():
                log(f"  Converting {key} (shape: {value.shape if hasattr(value, 'shape') else 'scalar'})...")
                try:
                    # Use device_put to explicitly place on GPU
                    samples[key] = jax.device_put(jnp.array(value))
                    log(f"  ✓ {key} converted")
                except Exception as e:
                    log(f"  ✗ Error converting {key}: {e}")
                    raise
            
            _metadata = model_data['metadata']
            team_to_id = model_data['team_to_id']
            id_to_team = model_data['id_to_team']
            
            log(f"Model loaded: {len(team_to_id)} teams, {samples['alpha'].shape[0]} posterior draws")
        except Exception as e:
            log(f"Error loading model: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            raise

        log("Loading schedule...")
        schedule_df = load_schedule(str(args["sport"]), int(args["year"]))
        log(f"Schedule loaded: {len(schedule_df)} games")
        
        next_week_games = filter_next_week_games(schedule_df, str(args["monday_date"]))
        if next_week_games.empty:
            return "No games in next week; nothing to simulate."

        log(f"Filtered to {len(next_week_games)} games in next week")

        # Filter unknown teams using the cached model's mapping
        valid_games = []
        for _, row in next_week_games.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            if home_team not in team_to_id:
                continue
            if away_team not in team_to_id:
                continue
            valid_games.append(row)

        if not valid_games:
            return "No games with known teams; nothing to simulate."

        log(f"Found {len(valid_games)} games with known teams")
        
        games_with_indices = map_schedule_teams_to_model(
            # preserve columns
            next_week_games.loc[[r.name for r in valid_games]].reset_index(drop=True),
            team_to_id,
        )

        log("Starting simulation...")
        out_path: Path = get_simulation_path(str(args["sport"]), str(args["model_name"]), str(args["monday_date"]))
        simulate_games_to_parquet(
            games_df=games_with_indices,
            samples=samples,
            model_name=str(args["model_name"]),
            n_sims=int(args["n_sims"]),
            draw_block_size=int(args["draw_block_size"]),
            output_path=out_path,
            rng_seed=42,
            sim_logger=None,
        )

        log("Committing volume...")
        # Persist to volume
        data_volume.commit()
        log("Simulation complete!")
        return f"Wrote {out_path}"
    
    except Exception as e:
        error_msg = f"Error in GPU function: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        # Also write to a file on the volume for debugging
        try:
            error_file = Path(f"{PROJECT_MOUNT_PATH}/data/error_log.txt")
            error_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_file, "w") as f:
                f.write(error_msg)
            data_volume.commit()
        except:
            pass
        raise


def _upload_data_to_volume(sport: str, model_name: str, monday_date: str, year: int) -> None:
    """
    Upload required data files to Modal volume before simulation.
    
    Uploads:
    - Model pickle file
    - Game data CSV file (for loading unfinished games/schedule)
    
    Always overwrites existing files to ensure updates propagate.
    """
    from pathlib import Path
    from src.utils.data_utils import get_model_data_path, get_game_data_path
    
    # Get local file paths
    model_path = get_model_data_path("ncaab", sport, model_name, monday_date)
    game_path = get_game_data_path("ncaab", year, sport, "d1")
    
    # Check files exist locally
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found locally: {model_path}")
    if not game_path.exists():
        raise FileNotFoundError(f"Game data file not found locally: {game_path}")
    
    # Define volume paths
    model_volume_path = f"ncaab/models/{model_path.name}"
    game_volume_path = f"ncaab/game/{game_path.name}"
    
    # Upload model file (always overwrite to ensure updates propagate)
    print(f"Uploading model file to Modal volume: {model_path.name} (will overwrite if exists)")
    with data_volume.batch_upload() as batch:
        batch.put_file(str(model_path), model_volume_path)
    print(f"✓ Model file uploaded")
    
    # Upload game data file (always overwrite to ensure updates propagate)
    print(f"Uploading game data file to Modal volume: {game_path.name} (will overwrite if exists)")
    with data_volume.batch_upload() as batch:
        batch.put_file(str(game_path), game_volume_path)
    print(f"✓ Game data file uploaded")
    
    print("Data upload complete! All files updated in Modal volume.")


def run_modal_simulation(
    *,
    sport: str,
    model_name: str,
    year: int,
    monday_date: str,
    n_sims: int,
    draw_block_size: int,
) -> None:
    """
    Submit a Modal GPU run for simulation.

    This is called by `src/ncaa/simulate_games.py` when `--backend modal`.
    """
    # Upload required data files to Modal volume
    _upload_data_to_volume(sport, model_name, monday_date, year)
    
    # Modal App must be running to call .remote()
    with app.run():
        result = _run_simulation_on_gpu.remote(  # type: ignore
            {
                "sport": sport,
                "model_name": model_name,
                "year": year,
                "monday_date": monday_date,
                "n_sims": n_sims,
                "draw_block_size": draw_block_size,
            }
        )
        print(f"Modal simulation result: {result}")
        
        # Download the simulation file from volume to local
        from src.utils.data_utils import get_simulation_data_path
        from pathlib import Path
        import subprocess
        
        sim_filename = f"{model_name}_{monday_date}.parquet"
        volume_path = f"ncaab/simulations/{sim_filename}"
        local_path = get_simulation_data_path("ncaab", sport, model_name, monday_date)
        
        print(f"Downloading simulation file from volume to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download from volume using Modal CLI
        # NOTE: For local development, ensure you are using the full physical paths:
        # modal volume get postpick-data ncaab/simulations/Vanilla_2026-01-19.parquet /Users/arhamhabib/Projects/PostPick/data/ncaab/simulations/Vanilla_2026-01-19.parquet
        try:
            # Always use absolute path for destination to avoid issues
            # If model_name == "Vanilla" and monday_date == "2026-01-19", we want:
            # modal volume get postpick-data ncaab/simulations/Vanilla_2026-01-19.parquet /Users/arhamhabib/Projects/PostPick/data/ncaab/simulations/Vanilla_2026-01-19.parquet
            modal_cmd = [
                "modal", "volume", "get", "postpick-data",
                f"ncaab/simulations/{model_name}_{monday_date}.parquet",
                str(local_path.resolve()),
            ]
            subprocess.run(
                modal_cmd,
                check=True,
                capture_output=True,
            )
            print(f"✓ Simulation file downloaded to {local_path.resolve()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠ Could not automatically download file: {e}")
            print(f"  The file is stored in the Modal volume at: ncaab/simulations/{model_name}_{monday_date}.parquet")
            print("  Download it manually using:")
            print(f"  modal volume get postpick-data ncaab/simulations/{model_name}_{monday_date}.parquet {local_path.resolve()}")
            print("  Or access it via the Modal dashboard")
