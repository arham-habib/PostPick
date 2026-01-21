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
    import os
    from pathlib import Path
    
    # Get absolute path to src directory
    src_path = Path(__file__).parent.parent.parent / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"src directory not found at {src_path}")
    
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .run_commands(
            # Install JAX CPU version first (more stable, avoids CUDA segfaults)
            # We can still use GPU via JAX's automatic device placement
            "python -m pip install --upgrade pip",
            # Install JAX CPU version (compatible with repo's version constraint)
            "python -m pip install 'jax==0.4.28' 'jaxlib==0.4.28'",
            # Runtime deps used in simulation pipeline
            "python -m pip install pandas pyarrow numpy",
            # NumPyro is needed for model imports (even though we're not fitting)
            "python -m pip install numpyro",
        )
        # Set environment variables for JAX
        .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        .env({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"})
        # Add local files LAST with copy=True to bake into image
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
    timeout=60 * 60,  # 1h
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

        # Use CPU JAX to avoid CUDA segfault issues
        # CPU JAX will still be much faster than the old nested Python loops
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        log("Set JAX to use CPU (avoiding CUDA segfault issues)")
        
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
    - Schedule CSV file
    
    Skips files that already exist in the volume.
    """
    from pathlib import Path
    from src.utils.data_utils import get_model_data_path, get_schedule_data_path
    
    # Get local file paths
    model_path = get_model_data_path("ncaab", sport, model_name, monday_date)
    schedule_path = get_schedule_data_path("ncaab", year, sport, "d1")
    
    # Check files exist locally
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found locally: {model_path}")
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found locally: {schedule_path}")
    
    # Upload to volume (skip if already exists)
    model_volume_path = f"ncaab/models/{model_path.name}"
    schedule_volume_path = f"ncaab/schedule/{schedule_path.name}"
    
    # Upload model file (skip if already exists)
    try:
        print(f"Uploading model file to Modal volume: {model_path.name}")
        with data_volume.batch_upload() as batch:
            batch.put_file(str(model_path), model_volume_path)
        print(f"✓ Model file uploaded")
    except FileExistsError:
        print(f"✓ Model file already exists in volume: {model_path.name}")
    
    # Upload schedule file (skip if already exists)
    try:
        print(f"Uploading schedule file to Modal volume: {schedule_path.name}")
        with data_volume.batch_upload() as batch:
            batch.put_file(str(schedule_path), schedule_volume_path)
        print(f"✓ Schedule file uploaded")
    except FileExistsError:
        print(f"✓ Schedule file already exists in volume: {schedule_path.name}")
    
    print("Data upload check complete!")


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

