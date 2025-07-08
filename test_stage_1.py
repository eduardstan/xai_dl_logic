import time
import shutil
from pathlib import Path
from research_analysis.config.loader import load_config
from research_analysis.stages.stage_1_topic_model import run_stage_1
from research_analysis.utils.logging import setup_logging

# --- Test Setup ---
# 1. Load config
config = load_config()

# 2. Create a timestamped output directory for this run
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = Path(config.pipeline.paths.outputs) / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Set up logging for the test
log_file = output_dir / config.pipeline.paths.logs / "stage_1_test.log"
setup_logging(level=config.pipeline.logging.level, log_file=log_file)

# 4. Simulate pre-existing cache by copying from legacy
cache_dir = output_dir / config.pipeline.paths.cache
cache_dir.mkdir(parents=True, exist_ok=True)
legacy_cache_dir = Path("legacy/cache")

print(f"Pre-populating cache in {cache_dir}...")
for legacy_cache_file in legacy_cache_dir.glob("*.pkl"):
    print(f"  -> Copying {legacy_cache_file.name}")
    shutil.copy(legacy_cache_file, cache_dir)

# --- Run the Test ---
try:
    print("🚀 Starting Stage 1 test run...")
    run_stage_1(config, output_dir)
    print(f"✅ Stage 1 test run completed successfully! Check outputs in: {output_dir}")
except Exception as e:
    print(f"❌ Stage 1 test run failed: {e}")
    # Print traceback for debugging
    import traceback
    traceback.print_exc()

