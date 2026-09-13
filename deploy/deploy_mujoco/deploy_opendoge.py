"""Backward-compatible alias for the OpenDoge MuJoCo deploy entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.deploy_mujoco.deploy_mujoco import main


if __name__ == "__main__":
    main()
