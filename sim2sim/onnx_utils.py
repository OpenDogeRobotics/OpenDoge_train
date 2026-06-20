"""
ONNX model path resolution for sim2sim.

Priority: CLI --onnx argument > OPENDOGE_ONNX_PATH env var > latest glob match.

Adapted from OpenDoge_deploy; now uses LEGGED_GYM_ROOT_DIR as the search base.
"""

import glob
import os

from legged_gym import LEGGED_GYM_ROOT_DIR


def _resolve_and_validate(path, source):
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"ONNX path from {source} does not exist: {resolved}")
    return resolved


def resolve_onnx_path(cli_onnx=None, env_vars=None, onnx_glob="flat_opendoge*.onnx",
                      robot_name="opendoge", root_dir=None):
    """
    Resolve ONNX path with priority: CLI > env var > latest matched glob.

    Parameters
    ----------
    cli_onnx : str or None
        Path supplied via --onnx command-line argument.
    env_vars : list[str] or None
        Environment variable names to inspect (e.g. ["OPENDOGE_ONNX_PATH"]).
    onnx_glob : str
        Glob pattern relative to ``root_dir/onnx/``.
    robot_name : str
        Used in error messages only.
    root_dir : str or None
        Search root. Defaults to ``LEGGED_GYM_ROOT_DIR``.

    Returns
    -------
    str
        Absolute path to the ONNX model file.
    """
    if root_dir is None:
        root_dir = LEGGED_GYM_ROOT_DIR
    if env_vars is None:
        env_vars = ["OPENDOGE_ONNX_PATH"]

    if cli_onnx:
        return _resolve_and_validate(cli_onnx, "--onnx")

    for env_name in env_vars:
        env_value = os.environ.get(env_name)
        if env_value:
            return _resolve_and_validate(env_value, f"${env_name}")

    pattern = os.path.join(root_dir, "onnx", onnx_glob)
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if candidates:
        # Newest by mtime; tie-break by basename for deterministic behaviour.
        candidates.sort(key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
        return candidates[-1]

    raise FileNotFoundError(
        f"No ONNX found for {robot_name}. "
        f"Checked env vars {env_vars} and glob {pattern}. "
        f"Please pass --onnx /abs/path/model.onnx."
    )
