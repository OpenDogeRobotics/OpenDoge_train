#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OpenDoge 一键环境安装脚本
# 用法:
#   bash setup_env.sh <isaacgym_path>
#
# 示例:
#   bash setup_env.sh /home/lain/IsaacGym_Preview_4_Package
# ============================================================

ISAACGYM_PATH="${1:-${ISAACGYM_PATH:-}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="himloco"
CONDA_ENV_DIR=""

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*"; }

abort() { err "$@"; exit 1; }

# ============================================================
# 1. 检查前置条件
# ============================================================
check_prereqs() {
  info "检查前置条件..."

  if ! command -v conda &>/dev/null; then
    abort "conda 未安装或不在 PATH 中，请先安装 Miniconda"
  fi
  info "  conda   ok"

  if ! command -v nvidia-smi &>/dev/null; then
    abort "nvidia-smi 未找到，请检查 NVIDIA 驱动是否安装"
  fi
  info "  nvidia  ok ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"

  if [ -z "$ISAACGYM_PATH" ]; then
    # 尝试常见路径
    for guess in \
      /home/lain/IsaacGym_Preview_4_Package \
      "$HOME/IsaacGym_Preview_4_Package" \
      /opt/isaacgym; do
      if [ -f "$guess/isaacgym/python/setup.py" ]; then
        ISAACGYM_PATH="$guess"
        break
      fi
    done
  fi

  if [ -z "$ISAACGYM_PATH" ] || [ ! -f "$ISAACGYM_PATH/isaacgym/python/setup.py" ]; then
    err "未找到 Isaac Gym，请指定路径："
    err "  bash setup_env.sh <isaacgym_path>"
    err ""
    err "从 NVIDIA 官网下载 Isaac Gym Preview 4:"
    err "  https://developer.nvidia.com/isaac-gym"
    exit 1
  fi
  info "  isaacgym  $ISAACGYM_PATH"
}

# ============================================================
# 2. 创建 conda 环境
# ============================================================
ensure_env() {
  CONDA_ENV_DIR="$(conda env list 2>/dev/null | grep "^${ENV_NAME} " | awk '{print $NF}' || true)"
  if [ -z "$CONDA_ENV_DIR" ]; then
    CONDA_ENV_DIR="$(conda info --base)/envs/${ENV_NAME}"
  fi

  if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    info "环境 $ENV_NAME 已存在: $CONDA_ENV_DIR"
  else
    info "创建 conda 环境: $ENV_NAME ..."
    conda env create -f "$SCRIPT_DIR/himloco.yml"
    CONDA_ENV_DIR="$(conda env list 2>/dev/null | grep "^${ENV_NAME} " | awk '{print $NF}')"
    if [ -z "$CONDA_ENV_DIR" ]; then
      CONDA_ENV_DIR="$(conda info --base)/envs/${ENV_NAME}"
    fi
    info "  环境路径: $CONDA_ENV_DIR"
  fi
}

# ============================================================
# 3. 安装 Isaac Gym
# ============================================================
install_isaacgym() {
  info "安装 Isaac Gym..."

  conda run -n "$ENV_NAME" pip install -e "$ISAACGYM_PATH/isaacgym/python"
  info "  Isaac Gym 安装完成"
}

# ============================================================
# 4. 安装项目依赖
# ============================================================
install_project() {
  info "安装项目依赖..."

  conda run -n "$ENV_NAME" pip install -e "$SCRIPT_DIR"
  info "  项目依赖安装完成"
}

# ============================================================
# 5. 配置环境变量 (仅 himloco 环境生效)
# ============================================================
setup_hook() {
  info "配置环境变量 hook..."

  local hook_dir="$CONDA_ENV_DIR/etc/conda/activate.d"
  mkdir -p "$hook_dir"

  cat > "$hook_dir/env_vars.sh" << HOOKEOF
export LD_LIBRARY_PATH=${CONDA_ENV_DIR}/lib:${ISAACGYM_PATH}/isaacgym/lib:\$LD_LIBRARY_PATH
HOOKEOF

  info "  已写入: $hook_dir/env_vars.sh"
}

# ============================================================
# 6. 验证环境
# ============================================================
verify_env() {
  info "验证环境..."

  conda run -n "$ENV_NAME" python -c "
import isaacgym
print(f'  IsaacGym:     ok')

import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'  PyTorch CUDA: {torch.cuda.is_available()} (v{torch.version.cuda})')

import mujoco
print(f'  MuJoCo:       {mujoco.__version__}')
"

  info "======================================"
  info "  环境就绪！"
  info ""
  info "  激活环境:  conda activate $ENV_NAME"
  info "  开始训练:"
  info "    cd $(realpath "$SCRIPT_DIR")"
  info "    export PYTHONPATH=\$PWD"
  info "    python legged_gym/scripts/train.py --task=opendoge --headless"
  info "======================================"
}

# ============================================================
# main
# ============================================================
main() {
  echo ""
  info "OpenDoge 环境安装开始"
  info "====================="
  check_prereqs
  ensure_env
  install_isaacgym
  install_project
  setup_hook
  verify_env
}

main
