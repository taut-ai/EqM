#!/usr/bin/env bash
set -euo pipefail

### CONFIG #############################################################

MINICONDA_DIR="${HOME}/miniconda3"
EQM_UPSTREAM_URL="https://github.com/raywang4/EqM.git"
EQM_ENV_NAME="EqM"   # Assumes environment.yml names it "EqM"

#######################################################################
# 0. Validate /opt/imagenet_hf
#######################################################################

if [ ! -d /opt/imagenet_hf ]; then
  cat >&2 <<'EOF'
ERROR: /opt/imagenet_hf does not exist.

Please create it (once) with:

  sudo mkdir -p /opt/imagenet_hf && \
  sudo chown -R $(whoami):$(whoami) /opt/imagenet_hf && \
  sudo chmod -R 777 /opt/imagenet_hf

Then re-run this script (no sudo required).
EOF
  exit 1
fi

#######################################################################
# 1. Basic dependency checks
#######################################################################

for cmd in wget git gcc; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command '$cmd' not found in PATH. Please install it and re-run." >&2
    exit 1
  fi
done

#######################################################################
# 2. Ensure conda is installed (or install Miniconda non-interactively)
#######################################################################

CONDA_EXE=""
if command -v conda >/dev/null 2>&1; then
  # Use existing conda
  CONDA_EXE="$(command -v conda)"
else
  # Check if a previous Miniconda install exists at $MINICONDA_DIR
  if [ -x "${MINICONDA_DIR}/bin/conda" ]; then
    CONDA_EXE="${MINICONDA_DIR}/bin/conda"
  else
    echo "conda not found; installing Miniconda to ${MINICONDA_DIR}..."

    cd "${HOME}"
    INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

    if [ ! -f "${INSTALLER}" ]; then
      wget "https://repo.anaconda.com/miniconda/${INSTALLER}"
    fi

    bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"

    CONDA_EXE="${MINICONDA_DIR}/bin/conda"
  fi
fi

if [ ! -x "${CONDA_EXE}" ]; then
  echo "ERROR: conda executable not found after installation attempt." >&2
  exit 1
fi

#######################################################################
# 3. Initialize conda for this script's shell (no .bashrc changes)
#######################################################################

__conda_setup="$("${CONDA_EXE}" shell.bash hook 2> /dev/null)"
eval "${__conda_setup}"
unset __conda_setup

#######################################################################
# 4. Confirm we are inside the EqM repository
#######################################################################

REPO_ROOT="$(pwd)"

if [ ! -f "${REPO_ROOT}/environment.yml" ] || [ ! -d "${REPO_ROOT}/.git" ]; then
  cat >&2 <<'EOF'
ERROR: Please run this script from the root of your already-cloned EqM repo.

Example:
  git clone https://github.com/taut-ai/EqM.git
  cd EqM
  ./install.sh
EOF
  exit 1
fi

#######################################################################
# 5. Ensure upstream remote exists
#######################################################################

if git remote | grep -qx 'upstream'; then
  echo "Git remote 'upstream' already exists; skipping add."
else
  echo "Adding upstream remote ${EQM_UPSTREAM_URL}..."
  git remote add upstream "${EQM_UPSTREAM_URL}"
fi

echo
echo "NOTE: To merge upstream changes in the future, you can run:"
echo "  git checkout main"
echo "  git fetch upstream && git merge upstream/main && git push"
echo

#######################################################################
# 6. Create / update conda environment from environment.yml
#######################################################################

echo "Ensuring conda environment '${EQM_ENV_NAME}' exists..."

if conda env list | awk '{print $1}' | grep -qx "${EQM_ENV_NAME}"; then
  echo "Environment '${EQM_ENV_NAME}' already exists; updating from environment.yml..."
  conda env update -f environment.yml -n "${EQM_ENV_NAME}" --prune -y
else
  echo "Creating environment '${EQM_ENV_NAME}' from environment.yml..."
  conda env create -f environment.yml -n "${EQM_ENV_NAME}" -y
fi

echo "Activating environment '${EQM_ENV_NAME}'..."
conda activate "${EQM_ENV_NAME}"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: CONDA_PREFIX is not set after activation; something is wrong with conda." >&2
  exit 1
fi

#######################################################################
# 7. Install extra Python deps
#######################################################################

echo "Installing additional Python dependencies into '${EQM_ENV_NAME}'..."

pip install --upgrade pip
pip install datasets
conda install -y -c conda-forge matplotlib scikit-learn

#######################################################################
# 8. Create stub libittnotify.so and preload hooks
#######################################################################

echo "Creating libittnotify.so stub in ${CONDA_PREFIX}/lib ..."

mkdir -p "${CONDA_PREFIX}/lib"

cat > "${CONDA_PREFIX}/lib/itt_stub.c" << 'EOF'
#ifdef __cplusplus
extern "C" {
#endif

void iJIT_NotifyEvent(...) {}
void iJIT_NotifyEventW(...) {}
int  iJIT_IsProfilingActive(void) { return 0; }
int  iJIT_GetNewMethodID(void) { return 1; }
int  iJIT_GetNewMethodIDEx(void) { return 1; }
void iJIT_NotifyEventStr(...) {}
void iJIT_NotifyEventEx(...) {}

#ifdef __cplusplus
}
#endif
EOF

gcc -shared -fPIC -O2 -o "${CONDA_PREFIX}/lib/libittnotify.so" "${CONDA_PREFIX}/lib/itt_stub.c"

echo "Setting up conda activate/deactivate hooks for LD_PRELOAD..."

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" \
         "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat > "${CONDA_PREFIX}/etc/conda/activate.d/itt_preload.sh" << 'EOF'
export _OLD_LD_PRELOAD="${LD_PRELOAD:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so${LD_PRELOAD:+:$LD_PRELOAD}"
EOF

cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/itt_preload.sh" << 'EOF'
export LD_PRELOAD="${_OLD_LD_PRELOAD}"
unset _OLD_LD_PRELOAD
EOF

#######################################################################
# 9. Set dummy Weights & Biases env vars via conda hooks
#######################################################################

echo "Configuring dummy Weights & Biases environment variables..."

cat > "${CONDA_PREFIX}/etc/conda/activate.d/wandb_env.sh" << 'EOF'
export ENTITY="local"
export PROJECT="EqM-test"
export WANDB_KEY="dummy"
EOF

cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/wandb_env.sh" << 'EOF'
unset ENTITY
unset PROJECT
unset WANDB_KEY
EOF

#######################################################################
# 10. Done
#######################################################################

cat <<EOF

============================================================
Setup complete.

Conda env:   ${EQM_ENV_NAME}
Repo path:   ${REPO_ROOT}
ImageNet dir: /opt/imagenet_hf (verified existing)

From a new shell, you can do:

  cd ${REPO_ROOT}
  conda activate ${EQM_ENV_NAME}

LD_PRELOAD and WANDB dummy env vars will be handled automatically
when you activate/deactivate the '${EQM_ENV_NAME}' environment.

============================================================
EOF
