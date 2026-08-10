#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/.pids"
CONDA_ENV_NAME="${GEN6D_CONDA_ENV:-gen6d}"
PYTHON=""

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-19000}"
MODEL_DIR="${MODEL_DIR:-/home/ubuntu/stephen/02-weight/depth-anything/DA3-SMALL}"
DEVICE="${DEVICE:-cuda}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"

# ~/.local 里的旧 nvidia-nccl-cu12 会抢先加载，导致
# libtorch_cuda.so: undefined symbol: ncclCommResume
# conda activate 的 activate.d 也会设置；这里再强制一层，确保 nohup 子进程继承。
export PYTHONNOUSERSITE=1

mkdir -p "${LOG_DIR}" "${PID_DIR}"

activate_gen6d() {
    if [[ -n "${GEN6D_PYTHON:-}" ]]; then
        PYTHON="${GEN6D_PYTHON}"
        return
    fi

    local conda_base=""
    if [[ -n "${CONDA_EXE:-}" ]]; then
        conda_base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
    fi
    if [[ -z "${conda_base}" ]]; then
        local candidate
        for candidate in /home/ubuntu/miniconda3 "${HOME}/miniconda3" "${HOME}/anaconda3"; do
            if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
                conda_base="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        echo "conda not found. Install miniconda or set GEN6D_PYTHON=/path/to/python"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    PYTHON="$(command -v python)"
    echo "[env] conda activate ${CONDA_ENV_NAME} -> ${PYTHON}"
}

get_lan_ip() {
    hostname -I 2>/dev/null | awk '{print $1}'
}

is_running() {
    local pid_file="$1"
    [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

port_in_use() {
    ss -tln | grep -q ":${1} "
}

wait_for_port() {
    local name="$1"
    local port="$2"
    local timeout="$3"
    local elapsed=0

    echo -n "[wait] ${name} ready on port ${port}"
    while (( elapsed < timeout )); do
        if port_in_use "${port}"; then
            echo " ok (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo " timeout"
    echo "[warn] ${name} not listening on port ${port} within ${timeout}s"
    echo "       check log: ${LOG_DIR}/${name}.log"
    return 1
}

free_port() {
    local port="$1"
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    if [[ -n "${pids}" ]]; then
        echo "[warn] port ${port} in use by pid(s): ${pids}"
        echo "       run 'bash start.sh stop' first, or: kill ${pids}"
    fi
}

start_service() {
    local name="$1"
    local pid_file="${PID_DIR}/${name}.pid"
    local log_file="${LOG_DIR}/${name}.log"
    shift

    if is_running "${pid_file}"; then
        echo "[skip] ${name} already running (pid $(cat "${pid_file}"))"
        return
    fi

    echo "[start] ${name}"
    : >"${log_file}"
    nohup "$@" >>"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
    echo "       pid=$(cat "${pid_file}") log=${log_file}"
}

stop_service() {
    local name="$1"
    local pid_file="${PID_DIR}/${name}.pid"

    if ! is_running "${pid_file}"; then
        rm -f "${pid_file}"
        echo "[skip] ${name} not running"
        return
    fi

    echo "[stop] ${name} (pid $(cat "${pid_file}"))"
    kill "$(cat "${pid_file}")" 2>/dev/null || true
    rm -f "${pid_file}"
}

start_all() {
    cd "${ROOT_DIR}"

    activate_gen6d
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Python not found: ${PYTHON}"
        echo "Set GEN6D_PYTHON or ensure conda env '${CONDA_ENV_NAME}' exists."
        exit 1
    fi

    free_port "${API_PORT}"

    start_service "server" \
        "${PYTHON}" run_server.py \
        --host "${API_HOST}" --port "${API_PORT}" \
        --model-dir "${MODEL_DIR}" --device "${DEVICE}"

    echo
    echo "Loading model (~30-120s on first start)..."
    wait_for_port "server" "${API_PORT}" "${STARTUP_TIMEOUT}" || true

    local lan_ip
    lan_ip="$(get_lan_ip)"
    echo
    echo "Gen6D started"
    if [[ -n "${lan_ip}" ]]; then
        echo "  UI:  http://${lan_ip}:${API_PORT}/ui"
        echo "  API: http://${lan_ip}:${API_PORT}/docs"
    fi
    echo "  UI:  http://127.0.0.1:${API_PORT}/ui"
    echo "  API: http://127.0.0.1:${API_PORT}/docs"
    echo
    echo "Logs: ${LOG_DIR}/server.log"
    echo "Stop:  bash start.sh stop"
}

stop_all() {
    stop_service "server"
    stop_service "ui"
    stop_service "api"
    echo "Gen6D stopped"
}

status_all() {
    local lan_ip
    lan_ip="$(get_lan_ip)"
    local pid_file="${PID_DIR}/server.pid"
    if is_running "${pid_file}"; then
        if port_in_use "${API_PORT}"; then
            echo "[running] server pid=$(cat "${pid_file}") port=${API_PORT}"
        else
            echo "[loading] server pid=$(cat "${pid_file}") port=${API_PORT} (model loading...)"
        fi
    else
        echo "[stopped] server"
    fi
    if [[ -n "${lan_ip}" ]]; then
        echo "  UI:  http://${lan_ip}:${API_PORT}/ui"
        echo "  API: http://${lan_ip}:${API_PORT}/docs"
    fi
}

case "${1:-start}" in
    start) start_all ;;
    stop) stop_all ;;
    restart) stop_all; sleep 2; start_all ;;
    status) status_all ;;
    *)
        echo "Usage: bash start.sh [start|stop|restart|status]"
        exit 1
        ;;
esac
