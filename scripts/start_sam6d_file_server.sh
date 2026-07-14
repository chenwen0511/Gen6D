#!/usr/bin/env bash
# 在 SAM-6D 实际输出目录启动 HTTP 文件服务（默认 :8005）
# Gen6D 通过 grasp_config.json 的 pem.sam6d_file_server_url 访问
set -euo pipefail

PORT="${SAM6D_FILE_SERVER_PORT:-8005}"
ROOT="${SAM6D_SAM_OUTPUT_ROOT:-/home/mui/projects/smt/SAM-6D/SAM-6D/service_outputs}"
LOG="${SAM6D_FILE_SERVER_LOG:-/tmp/sam6d_file_server_${PORT}.log}"
PID_FILE="/tmp/sam6d_file_server_${PORT}.pid"
RUN_USER="${SAM6D_FILE_SERVER_USER:-mui}"

if ss -tln | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已被占用（可能已在运行）"
  ss -tlnp | grep ":${PORT} " || true
  exit 0
fi

_start() {
  cd "${ROOT}"
  nohup python3 -m http.server "${PORT}" >>"${LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
}

echo "启动 SAM-6D 文件服务"
echo "  user=${RUN_USER}"
echo "  root=${ROOT}"
echo "  port=${PORT}"
echo "  log=${LOG}"

if [[ "$(id -un)" == "${RUN_USER}" ]]; then
  _start
elif id "${RUN_USER}" &>/dev/null; then
  sudo -u "${RUN_USER}" bash -c "
    cd '${ROOT}' &&
    nohup python3 -m http.server '${PORT}' >>'${LOG}' 2>&1 &
    echo \$! > '${PID_FILE}'
  "
else
  echo "用户 ${RUN_USER} 不存在，尝试当前用户启动..."
  _start
fi

echo "pid=$(cat "${PID_FILE}")"
echo "测试: curl -I http://127.0.0.1:${PORT}/"
echo "Gen6D 配置: pem.sam6d_file_server_url = http://192.168.100.220:${PORT}"
