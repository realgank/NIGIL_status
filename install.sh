#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="NIGIL Discord Bot"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
SERVICE_NAME="nigil-status-bot"

run_root() {
  if command -v sudo >/dev/null 2>&1 && [ "${SUDO_USER-}" != "root" ] && [ "$(id -u)" -ne 0 ]; then
    sudo "$@"
  else
    "$@"
  fi
}

printf '\n=== Установщик %s ===\n\n' "$PROJECT_NAME"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Ошибка] Python 3 не найден. Установите его командой: sudo apt-get install -y python3 python3-venv python3-pip" >&2
  exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "Устанавливаю python3-pip (потребуется sudo)..."
    run_root apt-get update
    run_root apt-get install -y python3-pip
  else
    echo "[Ошибка] pip3 не найден и не удалось автоматически установить." >&2
    exit 1
  fi
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "Устанавливаю python3-venv (потребуется sudo)..."
    run_root apt-get install -y python3-venv
  else
    echo "[Ошибка] Модуль venv недоступен. Установите пакет python3-venv." >&2
    exit 1
  fi
fi

cd "$SCRIPT_DIR"

normalize_requirements_url() {
  local input="$1"

  if [[ "$input" =~ ^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$ ]]; then
    printf 'https://raw.githubusercontent.com/%s/%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  elif [[ "$input" =~ ^https://github\.com/([^/]+)/([^/]+)/raw/(.+)$ ]]; then
    printf 'https://raw.githubusercontent.com/%s/%s/%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  else
    printf '%s\n' "$input"
  fi
}

validate_requirements_file() {
  local path="$1"

  if [ ! -s "$path" ]; then
    echo "[Ошибка] Получен пустой файл requirements.txt." >&2
    rm -f "$path"
    return 1
  fi

  local head_sample
  head_sample="$(head -c 512 "$path" 2>/dev/null || true)"

  if [[ "$head_sample" =~ <!DOCTYPE[[:space:]]+html ]] || \
     [[ "$head_sample" =~ <html ]] || \
     [[ "$head_sample" =~ <head ]] || \
     [[ "$head_sample" =~ <body ]] || \
     [[ "$head_sample" =~ \{"message" ]] || \
     [[ "$head_sample" =~ ^[[:space:]]*404: ]] || \
     [[ "$head_sample" =~ ^[[:space:]]*403: ]] || \
     [[ "$head_sample" =~ ^[[:space:]]*Not[[:space:]]+Found ]] || \
     [[ "$head_sample" =~ ^[[:space:]]*<![[:upper:]] ]] || \
     [[ "$head_sample" =~ ^<[[:alpha:]] ]] ; then
    echo "[Ошибка] Получен HTML или ответ об ошибке вместо requirements.txt. Укажите raw-ссылку на файл." >&2
    rm -f "$path"
    return 1
  fi

  return 0
}

download_requirements() {
  local default_url="https://raw.githubusercontent.com/NIGIL-status/NIGIL_status/main/requirements.txt"
  local url="${REQUIREMENTS_URL-}"

  echo "requirements.txt не найден. Попробую скачать с GitHub..."

  if [ -z "$url" ]; then
    read -rp "Введите URL до raw requirements.txt [$default_url]: " url
    url="${url:-$default_url}"
  fi

  if [ -z "$url" ]; then
    echo "[Ошибка] URL для скачивания requirements.txt не указан." >&2
    return 1
  fi

  local normalized_url
  normalized_url="$(normalize_requirements_url "$url")"
  if [ "$normalized_url" != "$url" ]; then
    echo "Обнаружена ссылка на GitHub. Использую raw-вариант: $normalized_url"
  fi
  url="$normalized_url"

  if command -v curl >/dev/null 2>&1; then
    if curl -fsSL "$url" -o requirements.txt && validate_requirements_file requirements.txt; then
      echo "requirements.txt успешно скачан."
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -qO requirements.txt "$url" && validate_requirements_file requirements.txt; then
      echo "requirements.txt успешно скачан."
      return 0
    fi
  else
    echo "[Ошибка] Не найден curl или wget для скачивания файла." >&2
    return 1
  fi

  echo "[Ошибка] Не удалось скачать requirements.txt по адресу: $url" >&2
  return 1
}

if [ ! -f requirements.txt ]; then
  if ! download_requirements; then
    cat <<'ERR' >&2
[Ошибка] Файл requirements.txt не найден и не удалось скачать автоматически.
Сначала скачайте весь проект (например, через git clone из GitHub),
или укажите корректный URL в переменной REQUIREMENTS_URL перед запуском.
ERR
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  echo "Создаю виртуальное окружение (.venv)..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

DEFAULT_CHANNEL_ID=1375890545364172981
DEFAULT_DB_PATH="$SCRIPT_DIR/nigil_monitor.sqlite3"
DEFAULT_WEEKLY_LIMIT=3
DEFAULT_COOLDOWN=20
DEFAULT_NOTIFY_MIN=60
DEFAULT_AUTO_CLEAN=10
DEFAULT_FUZZY=0.72
DEFAULT_SYSTEM_REGEX='^[A-Za-z0-9\-]{2,12}$'
DEFAULT_TIMEZONE="Europe/Moscow"
DEFAULT_PIN_STATUS=true
DEFAULT_LIVE_POST=true
DEFAULT_TTL=0

prompt_required() {
  local prompt="$1"
  local value
  while true; do
    read -rp "$prompt: " value
    value="${value//[$'\r\n']}"
    if [ -n "$value" ]; then
      printf '%s' "$value"
      return
    fi
    echo "Значение обязательно. Повторите ввод."
  done
}

prompt_with_default() {
  local prompt="$1"
  local default="$2"
  local value
  read -rp "$prompt [$default]: " value
  value="${value//[$'\r\n']}"
  if [ -z "$value" ]; then
    printf '%s' "$default"
  else
    printf '%s' "$value"
  fi
}

prompt_int() {
  local prompt="$1"
  local default="$2"
  local value
  while true; do
    value="$(prompt_with_default "$prompt" "$default")"
    if [[ "$value" =~ ^-?[0-9]+$ ]]; then
      printf '%s' "$value"
      return
    fi
    echo "Введите целое число."
  done
}

prompt_float() {
  local prompt="$1"
  local default="$2"
  local value
  while true; do
    value="$(prompt_with_default "$prompt" "$default")"
    value="${value/,/.}"
    if [[ "$value" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      printf '%s' "$value"
      return
    fi
    echo "Введите число."
  done
}

prompt_bool() {
  local prompt="$1"
  local default="$2"
  local hint
  if [ "$default" = "true" ]; then
    hint="Y/n"
  else
    hint="y/N"
  fi
  local value
  while true; do
    read -rp "$prompt [$hint]: " value
    value="${value//[$'\r\n']}"
    if [ -z "$value" ]; then
      printf '%s' "$default"
      return
    fi
    case "${value,,}" in
      y|yes|д|да|true|1) printf 'true'; return ;;
      n|no|н|нет|false|0) printf 'false'; return ;;
    esac
    echo "Введите y/n или да/нет."
  done
}

echo "Введите параметры бота (значения по умолчанию можно принять клавишей Enter)."

BOT_TOKEN="$(prompt_required "Discord-токен бота")"
CHANNEL_ID="$(prompt_int "ID канала для уведомлений" "$DEFAULT_CHANNEL_ID")"
DB_PATH="$(prompt_with_default "Путь к файлу базы данных" "$DEFAULT_DB_PATH")"
WEEKLY_LIMIT="$(prompt_int "Лимит вызовов в неделю" "$DEFAULT_WEEKLY_LIMIT")"
COOLDOWN_HOURS="$(prompt_int "Кулдаун между вызовами (часы)" "$DEFAULT_COOLDOWN")"
NOTIFY_MIN="$(prompt_int "Период уведомлений (минуты)" "$DEFAULT_NOTIFY_MIN")"
AUTO_CLEAN="$(prompt_int "Авто-очистка сообщений (секунды)" "$DEFAULT_AUTO_CLEAN")"
FUZZY_CUTOFF="$(prompt_float "Минимальный рейтинг fuzzy-сопоставления" "$DEFAULT_FUZZY")"
SYSTEM_REGEX="$(prompt_with_default "Регулярное выражение для имён систем" "$DEFAULT_SYSTEM_REGEX")"
TIMEZONE="$(prompt_with_default "Таймзона" "$DEFAULT_TIMEZONE")"
PIN_STATUS="$(prompt_bool "Закреплять статусное сообщение" "$DEFAULT_PIN_STATUS")"
LIVE_POST="$(prompt_bool "Использовать только живой пост" "$DEFAULT_LIVE_POST")"
COMMAND_TTL="$(prompt_int "TTL ответов на команды (секунды)" "$DEFAULT_TTL")"

echo

if ! command -v systemctl >/dev/null 2>&1; then
  echo "[Предупреждение] systemd недоступен. Пропускаю создание сервиса автозапуска." >&2
  exit 0
fi

if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
  echo "[Ошибка] Для настройки автозапуска требуются права суперпользователя (sudo или запуск от root)." >&2
  exit 1
fi

default_user="$(id -un)"
read -rp "Укажите пользователя для запуска сервиса [${default_user}]: " service_user
service_user="${service_user:-$default_user}"

if ! id "$service_user" >/dev/null 2>&1; then
  echo "[Ошибка] Пользователь ${service_user} не найден в системе." >&2
  exit 1
fi

cat <<EOF

Создаю systemd-сервис ${SERVICE_NAME}.service от имени пользователя ${service_user}.

EOF

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

escape_systemd_value() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//"/\\"}"
  printf '%s' "$value"
}

build_env_line() {
  local name="$1"
  local value="$2"
  printf 'Environment="%s=%s"\n' "$name" "$(escape_systemd_value "$value")"
}

UNIT_ENVIRONMENT=""
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_TOKEN" "$BOT_TOKEN")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_CHANNEL_ID" "$CHANNEL_ID")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_DB_PATH" "$DB_PATH")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_WEEKLY_LIMIT" "$WEEKLY_LIMIT")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_COOLDOWN_HOURS" "$COOLDOWN_HOURS")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_NOTIFY_PERIOD_MIN" "$NOTIFY_MIN")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_AUTO_CLEAN_SECONDS" "$AUTO_CLEAN")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_FUZZY_CUTOFF" "$FUZZY_CUTOFF")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_SYSTEM_NAME_REGEX" "$SYSTEM_REGEX")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_TIMEZONE" "$TIMEZONE")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_PIN_STATUS_MESSAGE" "$PIN_STATUS")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_LIVE_POST_ONLY" "$LIVE_POST")"
UNIT_ENVIRONMENT+="$(build_env_line "NIGIL_COMMAND_REPLY_TTL" "$COMMAND_TTL")"

UNIT_CONTENT="[Unit]
Description=${PROJECT_NAME}
After=network.target

[Service]
Type=simple
User=${service_user}
WorkingDirectory=${SCRIPT_DIR}
${UNIT_ENVIRONMENT}ExecStart="${PYTHON_BIN}" "${SCRIPT_DIR}/NIGIL_status.py" run
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"

echo "${UNIT_CONTENT}" | run_root tee "${SERVICE_FILE}" >/dev/null
run_root systemctl daemon-reload
run_root systemctl enable --now "${SERVICE_NAME}.service"

cat <<MSG

Готово. Сервис ${SERVICE_NAME}.service создан и запущен.
Убедитесь, что переменные окружения в unit-файле заданы корректно.
Проверить состояние: sudo systemctl status ${SERVICE_NAME}.service
Логи: journalctl -u ${SERVICE_NAME}.service -f

Для ручного запуска экспортируйте переменные окружения и выполните:
  export NIGIL_TOKEN='${BOT_TOKEN}'
  export NIGIL_CHANNEL_ID='${CHANNEL_ID}'
  export NIGIL_DB_PATH='${DB_PATH}'
  export NIGIL_WEEKLY_LIMIT='${WEEKLY_LIMIT}'
  export NIGIL_COOLDOWN_HOURS='${COOLDOWN_HOURS}'
  export NIGIL_NOTIFY_PERIOD_MIN='${NOTIFY_MIN}'
  export NIGIL_AUTO_CLEAN_SECONDS='${AUTO_CLEAN}'
  export NIGIL_FUZZY_CUTOFF='${FUZZY_CUTOFF}'
  export NIGIL_SYSTEM_NAME_REGEX='${SYSTEM_REGEX}'
  export NIGIL_TIMEZONE='${TIMEZONE}'
  export NIGIL_PIN_STATUS_MESSAGE='${PIN_STATUS}'
  export NIGIL_LIVE_POST_ONLY='${LIVE_POST}'
  export NIGIL_COMMAND_REPLY_TTL='${COMMAND_TTL}'
  ${PYTHON_BIN} ${SCRIPT_DIR}/NIGIL_status.py run

MSG
