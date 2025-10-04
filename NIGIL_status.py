import os
import sys
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import asyncio
import difflib
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

# ---------- Discord Bot ----------
import discord
from discord import app_commands
from discord.ext import commands
from discord import ui, Interaction, ButtonStyle
import aiosqlite
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dateutil import tz

# =========================
# ===== PATH / RESOURCES ==
# =========================

def resource_path(rel_path: str) -> str:
    """Путь к ресурсу в режиме PyInstaller (--onefile) и при обычном запуске."""
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, rel_path)

# Каталог данных (по умолчанию рядом с .py/.exe)
DATA_DIR = os.path.abspath(".")
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# ===== CONFIG LOGIC ======
# =========================

ENV_CONFIG_SPEC: Dict[str, Tuple[str, object, object]] = {
    "token": ("NIGIL_TOKEN", str, ""),
    "channel_id": ("NIGIL_CHANNEL_ID", int, 1375890545364172981),
    "db_path": ("NIGIL_DB_PATH", str, os.path.join(DATA_DIR, "nigil_monitor.sqlite3")),
    "weekly_limit": ("NIGIL_WEEKLY_LIMIT", int, 3),
    "cooldown_hours": ("NIGIL_COOLDOWN_HOURS", int, 20),
    "notify_period_min": ("NIGIL_NOTIFY_PERIOD_MIN", int, 60),
    "auto_clean_seconds": ("NIGIL_AUTO_CLEAN_SECONDS", int, 10),
    "fuzzy_cutoff": ("NIGIL_FUZZY_CUTOFF", float, 0.72),
    "system_name_regex": ("NIGIL_SYSTEM_NAME_REGEX", str, r"^[A-Za-z0-9\-]{2,12}$"),
    "timezone": ("NIGIL_TIMEZONE", str, "Europe/Moscow"),
    "pin_status_message": ("NIGIL_PIN_STATUS_MESSAGE", bool, True),
    "live_post_only": ("NIGIL_LIVE_POST_ONLY", bool, True),
    "command_reply_ttl": ("NIGIL_COMMAND_REPLY_TTL", int, 0),
}


def _parse_bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Не удалось интерпретировать '{value}' как булево значение")


def load_config_from_env() -> dict:
    cfg: Dict[str, object] = {}
    for key, (env_name, converter, default) in ENV_CONFIG_SPEC.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            value = default
        else:
            try:
                if converter is bool:
                    value = _parse_bool_env(raw)
                else:
                    value = converter(raw)
            except Exception as exc:
                print(f"[config] Переменная окружения {env_name} содержит некорректное значение: {exc}", file=sys.stderr)
                sys.exit(1)
        cfg[key] = value
    return cfg

def ensure_db_file_exists(db_path: str, template_name: str = "nigil_monitor.sqlite3"):
    """Если БД нет — скопировать шаблон (если упакован), иначе создать автоматически миграциями."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if not os.path.exists(db_path):
        try:
            shutil.copy2(resource_path(template_name), db_path)
        except Exception:
            # нет шаблона — таблицы создадутся миграциями при первом запуске
            pass


def self_update_from_github(repo: str, branch: str, files: Optional[List[str]] = None) -> List[str]:
    """Скачать свежие файлы из GitHub и перезаписать локальные."""
    files = files or ["NIGIL_status.py", "requirements.txt", "install.sh"]
    base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/"
    updated: List[str] = []
    script_dir = os.path.abspath(os.path.dirname(__file__))

    for rel_path in files:
        url = base_url + rel_path
        dest = os.path.join(script_dir, rel_path)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = resp.read()
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            print(f"[self-update] Не удалось скачать {url}: {exc}", file=sys.stderr)
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="nigil-update-")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(data)
            backup_path = dest + ".bak"
            if os.path.exists(dest):
                try:
                    shutil.copy2(dest, backup_path)
                except Exception as exc:
                    print(f"[self-update] Не удалось создать резервную копию {dest}: {exc}", file=sys.stderr)
            os.replace(tmp_path, dest)
            updated.append(rel_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return updated

# =========================
# ===== BOT BACKEND =======
# =========================

def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    if "Z" in ts or "+" in ts:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.fromisoformat(ts).replace(tzinfo=tz.UTC)

class NigilBot:
    """
    Discord-бот с единственным “живым постом” (LSP), который всегда редактируется.
    Сортировка систем, комментарии, единый список: Доступно/КД/Лимит.
    """
    def __init__(self, cfg: dict, log_fn=print):
        self.cfg = cfg
        self.log = log_fn

        self.MSK_TZ = tz.gettz(cfg.get("timezone", "Europe/Moscow"))
        self.WEEKLY_LIMIT = int(cfg.get("weekly_limit", 3))
        self.COOLDOWN_HOURS = int(cfg.get("cooldown_hours", 20))
        self.DB_PATH = cfg.get("db_path", os.path.join(DATA_DIR, "nigil_monitor.sqlite3"))
        self.CHANNEL_ID = int(cfg.get("channel_id"))
        self.AUTO_CLEAN = int(cfg.get("auto_clean_seconds", 10))
        self.NOTIFY_PERIOD_MIN = int(cfg.get("notify_period_min", 60))
        self.FUZZY_CUTOFF = float(cfg.get("fuzzy_cutoff", 0.72))
        self.SYSTEM_NAME_RE = re.compile(cfg.get("system_name_regex", r"^[A-Za-z0-9\-]{2,12}$"))
        self.PIN_STATUS = bool(cfg.get("pin_status_message", True))
        self.LIVE_POST_ONLY = bool(cfg.get("live_post_only", True))
        self.CMD_REPLY_TTL = int(cfg.get("command_reply_ttl", 0))

        intents = discord.Intents.default()
        intents.message_content = True  # НУЖНО включить в Dev Portal → Bot → Privileged Gateway Intents
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.db: Optional[aiosqlite.Connection] = None
        self.repo: Optional["Repo"] = None
        self.sched = AsyncIOScheduler(timezone=cfg.get("timezone", "Europe/Moscow"))

        self._bind_events_and_commands()

    # ---------- Utilities ----------
    def now_utc(self) -> datetime:
        return datetime.utcnow().replace(tzinfo=tz.UTC)

    def to_msk(self, dt_utc: datetime) -> datetime:
        return dt_utc.astimezone(self.MSK_TZ)

    def fmt_dt(self, dt: datetime) -> str:
        msk = self.to_msk(dt)
        return msk.strftime("%d.%m.%Y %H:%M") + " МСК"

    def human_delta_minutes(self, delta: timedelta) -> str:
        total = int(max(0, delta.total_seconds()))
        m = (total + 59) // 60  # округление вверх до минут
        h, mm = divmod(m, 60)
        if h and mm:
            return f"{h} ч {mm} мин"
        if h:
            return f"{h} ч"
        return f"{mm} мин"

    # ---------- Fuzzy ----------
    def levenshtein_le1(self, a: str, b: str) -> bool:
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) <= 1
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        i = j = mismatch = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1; j += 1
            else:
                mismatch += 1
                if mismatch > 1:
                    return False
                j += 1
        return True

    def find_best_system(self, user_text: str, systems: List[str]) -> Tuple[Optional[str], bool]:
        if not systems:
            return None, False
        up = user_text.upper()
        if up in systems:
            return up, False
        cand_lv1 = [s for s in systems if self.levenshtein_le1(up, s)]
        if cand_lv1:
            best = max(cand_lv1, key=lambda s: difflib.SequenceMatcher(None, up, s).ratio())
            return best, True
        close = difflib.get_close_matches(up, systems, n=1, cutoff=self.FUZZY_CUTOFF)
        if close:
            return close[0], True
        return None, False

    # ---------- Repo / DB ----------
    async def ensure_migrations(self, db: aiosqlite.Connection):
        # guild_settings columns
        cur = await db.execute("PRAGMA table_info(guild_settings)")
        cols = [r[1] for r in await cur.fetchall()]
        await cur.close()
        if "auto_clean_seconds" not in cols:
            await db.execute("ALTER TABLE guild_settings ADD COLUMN auto_clean_seconds INTEGER DEFAULT 0")
        if "status_week_anchor" not in cols:
            await db.execute("ALTER TABLE guild_settings ADD COLUMN status_week_anchor TEXT")

        # systems columns
        cur = await db.execute("PRAGMA table_info(systems)")
        scols = [r[1] for r in await cur.fetchall()]
        await cur.close()
        if "position" not in scols:
            await db.execute("ALTER TABLE systems ADD COLUMN position INTEGER DEFAULT 0")
        if "comment" not in scols:
            await db.execute("ALTER TABLE systems ADD COLUMN comment TEXT DEFAULT ''")
        await db.commit()

        # заполнить position по умолчанию (id-порядок) для тех, у кого 0/NULL
        await db.execute("""
            UPDATE systems SET position = id
            WHERE (position IS NULL OR position = 0)
        """)
        await db.commit()

    async def init_db(self):
        INIT_SQL = """
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS guild_settings (
          guild_id            INTEGER PRIMARY KEY,
          notify_channel_id   INTEGER,
          input_channel_id    INTEGER,
          notify_period_min   INTEGER DEFAULT 60,
          status_channel_id   INTEGER,
          status_message_id   INTEGER,
          status_week_anchor  TEXT,
          auto_clean_seconds  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS systems (
          id       INTEGER PRIMARY KEY AUTOINCREMENT,
          guild_id INTEGER NOT NULL,
          name     TEXT NOT NULL,
          position INTEGER DEFAULT 0,
          comment  TEXT DEFAULT '',
          UNIQUE(guild_id, name)
        );

        CREATE TABLE IF NOT EXISTS system_state (
          system_id     INTEGER PRIMARY KEY,
          weekly_count  INTEGER NOT NULL DEFAULT 0,
          last_call_ts  TEXT    DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS calls (
          id         INTEGER PRIMARY KEY AUTOINCREMENT,
          system_id  INTEGER NOT NULL,
          user_id    INTEGER NOT NULL,
          ts         TEXT    NOT NULL
        );
        """
        self.db = await aiosqlite.connect(self.DB_PATH)
        await self.db.executescript(INIT_SQL)
        await self.db.commit()
        await self.ensure_migrations(self.db)
        self.repo = Repo(self.db)

    # ---------- Time anchors ----------
    def week_reset_anchor(self, ref: Optional[datetime] = None) -> datetime:
        ref = ref or self.now_utc()
        msk = self.to_msk(ref)
        weekday = msk.weekday()  # Monday=0
        anchor = msk.replace(hour=3, minute=0, second=0, microsecond=0) - timedelta(days=weekday)
        if msk < anchor:
            anchor -= timedelta(days=7)
        return anchor.astimezone(tz.UTC)

    def next_available_time(self, weekly_used: int, last_call: Optional[datetime]) -> Optional[datetime]:
        cd_time = last_call + timedelta(hours=self.COOLDOWN_HOURS) if last_call else None
        limit_time = (self.week_reset_anchor() + timedelta(days=7)) if weekly_used >= self.WEEKLY_LIMIT else None
        candidates = [t for t in (cd_time, limit_time) if t is not None]
        return max(candidates) if candidates else None

    # ---------- Icons helper ----------
    def build_week_icons(self, used: int, cooldown_active: bool) -> str:
        """Возвращает строку из 3 значков: 🔴 (использ.), 🟡 (кд ближайшего), ✅ (доступно)."""
        L = self.WEEKLY_LIMIT
        used = max(0, min(L, used))
        if used >= L:
            return "🔴" * L
        if cooldown_active:
            # первый — КД, далее израсходованные (красные), потом доступные
            red_after = max(0, used - 1)
            green_after = max(0, L - 1 - red_after)
            return "🟡" + "🔴" * red_after + "✅" * green_after
        return "🔴" * used + "✅" * (L - used)

    # ---------- Stats (две недели) ----------
    async def build_two_weeks_stats_text(self, guild: discord.Guild) -> str:
        now = self.now_utc()
        anchor = self.week_reset_anchor(now)
        prev_anchor = anchor - timedelta(days=7)
        data_this = await self.repo.weekly_stats(guild.id, None, anchor)
        data_prev = await self.repo.weekly_stats_between(guild.id, None, prev_anchor, anchor)

        def fmt_block(title: str, data: Dict[int, int], start: datetime, end: datetime) -> List[str]:
            lines = [f"**{title}**  _(с {self.fmt_dt(start)} по {self.fmt_dt(end)})_"]
            if not data:
                lines.append("• нет вызовов")
            else:
                total = sum(data.values())
                lines.append(f"Всего: **{total}**")
                for uid, cnt in sorted(data.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"• <@{uid}> — **{cnt}**")
            return lines

        lines = ["**📊 Вызовы нигилов**"]
        lines += fmt_block("Текущая неделя", data_this, anchor, now)
        lines.append("")
        lines += fmt_block("Прошлая неделя", data_prev, prev_anchor, anchor)
        return "\n".join(lines)

    # ---------- UI: одна кнопка «Статистика» ----------
    class StatsView(ui.View):
        def __init__(self, outer: "NigilBot"):
            super().__init__(timeout=None)
            self.outer = outer

        @ui.button(label="📊 Статистика", style=ButtonStyle.primary, custom_id="stats:two_weeks")
        async def show_two_weeks(self, interaction: Interaction, button: ui.Button):
            guild = interaction.guild
            if not guild:
                return
            text = await self.outer.build_two_weeks_stats_text(guild)
            await interaction.response.send_message(text, ephemeral=True)

    # ---------- Discord binding ----------
    def _bind_events_and_commands(self):
        bot = self.bot

        @bot.event
        async def on_ready():
            try:
                await self.init_db()
                await bot.tree.sync()
                self.log(f"[READY] Вошёл как {bot.user} (id: {bot.user.id})")
                bot.add_view(self.StatsView(self))

                for guild in bot.guilds:
                    ch = guild.get_channel(self.CHANNEL_ID)
                    if isinstance(ch, discord.TextChannel):
                        await self.repo.set_status_channel(guild.id, self.CHANNEL_ID)
                        await self.repo.set_input_channel(guild.id, self.CHANNEL_ID)
                        await self.ensure_live_post(guild)
                        await self.refresh_live_post(guild)

                self.sched.start()
                self.sched.add_job(
                    self.weekly_rollover,
                    CronTrigger(day_of_week="mon", hour=3, minute=0, second=0, timezone=self.cfg.get("timezone", "Europe/Moscow"))
                )
                self.sched.add_job(
                    self.periodic_tick,
                    IntervalTrigger(minutes=1, timezone=self.cfg.get("timezone", "Europe/Moscow"))
                )
            except Exception as e:
                self.log(f"[ERROR on_ready] {e}")

        # -------- autocomplete --------
        async def systems_autocomplete(interaction: discord.Interaction, current: str):
            names = await self.repo.list_systems(interaction.guild_id)
            q = (current or "").upper()
            if q:
                names = [n for n in names if q in n.upper()]
            return [app_commands.Choice(name=n, value=n) for n in names[:25]]

        # -------- СИСТЕМЫ (группа) --------
        class SystemGroup(app_commands.Group):
            def __init__(self, outer: "NigilBot"):
                super().__init__(name="systems", description="Управление списком систем")
                self.outer = outer

            @app_commands.command(name="add", description="Добавить систему")
            async def add(self, interaction: discord.Interaction, name: str):
                await interaction.response.defer(ephemeral=True)
                ok = await self.outer.repo.add_system(interaction.guild_id, name)
                await self.outer.refresh_live_post(interaction.guild)
                await interaction.followup.send("✅ добавлена" if ok else "⚠️ уже есть", ephemeral=True)

            @app_commands.command(name="remove", description="Удалить систему")
            @app_commands.autocomplete(name=systems_autocomplete)
            async def remove(self, interaction: discord.Interaction, name: str):
                await interaction.response.defer(ephemeral=True)
                ok = await self.outer.repo.remove_system(interaction.guild_id, name)
                await self.outer.refresh_live_post(interaction.guild)
                await interaction.followup.send("🗑️ удалена" if ok else "❌ не найдена", ephemeral=True)

            @app_commands.command(name="list", description="Список систем (с порядком)")
            async def list_(self, interaction: discord.Interaction):
                await interaction.response.defer(ephemeral=True)
                lst = await self.outer.repo.list_systems_with_meta(interaction.guild_id)
                if not lst:
                    await interaction.followup.send("Список пуст. `/systems add <имя>`.", ephemeral=True); return
                lines = [f"{pos:>2}. {name.upper()}" + (f" — _{comment}_" if comment else "") for (name, pos, comment) in lst]
                await interaction.followup.send("\n".join(lines), ephemeral=True)

            # ВНИМАНИЕ: тут НЕТ субкоманды move (with where) — используем только ярлык /systems_move

            @app_commands.command(name="comment", description="Задать комментарий системе")
            @app_commands.autocomplete(name=systems_autocomplete)
            async def comment(self, interaction: discord.Interaction, name: str, text: str):
                await interaction.response.defer(ephemeral=True)
                ok = await self.outer.repo.set_comment(interaction.guild_id, name, text)
                await self.outer.refresh_live_post(interaction.guild)
                await interaction.followup.send("💬 комментарий обновлён" if ok else "❌ система не найдена", ephemeral=True)

            @app_commands.command(name="comment_clear", description="Очистить комментарий системы")
            @app_commands.autocomplete(name=systems_autocomplete)
            async def comment_clear(self, interaction: discord.Interaction, name: str):
                await interaction.response.defer(ephemeral=True)
                ok = await self.outer.repo.set_comment(interaction.guild_id, name, "")
                await self.outer.refresh_live_post(interaction.guild)
                await interaction.followup.send("🧹 комментарий удалён" if ok else "❌ система не найдена", ephemeral=True)

        bot.tree.add_command(SystemGroup(self))

        # -------- ЯРЛЫКИ --------
        @app_commands.command(name="systems_move", description="Поставить систему на позицию (сдвигая остальных)")
        @app_commands.autocomplete(name=systems_autocomplete)
        async def systems_move(interaction: discord.Interaction, name: str, pos: int):
            await interaction.response.defer(ephemeral=True)
            ok = await self.repo.move_to_position(interaction.guild_id, name, pos)
            await self.refresh_live_post(interaction.guild)
            await interaction.followup.send("↕️ поставлено на позицию" if ok else "❌ не удалось", ephemeral=True)
        bot.tree.add_command(systems_move)

        @app_commands.command(name="systems_comment", description="Комментарий к системе")
        @app_commands.autocomplete(name=systems_autocomplete)
        async def systems_comment(interaction: discord.Interaction, name: str, text: str):
            await interaction.response.defer(ephemeral=True)
            ok = await self.repo.set_comment(interaction.guild_id, name, text)
            await self.refresh_live_post(interaction.guild)
            await interaction.followup.send("💬 комментарий обновлён" if ok else "❌ система не найдена", ephemeral=True)
        bot.tree.add_command(systems_comment)

        # -------- Остальные команды --------
        @app_commands.command(name="call", description="Отметить вызов по системе")
        @app_commands.autocomplete(system=systems_autocomplete)
        async def call_cmd(interaction: discord.Interaction, system: str):
            ttl = self.CMD_REPLY_TTL
            if ttl == 0:
                await interaction.response.defer(ephemeral=True)
            all_names = await self.repo.list_systems(interaction.guild_id)
            best, fuzzy = self.find_best_system(system, all_names)
            if not best:
                msg = f"❌ Система **{system.upper()}** не найдена."
                if ttl == 0:
                    await interaction.followup.send(msg, ephemeral=True)
                else:
                    await interaction.response.send_message(msg, ephemeral=False)
                    try: await (await interaction.original_response()).delete(delay=ttl)
                    except: pass
                return
            msg, _ok = await self.register_call(interaction.guild_id, best, interaction.user.id)
            if fuzzy: msg = f"(опечатка исправлена на **{best}**) " + msg
            if ttl == 0:
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=False)
                try: await (await interaction.original_response()).delete(delay=ttl)
                except: pass
            await self.refresh_live_post(interaction.guild)
        bot.tree.add_command(call_cmd)

        @app_commands.command(name="undo", description="Отменить последний вызов по системе")
        @app_commands.autocomplete(system=systems_autocomplete)
        async def undo_cmd(interaction: discord.Interaction, system: str):
            ttl = self.CMD_REPLY_TTL
            if ttl == 0:
                await interaction.response.defer(ephemeral=True)
            all_names = await self.repo.list_systems(interaction.guild_id)
            best, fuzzy = self.find_best_system(system, all_names)
            if not best:
                msg = f"❌ Система **{system.upper()}** не найдена."
                if ttl == 0:
                    await interaction.followup.send(msg, ephemeral=True)
                else:
                    await interaction.response.send_message(msg, ephemeral=False)
                    try: await (await interaction.original_response()).delete(delay=ttl)
                    except: pass
                return
            msg = await self.undo_last(interaction.guild_id, best)
            if fuzzy and "не найдена" not in msg:
                msg = f"(опечатка исправлена на **{best}**) " + msg
            if ttl == 0:
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=False)
                try: await (await interaction.original_response()).delete(delay=ttl)
                except: pass
            await self.refresh_live_post(interaction.guild)
        bot.tree.add_command(undo_cmd)

        @app_commands.command(name="status", description="Статус по системе или сводка по всем")
        @app_commands.autocomplete(system=systems_autocomplete)
        async def status_cmd(interaction: discord.Interaction, system: Optional[str] = None):
            ttl = self.CMD_REPLY_TTL
            if ttl == 0:
                await interaction.response.defer(ephemeral=True)

            if system:
                all_names = await self.repo.list_systems(interaction.guild_id)
                best, fuzzy = self.find_best_system(system, all_names)
                if not best:
                    msg = f"❌ Система **{system.upper()}** не найдена."
                    if ttl == 0:
                        await interaction.followup.send(msg, ephemeral=True)
                    else:
                        await interaction.response.send_message(msg, ephemeral=False)
                        try: await (await interaction.original_response()).delete(delay=ttl)
                        except: pass
                    return
                msg, _ = await self.system_status(interaction.guild_id, best)
                if fuzzy:
                    msg = f"(опечатка исправлена на **{best}**)\n" + msg
            else:
                entries = await self.repo.list_systems_with_meta(interaction.guild_id)
                if not entries:
                    msg = "Список систем пуст. Добавь через `/systems add <имя>`."
                else:
                    anchor = self.week_reset_anchor()
                    next_reset = anchor + timedelta(days=7)
                    now = self.now_utc()
                    lines = []
                    for (n, _pos, comment) in entries:
                        sid = await self.repo.get_system_id(interaction.guild_id, n)
                        used = await self.repo.count_calls_in_week(sid, anchor)
                        _, last_call = await self.repo.get_state(sid)
                        cooldown_active = bool(last_call and now < (last_call + timedelta(hours=self.COOLDOWN_HOURS)))
                        icons = self.build_week_icons(used, cooldown_active)
                        cmt = f" _({comment})_" if (comment or "").strip() else ""
                        if used >= self.WEEKLY_LIMIT:
                            lines.append(f"**Лимит:** {icons} {n.upper()}{cmt}")
                        elif cooldown_active:
                            ready = last_call + timedelta(hours=self.COOLDOWN_HOURS)
                            rem = self.human_delta_minutes(ready - now)
                            lines.append(f"**КД:** {icons} {n.upper()}{cmt} — в {self.fmt_dt(ready)} (осталось {rem})")
                        else:
                            lines.append(f"**Доступно:** {icons} {n.upper()}{cmt}")
                    lines.append("")
                    lines.append("✅ Комментарии к системе (через `/systems_comment`).")
                    lines.append("✅ Перемещение системы по позиции (через `/systems_move`).")
                    lines.append(f"_Сброс лимитов: {self.fmt_dt(next_reset)}_")
                    msg = "\n".join(lines[:120])

            if ttl == 0:
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=False)
                try: await (await interaction.original_response()).delete(delay=ttl)
                except: pass

            await self.refresh_live_post(interaction.guild)
        bot.tree.add_command(status_cmd)

        @app_commands.command(name="stats", description="Статистика за текущую и прошлую недели")
        async def stats_cmd(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            text = await self.build_two_weeks_stats_text(interaction.guild)
            await interaction.followup.send(text, ephemeral=True)
        bot.tree.add_command(stats_cmd)

        # текстовый ввод в канале
        @bot.event
        async def on_message(message: discord.Message):
            if not message.guild:
                return
            if message.channel.id == self.CHANNEL_ID:
                if not message.author.bot:
                    text = message.content.strip().upper()
                    if self.SYSTEM_NAME_RE.fullmatch(text):
                        all_names = await self.repo.list_systems(message.guild.id)
                        best, _ = self.find_best_system(text, all_names)
                        if best:
                            _, ok = await self.register_call(message.guild.id, best, message.author.id)
                            try: await message.add_reaction("✅" if ok else "⛔")
                            except: pass
                            await self.refresh_live_post(message.guild)
                        else:
                            try: await message.add_reaction("❓")
                            except: pass
                    seconds = max(1, int(self.AUTO_CLEAN or 10))
                    asyncio.create_task(self.schedule_cleanup(message.channel, user_msg=message, bot_msg=None, seconds=seconds))
            await bot.process_commands(message)

    # ---------- Domain logic ----------
    async def system_status(self, guild_id: int, name: str) -> Tuple[str, bool]:
        sid = await self.repo.get_system_id(guild_id, name)
        if not sid:
            return f"Система **{name.upper()}** не найдена в списке.", False
        anchor = self.week_reset_anchor()
        used = await self.repo.count_calls_in_week(sid, anchor)
        _, last_call = await self.repo.get_state(sid)
        now = self.now_utc()
        cooldown_active = bool(last_call and now < (last_call + timedelta(hours=self.COOLDOWN_HOURS)))
        icons = self.build_week_icons(used, cooldown_active)
        if used >= self.WEEKLY_LIMIT:
            lines = [f"**Лимит:** {icons} {name.upper()}",
                     f"• Последний вызов: {self.fmt_dt(last_call) if last_call else '—'}"]
            return "\n".join(lines), False
        elif cooldown_active:
            ready = last_call + timedelta(hours=self.COOLDOWN_HOURS)
            rem = self.human_delta_minutes(ready - now)
            lines = [f"**КД:** {icons} {name.upper()} — в {self.fmt_dt(ready)} (осталось {rem})",
                     f"• Последний вызов: {self.fmt_dt(last_call) if last_call else '—'}"]
            return "\n".join(lines), False
        else:
            lines = [f"**Доступно:** {icons} {name.upper()}",
                     f"• Последний вызов: {self.fmt_dt(last_call) if last_call else '—'}"]
            return "\n".join(lines), True

    async def register_call(self, guild_id: int, name: str, user_id: int) -> Tuple[str, bool]:
        sid = await self.repo.get_system_id(guild_id, name)
        if not sid:
            return f"Система **{name.upper()}** не найдена. Добавьте её через `/systems add {name.upper()}`.", False
        anchor = self.week_reset_anchor()
        used = await self.repo.count_calls_in_week(sid, anchor)
        if used >= self.WEEKLY_LIMIT:
            return f"⛔ Лимит исчерпан на эту неделю для **{name.upper()}**.", False
        _, last_call = await self.repo.get_state(sid)
        if last_call:
            ready = last_call + timedelta(hours=self.COOLDOWN_HOURS)
            if self.now_utc() < ready:
                return (f"⛔ Рано: по **{name.upper()}** кулдаун ещё **{self.human_delta_minutes(ready - self.now_utc())}** "
                        f"(до {self.fmt_dt(ready)}).", False)
        ts = self.now_utc()
        await self.repo.add_call(sid, user_id, ts)
        await self.repo.set_state(sid, weekly_count=used + 1, last_call_ts=ts)
        return f"✅ Записал вызов по **{name.upper()}** от <@{user_id}> в {self.fmt_dt(ts)}.", True

    async def undo_last(self, guild_id: int, name: str) -> str:
        sid = await self.repo.get_system_id(guild_id, name)
        if not sid:
            return f"Система **{name.upper()}** не найдена."
        popped = await self.repo.pop_last_call(sid)
        if not popped:
            return f"По **{name.upper()}** нет записанных вызовов."
        _, user_id, ts = popped
        anchor = self.week_reset_anchor()
        used = await self.repo.count_calls_in_week(sid, anchor)
        last = await self.repo.last_call(sid)
        await self.repo.set_state(sid, weekly_count=used, last_call_ts=last)
        return f"↩️ Отменил последний вызов по **{name.upper()}** (был от <@{user_id}> в {self.fmt_dt(ts)})."

    # ---------- Live Status Post ----------
    async def ensure_live_post(self, guild: discord.Guild) -> Optional[discord.Message]:
        channel = guild.get_channel(self.CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):
            return None
        st = await self.repo.get_settings(guild.id)
        msg_id = st.get("status_message_id")
        view = self.StatsView(self)
        embeds = await self.build_status_embeds(guild)
        if msg_id:
            try:
                msg = await channel.fetch_message(msg_id)
                await msg.edit(embeds=embeds, view=view)
                return msg
            except discord.NotFound:
                pass
        msg = await channel.send(embeds=embeds, view=view)
        if self.PIN_STATUS:
            try: await msg.pin(reason="Живой статус-пост")
            except Exception: pass
        await self.repo.set_status_message(guild.id, msg.id, self.week_reset_anchor())
        return msg

    async def refresh_live_post(self, guild: discord.Guild):
        channel = guild.get_channel(self.CHANNEL_ID)
        if not isinstance(channel, discord.TextChannel):
            return
        st = await self.repo.get_settings(guild.id)
        msg_id = st.get("status_message_id")
        embeds = await self.build_status_embeds(guild)
        view = self.StatsView(self)
        if not msg_id:
            await self.ensure_live_post(guild); return
        try:
            msg = await channel.fetch_message(msg_id)
            await msg.edit(embeds=embeds, view=view)
        except discord.NotFound:
            await self.ensure_live_post(guild)

    async def schedule_cleanup(self, channel: discord.TextChannel, user_msg: Optional[discord.Message], bot_msg: Optional[discord.Message], seconds: int):
        if seconds <= 0:
            return
        await asyncio.sleep(seconds)
        try:
            if user_msg and channel.permissions_for(channel.guild.me).manage_messages:
                await user_msg.delete()
        except Exception:
            pass
        try:
            if bot_msg:
                await bot_msg.delete()
        except Exception:
            pass

    async def weekly_rollover(self):
        for guild in self.bot.guilds:
            await self.refresh_live_post(guild)

    async def periodic_tick(self):
        for guild in self.bot.guilds:
            await self.refresh_live_post(guild)

    # ---------- Embeds (единый список) ----------
    async def build_status_embeds(self, guild: discord.Guild) -> List[discord.Embed]:
        entries = await self.repo.list_systems_with_meta(guild.id)  # [(name, pos, comment)]
        anchor = self.week_reset_anchor()
        next_reset = anchor + timedelta(days=7)
        now = self.now_utc()

        title = f"Недельная сводка нигилов ({self.fmt_dt(anchor)} — {self.fmt_dt(next_reset)})"
        embed = discord.Embed(title=title, colour=discord.Colour.green())
        embed.set_footer(text="Обновлено")
        embed.timestamp = now

        if not entries:
            embed.description = ("Систем пока нет. Добавьте: `/systems add JITA`.\n"
                                 "Чтобы отметить вызов: напишите имя системы в канал или `/call <система>`.\n"
                                 f"_Сброс лимитов: {self.fmt_dt(next_reset)}_")
            return [embed]

        lines: List[str] = []
        for (name, _pos, comment) in entries:
            sid = await self.repo.get_system_id(guild.id, name)
            used = await self.repo.count_calls_in_week(sid, anchor)
            _, last_call = await self.repo.get_state(sid)
            cooldown_active = bool(last_call and now < (last_call + timedelta(hours=self.COOLDOWN_HOURS)))
            icons = self.build_week_icons(used, cooldown_active)
            cmt = f" _({comment})_" if (comment or "").strip() else ""

            if used >= self.WEEKLY_LIMIT:
                lines.append(f"**Лимит:** {icons} {name.upper()}{cmt}")
            elif cooldown_active:
                ready = last_call + timedelta(hours=self.COOLDOWN_HOURS)
                rem = self.human_delta_minutes(ready - now)
                lines.append(f"**КД:** {icons} {name.upper()}{cmt} — в {self.fmt_dt(ready)} (осталось {rem})")
            else:
                lines.append(f"**Доступно:** {icons} {name.upper()}{cmt}")

        lines.append("")
        lines.append("✅ Комментарии к системе (через `/systems_comment`).")
        lines.append("✅ Перемещение системы по позиции (через `/systems_move`).")
        lines.append(f"_Сброс лимитов: {self.fmt_dt(next_reset)}_")

        embed.description = "\n".join(lines)
        return [embed]

    # ---------- Start / Stop ----------
    async def run_forever(self, token: str):
        os.environ["DISCORD_TOKEN"] = token
        try:
            await self.bot.start(token)
        finally:
            if self.sched.running:
                self.sched.shutdown(wait=False)
            if not self.bot.is_closed():
                await self.bot.close()

# ---------- Repo ----------
class Repo:
    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def get_settings(self, guild_id: int) -> Dict:
        cur = await self.db.execute(
            "SELECT guild_id, notify_channel_id, input_channel_id, notify_period_min, "
            "status_channel_id, status_message_id, status_week_anchor, auto_clean_seconds "
            "FROM guild_settings WHERE guild_id = ?",
            (guild_id,)
        )
        row = await cur.fetchone(); await cur.close()
        if not row:
            await self.db.execute("INSERT INTO guild_settings (guild_id) VALUES (?)", (guild_id,))
            await self.db.commit()
            return {"guild_id": guild_id, "notify_channel_id": None, "input_channel_id": None,
                    "notify_period_min": 60, "status_channel_id": None, "status_message_id": None,
                    "status_week_anchor": None, "auto_clean_seconds": 0}
        return {"guild_id": row[0], "notify_channel_id": row[1], "input_channel_id": row[2],
                "notify_period_min": row[3] if row[3] else 60, "status_channel_id": row[4],
                "status_message_id": row[5], "status_week_anchor": row[6],
                "auto_clean_seconds": row[7] if row[7] is not None else 0}

    async def set_status_channel(self, guild_id: int, channel_id: int):
        await self.db.execute(
            "INSERT INTO guild_settings (guild_id, status_channel_id) VALUES (?, ?) "
            "ON CONFLICT(guild_id) DO UPDATE SET status_channel_id = excluded.status_channel_id",
            (guild_id, channel_id)
        )
        await self.db.commit()

    async def set_input_channel(self, guild_id: int, channel_id: int):
        await self.db.execute(
            "INSERT INTO guild_settings (guild_id, input_channel_id) VALUES (?, ?) "
            "ON CONFLICT(guild_id) DO UPDATE SET input_channel_id = excluded.input_channel_id",
            (guild_id, channel_id)
        )
        await self.db.commit()

    async def set_status_message(self, guild_id: int, message_id: Optional[int], week_anchor: Optional[datetime]):
        anchor_str = week_anchor.astimezone(tz.UTC).isoformat() if week_anchor else None
        await self.db.execute(
            "UPDATE guild_settings SET status_message_id = ?, status_week_anchor = ? WHERE guild_id = ?",
            (message_id, anchor_str, guild_id)
        )
        await self.db.commit()

    # ------ Systems CRUD + meta ------
    async def add_system(self, guild_id: int, name: str) -> bool:
        try:
            name_u = name.upper()
            # позиция = 1 + макс позиция
            cur = await self.db.execute("SELECT COALESCE(MAX(position),0) FROM systems WHERE guild_id = ?", (guild_id,))
            row = await cur.fetchone(); await cur.close()
            next_pos = (row[0] or 0) + 1
            await self.db.execute("INSERT INTO systems (guild_id, name, position, comment) VALUES (?, ?, ?, '')",
                                  (guild_id, name_u, next_pos))
            await self.db.commit()
            # state
            cur = await self.db.execute("SELECT id FROM systems WHERE guild_id = ? AND name = ?", (guild_id, name_u))
            sid = (await cur.fetchone())[0]; await cur.close()
            await self.db.execute("INSERT OR IGNORE INTO system_state (system_id) VALUES (?)", (sid,))
            await self.db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def remove_system(self, guild_id: int, name: str) -> bool:
        cur = await self.db.execute("SELECT id FROM systems WHERE guild_id = ? AND name = ?", (guild_id, name.upper()))
        row = await cur.fetchone(); await cur.close()
        if not row:
            return False
        sid = row[0]
        await self.db.execute("DELETE FROM calls WHERE system_id = ?", (sid,))
        await self.db.execute("DELETE FROM system_state WHERE system_id = ?", (sid,))
        await self.db.execute("DELETE FROM systems WHERE id = ?", (sid,))
        await self.db.commit()
        # после удаления — нормализовать позиции
        await self.compact_positions(guild_id)
        return True

    async def compact_positions(self, guild_id: int):
        cur = await self.db.execute("SELECT id FROM systems WHERE guild_id = ? ORDER BY position, id", (guild_id,))
        rows = await cur.fetchall(); await cur.close()
        pos = 1
        for (sid,) in rows:
            await self.db.execute("UPDATE systems SET position = ? WHERE id = ?", (pos, sid))
            pos += 1
        await self.db.commit()

    async def list_systems(self, guild_id: int) -> List[str]:
        cur = await self.db.execute("SELECT name FROM systems WHERE guild_id = ? ORDER BY position, id", (guild_id,))
        rows = await cur.fetchall(); await cur.close()
        return [r[0] for r in rows]

    async def list_systems_with_meta(self, guild_id: int) -> List[Tuple[str,int,str]]:
        cur = await self.db.execute("SELECT name, position, COALESCE(comment,'') FROM systems WHERE guild_id = ? ORDER BY position, id", (guild_id,))
        rows = await cur.fetchall(); await cur.close()
        return [(r[0], r[1], r[2]) for r in rows]

    async def get_system_id(self, guild_id: int, name: str) -> Optional[int]:
        cur = await self.db.execute("SELECT id FROM systems WHERE guild_id = ? AND name = ?", (guild_id, name.upper()))
        row = await cur.fetchone(); await cur.close()
        return row[0] if row else None

    async def get_state(self, system_id: int) -> Tuple[int, Optional[datetime]]:
        cur = await self.db.execute("SELECT weekly_count, last_call_ts FROM system_state WHERE system_id = ?", (system_id,))
        row = await cur.fetchone(); await cur.close()
        if not row:
            return 0, None
        weekly_count = row[0] or 0
        last_ts = parse_iso(row[1]) if row[1] else None
        return weekly_count, last_ts

    async def set_state(self, system_id: int, weekly_count: int, last_call_ts: Optional[datetime]):
        ts_str = last_call_ts.astimezone(tz.UTC).isoformat() if last_call_ts else None
        await self.db.execute(
            "INSERT INTO system_state (system_id, weekly_count, last_call_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(system_id) DO UPDATE SET weekly_count = excluded.weekly_count, last_call_ts = excluded.last_call_ts",
            (system_id, weekly_count, ts_str)
        )
        await self.db.commit()

    async def add_call(self, system_id: int, user_id: int, when: datetime):
        await self.db.execute(
            "INSERT INTO calls (system_id, user_id, ts) VALUES (?, ?, ?)",
            (system_id, user_id, when.astimezone(tz.UTC).isoformat())
        )
        await self.db.commit()

    async def last_call(self, system_id: int) -> Optional[datetime]:
        cur = await self.db.execute(
            "SELECT ts FROM calls WHERE system_id = ? ORDER BY ts DESC LIMIT 1",
            (system_id,)
        )
        row = await cur.fetchone(); await cur.close()
        return parse_iso(row[0]) if row else None

    async def pop_last_call(self, system_id: int) -> Optional[Tuple[int, int, datetime]]:
        cur = await self.db.execute(
            "SELECT id, user_id, ts FROM calls WHERE system_id = ? ORDER BY ts DESC LIMIT 1",
            (system_id,)
        )
        row = await cur.fetchone(); await cur.close()
        if not row:
            return None
        call_id, user_id, ts = row[0], row[1], parse_iso(row[2])
        await self.db.execute("DELETE FROM calls WHERE id = ?", (call_id,))
        await self.db.commit()
        return call_id, user_id, ts

    async def weekly_stats(self, guild_id: int, system_id: Optional[int], start_utc: datetime) -> Dict[int, int]:
        """Сумма вызовов по пользователям с момента start_utc (без верхней границы)."""
        params = [start_utc.astimezone(tz.UTC).isoformat(), guild_id]
        sql = ("SELECT user_id, COUNT(*) FROM calls c "
               "JOIN systems s ON s.id = c.system_id "
               "WHERE c.ts >= ? AND s.guild_id = ?")
        if system_id:
            sql += " AND c.system_id = ?"
            params.append(system_id)
        sql += " GROUP BY user_id"
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall(); await cur.close()
        return {r[0]: r[1] for r in rows}

    async def weekly_stats_between(self, guild_id: int, system_id: Optional[int], start_utc: datetime, end_utc: datetime) -> Dict[int, int]:
        """Сумма вызовов по пользователям в интервале [start_utc, end_utc)."""
        params = [start_utc.astimezone(tz.UTC).isoformat(), end_utc.astimezone(tz.UTC).isoformat(), guild_id]
        sql = ("SELECT user_id, COUNT(*) FROM calls c "
               "JOIN systems s ON s.id = c.system_id "
               "WHERE c.ts >= ? AND c.ts < ? AND s.guild_id = ?")
        if system_id:
            sql += " AND c.system_id = ?"
            params.append(system_id)
        sql += " GROUP BY user_id"
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall(); await cur.close()
        return {r[0]: r[1] for r in rows}

    async def count_calls_in_week(self, system_id: int, start_utc: datetime) -> int:
        """Сколько вызовов было у системы с момента недельного якоря (UTC)."""
        cur = await self.db.execute(
            "SELECT COUNT(*) FROM calls WHERE system_id = ? AND ts >= ?",
            (system_id, start_utc.astimezone(tz.UTC).isoformat())
        )
        row = await cur.fetchone(); await cur.close()
        return row[0] if row else 0

    # ------ Комментарии ------
    async def set_comment(self, guild_id: int, name: str, text: str) -> bool:
        cur = await self.db.execute("UPDATE systems SET comment = ? WHERE guild_id = ? AND name = ?",
                                    (text, guild_id, name.upper()))
        await self.db.commit()
        return cur.rowcount > 0

    # ------ Перемещение систем ------
    async def move_system(self, guild_id: int, name: str, where: str, pos: Optional[int]) -> bool:
        cur = await self.db.execute("SELECT id, name FROM systems WHERE guild_id = ? ORDER BY position, id", (guild_id,))
        rows = await cur.fetchall(); await cur.close()
        if not rows:
            return False
        ids = [r[0] for r in rows]
        names = [r[1] for r in rows]
        name_u = name.upper()
        if name_u not in names:
            return False
        idx = names.index(name_u)
        n = len(names)

        if where == "up" and idx > 0:
            new_idx = idx - 1
        elif where == "down" and idx < n - 1:
            new_idx = idx + 1
        elif where == "top":
            new_idx = 0
        elif where == "bottom":
            new_idx = n - 1
        elif where == "pos":
            if pos is None or pos < 1:
                new_idx = 0
            elif pos > n:
                new_idx = n - 1
            else:
                new_idx = pos - 1
        else:
            return False

        item_id, item_name = ids[idx], names[idx]
        del ids[idx], names[idx]
        ids.insert(new_idx, item_id); names.insert(new_idx, item_name)

        for i, sid in enumerate(ids, start=1):
            await self.db.execute("UPDATE systems SET position = ? WHERE id = ?", (i, sid))
        await self.db.commit()
        return True

    async def move_to_position(self, guild_id: int, name: str, pos: int) -> bool:
        return await self.move_system(guild_id, name, "pos", pos)

# =========================
# ========= CLI ===========
# =========================


def run_bot(override_token: Optional[str]):
    cfg = load_config_from_env()
    if override_token:
        cfg["token"] = override_token.strip()

    token = (cfg.get("token") or "").strip()
    if not token:
        print(
            "Токен Discord не задан. Установите переменную окружения NIGIL_TOKEN или передайте --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = cfg.get("db_path", os.path.join(DATA_DIR, "nigil_monitor.sqlite3"))
    ensure_db_file_exists(db_path)

    bot = NigilBot(cfg)
    try:
        asyncio.run(bot.run_forever(token))
    except KeyboardInterrupt:
        print("Остановка по запросу пользователя.")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Управление Discord-ботом NIGIL")
    parser.add_argument("--token", help="Переопределить токен, заданный в переменной окружения NIGIL_TOKEN")
    parser.add_argument("--self-update", action="store_true", help="Обновить файлы проекта с GitHub и выйти")
    parser.add_argument("--update-repo", default=os.getenv("NIGIL_UPDATE_REPO", "NIGIL-status/NIGIL_status"),
                        help="GitHub-репозиторий в формате owner/repo для self-update")
    parser.add_argument("--update-branch", default=os.getenv("NIGIL_UPDATE_BRANCH", "main"),
                        help="Ветка GitHub для self-update")
    parser.add_argument("--update-files", nargs="*",
                        help="Список файлов (относительно корня репозитория) для self-update")

    args = parser.parse_args(argv)

    if getattr(args, "self_update", False):
        files = args.update_files or None
        updated = self_update_from_github(args.update_repo, args.update_branch, files)
        if updated:
            print("[self-update] Обновлены файлы:")
            for name in updated:
                print(f" • {name}")
            print("[self-update] Готово. Перезапустите установку зависимостей при необходимости.")
            return
        print("[self-update] Не удалось обновить ни один файл.", file=sys.stderr)
        sys.exit(1)

    override_token = getattr(args, "token", None)
    run_bot(override_token)


if __name__ == "__main__":
    main()
