#!/usr/bin/env python3
"""
Telegram Bot для мониторинга тренировки Bulba1 (исправленный)
"""

import os, sys, re, json, time, asyncio, signal, subprocess
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "logs" / "bulba1.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "run_bulba1_340m"
SERVICE_NAME = "bulba1"

try:
    from telegram_bot.bot_config import BOT_TOKEN, ADMIN_IDS
    _config_source = "bot_config.py"
except ImportError:
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_TOKEN", ""))
    ADMIN_IDS = set()
    if os.environ.get("TELEGRAM_ADMIN_ID"):
        ADMIN_IDS = {int(x) for x in os.environ["TELEGRAM_ADMIN_ID"].split(",")}
    _config_source = "environment"

# ── Helpers ─────────────────────────────────────────────────────────
async def run_systemctl(*args: str) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode == 0, (stdout + stderr).decode()
    except Exception as e:
        return False, str(e)

async def get_service_status() -> str:
    ok, stdout = await run_systemctl("is-active", SERVICE_NAME)
    return stdout.strip() if ok else "inactive"

async def get_gpu_info() -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        parts = stdout.decode().strip().split(",")
        name, util, temp, mem_used, mem_total, power = parts
        return (
            f"🎮 *{name.strip()}*\n"
            f"Util: `{util.strip()}` | Temp: `{temp.strip()}°C`\n"
            f"VRAM: `{mem_used.strip()}/{mem_total.strip()} MB`\n"
            f"Power: `{power.strip()}`"
        )
    except Exception:
        return "❌ GPU not available"

async def get_system_info() -> str:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = int(lines[0].split()[1]) // 1024
        avail = int(lines[2].split()[1]) // 1024
        used = total - avail
        with open("/proc/loadavg") as f:
            load = f.read().split()[0]
        with open("/proc/cpuinfo") as f:
            cores = f.read().count("processor\t:")
        proc = await asyncio.create_subprocess_exec(
            "df", "-h", "/", stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        disk = stdout.decode().splitlines()[1].split()[2:4]
        return (
            f"💻 *System*\n"
            f"CPU: `{cores}` cores | Load: `{load}`\n"
            f"RAM: `{used}/{total} MB` ({used*100//total}%)\n"
            f"Disk: `{disk[0]}/{disk[1]}`"
        )
    except Exception:
        return "❌ System info unavailable"

def read_jsonl(n: int = 100) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-n:]

def get_current_step() -> int:
    recs = read_jsonl(200)
    for r in reversed(recs):
        if "step" in r:
            return r["step"]
    return 0

def get_loss() -> float:
    recs = read_jsonl(200)
    for r in reversed(recs):
        if "loss" in r:
            return r["loss"]
    return 0.0

def get_ema_loss() -> float:
    recs = read_jsonl(200)
    for r in reversed(recs):
        if "ema_loss" in r:
            return r["ema_loss"] or 0.0
    return 0.0

def get_tokens_per_sec() -> int:
    recs = read_jsonl(200)
    for r in reversed(recs):
        if "tok_per_sec" in r:
            return r["tok_per_sec"]
    return 0

def get_latest_checkpoint() -> int | None:
    if not CHECKPOINT_DIR.exists():
        return None
    files = list(CHECKPOINT_DIR.glob("checkpoint_step_*.safetensors"))
    if not files:
        return None
    best = max(files, key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)))
    return int(re.search(r"step_(\d+)", best.name).group(1))

class RateLimiter:
    def __init__(self, max_calls: int = 15, window: float = 60.0):
        self.max_calls = max_calls
        self.window = window
        self.history: dict[int, list[float]] = {}

    def check(self, user_id: int) -> bool:
        now = time.time()
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id] = [t for t in self.history[user_id] if now - t < self.window]
        if len(self.history[user_id]) >= self.max_calls:
            return False
        self.history[user_id].append(now)
        return True

limiter = RateLimiter(max_calls=15, window=60.0)

# Безопасный способ отправить сообщение
async def _reply(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs) -> None:
    """Отправляет сообщение в тот же чат, откуда пришёл запрос."""
    if update.callback_query:
        msg = update.callback_query.message
    else:
        msg = update.message
    if msg:
        await msg.reply_text(text, **kwargs)

# ── Keyboards ───────────────────────────────────────────────────────
def main_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton("🖥 GPU", callback_data="gpu")],
        [InlineKeyboardButton("💻 System", callback_data="sys"),
         InlineKeyboardButton("⏱ ETA", callback_data="eta")],
        [InlineKeyboardButton("📝 Logs", callback_data="logs"),
         InlineKeyboardButton("📈 Plot", callback_data="plot")],
        [InlineKeyboardButton("📦 Checkpoint", callback_data="checkpoint")],
    ]
    return InlineKeyboardMarkup(buttons)

# ── Command handlers ────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = await get_service_status()
    step = get_current_step()
    loss = get_loss()
    ema = get_ema_loss()
    tok_s = get_tokens_per_sec()
    ckpt = get_latest_checkpoint()
    gpu = await get_gpu_info()
    text = (
        f"{'✅' if status == 'active' else '❌'} *Bulba1 Control Panel*\n\n"
        f"Service: `{status}`\n"
        f"Step: `{step}/100000`\n"
        f"Loss: `{loss:.4f}`\n"
        f"EMA Loss: `{ema:.4f}`\n"
        f"Speed: `{tok_s} tok/s`\n"
        f"Checkpoint: `{ckpt or 'N/A'}`\n\n"
        f"{gpu}"
    )
    await _reply(update, context, text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_keyboard())

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    status = await get_service_status()
    step = get_current_step()
    loss = get_loss()
    ema = get_ema_loss()
    tok_s = get_tokens_per_sec()
    ckpt = get_latest_checkpoint()
    gpu = await get_gpu_info()
    text = (
        f"{'✅' if status == 'active' else '❌'} *Training Status*\n\n"
        f"Service: `{status}`\n{gpu}\n"
        f"Step: `{step}/100000`\n"
        f"Loss: `{loss:.4f}` | EMA: `{ema:.4f}`\n"
        f"Speed: `{tok_s} tok/s`\n"
        f"Checkpoint: `{ckpt or 'N/A'}`"
    )
    await _reply(update, context, text, parse_mode=ParseMode.MARKDOWN)

async def gpu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    msg = await get_gpu_info()
    await _reply(update, context, msg, parse_mode=ParseMode.MARKDOWN)

async def sys_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    msg = await get_system_info()
    await _reply(update, context, msg, parse_mode=ParseMode.MARKDOWN)

async def eta_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    step = get_current_step()
    loss = get_loss()
    remaining = 100000 - step
    tok_s = get_tokens_per_sec()
    if tok_s > 0:
        eta_sec = remaining * 360 / tok_s
        hours, mins = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
    else:
        hours, mins = 0, 0
    text = f"⏱ *ETA*\nStep: `{step}/100000`\nLoss: `{loss:.4f}`\nRemaining: `{hours}h {mins}m`"
    await _reply(update, context, text, parse_mode=ParseMode.MARKDOWN)

async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    n = min(int(context.args[0]) if context.args else 10, 50)
    recs = read_jsonl(n)
    if not recs:
        return await _reply(update, context, "❌ Log file not found")
    lines = [f"Step {r['step']}: loss={r['loss']:.4f} ema={r.get('ema_loss',0):.4f} tok/s={r['tok_per_sec']}" for r in recs]
    await _reply(update, context, f"📝 Last {n} records:\n\n`{chr(10).join(lines[-n:])}`", parse_mode=ParseMode.MARKDOWN)

async def plot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    await _reply(update, context, "🎨 Generating plot...")
    result = subprocess.run(
        ["uv", "run", "tools/log_viz.py", "-f", str(LOG_FILE), "-o", str(PROJECT_ROOT / "logs" / "bulba1_plot.png")],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return await _reply(update, context, f"❌ Plot error: {result.stderr}")
    plot_path = PROJECT_ROOT / "logs" / "bulba1_plot.png"
    if plot_path.exists():
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(plot_path, "rb"))
    else:
        await _reply(update, context, "❌ Plot file not created")

async def checkpoint_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id): return
    ckpt = get_latest_checkpoint()
    step = get_current_step()
    loss = get_loss()
    await _reply(update, context,
                 f"📦 *Checkpoint*\nLatest: `{ckpt or 'N/A'}`\nStep: `{step}`\nLoss: `{loss:.4f}`",
                 parse_mode=ParseMode.MARKDOWN)

# ── Button handler ──────────────────────────────────────────────────
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    # Создаём фейковый update с сообщением для команд
    # Просто передаём управление той же функции, она справится через _reply
    if data == "status":
        await status_cmd(update, context)
    elif data == "gpu":
        await gpu_cmd(update, context)
    elif data == "sys":
        await sys_cmd(update, context)
    elif data == "eta":
        await eta_cmd(update, context)
    elif data == "logs":
        await logs_cmd(update, context)
    elif data == "plot":
        await plot_cmd(update, context)
    elif data == "checkpoint":
        await checkpoint_cmd(update, context)

# ── Admin commands ──────────────────────────────────────────────────
def admin_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id not in ADMIN_IDS:
            await _reply(update, context, "⛔ Admin only")
            return
        return await func(update, context)
    return wrapper

@admin_only
async def train_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("start", SERVICE_NAME)
    await _reply(update, context, "▶ Training started" if ok else "❌ Failed")

@admin_only
async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("stop", SERVICE_NAME)
    await _reply(update, context, "⏹ Stopped" if ok else "❌ Failed")

@admin_only
async def restart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("restart", SERVICE_NAME)
    await _reply(update, context, "🔄 Restarted" if ok else "❌ Failed")

@admin_only
async def save_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = subprocess.run(["pgrep", "-f", "bulba1.cli"], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        if not pids:
            return await _reply(update, context, "❌ Training not running")
        os.kill(int(pids[0]), signal.SIGUSR1)
        await _reply(update, context, "💾 Checkpoint saved")
    except Exception as e:
        await _reply(update, context, f"❌ {e}")

# ── Main ────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    args = parser.parse_args()

    token = args.token or BOT_TOKEN
    if not token:
        print("No token"); sys.exit(1)

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("gpu", gpu_cmd))
    app.add_handler(CommandHandler("sys", sys_cmd))
    app.add_handler(CommandHandler("eta", eta_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("plot", plot_cmd))
    app.add_handler(CommandHandler("checkpoint", checkpoint_cmd))
    # Admin
    app.add_handler(CommandHandler("train", train_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("restart", restart_cmd))
    app.add_handler(CommandHandler("save", save_cmd))
    # Buttons
    app.add_handler(CallbackQueryHandler(button_handler))

    print(f"🤖 Bulba1 Bot ({_config_source})")
    app.run_polling()

if __name__ == "__main__":
    main()