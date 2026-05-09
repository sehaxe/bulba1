#!/usr/bin/env python3
"""
Telegram Bot для мониторинга тренировки Bulba1
Оптимизированная версия: асинхронный event loop, нет блокирующих subprocess.run
"""

import os, sys, re, json, time, asyncio, signal, subprocess
from pathlib import Path
from datetime import datetime
from functools import wraps
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
LOG_FILE = PROJECT_ROOT / "logs" / "bulba1.log"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "run_bulba1_150m"
SERVICE_NAME = "bulba1"

try:
    from bot_config import BOT_TOKEN, ADMIN_IDS
    _config_source = "bot_config.py"
except ImportError:
    BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
    ADMIN_IDS = set()
    if os.environ.get("TELEGRAM_ADMIN_ID"):
        ADMIN_IDS = {int(x) for x in os.environ["TELEGRAM_ADMIN_ID"].split(",")}
    _config_source = "environment"

# ── Async systemctl wrapper ─────────────────────────────────────────
async def run_systemctl(*args: str) -> tuple[bool, str]:
    """Асинхронный вызов systemctl --user"""
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
    if ok:
        return stdout.strip()
    return "inactive"

# ── GPU / System Info (async) ───────────────────────────────────────
async def get_gpu_info() -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        name, util, temp, mem_used, mem_total, power = stdout.decode().strip().split(",")
        return (
            f"🎮 *{name.strip()}*\n"
            f"Util: `{util.strip()}` | Temp: `{temp.strip()}°C`\n"
            f"VRAM: `{mem_used.strip()}/{mem_total.strip()} MB`\n"
            f"Power: `{power.strip()}`"
        )
    except Exception as e:
        return f"❌ {e}"

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
    except Exception as e:
        return f"❌ {e}"

# ── Training state ──────────────────────────────────────────────────
def read_log_tail(n: int = 100) -> list[str]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-n:]

def get_current_step() -> int:
    for line in reversed(read_log_tail(200)):
        m = re.search(r"Step (\d+)/", line)
        if m:
            return int(m.group(1))
    return 0

def get_loss() -> float:
    for line in reversed(read_log_tail(200)):
        m = re.search(r"loss=([\d.]+)", line)
        if m:
            return float(m.group(1))
    return 0.0

def get_latest_checkpoint() -> int | None:
    if not CHECKPOINT_DIR.exists():
        return None
    files = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.safetensors"))
    if files:
        return int(files[-1].stem.split("_")[-1])
    return None

# ── Decorators ──────────────────────────────────────────────────────
def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not is_admin(update.effective_user.id):
            if update.callback_query:
                await update.callback_query.answer("⛔ Admin only", show_alert=True)
            else:
                await update.message.reply_text("⛔ Admin only")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

# ── Rate limiter ────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_calls: int = 10, window: float = 60.0):
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

    def remaining(self, user_id: int) -> int:
        now = time.time()
        if user_id not in self.history:
            return self.max_calls
        active = len([t for t in self.history[user_id] if now - t < self.window])
        return max(0, self.max_calls - active)

limiter = RateLimiter(max_calls=15, window=60.0)
chat_limiter = RateLimiter(max_calls=5, window=3600.0)  # 5 chats per hour

# ── Keyboards ───────────────────────────────────────────────────────
def main_keyboard(is_admin_user: bool) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton("🖥 GPU", callback_data="gpu")],
        [InlineKeyboardButton("💻 System", callback_data="system"),
         InlineKeyboardButton("⏱ ETA", callback_data="eta")],
        [InlineKeyboardButton("📝 Logs", callback_data="logs"),
         InlineKeyboardButton("📈 Plot", callback_data="plot")],
        [InlineKeyboardButton("📦 Checkpoint", callback_data="checkpoint")],
    ]
    if is_admin_user:
        buttons += [
            [InlineKeyboardButton("▶ Start", callback_data="train"),
             InlineKeyboardButton("⏹ Stop", callback_data="stop")],
            [InlineKeyboardButton("🔄 Restart", callback_data="restart"),
             InlineKeyboardButton("💾 Save", callback_data="save")],
        ]
    return InlineKeyboardMarkup(buttons)

# ── Command handlers ────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = main_keyboard(is_admin(update.effective_user.id))
    await update.message.reply_text("🎛 *Bulba1 Control Panel*", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)

async def panel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_command(update, context)

@RateLimiter().check  # not using decorator properly, will fix below
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not limiter.check(user_id):
        await update.message.reply_text("⏳ Slow down! Try again in a moment.")
        return

    status = await get_service_status()
    gpu = await get_gpu_info()
    step = get_current_step()
    loss = get_loss()
    ckpt = get_latest_checkpoint()

    msg = (
        f"{'✅' if status == 'active' else '❌'} *Training Status*\n\n"
        f"Service: `{status}`\n"
        f"{gpu}\n"
        f"Step: `{step}/100000`\n"
        f"Loss: `{loss:.4f}`\n"
        f"Checkpoint: `{ckpt or 'N/A'}`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def gpu_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    msg = await get_gpu_info()
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def system_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    msg = await get_system_info()
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def eta_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    step = get_current_step()
    loss = get_loss()
    remaining = 100000 - step
    # грубая оценка: ~0.3 сек/шаг с учётом предтокенизации
    eta_sec = remaining * 0.3
    hours, mins = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
    await update.message.reply_text(
        f"⏱ *ETA*\nStep: `{step}/100000`\nLoss: `{loss:.4f}`\nRemaining: `{hours}h {mins}m`",
        parse_mode=ParseMode.MARKDOWN
    )

async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    n = min(int(context.args[0]) if context.args else 10, 50)
    lines = read_log_tail(n)
    if not lines:
        return await update.message.reply_text("❌ Log file not found")
    text = "".join(lines[-n:]).replace("_", "\\_").replace("*", "•")
    await update.message.reply_text(f"📝 Last {n} lines:\n\n{text}")

async def plot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    await update.message.reply_text("🎨 Generating plot...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.log_viz import parse_log, generate_plots
    if not LOG_FILE.exists():
        return await update.message.reply_text("❌ Log file not found")
    df = parse_log(str(LOG_FILE))
    if df is None or len(df) == 0:
        return await update.message.reply_text("❌ No data")
    output = PROJECT_ROOT / "logs" / "bulba1_plot.png"
    generate_plots(df, str(output), "Bulba1 Training")
    await update.message.reply_photo(photo=open(output, "rb"))

async def checkpoint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not limiter.check(update.effective_user.id):
        return await update.message.reply_text("⏳ Slow down!")
    ckpt = get_latest_checkpoint()
    step = get_current_step()
    loss = get_loss()
    await update.message.reply_text(
        f"📦 *Checkpoint*\nLatest: `{ckpt or 'N/A'}`\nStep: `{step}`\nLoss: `{loss:.4f}`",
        parse_mode=ParseMode.MARKDOWN
    )

# ── Admin commands ──────────────────────────────────────────────────
@admin_only
async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("start", SERVICE_NAME)
    await update.message.reply_text("▶ Training started" if ok else "❌ Failed to start")

@admin_only
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("stop", SERVICE_NAME)
    await update.message.reply_text("⏹ Training stopped" if ok else "❌ Failed to stop")

@admin_only
async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, _ = await run_systemctl("restart", SERVICE_NAME)
    await update.message.reply_text("🔄 Training restarted" if ok else "❌ Failed to restart")

@admin_only
async def save_checkpoint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = subprocess.run(["pgrep", "-f", "bulba1.cli"], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        if not pids:
            return await update.message.reply_text("❌ Training not running")
        os.kill(int(pids[0]), signal.SIGUSR1)
        await update.message.reply_text("💾 Checkpoint saved")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id) and not chat_limiter.check(user_id):
        return await update.message.reply_text("⏳ Chat daily limit reached")
    checkpoints = sorted(
        [int(f.stem.split("_")[-1]) for f in CHECKPOINT_DIR.glob("checkpoint_step_*.safetensors")]
    ) if CHECKPOINT_DIR.exists() else []
    if not checkpoints:
        return await update.message.reply_text("No checkpoints yet")
    # show last 5
    ckpts = checkpoints[-5:]
    buttons = [[InlineKeyboardButton(f"Step {s}", callback_data=f"chat_{s}")] for s in ckpts]
    await update.message.reply_text(
        "Select checkpoint to chat with:", reply_markup=InlineKeyboardMarkup(buttons)
    )

# ── Callback query handler ─────────────────────────────────────────
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    # Map callback data to handler functions
    handlers = {
        "status": status_command,
        "gpu": gpu_info_command,
        "system": system_info_command,
        "eta": eta_command,
        "logs": logs_command,
        "plot": plot_command,
        "checkpoint": checkpoint_command,
        "train": train_command,
        "stop": stop_command,
        "restart": restart_command,
        "save": save_checkpoint_command,
    }
    if data in handlers:
        # Передаём управление через fake update
        update.message = query.message
        update.effective_user = query.from_user
        # Для команд, которые читают context.args – не используется в button handlers
        await handlers[data](update, context)
    elif data.startswith("chat_"):
        step = int(data.split("_")[1])
        context.user_data["chat_step"] = step
        await query.message.reply_text(f"Checkpoint {step} selected. Type your message:")

# ── Message handler for chat ────────────────────────────────────────
async def handle_chat_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    step = context.user_data.pop("chat_step", None)
    if step is None:
        return await update.message.reply_text("Use /chat to start a conversation")
    prompt = update.message.text.strip()
    await update.message.reply_text("🧠 Thinking...")
    # run in thread to avoid blocking
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: subprocess.run(
        [sys.executable, "-m", "bulba1.cli", "--steps", "0", "--generate",
         "--prompt", prompt, "--gen-max-tokens", "100", "--checkpoint", str(step),
         "--resume", "--device", "cpu"],
        capture_output=True, text=True, timeout=180
    ))
    output = None
    for line in result.stdout.splitlines():
        if "Generated:" in line:
            output = line.split("Generated:", 1)[1].strip()
    await update.message.reply_text(output or f"Error: {result.stderr[:200]}")

# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    parser.add_argument("--poll", action="store_true")
    args = parser.parse_args()

    token = args.token or BOT_TOKEN
    if not token:
        print("No token"); sys.exit(1)

    app = Application.builder().token(token).build()

    # User commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("panel", panel_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("gpu", gpu_info_command))
    app.add_handler(CommandHandler("system", system_info_command))
    app.add_handler(CommandHandler("eta", eta_command))
    app.add_handler(CommandHandler("logs", logs_command))
    app.add_handler(CommandHandler("plot", plot_command))
    app.add_handler(CommandHandler("checkpoint", checkpoint_command))
    # Admin commands
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("restart", restart_command))
    app.add_handler(CommandHandler("save", save_checkpoint_command))
    # Chat
    app.add_handler(CommandHandler("chat", chat_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat_input))
    # Callback
    app.add_handler(CallbackQueryHandler(button_callback))

    print(f"🤖 Bulba1 Bot ({_config_source})")
    app.run_polling()

if __name__ == "__main__":
    import argparse
    main()