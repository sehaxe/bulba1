#!/usr/bin/env python3
"""
Telegram Bot for Bulba1 Training Monitor
Run with: python bot.py [--token TOKEN] [--chat-id CHAT_ID]
Or set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Config paths
LOG_FILE = PROJECT_ROOT / "logs" / "bulba1_225m.log"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "run_bulba1_225m_clean"
SERVICE_NAME = "bulba1"
CLI_SCRIPT = PROJECT_ROOT / "bulba1" / "cli.py"


def run_cli(args: list, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run CLI command using uv"""
    cmd = [
        "uv",
        "run",
        "--with",
        "torch",
        "--with",
        "safetensors",
        "python",
        str(CLI_SCRIPT),
    ] + args
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT)
    )


# Rate limiting
import time

RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_COMMANDS = 10  # max commands per window
RATE_LIMIT_PLOT_MAX = 2  # max /plot per window (expensive)
command_history = {}  # user_id -> [(timestamp, command)]

# Load config - prefer bot_config.py, fall back to env vars
try:
    from bot_config import BOT_TOKEN, ADMIN_IDS
    _config_loaded = "bot_config.py"
except ImportError:
    # Fall back to environment variables
    BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
    ADMIN_IDS = set()
    if os.environ.get("TELEGRAM_ADMIN_ID"):
        ADMIN_IDS = {int(x) for x in os.environ.get("TELEGRAM_ADMIN_ID").split(",")}
    _config_loaded = "environment"


def is_admin(user_id: int) -> bool:
    """Check if user is admin"""
    return user_id in ADMIN_IDS


def get_main_keyboard(is_admin_user: bool = False) -> InlineKeyboardMarkup:
    """Get main keyboard with buttons"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("🎮 GPU", callback_data="gpu"),
        ],
        [
            InlineKeyboardButton("💻 System", callback_data="system"),
            InlineKeyboardButton("⏱️ ETA", callback_data="eta"),
        ],
        [
            InlineKeyboardButton("📝 Logs", callback_data="logs"),
            InlineKeyboardButton("📈 Plot", callback_data="plot"),
        ],
        [InlineKeyboardButton("📦 Checkpoint", callback_data="checkpoint")],
    ]
    if is_admin_user:
        keyboard.extend(
            [
                [
                    InlineKeyboardButton("▶️ Start", callback_data="train"),
                    InlineKeyboardButton("⏹ Stop", callback_data="stop"),
                ],
                [
                    InlineKeyboardButton("🔄 Restart", callback_data="restart"),
                    InlineKeyboardButton("🗑 Reset", callback_data="reset"),
                ],
                [InlineKeyboardButton("🛑 Off", callback_data="shutdown")],
            ]
        )
    return InlineKeyboardMarkup(keyboard)


async def panel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show control panel"""
    user_id = update.effective_user.id
    keyboard = get_main_keyboard(is_admin(user_id))

    await update.message.reply_text(
        "🎛️ *Control Panel*", reply_markup=keyboard, parse_mode="Markdown"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    # Route to appropriate handler
    if data == "status":
        await status_command(update, context)
    elif data == "logs":
        await logs_command(update, context)
    elif data == "plot":
        await plot_command(update, context)
    elif data == "checkpoint":
        await checkpoint_command(update, context)
    elif data == "train":
        await train_command(update, context)
    elif data == "stop":
        await stop_command(update, context)
    elif data == "restart":
        await restart_command(update, context)
    elif data == "reset":
        await reset_command(update, context)
    elif data == "gpu":
        await gpu_info_command(update, context)
    elif data == "system":
        await system_info_command(update, context)
    elif data == "eta":
        await eta_command(update, context)
    elif data == "shutdown":
        await shutdown_command(update, context)


def check_rate_limit(user_id: int, command: str) -> tuple[bool, str]:
    """Check if user is rate limited. Returns (allowed, message)"""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    if user_id not in command_history:
        command_history[user_id] = []

    # Clean old entries
    command_history[user_id] = [
        (ts, cmd) for ts, cmd in command_history[user_id] if ts > window_start
    ]

    # Check /plot separately (more expensive)
    if command == "plot":
        plot_count = sum(1 for ts, cmd in command_history[user_id] if cmd == "plot")
        if plot_count >= RATE_LIMIT_PLOT_MAX:
            return False, f"⏳ /plot limited to {RATE_LIMIT_PLOT_MAX} times per minute. Wait a bit."

    # Check total commands
    total_count = len(command_history[user_id])
    if total_count >= RATE_LIMIT_MAX_COMMANDS:
        return False, f"⏳ Rate limit: {RATE_LIMIT_MAX_COMMANDS} commands per minute. Wait a bit."

    # Record this command
    command_history[user_id].append((now, command))
    return True, ""


def get_service_status() -> str:
    """Get systemd service status via CLI"""
    result = run_cli(["status"])
    try:
        data = json.loads(result.stdout)
        return data.get("status", "unknown")
    except:
        return "unknown"


def get_gpu_info() -> str:
    """Get GPU utilization and memory"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        util, used, total = result.stdout.strip().split(",")
        return f"GPU: {util.strip()}%, {used.strip()}/{total.strip()}MB"
    except Exception as e:
        return f"GPU: N/A ({e})"


def get_current_step() -> int:
    """Get current training step from log file"""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "Step" in line and "/" in line:
                    import re
                    match = re.search(r"Step (\d+)/", line)
                    if match:
                        return int(match.group(1))
        return 0
    except:
        return 0


def get_loss() -> float:
    """Get current loss from logs"""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "loss=" in line:
                    match = line.split("loss=")
                    if len(match) > 1:
                        return float(match[1].split()[0])
    except Exception:
        pass
    return 0.0


def get_latest_checkpoint() -> str:
    """Get latest checkpoint step via CLI"""
    result = run_cli(["checkpoint"])
    try:
        data = json.loads(result.stdout)
        step = data.get("step", 0)
        return str(step) if step > 0 else "N/A"
    except:
        return "N/A"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🚀 *Bulba1 Bot*\n\n"
        "Commands:\n"
        "/status - Training status\n"
        "/gpu - GPU info\n"
        "/system - System info\n"
        "/logs - Last log lines\n"
        "/plot - Training graph\n"
        "/checkpoint - Latest checkpoint\n"
        "/eta - Time remaining\n"
        "/chat [msg] - Chat (3/day public)\n\n"
        "Admin: /stop /train /restart /reset /quit /shutdown /pull /restart_bot",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await start_command(update, context)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    user_id = update.effective_user.id
    allowed, msg = check_rate_limit(user_id, "status")
    if not allowed:
        await update.message.reply_text(msg)
        return

    status = get_service_status()
    gpu = get_gpu_info()
    step = get_current_step()
    loss = get_loss()
    checkpoint = get_latest_checkpoint()

    status_emoji = "✅" if status == "active" else "❌"

    msg = f"""
{status_emoji} *Training Status*

Service: `{status}`
{gpu}
Step: `{step}/100000`
Loss: `{loss:.4f}`
Checkpoint: `{checkpoint}`
"""
    await update.message.reply_text(msg, parse_mode="Markdown")


async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /logs command - show last N lines"""
    user_id = update.effective_user.id
    allowed, msg = check_rate_limit(user_id, "logs")
    if not allowed:
        await update.message.reply_text(msg)
        return

    try:
        n = 10
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                pass

        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()[-n:]
            # Escape special markdown chars, use plain text
            log_text = "".join(lines).replace("_", "❟").replace("*", "•").replace("`", "›")
            await update.message.reply_text(f"📝 Last {n} lines from logs:\n\n{log_text}")
        else:
            await update.message.reply_text("❌ Log file not found")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def checkpoint_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /checkpoint command"""
    user_id = update.effective_user.id
    allowed, msg = check_rate_limit(user_id, "checkpoint")
    if not allowed:
        await update.message.reply_text(msg)
        return

    checkpoint = get_latest_checkpoint()
    step = get_current_step()
    loss = get_loss()

    msg = f"""
📦 *Checkpoint Info*

Latest: `{checkpoint}`
Current step: `{step}`
Current loss: `{loss:.4f}`
"""
    await update.message.reply_text(msg, parse_mode="Markdown")


async def plot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /plot command - generate and send visualization"""
    user_id = update.effective_user.id
    allowed, msg = check_rate_limit(user_id, "plot")
    if not allowed:
        await update.message.reply_text(msg)
        return

    await update.message.reply_text("🎨 Generating plot...")

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from tools.log_viz import parse_log, generate_plots

        if not LOG_FILE.exists():
            await update.message.reply_text("❌ Log file not found")
            return

        df = parse_log(str(LOG_FILE))
        if df is None or len(df) == 0:
            await update.message.reply_text("❌ No data in log file")
            return

        output_path = PROJECT_ROOT / "logs" / "bulba1_plot.png"
        generate_plots(df, str(output_path), "Bulba1 Training")

        # Send the image
        await update.message.reply_photo(photo=open(output_path, "rb"))
        await update.message.reply_text(
            f"✅ Plot generated!\nSteps: {len(df)}, Loss: {df['loss'].iloc[-1]:.4f}"
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating plot: {e}")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown messages"""
    await update.message.reply_text("Unknown command. Use /help for available commands.")


# Admin-only commands
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop training - admin only with confirmation"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    # Check if already stopped
    if get_service_status() != "active":
        await update.message.reply_text("⚠️ Training already stopped")
        return

    await update.message.reply_text(
        "⏹ *Confirm stop training?*\n\n"
        "Training will pause. Resume with /train\n\n"
        "Reply /confirm_stop to confirm",
        parse_mode="Markdown",
    )


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start training - admin only"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    result = run_cli(["start"])
    await update.message.reply_text("▶ Training started")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart training - admin only"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    result = run_cli(["restart"])
    await update.message.reply_text("🔄 Training restarted")


async def pull_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Git pull latest code - admin only"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text("📥 Pulling latest code...")
    result = subprocess.run(
        ["git", "pull", "origin", "master"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        await update.message.reply_text(f"✅ Pulled:\n{result.stdout}")
    else:
        await update.message.reply_text(f"❌ Error:\n{result.stderr}")


async def restart_bot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart bot service - admin only"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text("🔄 Restarting bot...")
    subprocess.run(["systemctl", "--user", "restart", "bulba1-bot"])
    await update.message.reply_text("✅ Bot restarting...")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset training from step 0 - admin only with confirmation"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text(
        "🗑 *Confirm reset training?*\n\n"
        "⚠️ ALL checkpoints will be DELETED!\n"
        "Training starts from step 0\n\n"
        "Reply /confirm_reset to confirm",
        parse_mode="Markdown",
    )


async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shutdown computer - admin only with confirmation"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text(
        "⚠️ *Shutdown computer?*\n\nReply /confirm_shutdown to confirm.", parse_mode="Markdown"
    )


async def confirm_shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm shutdown - admin only"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text("🛑 Shutting down...")
    subprocess.run(["systemctl", "poweroff"])


async def gpu_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get detailed GPU info directly"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        name, util, temp, mem_used, mem_total, power = result.stdout.strip().split(",")

        msg = f"🎮 *GPU Info*\n\n"
        msg += f"Name: `{name.strip()}`\n"
        msg += f"Util: `{util.strip()}`\n"
        msg += f"Temp: `{temp.strip()}°C`\n"
        msg += f"Memory: `{mem_used.strip()}/{mem_total.strip()} MB`\n"
        msg += f"Power: `{power.strip()} W`"

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def system_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get system info directly"""
    try:
        # RAM
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    free = int(line.split()[1]) // 1024
        used = total - free

        # CPU
        with open("/proc/loadavg") as f:
            load = f.read().split()[0]
        with open("/proc/cpuinfo") as f:
            cpu_count = f.read().count("processor")

        # Disk
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        disk = result.stdout.split("\n")[1].split()[2:4]

        msg = f"💻 *System Info*\n\n"
        msg += f"CPU: `{cpu_count} cores` | Load: `{load}`\n"
        msg += f"RAM: `{used}/{total} MB` ({used * 100 // total}%)\n"
        msg += f"Disk: `{disk[0]}/{disk[1]}`"

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


# Notification system - poll for events
EVENT_FILE = PROJECT_ROOT / "logs" / "events.json"
last_event_time = 0


async def check_notifications(app: Application):
    """Check for new events and notify"""
    global last_event_time
    try:
        if EVENT_FILE.exists():
            mtime = os.path.getmtime(EVENT_FILE)
            if mtime > last_event_time:
                last_event_time = mtime
                with open(EVENT_FILE) as f:
                    events = json.load(f)
                # Send latest events
                for event in events[-3:]:
                    await app.bot.send_message(
                        chat_id=ADMIN_IDS, text=event.get("message", "Event"), parse_mode="Markdown"
                    )
                # Clear old events
                open(EVENT_FILE, "w").close()
    except Exception:
        pass


def add_event(message: str):
    """Add event to notification queue"""
    try:
        events = []
        if EVENT_FILE.exists():
            with open(EVENT_FILE) as f:
                events = json.load(f)
        events.append({"message": message, "time": time.time()})
        events = events[-10:]
        with open(EVENT_FILE, "w") as f:
            json.dump(events, f)
    except Exception:
        pass


async def eta_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get ETA to completion"""
    step = get_current_step()
    loss = get_loss()

    if step == 0:
        await update.message.reply_text("❌ Training not started")
        return

    # Estimate from loss rate
    # Assume ~70 tok/s, 512 seq, batch 5 = ~1400 tokens/step
    # 100k steps remaining, ~70 sec/step = 194 hours (too high)
    # Better: use avg time per step from logs
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                lines = f.readlines()
            # Find last line with timing
            for line in reversed(lines):
                if "tok/s" in line:
                    import re

                    match = re.search(r"tok/s=(\d+)", line)
                    if match:
                        tok_rate = int(match.group(1))
                        remaining_steps = 100000 - step
                        # avg 100 sec/step (from log), tok/s not accurate
                        # Estimate: ~90 sec/step
                        eta_seconds = remaining_steps * 90
                        hours = eta_seconds // 3600
                        minutes = (eta_seconds % 3600) // 60
                        msg = f"⏱️ *ETA*\n\n"
                        msg += f"Current: `{step}/100000` ({step / 1000}%)\n"
                        msg += f"Loss: `{loss:.4f}`\n"
                        msg += f"Remaining: `{hours}h {minutes}m`"
                        await update.message.reply_text(msg, parse_mode="Markdown")
                        return
    except Exception as e:
        pass

    await update.message.reply_text(
        f"⏱️ Step: `{step}/100000` | Loss: `{loss:.4f}`", parse_mode="Markdown"
    )


# Remote config - save desired config, training reads it on restart
CONFIG_FILE = PROJECT_ROOT / "logs" / "desired_config.json"


async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set remote config - batch_size, lr etc"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "⚙️ *Remote Config*\n\n"
            "Usage: /config batch_size=5 lr=1e-4\n\n"
            "Available: batch_size, lr, seq_len",
            parse_mode="Markdown",
        )
        return

    for arg in args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            try:
                if "." in val:
                    value = float(val)
                else:
                    value = int(val)
                result = run_cli(["config", "--key", key, "--value", str(value)])
            except:
                await update.message.reply_text(f"❌ Invalid value: {val}")
                return

    msg = "✅ *Config updated*\n\n" + " ".join(args) + "\nWill apply on restart"
    await update.message.reply_text(msg, parse_mode="Markdown")


# Chat rate limiting for public users
CHAT_LIMIT_DAILY = 3  # Max 3 chats per day for non-admin
chat_usage = {}  # user_id -> [(day, count)]


def check_chat_limit(user_id: int) -> tuple[bool, int]:
    """Check daily chat limit for user. Returns (allowed, remaining)"""
    from datetime import datetime

    today = datetime.now().day

    if user_id not in chat_usage:
        chat_usage[user_id] = []

    # Clean old entries
    chat_usage[user_id] = [(day, cnt) for day, cnt in chat_usage[user_id] if day == today]

    total = sum(cnt for _, cnt in chat_usage[user_id])
    remaining = CHAT_LIMIT_DAILY - total

    if remaining <= 0:
        return False, 0

    # Record this usage
    chat_usage[user_id].append((today, 1))
    return True, remaining


async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chat with the model - limited for public"""
    user_id = update.effective_user.id
    is_admin_user = is_admin(user_id)

    # Check rate limit for non-admins
    if not is_admin_user:
        allowed, remaining = check_chat_limit(user_id)
        if not allowed:
            await update.message.reply_text(
                f"⏳ Daily limit reached (3 chats/day).\nUpgrade to admin or wait tomorrow.",
                parse_mode="Markdown",
            )
            return
        await update.message.reply_text(f"💬 ({remaining} left today)")

    args = context.args
    if not args:
        await update.message.reply_text("Usage: /chat [your message]")
        return

    prompt = " ".join(args)

    # Check if training is active
    training_active = get_service_status() == "active"
    mode_msg = " (CPU mode - training active)" if training_active else ""
    await update.message.reply_text(f"💬 Thinking{mode_msg}...")

    result = run_cli(["generate", "--prompt", prompt, "--tokens", "100"], timeout=180)

    try:
        data = json.loads(result.stdout)
        if "error" in data:
            await update.message.reply_text(f"❌ {data['error']}")
        else:
            await update.message.reply_text(
                f"📝 *You:* {prompt}\n\n💬 *Bot:*\n{data.get('output', 'N/A')}",
                parse_mode="Markdown",
            )
    except:
        await update.message.reply_text(f"❌ Error")


async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop training - admin only with confirmation"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text(
        "🛑 *Confirm quit & stop training?*\n\n"
        "Training stops, bot disconnects.\n"
        "To restart: systemctl --user start bulba1\n\n"
        "Reply /confirm_quit to confirm",
        parse_mode="Markdown",
    )


# Download checkpoint
async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Download latest checkpoint"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    checkpoint = get_latest_checkpoint()
    if checkpoint == "N/A":
        await update.message.reply_text("❌ No checkpoints")
        return

    # Find best checkpoint
    best = CHECKPOINT_DIR / "best.safetensors"
    if not best.exists():
        best = CHECKPOINT_DIR / f"checkpoint_step_{checkpoint}.safetensors"

    if not best.exists():
        await update.message.reply_text("❌ Checkpoint file not found")
        return

    await update.message.reply_text(f"📦 Sending checkpoint {checkpoint}...")
    try:
        await update.message.reply_document(document=open(best, "rb"))
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def export_hf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export checkpoint to HuggingFace format"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    await update.message.reply_text("📤 Exporting to HuggingFace format...")

    result = run_cli(["--export-hf", "--export-dir", "hf_export"], timeout=120)

    if result.returncode == 0:
        await update.message.reply_text("✅ Exported to hf_export/")
    else:
        await update.message.reply_text(f"❌ Error: {result.stderr or result.stdout}")


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Schedule training start"""
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ Admin only")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "⏰ *Schedule*\n\nUsage: /schedule 14:30 (24h format)\nOr /schedule now (immediate)",
            parse_mode="Markdown",
        )
        return

    result = run_cli(["--schedule", args[0]], timeout=10)

    if result.returncode == 0:
        await update.message.reply_text(f"✅ {result.stdout.strip()}")
    else:
        await update.message.reply_text(f"❌ Error: {result.stderr or result.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Bulba1 Telegram Bot")
    parser.add_argument("--token", help="Telegram bot token")
    parser.add_argument("--chat-id", help="Telegram chat ID to send messages to")
    parser.add_argument("--poll", action="store_true", help="Run in polling mode (for testing)")
    args = parser.parse_args()

    # Get token from args, config, or environment
    token = args.token or BOT_TOKEN or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("ERROR: No token. Create bot_config.py or set TELEGRAM_TOKEN")
        sys.exit(1)

    chat_id = args.chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    # Build application
    app = Application.builder().token(token).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("panel", panel_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("logs", logs_command))
    app.add_handler(CommandHandler("plot", plot_command))
    app.add_handler(CommandHandler("checkpoint", checkpoint_command))
    # Admin commands
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("training", train_command))  # alias
    app.add_handler(CommandHandler("restart", restart_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("shutdown", shutdown_command))
    app.add_handler(CommandHandler("confirm_shutdown", confirm_shutdown_command))
    app.add_handler(CommandHandler("gpu", gpu_info_command))
    app.add_handler(CommandHandler("system", system_info_command))
    app.add_handler(CommandHandler("eta", eta_command))
    app.add_handler(CommandHandler("config", config_command))
    app.add_handler(CommandHandler("chat", chat_command))
    app.add_handler(CommandHandler("quit", quit_command))
    app.add_handler(CommandHandler("pull", pull_command))
    app.add_handler(CommandHandler("restart_bot", restart_bot_command))

    # Confirmation handlers
    async def confirm_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            return
        result = run_cli(["stop"])
        await update.message.reply_text("⏹ Training stopped")

    async def confirm_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            return
        result = run_cli(["reset"])
        await update.message.reply_text("🗑 Reset! Training from step 0")

    async def confirm_quit(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update.effective_user.id):
            return
        subprocess.run(["systemctl", "--user", "stop", "bulba1"])
        await update.message.reply_text("🛑 Stopped. Bye!")

    app.add_handler(CommandHandler("confirm_stop", confirm_stop))
    app.add_handler(CommandHandler("confirm_reset", confirm_reset))
    app.add_handler(CommandHandler("confirm_quit", confirm_quit))

    app.add_handler(CommandHandler("download", download_command))
    app.add_handler(CommandHandler("export_hf", export_hf_command))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print(f"🤖 Bulba1 Bot started! (config: {_config_loaded})")
    print(f"Commands: /start, /status, /logs, /plot, /checkpoint, /chat, /help")
    print(f"Admin: /stop, /train, /restart, /reset, /quit, /shutdown")

    if chat_id:
        # One-way mode: just send startup message and exit
        import asyncio

        async def send_startup():
            await app.bot.send_message(
                chat_id=chat_id, text="🚀 Bulba1 Bot connected!\nUse /status to check training."
            )

        asyncio.run(send_startup())
        print(f"📤 Sent startup message to chat {chat_id}")
    else:
        # Polling mode
        app.run_polling()


if __name__ == "__main__":
    main()
