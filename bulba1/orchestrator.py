import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime


class TrainingOrchestrator:
    PHASES = [
        "data_download",
        "data_filter",
        "tokenizer_train",
        "model_train",
        "final_eval",
    ]

    SIZE_TO_PARAMS = {
        "125M": 125_000_000,
        "350M": 350_000_000,
        "766M": 766_000_000,
        "1B": 1_000_000_000,
    }

    @staticmethod
    def parse_model_size(size_str: str) -> int:
        s = size_str.strip().upper()
        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        for suffix, mult in multipliers.items():
            if s.endswith(suffix):
                return int(float(s[:-1]) * mult)
        return int(s)

    def __init__(
        self,
        model_size: str = "766M",
        total_steps: int = 120_000,
        data_dir: str = "data",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        resume: bool = True,
        max_docs: int = None,
        max_items: int = None,
        batch_size: int = 1,
        seq_len: int = 128,
        compile_model: bool = True,
        remove_templated: bool = True,
        remove_auto_generated: bool = True,
        min_doc_length: int = 100,
        max_doc_length: int = 500_000,
    ):
        self.model_size = model_size
        self.total_steps = total_steps
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.filtered_dir = self.data_dir / "filtered"
        self.train_dir = self.data_dir / "train"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.resume = resume
        self.max_docs = max_docs
        self.max_items = max_items
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.compile_model = compile_model
        self.remove_templated = remove_templated
        self.remove_auto_generated = remove_auto_generated
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.state_file = self.log_dir / "orchestrator_state.json"
        self.running = True

        self._setup_dirs()
        self._setup_signal_handlers()
        self.state = self._load_state()

    def _setup_dirs(self):
        for d in [
            self.data_dir,
            self.raw_dir,
            self.filtered_dir,
            self.train_dir,
            self.checkpoint_dir,
            self.log_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_signal_handlers(self):
        def graceful_shutdown(signum, frame):
            print(f"\n[Orchestrator] Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "current_phase": None,
            "completed_phases": [],
            "start_time": None,
            "last_update": None,
            "errors": [],
        }

    def _save_state(self):
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)

        log_file = self.log_dir / "orchestrator.log"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    def _run_command(self, cmd: list, desc: str, timeout: int = None) -> bool:
        self._log(f"Starting: {desc}")
        self._log(f"Command: {' '.join(cmd)}")

        log_file = self.log_dir / f"{desc.replace(' ', '_')}.log"

        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                start_time = time.time()
                while process.poll() is None:
                    if not self.running:
                        process.terminate()
                        process.wait(timeout=30)
                        if process.poll() is None:
                            process.kill()
                        self._log(f"Interrupted: {desc}")
                        return False

                    if timeout and (time.time() - start_time) > timeout:
                        process.terminate()
                        self._log(f"Timeout after {timeout}s: {desc}")
                        return False

                    time.sleep(1)

                if process.returncode == 0:
                    self._log(f"Completed: {desc}")
                    return True
                else:
                    self._log(f"Failed (exit {process.returncode}): {desc}")
                    return False

        except Exception as e:
            self._log(f"Error in {desc}: {e}")
            return False

    def _check_data_exists(self) -> bool:
        txt_files = list(self.raw_dir.glob("*.txt"))
        return len(txt_files) > 0

    def _check_filtered_exists(self) -> bool:
        return self.filtered_dir.exists() and len(list(self.filtered_dir.glob("*.txt"))) > 100

    def _check_tokenizer_exists(self) -> bool:
        return (self.data_dir / "tokenizer.json").exists()

    def _check_checkpoint_exists(self) -> bool:
        return len(list(self.checkpoint_dir.glob("*.safetensors"))) > 0

    def phase_data_download(self) -> bool:
        if self._check_data_exists():
            self._log("Raw data already exists, skipping download")
            return True

        self._log("=== PHASE: Data Download ===")

        cmd = [sys.executable, "-m", "scripts.prepare_data", "--output-dir", str(self.raw_dir)]
        if self.max_items is not None:
            cmd.extend(["--max-items", str(self.max_items)])
        success = self._run_command(cmd, "data download", timeout=3600)

        if not success or not self._check_data_exists():
            self._log("Download failed or no internet, generating synthetic data...")
            self._generate_synthetic_data()

        return self._check_data_exists()

    def _generate_synthetic_data(self):
        self._log("Generating synthetic training data...")
        synth_file = self.raw_dir / "synthetic.txt"

        code_snippets = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
            "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n    def forward(self, x):\n        for layer in self.layers:\n            x = layer(x)\n        return x\n",
            "import torch\nimport torch.nn as nn\n\nmodel = nn.Transformer()\noptimizer = torch.optim.Adam(model.parameters())\n",
        ]

        with open(synth_file, "w") as f:
            for i in range(50_000):
                snippet = code_snippets[i % len(code_snippets)]
                f.write(f"# Example {i}\n{snippet}\n")

        self._log(f"Generated synthetic data: {synth_file}")

    def phase_data_filter(self) -> bool:
        if self._check_filtered_exists():
            self._log("Filtered data already exists, skipping filtering")
            return True

        self._log("=== PHASE: Data Filtering ===")

        if not self._check_data_exists():
            self._log("No raw data found!")
            return False

        cmd = [
            sys.executable,
            "-m",
            "bulba1.data.quality",
            "--raw-dir",
            str(self.raw_dir),
            "--output-dir",
            str(self.filtered_dir),
            "--remove-templated",
            "1" if self.remove_templated else "0",
            "--remove-auto-generated",
            "1" if self.remove_auto_generated else "0",
            "--min-doc-length",
            str(self.min_doc_length),
            "--max-doc-length",
            str(self.max_doc_length),
        ]
        if self.max_docs is not None:
            cmd.extend(["--max-docs", str(self.max_docs)])

        success = self._run_command(cmd, "data filtering", timeout=7200)
        return success or self._check_filtered_exists()

    def phase_tokenizer_train(self) -> bool:
        if self._check_tokenizer_exists():
            self._log("Tokenizer already exists, skipping training")
            return True

        self._log("=== PHASE: Tokenizer Training ===")

        target_params = self.parse_model_size(self.model_size)

        script = f"""
import glob
from bulba1.data.tokenizer import SmartTokenizer

files = glob.glob("{self.filtered_dir}/**/*.txt", recursive=True)
if not files:
    files = glob.glob("{self.raw_dir}/**/*.txt", recursive=True)

print(f"[Tokenizer] Found {{len(files)}} files for training")

tokenizer = SmartTokenizer(
    vocab_size=None,
    model_path="{self.data_dir}/tokenizer.json",
    target_params={target_params},
    auto_detect=True,
    sample_size=10_000_000,
)

tokenizer.train(files)
report = tokenizer.get_analysis_report()
print(report)
print(f"[Tokenizer] Final vocab_size: {{tokenizer.get_vocab_size()}}")
"""

        script_file = self.log_dir / "train_tokenizer.py"
        with open(script_file, "w") as f:
            f.write(script)

        cmd = [sys.executable, str(script_file)]
        success = self._run_command(cmd, "tokenizer training", timeout=3600)

        return success or self._check_tokenizer_exists()

    def phase_model_train(self) -> bool:
        self._log("=== PHASE: Model Training ===")

        cmd = [
            sys.executable,
            "-m",
            "bulba1.cli",
            "--params",
            self.model_size,
            "--steps",
            str(self.total_steps),
            "--batch-size",
            str(self.batch_size),
            "--seq-len",
            str(self.seq_len),
            "--data-dir",
            str(self.train_dir),
            "--remove-templated",
            "1" if self.remove_templated else "0",
            "--remove-auto-generated",
            "1" if self.remove_auto_generated else "0",
            "--min-doc-length",
            str(self.min_doc_length),
            "--max-doc-length",
            str(self.max_doc_length),
        ]

        if self.compile_model:
            cmd.append("--compile")

        if self.resume and self._check_checkpoint_exists():
            cmd.append("--resume")
            self._log("Resuming from checkpoint")

        success = self._run_command(cmd, "model training", timeout=None)

        return success

    def phase_final_eval(self) -> bool:
        self._log("=== PHASE: Final Evaluation ===")

        checkpoints = list(self.checkpoint_dir.glob("*.safetensors"))
        if not checkpoints:
            self._log("No checkpoints found for evaluation")
            return False

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        self._log(f"Evaluating checkpoint: {latest.name}")

        target_params = self.parse_model_size(self.model_size)

        eval_script = f"""
import torch
from bulba1.model.minichat import MiniChat
from bulba1.data.tokenizer import SmartTokenizer
from bulba1.utils.config import find_architecture

cfg = find_architecture({target_params})
model = MiniChat(cfg).cuda()
tokenizer = SmartTokenizer(model_path="data/tokenizer.json").load()

import safetensors.torch
state = safetensors.torch.load_file("{latest}")
model.load_state_dict(state, strict=False)

model.eval()
prompts = [
    "The theory of relativity states that",
    "def quicksort(arr):",
    "In the field of machine learning,",
    "Once upon a time in a distant galaxy,",
]

for prompt in prompts:
    ids = torch.tensor([tokenizer.encode(prompt)], device='cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, temperature=0.8)
    text = tokenizer.decode(out[0].tolist())
    print(f"\\nPrompt: {{prompt}}")
    print(f"Output: {{text}}")
"""

        script_file = self.log_dir / "final_eval.py"
        with open(script_file, "w") as f:
            f.write(eval_script)

        cmd = [sys.executable, str(script_file)]
        return self._run_command(cmd, "final evaluation", timeout=600)

    def run(self):
        self._log("=" * 60)
        self._log("BULBA 1 TRAINING ORCHESTRATOR")
        self._log("Fully Automatic Mode - Launch and Wait")
        self._log("=" * 60)
        self._log(f"Model size: {self.model_size}")
        self._log(f"Total steps: {self.total_steps:,}")
        self._log(f"Data dir: {self.data_dir}")
        self._log(f"Resume: {self.resume}")
        self._log("")

        self.state["start_time"] = datetime.now().isoformat()

        phases_to_run = []
        for phase_name in self.PHASES:
            if phase_name not in self.state["completed_phases"]:
                phases_to_run.append(phase_name)

        if not phases_to_run:
            self._log("All phases already completed!")
            return True

        self._log(f"Phases to run: {phases_to_run}")

        for phase_name in phases_to_run:
            if not self.running:
                self._log("Orchestrator stopped by user")
                return False

            self.state["current_phase"] = phase_name
            self._save_state()

            phase_method = getattr(self, f"phase_{phase_name}")
            success = phase_method()

            if success:
                self.state["completed_phases"].append(phase_name)
                self._save_state()
            else:
                self._log(f"Phase {phase_name} failed!")
                self.state["errors"].append(
                    {
                        "phase": phase_name,
                        "time": datetime.now().isoformat(),
                    }
                )
                self._save_state()

                if phase_name in ["data_download", "data_filter", "tokenizer_train"]:
                    self._log("Critical phase failed, stopping pipeline")
                    return False

        self.state["current_phase"] = "complete"
        self._save_state()

        self._log("=" * 60)
        self._log("ALL PHASES COMPLETE!")
        self._log("=" * 60)

        if self.state["start_time"]:
            start = datetime.fromisoformat(self.state["start_time"])
            elapsed = datetime.now() - start
            self._log(f"Total time: {elapsed}")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bulba 1 Automatic Training Orchestrator")
    parser.add_argument(
        "--model-size", default="766M", help="Model size (e.g., 125M, 766M, 1B, 7B, 13B)"
    )
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (can point to large drive)"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Max docs after filtering (None = no limit)"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Max items per source (None = no limit)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--remove-templated", type=int, default=1)
    parser.add_argument("--remove-auto-generated", type=int, default=1)
    parser.add_argument("--min-doc-length", type=int, default=100)
    parser.add_argument("--max-doc-length", type=int, default=500_000)
    args = parser.parse_args()

    orchestrator = TrainingOrchestrator(
        model_size=args.model_size,
        total_steps=args.steps,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume=not args.no_resume,
        max_docs=args.max_docs,
        max_items=args.max_items,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        compile_model=not args.no_compile,
        remove_templated=bool(args.remove_templated),
        remove_auto_generated=bool(args.remove_auto_generated),
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
    )

    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
