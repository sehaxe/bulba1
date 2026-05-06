import os
import json
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", keep_top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoints = []

    def _extra_paths(self, path: str):
        return {
            "opt": path.replace(".safetensors", "_optimizer.pt"),
            "ema": path.replace(".safetensors", "_ema.pt"),
        }

    def save(
        self, model: nn.Module, optimizer, step: int, loss: float, config: dict = None, ema=None
    ):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.safetensors")
        state_dict = model.state_dict()
        save_file(state_dict, path)

        extras = self._extra_paths(path)
        torch.save(optimizer.state_dict(), extras["opt"])
        if ema is not None:
            torch.save(ema.state_dict(), extras["ema"])

        meta_path = path.replace(".safetensors", ".json")
        metadata = {"step": step, "loss": loss}
        if config:
            metadata["config"] = config
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        self.checkpoints.append((step, loss, path))
        self.checkpoints.sort(key=lambda x: x[1])

        best_path = os.path.join(self.checkpoint_dir, "best.safetensors")
        if step == self.checkpoints[0][0]:
            save_file(state_dict, best_path)
            best_extras = self._extra_paths(best_path)
            torch.save(optimizer.state_dict(), best_extras["opt"])
            if ema is not None:
                torch.save(ema.state_dict(), best_extras["ema"])

        while len(self.checkpoints) > self.keep_top_k:
            old = self.checkpoints.pop()
            for suffix in [".safetensors", ".json", "_optimizer.pt", "_ema.pt"]:
                f = old[2].replace(".safetensors", suffix)
                if os.path.exists(f):
                    os.remove(f)

        return step == self.checkpoints[0][0]

    def find_latest(self) -> str:
        import glob

        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.json")
        files = glob.glob(pattern)
        if not files:
            return ""
        files.sort(key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
        latest = files[-1]
        return latest.replace(".json", ".safetensors")

    def load(self, model: nn.Module, path: str, optimizer=None, ema=None):
        if path == "best":
            path = os.path.join(self.checkpoint_dir, "best.safetensors")
        if path == "latest" or not path:
            path = self.find_latest()
        if not path or not os.path.exists(path):
            return None
        state_dict = load_file(path)
        model.load_state_dict(state_dict, strict=False)

        extras = self._extra_paths(path)
        if optimizer is not None and os.path.exists(extras["opt"]):
            optimizer.load_state_dict(torch.load(extras["opt"], weights_only=False))
        if ema is not None and os.path.exists(extras["ema"]):
            ema.load_state_dict(torch.load(extras["ema"], weights_only=False))

        meta_path = path.replace(".safetensors", ".json")
        step = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get("step", 0)
        return step
