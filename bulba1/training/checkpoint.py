import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


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
        self, model: nn.Module, optimizer, step: int, loss: float, config: dict | None = None, ema=None
    ):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.safetensors")
        state_dict = model.state_dict()
        
        # Clone shared tensors to prevent safetensors saving failure
        save_dict = {}
        seen_storages = set()
        for k, v in state_dict.items():
            storage_ptr = v.untyped_storage().data_ptr() if hasattr(v, "untyped_storage") else None
            if storage_ptr is not None:
                if storage_ptr in seen_storages:
                    v = v.clone()
                else:
                    seen_storages.add(storage_ptr)
            save_dict[k] = v

        save_file(save_dict, path)

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

        best_path = os.path.join(self.checkpoint_dir, "best.safetensors")
        if loss <= (self.checkpoints[0][1] if self.checkpoints else float("inf")):
            save_file(save_dict, best_path)
            best_extras = self._extra_paths(best_path)
            torch.save(optimizer.state_dict(), best_extras["opt"])
            if ema is not None:
                torch.save(ema.state_dict(), best_extras["ema"])

        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.keep_top_k:
            old = self.checkpoints.pop()
            for suffix in [".safetensors", ".json", "_optimizer.pt", "_ema.pt"]:
                f = old[2].replace(".safetensors", suffix)
                if os.path.exists(f):
                    os.remove(f)

        return loss <= min(x[1] for x in self.checkpoints)

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
        elif path == "latest" or not path:
            path = self.find_latest()
        elif isinstance(path, int) or (isinstance(path, str) and path.isdigit()):
            step = int(path)
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.safetensors")
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


