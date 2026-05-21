"""
High-performance structured logger with async batch writing.
"""
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

try:
    import orjson
    def _dumps(obj): return orjson.dumps(obj).decode("utf-8")
except ImportError:
    def _dumps(obj): return json.dumps(obj, separators=(',', ':'))


class AsyncLogger:
    def __init__(self, log_dir="logs", flush_interval=1.0, batch_size=100, max_file_size_mb=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self._queues, self._files, self._file_sizes = {}, {}, {}
        self._lock = threading.Lock()
        self._running = False
        self._flush_event = threading.Event()
        self._recent = deque(maxlen=1000)
        self.start()
    
    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        self._flush_event.set()
        if self._thread: self._thread.join(timeout=2.0)
        with self._lock:
            for f in self._files.values():
                if not f.closed: f.close()
    
    def _get_queue(self, name):
        if name not in self._queues: self._queues[name] = deque(maxlen=10000)
        return self._queues[name]
    
    def _get_file(self, name):
        if name not in self._files or self._files[name].closed:
            path = self.log_dir / f"{name}.jsonl"
            if path.exists() and path.stat().st_size > self.max_file_size:
                for i in range(5, 0, -1):
                    old, new = self.log_dir / f"{name}.{i}.jsonl", self.log_dir / f"{name}.{i+1}.jsonl"
                    if old.exists(): old.rename(new)
                path.rename(self.log_dir / f"{name}.1.jsonl")
            f = open(path, "a", encoding="utf-8")
            self._files[name] = f
            self._file_sizes[name] = path.stat().st_size if path.exists() else 0
        return self._files[name]
    
    def _writer_loop(self):
        last_flush = time.time()
        buffers = {}
        while self._running or any(self._queues.get(n) for n in self._queues):
            self._flush_event.wait(timeout=self.flush_interval)
            self._flush_event.clear()
            has_data = False
            for name, q in list(self._queues.items()):
                if name not in buffers: buffers[name] = []
                while len(buffers[name]) < self.batch_size and q:
                    try:
                        buffers[name].append(q.popleft())
                        has_data = True
                    except IndexError:
                        break
            now = time.time()
            if has_data and (now - last_flush >= self.flush_interval or any(len(b) >= self.batch_size for b in buffers.values()) or not self._running):
                with self._lock:
                    for name, buf in buffers.items():
                        if buf:
                            f = self._get_file(name)
                            data = "".join([_dumps(r) + "\n" for r in buf])
                            f.write(data); f.flush()
                            self._file_sizes[name] = self._file_sizes.get(name, 0) + len(data)
                            buf.clear()
                            if self._file_sizes[name] > self.max_file_size:
                                f.close(); del self._files[name]; self._get_file(name)
                last_flush = now
    
    def log(self, stream, **kwargs):
        record = {"ts": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
        if stream == "train": self._recent.append(record)
        q = self._get_queue(stream)
        if len(q) < q.maxlen: q.append(record)
    
    def flush(self): self._flush_event.set()
    
    def get_recent(self, n=100, stream="train"):
        return list(self._recent)[-n:] if stream == "train" else []

_logger = None
def get_logger(log_dir="logs"):
    global _logger
    if _logger is None: _logger = AsyncLogger(log_dir)
    return _logger
