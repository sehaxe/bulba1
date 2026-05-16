import os
import glob
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional, Tuple, Dict
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from collections import defaultdict


class SmartTokenizer:
    """Tokenizer with automatic vocabulary size optimization.
    
    Automatically finds the ideal vocab_size based on:
    - Dataset characteristics (language mix, domain)
    - Tokenization efficiency (bytes per token)
    - Model size constraints
    - Compression ratio knee point detection
    """
    
    # Recommended vocab sizes for different model scales
    VOCAB_SIZE_GUIDELINES = {
        # (min_params, max_params): (min_vocab, max_vocab, default)
        (0, 100_000_000): (8000, 16000, 12000),
        (100_000_000, 500_000_000): (16000, 32000, 24000),
        (500_000_000, 1_500_000_000): (24000, 48000, 32000),
        (1_500_000_000, 5_000_000_000): (32000, 64000, 48000),
        (5_000_000_000, float('inf')): (48000, 100000, 64000),
    }
    
    # Language-specific adjustments
    LANGUAGE_MULTIPLIERS = {
        'en': 1.0,
        'code': 1.3,      # Code needs more tokens for identifiers
        'multilingual': 1.5,
        'ja': 1.4,        # Japanese/Chinese need larger vocabs
        'zh': 1.4,
        'ko': 1.3,
        'ar': 1.2,
        'ru': 1.1,
    }
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,  # None = auto-detect
        model_path: str = "data/tokenizer.json",
        target_params: Optional[int] = None,  # For scaling guidelines
        auto_detect: bool = True,
        sample_size: int = 10_000_000,  # Bytes to sample for analysis
        vocab_candidates: Optional[List[int]] = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.target_params = target_params
        self.auto_detect = auto_detect and vocab_size is None
        self.sample_size = sample_size
        self.tokenizer = None
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        self._analysis_results = None
        
        # Candidate vocab sizes for auto-detection
        if vocab_candidates is None:
            self.vocab_candidates = [8000, 12000, 16000, 24000, 32000, 48000, 64000]
        else:
            self.vocab_candidates = vocab_candidates
    
    def _sample_data(self, files: List[str]) -> str:
        """Sample text from training files for analysis."""
        total_bytes = 0
        samples = []
        
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    # Read in chunks to handle large files
                    while total_bytes < self.sample_size:
                        chunk = f.read(100_000)
                        if not chunk:
                            break
                        samples.append(chunk)
                        total_bytes += len(chunk.encode('utf-8'))
            except Exception:
                continue
            
            if total_bytes >= self.sample_size:
                break
        
        return "\n".join(samples)
    
    def _detect_language_mix(self, text: str) -> Dict[str, float]:
        """Detect language composition of the dataset."""
        # Simple heuristic based on character ranges
        total_chars = len(text)
        if total_chars == 0:
            return {'en': 1.0}
        
        lang_counts = defaultdict(int)
        
        for char in text[:100_000]:  # Sample for speed
            cp = ord(char)
            if char.isascii():
                lang_counts['en'] += 1
            elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                lang_counts['zh'] += 1
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                lang_counts['ja'] += 1
            elif 0xAC00 <= cp <= 0xD7AF:
                lang_counts['ko'] += 1
            elif 0x0600 <= cp <= 0x06FF:
                lang_counts['ar'] += 1
            elif 0x0400 <= cp <= 0x04FF:
                lang_counts['ru'] += 1
            else:
                lang_counts['other'] += 1
        
        # Normalize
        total = sum(lang_counts.values())
        return {k: v/total for k, v in lang_counts.items()}
    
    def _detect_code_ratio(self, text: str) -> float:
        """Estimate what fraction of text is code."""
        # Heuristics: common code patterns
        code_indicators = [
            'def ', 'class ', 'function', 'return ', 'import ', 'from ',
            'if __name__', 'const ', 'let ', 'var ', '#include', 'public class',
            '=>', '->', '{}', '[]', '===', '!==', '// ', '/*', '*/',
        ]
        
        sample = text[:500_000]
        indicator_count = sum(1 for ind in code_indicators if ind in sample)
        return min(indicator_count / len(code_indicators), 1.0)
    
    def _create_base_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+|[^\s\w]",
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer
    
    def _train_candidate(self, text: str, vocab_size: int) -> Tuple[Tokenizer, Dict]:
        """Train a candidate tokenizer and return metrics."""
        tokenizer = self._create_base_tokenizer()
        
        # Write sample to temp file
        temp_path = "/tmp/smart_tokenizer_sample.txt"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            min_frequency=2,
        )
        
        tokenizer.train([temp_path], trainer)
        
        # Evaluate tokenization efficiency
        encoding = tokenizer.encode(text[:100_000])
        tokens = encoding.ids
        
        bytes_per_token = len(text[:100_000].encode('utf-8')) / max(len(tokens), 1)
        chars_per_token = len(text[:100_000]) / max(len(tokens), 1)
        
        # Calculate entropy of token distribution
        token_counts = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1
        
        total_tokens = len(tokens)
        entropy = 0.0
        for count in token_counts.values():
            p = count / total_tokens
            entropy -= p * math.log2(p)
        
        # Coverage: what fraction of characters are covered (not <unk>)
        unk_count = tokens.count(tokenizer.token_to_id("<unk>"))
        coverage = 1.0 - (unk_count / max(len(tokens), 1))
        
        metrics = {
            'vocab_size': vocab_size,
            'bytes_per_token': bytes_per_token,
            'chars_per_token': chars_per_token,
            'entropy': entropy,
            'coverage': coverage,
            'efficiency_score': bytes_per_token * coverage,  # Higher is better
        }
        
        return tokenizer, metrics
    
    def _find_knee_point(self, metrics_list: List[Dict]) -> int:
        """Find the knee point where marginal gain diminishes.
        
        Uses the "elbow method": find point with maximum curvature.
        """
        if len(metrics_list) < 3:
            return metrics_list[-1]['vocab_size']
        
        vocab_sizes = [m['vocab_size'] for m in metrics_list]
        efficiency_scores = [m['efficiency_score'] for m in metrics_list]
        
        # Normalize to [0, 1]
        v_min, v_max = min(vocab_sizes), max(vocab_sizes)
        e_min, e_max = min(efficiency_scores), max(efficiency_scores)
        
        if e_max == e_min:
            return metrics_list[len(metrics_list)//2]['vocab_size']
        
        points = []
        for v, e in zip(vocab_sizes, efficiency_scores):
            x = (v - v_min) / (v_max - v_min)
            y = (e - e_min) / (e_max - e_min)
            points.append((x, y))
        
        # Find point with maximum distance from line between first and last points
        x1, y1 = points[0]
        x2, y2 = points[-1]
        
        max_distance = 0
        knee_idx = 0
        
        for i, (x0, y0) in enumerate(points):
            # Distance from point to line
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distance = numerator / max(denominator, 1e-10)
            
            if distance > max_distance:
                max_distance = distance
                knee_idx = i
        
        return metrics_list[knee_idx]['vocab_size']
    
    def _apply_guidelines(
        self, 
        base_vocab_size: int, 
        target_params: Optional[int],
        language_mix: Dict[str, float],
        code_ratio: float
    ) -> int:
        """Apply model size and language guidelines."""
        adjusted = base_vocab_size
        
        # Apply language multipliers
        dominant_multiplier = 1.0
        for lang, ratio in language_mix.items():
            if lang in self.LANGUAGE_MULTIPLIERS and ratio > 0.1:
                # Weighted by ratio
                multiplier = 1.0 + (self.LANGUAGE_MULTIPLIERS[lang] - 1.0) * ratio
                dominant_multiplier = max(dominant_multiplier, multiplier)
        
        adjusted = int(adjusted * dominant_multiplier)
        
        # Apply code multiplier
        if code_ratio > 0.3:
            adjusted = int(adjusted * (1.0 + code_ratio * 0.3))
        
        # Constrain by model size guidelines
        if target_params is not None:
            for (min_p, max_p), (min_v, max_v, default_v) in self.VOCAB_SIZE_GUIDELINES.items():
                if min_p <= target_params < max_p:
                    # If auto-detected is outside recommended range, clip it
                    adjusted = max(min_v, min(max_v, adjusted))
                    break
        
        # Round to nearest 1000 for clean numbers
        adjusted = round(adjusted / 1000) * 1000
        
        return adjusted
    
    def analyze(self, files: List[str]) -> Dict:
        """Analyze dataset and determine optimal vocab size."""
        print("[SmartTokenizer] Starting vocabulary size analysis...")
        
        # Sample data
        sample_text = self._sample_data(files)
        if len(sample_text) < 1000:
            print("[SmartTokenizer] Not enough data for analysis, using default 32000")
            self.vocab_size = 32000
            return {'vocab_size': 32000, 'reason': 'insufficient_data'}
        
        print(f"[SmartTokenizer] Analyzing {len(sample_text):,} characters...")
        
        # Detect language and domain
        language_mix = self._detect_language_mix(sample_text)
        code_ratio = self._detect_code_ratio(sample_text)
        
        print(f"[SmartTokenizer] Language mix: {language_mix}")
        print(f"[SmartTokenizer] Code ratio: {code_ratio:.2%}")
        
        # Train candidate tokenizers
        print("[SmartTokenizer] Training candidate tokenizers...")
        metrics_list = []
        
        for candidate_size in self.vocab_candidates:
            print(f"  Testing vocab_size={candidate_size}...", end=" ")
            _, metrics = self._train_candidate(sample_text, candidate_size)
            metrics_list.append(metrics)
            print(f"bpt={metrics['bytes_per_token']:.2f}, coverage={metrics['coverage']:.2%}")
        
        # Find knee point
        knee_vocab = self._find_knee_point(metrics_list)
        print(f"[SmartTokenizer] Knee point detected at vocab_size={knee_vocab}")
        
        # Apply guidelines
        final_vocab = self._apply_guidelines(
            knee_vocab,
            self.target_params,
            language_mix,
            code_ratio
        )
        
        print(f"[SmartTokenizer] After guideline adjustments: vocab_size={final_vocab}")
        
        self._analysis_results = {
            'recommended_vocab_size': final_vocab,
            'knee_point': knee_vocab,
            'language_mix': dict(language_mix),
            'code_ratio': code_ratio,
            'candidates': metrics_list,
        }
        
        self.vocab_size = final_vocab
        return self._analysis_results
    
    def _file_line_iterator(self, files, max_lines=None):
        """Генератор строк из многих файлов, без загрузки в память."""
        count = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        count += 1
                        if max_lines and count >= max_lines:
                            return

    def train(self, files: List[str], force_vocab_size: Optional[int] = None):
        """Тренирует токенизатор потоково, без OOM."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if self.auto_detect and force_vocab_size is None:
            analysis = self.analyze(files)
            print(f"[SmartTokenizer] Auto-selected vocab_size={self.vocab_size}")
        elif force_vocab_size is not None:
            self.vocab_size = force_vocab_size

        print(f"[SmartTokenizer] Training final tokenizer with vocab_size={self.vocab_size} (streaming)...")

        tokenizer = self._create_base_tokenizer()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            min_frequency=2,
        )

        # Потоковая тренировка: берём первые 5 миллионов строк для скорости,
        # этого более чем достаточно для BPE и не перегружает память.
        max_examples = 5_000_000
        iterator = self._file_line_iterator(files, max_lines=max_examples)
        tokenizer.train_from_iterator(iterator, trainer)

        tokenizer.save(self.model_path)
        self.tokenizer = tokenizer
        print(f"[SmartTokenizer] Saved to {self.model_path}")
    
    def load(self):
        """Load tokenizer from disk."""
        if os.path.exists(self.model_path):
            self.tokenizer = Tokenizer.from_file(self.model_path)
            # Update vocab_size from loaded tokenizer
            self.vocab_size = self.tokenizer.get_vocab_size()
        return self
    
    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        return self.vocab_size or 32000
    
    def get_analysis_report(self) -> str:
        """Get a human-readable analysis report."""
        if self._analysis_results is None:
            return "No analysis performed yet."
        
        r = self._analysis_results
        lines = [
            "=" * 60,
            "SMART TOKENIZER ANALYSIS REPORT",
            "=" * 60,
            f"Recommended vocab_size: {r['recommended_vocab_size']:,}",
            f"Raw knee point:         {r['knee_point']:,}",
            f"Code ratio:             {r['code_ratio']:.1%}",
            "",
            "Language mix:",
        ]
        for lang, ratio in r['language_mix'].items():
            lines.append(f"  {lang}: {ratio:.1%}")
        
        lines.extend([
            "",
            "Candidate performance:",
            "-" * 60,
            f"{'Vocab':>8} {'BPT':>8} {'CPT':>8} {'Entropy':>10} {'Coverage':>10}",
        ])
        for m in r['candidates']:
            lines.append(
                f"{m['vocab_size']:>8,} {m['bytes_per_token']:>8.2f} "
                f"{m['chars_per_token']:>8.2f} {m['entropy']:>10.2f} {m['coverage']:>9.1%}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Backwards compatibility: HFTokenizer is now an alias for SmartTokenizer with auto_detect=False
class HFTokenizer(SmartTokenizer):
    def __init__(self, vocab_size: int = 32000, model_path: str = "data/tokenizer.json"):
        super().__init__(
            vocab_size=vocab_size,
            model_path=model_path,
            auto_detect=False,
        )


class FastTokenizer:
    def __init__(self, model_path: str = "data/tokenizer_fast.json"):
        self.model_path = model_path
        self._tok = None

    def load(self):
        from tokenizers import Tokenizer as Tkz
        self._tok = Tkz.from_file(self.model_path)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str):
        return self._tok.encode(text).ids

    def encode_batch(self, texts):
        return [r.ids for r in self._tok.encode_batch(texts)]

    def decode(self, ids):
        return self._tok.decode(ids)


class TextDataset(IterableDataset):
    def __init__(self, tokenizer: SmartTokenizer, data_dir: str, seq_len: int, return_target: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.return_target = return_target
        self.files = self._find_files(data_dir)
        self.stride = max(1, seq_len // 2)

    def _find_files(self, data_dir: str) -> List[str]:
        patterns = [os.path.join(data_dir, "**/*.txt")]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        return sorted(files)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]

        buffer = []
        max_buffer_len = self.seq_len * 20
        for path in files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    data = f.read(50000)
                    if not data:
                        break
                    ids = self.tokenizer.encode(data)
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[:self.seq_len + 1]
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        if self.return_target:
                            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                            yield input_ids, target_ids
                        else:
                            yield input_ids
                        buffer = buffer[self.stride:]
                    if len(buffer) > max_buffer_len:
                        buffer = buffer[-max_buffer_len:]
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    if self.return_target:
                        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                        yield input_ids, target_ids
                    else:
                        yield input_ids
                    buffer = buffer[self.stride:]


class BinaryDataset(IterableDataset):
    def __init__(self, data_dir: str, seq_len: int, return_target: bool = True, use_mmap: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.return_target = return_target
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.bin"), recursive=True))
        self.stride = max(1, seq_len // 2)
        self.use_mmap = use_mmap

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]

        for path in files:
            if self.use_mmap:
                arr = np.memmap(path, dtype=np.int32, mode='r')
            else:
                arr = np.fromfile(path, dtype=np.int32)
            idx = 0
            while idx + self.seq_len + 1 <= len(arr):
                chunk = arr[idx:idx + self.seq_len + 1]
                input_ids = torch.from_numpy(chunk[:-1].copy()).long()
                if self.return_target:
                    target_ids = torch.from_numpy(chunk[1:].copy()).long()
                    yield input_ids, target_ids
                else:
                    yield input_ids
                idx += self.stride


def create_dataloader(tokenizer: SmartTokenizer, data_dir: str, batch_size: int, seq_len: int, num_workers: int = 0, shuffle: bool = True, return_target: bool = True, prefetch_factor: int = 2):
    bin_files = glob.glob(os.path.join(data_dir, "**/*.bin"), recursive=True)
    if bin_files:
        dataset = BinaryDataset(data_dir, seq_len, return_target=return_target)
    else:
        dataset = TextDataset(tokenizer, data_dir, seq_len, return_target=return_target)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    loader = DataLoader(**loader_kwargs)
    return loader
