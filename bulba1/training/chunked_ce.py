import torch


def chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 8192, ignore_index: int = -100
):
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets != ignore_index

    global_max = torch.full(
        (flat_logits.size(0),), -float("inf"), device=logits.device, dtype=logits.dtype
    )
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        local_max = flat_logits[:, start:end].max(dim=-1).values
        global_max = torch.maximum(global_max, local_max)

    exp_sum = torch.zeros_like(global_max)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        exp_sum += torch.exp(flat_logits[:, start:end] - global_max.unsqueeze(-1)).sum(dim=-1)

    log_denom = global_max + torch.log(exp_sum)

    correct_logits = torch.zeros_like(global_max)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_mask = valid_mask & (flat_targets >= start) & (flat_targets < end)
        if chunk_mask.any():
            chunk_targets = flat_targets[chunk_mask] - start
            correct_logits[chunk_mask] = (
                flat_logits[chunk_mask, start:end].gather(1, chunk_targets.unsqueeze(1)).squeeze(1)
            )

    nll = log_denom - correct_logits
    nll = nll[valid_mask]
    return nll.mean()
