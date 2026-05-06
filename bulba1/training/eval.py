import torch
import torch.nn.functional as F
from typing import Optional, List


@torch.no_grad()
def compute_perplexity(model, data_loader, device, max_batches: int = None) -> float:
    if max_batches is None:
        max_batches = getattr(model.cfg, "eval_max_batches", 10)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    clr = model.cfg.num_clr_tokens

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            input_ids, targets = batch[0].to(device), batch[1].to(device)
        else:
            input_ids = batch.to(device)
            targets = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        dev_type = getattr(device, "type", str(device).split(":")[0])
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=model.cfg.use_f16):
            logits, _, _, aux = model(input_ids)

        T_eff = min(input_ids.size(1), logits.size(1) - clr)
        text_logits = logits[:, clr : (clr + T_eff), :].reshape(-1, model.cfg.vocab_size)
        targets_flat = targets[:, :T_eff].reshape(-1)
        loss = F.cross_entropy(text_logits, targets_flat, reduction="sum")
        total_loss += loss.item()
        total_tokens += targets_flat.numel()

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


@torch.no_grad()
def generate_samples(
    model, tokenizer, prompts: List[str], device, max_new_tokens: int = 50, temperature: float = 0.8
) -> List[str]:
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        text = tokenizer.decode(output[0].tolist())
        results.append(text)
    return results


def run_eval(model, tokenizer, eval_loader, device, prompts: Optional[List[str]] = None):
    ppl = compute_perplexity(model, eval_loader, device)
    print(f"Perplexity: {ppl:.2f}")

    if prompts:
        samples = generate_samples(model, tokenizer, prompts, device)
        for prompt, sample in zip(prompts, samples):
            print(f"\nPrompt: {prompt}")
            print(f"Sample: {sample}")

    return {"perplexity": ppl}
