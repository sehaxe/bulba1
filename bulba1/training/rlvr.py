import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RLVRTrainer:
    def __init__(self, model, cfg, tokenizer, entropy_threshold=0.8):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.entropy_threshold = entropy_threshold
        self.gamma = getattr(cfg, "rl_gamma", 0.99)
        self.entropy_coef = getattr(cfg, "rl_entropy_coef", 0.01)
        self.kl_coef = getattr(cfg, "rl_kl_coef", 0.01)

    def compute_rewards(self, generated_texts, ground_truth):
        rewards = []
        for text in generated_texts:
            reward = 1.0 if ground_truth in text else 0.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

    def generate_with_entropy(self, input_ids, max_new_tokens=128, temperature=1.0):
        self.model.eval()
        B = input_ids.shape[0]
        all_logits = []
        all_tokens = []
        all_entropies = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _, _, _ = self.model(input_ids)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()
                entropy = dist.entropy()

                all_logits.append(next_logits)
                all_tokens.append(next_token)
                all_entropies.append(entropy)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        return (
            torch.stack(all_tokens, dim=1),
            torch.stack(all_logits, dim=1),
            torch.stack(all_entropies, dim=1),
        )

    def compute_8020_mask(self, entropies):
        B, T = entropies.shape
        k = max(1, int(T * 0.2))
        threshold = torch.topk(entropies, k, dim=-1)[0][:, -1:]
        return (entropies >= threshold).float()

    def train_step(self, prompts, ground_truths, optimizer, num_samples=4):
        self.model.train()
        total_loss = 0.0

        for prompt, gt in zip(prompts, ground_truths):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.cfg.device)

            rewards = []
            log_probs = []
            entropies_list = []

            for _ in range(num_samples):
                tokens, logits, entropy = self.generate_with_entropy(input_ids)
                text = self.tokenizer.decode(tokens[0])
                reward = self.compute_rewards([text], gt)[0]

                log_prob = F.log_softmax(logits, dim=-1)
                token_log_probs = log_prob.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

                rewards.append(reward)
                log_probs.append(token_log_probs)
                entropies_list.append(entropy)

            rewards = torch.stack(rewards)
            mean_reward = rewards.mean()
            advantages = rewards - mean_reward

            loss = 0.0
            for lp, adv, ent in zip(log_probs, advantages, entropies_list):
                mask = self.compute_8020_mask(ent)
                policy_loss = -(lp * adv * mask).sum() / (mask.sum() + 1e-8)
                entropy_bonus = -(ent * mask).mean() * self.entropy_coef
                loss += policy_loss - entropy_bonus

            loss = loss / num_samples
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return total_loss / len(prompts)
