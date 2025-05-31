import torch
from torch.nn import Module
from utils.logits_processor import GreedyProcessor, LogitsProcessor
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple, Dict, Any

def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    More stable version: apply softmax to prevent repeated tokens.
    """
    x = x - x.max()
    return torch.softmax(x, dim=-1)

@torch.no_grad()
def cascade_speculative_generate(
    inputs: List[int],
    drafters: List[Tuple[str, Module]],
    num_drafters: int | None,
    target: Module,
    tokenizer: Any = None,
    gamma: int = 5,
    torch_dtype: torch.dtype = torch.float16,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = True,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
    switch_threshold: int = 150,
) -> Tuple[List[int], Dict[str, float]]:
    device = torch.device('cuda:1')
    target.to(device)
    for name, model in drafters:
        model.to(device)

    list_tokens = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens, dtype=torch.long, device=device).view(-1, 1)

    prompt_len = len(inputs)
    max_ctx = getattr(target.config, 'max_position_embeddings',
                      getattr(target.config, 'max_context_length', 1024))
    total_len = min(max_ctx, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=device)
    current_pos = prompt_len

    drafter_caches: Dict[str, Any] = {name: None for name, _ in drafters}
    target_cache: Any = None
    drafts_spec = 0.0
    drafts_acc = 0.0
    usage_counts = {name: 0 for name, _ in drafters}

    if first_target:
        out0 = target(
            input_ids=input_ids[:, :current_pos],
            torch_dtype=torch_dtype,
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = out0.past_key_values
        p0 = logits_processor(out0.logits[..., -1, :])
        t0 = logits_processor.sample(p0)
        input_ids[0, current_pos] = t0
        current_pos += 1
        if torch.isin(t0, stop_tokens)[0]:
            return [], {"acc_rate": 0.0, "drafter_usage": usage_counts}
        if debug:
            printing.initial_step(t0, tokenizer)

    vocab_size = target.config.vocab_size
    processor = logits_processor

    while current_pos < total_len:
        g = min(gamma, total_len - current_pos - 1)
        if g <= 0:
            break

        gen_tokens = current_pos - prompt_len
        drafter_idx = 0 if gen_tokens < switch_threshold else min(1, len(drafters)-1)
        drafter_name, drafter_model = drafters[drafter_idx]

        seq_ids = input_ids.clone().to(device)
        q_buf = torch.zeros((g, vocab_size), device=device)
        for k in range(g):
            out_d = drafter_model(
                input_ids=seq_ids[:, :current_pos+k],
                torch_dtype=torch_dtype,
                past_key_values=drafter_caches[drafter_name],
                use_cache=use_cache,
            )
            drafter_caches[drafter_name] = out_d.past_key_values  # ⬅️ 修改：保留完整 cache
            probs = processor(out_d.logits[..., -1, :])
            q_buf[k] = probs
            xi = processor.sample(probs)
            seq_ids[0, current_pos + k] = xi

        out_t = target(
            input_ids=seq_ids[:, :current_pos+g],
            torch_dtype=torch_dtype,
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        logits = out_t.logits[0, current_pos-1:current_pos+g-1, :]

        p = processor(logits)
        fractions = torch.clamp(p / (q_buf + 1e-6), max=10.0)  # ⬅️ 修改：避免爆炸性接受
        r = torch.rand(g, device=device)
        n = g
        for j in range(g):
            token_id = seq_ids[0, current_pos + j].item()
            if r[j] > fractions[j, token_id]:
                n = j
                break

        usage_counts[drafter_name] += 1
        drafts_spec += g
        drafts_acc += n
        # print("current_acc_rate:", drafts_acc / drafts_spec if drafts_spec > 0 else 0.0)

        input_ids = seq_ids[:1].clone()

        # if n == g:
        #     p_next = processor(out_t.logits[0, current_pos+g-1, :])
        # else:
        #     target_cache = prune_cache(out_t.past_key_values, n+1)
        #     if use_cache:
        #         drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
        #         target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
        #     if skip_sample_adjustment:
        #         p_next = processor(out_t.logits[0, n, :] - q_buf[0, n , :])
        #     else:
        #         p_next = max_fn(out_t.logits[0, n, :])  # ⬅️ 已改為 softmax
        if n == g:
            target_cache = prune_cache(out_t.past_key_values, 0)
            p_next = processor(out_t.logits[0, current_pos+g-1, :])
        else:
            target_cache = prune_cache(out_t.past_key_values, n+1)
            if skip_sample_adjustment:
                p_next = processor(out_t.logits[0, current_pos+n, :])
            else:
                p_next = max_fn(out_t.logits[0, current_pos+n, :])  # ⬅️ 已改為 softmax

        x = processor.sample(p_next)
        if debug:
            printing.speculative_step(tokenizer, input_ids.clone(), x, n, prompt_len, current_pos, g)

        input_ids[0, current_pos + n] = x
        current_pos += n + 1

        if torch.isin(x, stop_tokens)[0]:
            break

    seq = input_ids[0, prompt_len:current_pos].tolist()
    acc_rate = drafts_acc / drafts_spec if drafts_spec > 0 else 0.0
    return seq, {"acc_rate": acc_rate, "drafter_usage": usage_counts}
