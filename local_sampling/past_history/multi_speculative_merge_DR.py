import torch
from torch.nn import Module
from utils.logits_processor import GreedyProcessor, LogitsProcessor
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple, Dict, Any


def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function for sample adjustment: keep positive parts and normalize.
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum

@torch.no_grad()
def multi_draft_speculative_generate(
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
) -> Tuple[List[int], Dict[str, float]]:

    device = target.device
    list_tokens = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens, dtype=torch.long, device=device).view(-1, 1)

    # prepare input buffer
    prompt_len = len(inputs)
    max_seq = getattr(target.config, 'max_position_embeddings',
                    getattr(target.config, 'max_context_length', 1024))
    total_len = min(max_seq, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=device)
    current_pos = prompt_len

    # caches
    drafter_caches: Dict[str, Any] = {name: None for name, _ in drafters}
    target_cache: Any = None

    # stats
    drafts_spec = 0.0
    drafts_acc = 0.0
    usage_counts = {name: 0 for name, _ in drafters}

    # === GPU allocation for drafters ===
    num_gpus = torch.cuda.device_count()
    # Assign each drafter model to a GPU
    for idx, (name, model) in enumerate(drafters):
        gpu_id = idx % num_gpus
        model.to(f"cuda:{gpu_id}")
        # Update drafters list to reflect model device
        drafters[idx] = (name, model)
    # End GPU allocation


    # parameters
    vocab_size = target.config.vocab_size
    processor = logits_processor
    B = len(drafters)

    # buffers
    batched_ids = torch.empty((B, total_len), dtype=torch.long, device=device)
    q_buf = torch.empty((B, gamma, vocab_size), dtype=torch.float16, device=device)
    idx_buf = torch.empty((B, gamma), dtype=torch.long, device=device)

    # first target step
    if first_target:
        out0 = target(
            input_ids=input_ids[:, :current_pos],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = out0.past_key_values
        p0 = processor(out0.logits[..., -1, :])
        t = processor.sample(p0)
        input_ids[0, current_pos] = t
        current_pos += 1
        if torch.isin(t, stop_tokens)[0]:
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:current_pos].tolist(), {"acc_rate": 0.0, "drafter_usage": usage_counts}
        if debug:
            printing.initial_step(t, tokenizer)

    # main loop
    while current_pos < total_len:
                # ensure room for at least one generated token
        g = min(gamma, total_len - current_pos - 1)
        if g <= 0:
            break
        # drafter proposals
        for i, (name, model) in enumerate(drafters):
            seq_ids = input_ids.clone().to(model.device)
            for k in range(g):
                out = model(
                    input_ids=seq_ids[:, :current_pos + k],
                    past_key_values=drafter_caches[name],
                    use_cache=use_cache,
                    torch_dtype=torch_dtype,
                )
                if use_cache:
                    drafter_caches[name] = prune_cache(out.past_key_values, g)
                draft_probs = processor(out.logits[..., -1, :])
                q_buf[i, k] = draft_probs.to(q_buf.dtype)
                xi = processor.sample(draft_probs)
                idx_buf[i, k] = xi
                seq_ids[0, current_pos + k] = xi
            batched_ids[i] = seq_ids.to(device)
        
        # evaluate batch
        # 論文中說要把每個 token 平行化驗證，其實就只是做一次 transformer 模型自然會得到的結果
        # 因為 transformer 會自然地輸出每個字的 logits
        # 要取得第 t 步的分布，你只要切：
        # p_t = batch_out.logits[:, t, :]   # shape = (batch_size, vocab_size)
        batch_out = target( 
            input_ids=batched_ids[:, :current_pos + g],
            use_cache=False,
        )
        logits_batch = batch_out.logits[..., current_pos - 1:current_pos + g - 1, :]

        # select best
        best_acc, best_i, best_n = -1, 0, 0
        for i in range(B):
            # Compute probabilities and fractions for drafter i
            p = processor(logits_batch[i])  # shape (g, vocab)
            fractions = p / q_buf[i, :g, :]
            r = torch.rand(g, device=device)  # uniform samples for rejection
            # Find first rejection index based on the sampled tokens
            n_i = g
            for j in range(g):
                token_id = idx_buf[i, j].item()
                if r[j] > fractions[j, token_id]:
                    n_i = j
                    break
            # Update best if this drafter has higher acceptance count
            if n_i > best_acc:
                best_acc, best_i, best_n = n_i, i, n_i
        
        usage_counts[drafters[best_i][0]] += 1
        drafts_spec += g
        drafts_acc += best_n

        # adopt
        input_ids = batched_ids[best_i:best_i+1]

        # update target
        out_t = target(
            input_ids=input_ids[:, :current_pos + g],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        if best_n == g:
            target_cache = prune_cache(out_t.past_key_values, g)
            p_next = processor(out_t.logits[..., -1, :])
        else:
            target_cache = prune_cache(out_t.past_key_values, best_n + 1)
            if not skip_sample_adjustment:
                p_next = max_fn(out_t.logits[..., -1, :])
            else:
                p_next = processor(out_t.logits[..., -1, :])

        x = processor.sample(p_next)
        if debug:
            printing.speculative_step(tokenizer, input_ids.clone(), x, best_n, prompt_len, current_pos, g)

        input_ids[0, current_pos + best_n] = x
        current_pos += best_n + 1

        if torch.isin(x, stop_tokens)[0]:
            if debug:
                printing.end_token_found(best_n)
            break

    # return
    seq = input_ids[0, prompt_len:current_pos].tolist()
    acc_rate = drafts_acc / drafts_spec if drafts_spec > 0 else 0.0
    return seq, {"acc_rate": acc_rate, "drafter_usage": usage_counts}
