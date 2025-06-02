
import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
from transformers.cache_utils import DynamicCache
from utils.caching import prune_cache
import utils.printing as printing
from typing import List, Tuple


def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


@torch.no_grad()
def speculative_generate_cascade(
    inputs: List[int],
    drafter0: Module,
    drafter1: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    switch_threshold: float = 0.4,
    debug: bool = False,
) -> Tuple[List[int], float]:

    drafter_cache, target_cache = None, None
    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)

    drafts_accepted, drafts_speculated = 0.0, 0.0
    vocabulary_size = target.config.vocab_size

    prompt_len = len(inputs)
    max_seq_length = getattr(target.config, 'max_position_embeddings', 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)

    current_position = prompt_len
    current_drafter = drafter0
    switched = False

    if first_target:
        Mp = target(input_ids=input_ids[..., :current_position], past_key_values=None, use_cache=use_cache)
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1

        if torch.isin(t, stop_tokens):
            if debug: printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0.0
        if debug: printing.initial_step(t, tokenizer)

    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)

        input_ids = input_ids.to(current_drafter.device)

        for k in range(corrected_gamma):
            Mq = current_drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi

        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)

        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
        p = logits_processor(draft_logits)

        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break

        drafts_accepted += n

        # === switch drafter if needed ===
        acc_rate_so_far = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 1.0
        if not switched and acc_rate_so_far < switch_threshold:
            print(f"[Switch] Acceptance rate dropped to {acc_rate_so_far:.3f}. Switching drafter.")
            current_drafter = drafter1
            drafter_cache = None  # optional: reset cache if architectures differ
            switched = True

        stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            if debug: printing.end_token_found(stop_location)
            return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), acc_rate_so_far

        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)

        if debug:
            generated = input_ids.clone().detach()

        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x

        if debug:
            printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, current_position, corrected_gamma)

        current_position += n + 1

        if torch.isin(x, stop_tokens):
            if debug: printing.end_token_found(n)
            return input_ids[0, prompt_len:current_position].tolist(), acc_rate_so_far

    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated
