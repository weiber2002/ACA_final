import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
from termcolor import colored

# ---- 專案本地模組 ------------------------------------------------------
# from local_sampling import autoregressive_generate, speculative_generate
from local_sampling import  autoregressive_generate
from local_sampling.cascade_speculative_decoding import cascade_speculative_generate
from utils.logits_processor import (
    GreedyProcessor
)

from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================================================================
# InferenceRunner
# ======================================================================
class InferenceRunner:
    """Thin wrapper around Qwen models for quick experiments (multi-drafter)."""

    # ------------------------------ INIT ------------------------------
    def __init__(self, device: str = "cuda", drafter_names: List[str] | None = None):
        self.device = device
        self.drafter_names = drafter_names or ["Qwen/Qwen3-0.6B"]

        # --- runtime flags (可由 CLI 覆寫) -----------------------------
        self.gamma: int = 4
        self.gen_len: int = 35
        self.debug: bool = False
        self.spec: bool = True          # speculative decoding
        self.dr: bool = True            # drafter AR generation (for baseline)
        self.cache: bool = False
        self.target_gen: bool = True    # target AR baseline
        self.chat: bool = True          # 使用 Qwen chat template
        self.measure_prefill: bool = False
        self.seed = 42
        self.dtype = torch.float16
        self.threshold = 150
        

        # ---- Logits processors ---------------------------------------
        self.processors = {
            "greedy": {"processor": GreedyProcessor, "building_args": {"temperature": float}}
        }
        self.processor = GreedyProcessor()

        # ---- Load models ---------------------------------------------
        self._load_models()

    # --------------------------- MODEL LOADING -------------------------
    def _load_models(self) -> None:
        target_model = "Qwen/Qwen3-8B"

        print(colored("Target model :", "yellow"), target_model)
        print(colored("Drafter model:", "yellow"), ", ".join(self.drafter_names))
        print(colored("Loading models ...", "light_grey"))

        # ---- Target ---------------------------------------------------
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).eval()

        # ---- Tokenizer (沿用 target) ----------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Drafter list --------------------------------------------
        self.drafters: List[Tuple[str, torch.nn.Module]] = []
        for name in self.drafter_names:
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
            self.drafters.append((name, mdl))

        # ---- End tokens ----------------------------------------------
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.end_tokens: List[int] = [self.tokenizer.eos_token_id]
        if im_end is not None:
            self.end_tokens.append(im_end)

    # --------------------------- PRIVATE HELPERS ----------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _time_prefill(self, model: torch.nn.Module, ids: List[int], label: str) -> None:
        if not self.measure_prefill:
            return
        with torch.no_grad():
            t0 = time.time()
            _ = model(input_ids=torch.tensor([ids], device=self.device), use_cache=self.cache)
            tp = len(ids) / (time.time() - t0)
        print(colored(f"Prefill throughput ({label}): {tp:.1f} tok/s", "cyan"))

    # ------------------------------ INFERENCE -------------------------
    def _infer(self, prompt: str) -> None:
        # ---- Apply chat template -------------------------------------
        if self.chat:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as exc:
                print(colored(f"Warning: chat template failed: {exc}", "red"))
                self.chat = False

        prefix_ids: List[int] = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        # ---- Prefill measurement -------------------------------------
        self._time_prefill(self.target, prefix_ids, "Target")
        if self.measure_prefill:
            for dn, dm in self.drafters:
                self._time_prefill(dm, prefix_ids, f"Drafter-{dn}")

        # ==============================================================
        # MULTI-DRAFTER SPECULATIVE DECODING
        # ==============================================================
        
        if self.spec:
            
            self._set_seed(self.seed)
            print(colored(f"\n--- Multi-Drafter Speculative Decoding ---", "green"))
            
            t0 = time.time()
            out_ids, stats = cascade_speculative_generate(
                prefix_ids,
                self.drafters,
                len(self.drafters),
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
                switch_threshold = self.threshold,
            )
            
            new_tok = len(out_ids)
            tp = new_tok / (time.time() - t0)
            
            print(colored(f"Generated: {new_tok} tokens  |  AccRate: {stats['acc_rate']:.3f}", "green"))
            print(colored(f"Throughput: {tp:.1f} tok/s", "green"))
            print(colored(f"Drafter usage:", "green"))
            for drafter_name, count in stats['drafter_usage'].items():
                print(colored(f"  - {drafter_name}: {count} times ({count/sum(stats['drafter_usage'].values())*100:.1f}%)", "green"))
            print(colored(self.tokenizer.decode(out_ids, skip_special_tokens=True), "green"))
            
        
        # ==============================================================
        # 2. BASELINES (Target AR / Drafter AR)
        # ==============================================================

        if self.target_gen:
            self._set_seed(42)
            print(colored("\n--- Target AR ---", "blue"))
            t0 = time.time()
            out_ids = autoregressive_generate(
                prefix_ids,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            tp = len(out_ids) / (time.time() - t0)
            print(colored(f"Throughput: {tp:.1f} tok/s", "blue"))
            print(colored(self.tokenizer.decode(out_ids[len(prefix_ids):], skip_special_tokens=True), "blue"))

        if self.dr:
            for dname, drafter in self.drafters:
                self._set_seed(42)
                print(colored(f"\n--- Drafter AR ({dname}) ---", "cyan"))
                t0 = time.time()
                out_ids = autoregressive_generate(
                    prefix_ids,
                    drafter,
                    use_cache=self.cache,
                    max_gen_len=self.gen_len,
                    eos_tokens_id=self.end_tokens,
                    logits_processor=self.processor,
                    debug=self.debug,
                )
                tp = len(out_ids) / (time.time() - t0)
                print(colored(f"Throughput: {tp:.1f} tok/s", "cyan"))
                print(colored(self.tokenizer.decode(out_ids[len(prefix_ids):], skip_special_tokens=True), "cyan"))

        print("\n" + "=" * 48 + "\n")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qwen multi-drafter speculative runner")

    # ---- 基本參數 -----------------------------------------------------
    parser.add_argument("--prompt", type=str, default="what's the weather like today in Taipei? predict it",)
    # parser.add_argument("--prompt", type=str, default="who's the most beautiful woman around the world",)
    parser.add_argument("--device", type=str, default="cuda:1")

    # ---- drafter 清單 -------------------------------------------------
    parser.add_argument(
        "--drafters",
        type=str,
        # default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B, Qwen/Qwen3-4B"),
        # default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B"),
        # default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-4B"),
        default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-1.7B"),
        # default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-0.6B"),
        help="逗號分隔的 drafter 型號 (亦可用環境變數 DRAFTER_MODELS)",
    )

    # ---- 其他旗標 -----------------------------------------------------
    parser.add_argument("--measure-prefill", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--chat", action="store_false", help="停用 chat template")

    # ---- 生成超參 -----------------------------------------------------
    parser.add_argument("--gen-len", type=int, default=180)
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=150)

    # ---- Processor 參數 ---------------------------------------------
    parser.add_argument(
        "--processor-name",
        type=str,
        default="greedy",
        choices=["greedy", "multinomial", "nucleus", "topknucleus"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)

    # ---- 預設值 (維持舊行為) -----------------------------------------
    parser.set_defaults(speculative=True, target_gen=True, dr=True, chat=True, measure_prefill=True, cache=False)

    args = parser.parse_args()

    # ---- Instantiate --------------------------------------------------
    drafter_list = [d.strip() for d in args.drafters.split(",") if d.strip()]
    runner = InferenceRunner(device=args.device, drafter_names=drafter_list)

    # ---- 覆寫 runtime flags ------------------------------------------
    runner.gamma = args.gamma
    runner.gen_len = args.gen_len
    runner.debug = args.debug
    runner.spec = args.speculative
    runner.dr = args.dr
    runner.cache = args.cache
    runner.target_gen = args.target_gen
    runner.chat = args.chat
    runner.measure_prefill = args.measure_prefill
    runner.threshold = args.threshold

    # ---- 設定 logits processor ---------------------------------------
    proc_cfg = runner.processors[args.processor_name]
    kwargs = {}
    if "temperature" in proc_cfg["building_args"]:
        kwargs["temperature"] = args.temperature
    runner.processor = proc_cfg["processor"](**kwargs)
    print(colored(f"Processor: {args.processor_name} {kwargs}", "blue"))

    # ---- GO! ----------------------------------------------------------
    runner._infer(args.prompt)