import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
from termcolor import colored

# ---- 專案本地模組 ------------------------------------------------------
from local_sampling import autoregressive_generate, speculative_generate
from utils.logits_processor import (
    GreedyProcessor,
    MultinomialProcessor,
    TopKProcessor,
    NucleusProcessor,
    TopKNucleusProcessor,
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

        # ---- Logits processors ---------------------------------------
        self.processors = {
            "greedy": {"processor": GreedyProcessor, "building_args": {"temperature": float}},
            "multinomial": {"processor": MultinomialProcessor, "building_args": {"temperature": float}},
            "topk": {"processor": TopKProcessor, "building_args": {"temperature": float, "top_k": int}},
            "nucleus": {"processor": NucleusProcessor, "building_args": {"temperature": float, "top_p": float}},
            "topknucleus": {
                "processor": TopKNucleusProcessor,
                "building_args": {"temperature": float, "top_k": int, "top_p": float},
            },
        }
        self.processor = GreedyProcessor()

        # ---- Load models ---------------------------------------------
        self._load_models()

    # --------------------------- MODEL LOADING -------------------------
    def _load_models(self) -> None:
        target_model = "Qwen/Qwen3-4B"

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
        # 1. MULTI-DRAFTER SPECULATIVE DECODING
        # ==============================================================

        if self.spec:
            best_ids: List[int] = []
            best_name: str = ""
            best_new: int = 0
            best_tp: float = 0.0

            for dname, drafter in self.drafters:
                self._set_seed(42)
                print(colored(f"\n--- Speculative ({dname}) ---", "green"))
                t0 = time.time()
                out_ids, acc_rate = speculative_generate(
                    prefix_ids,
                    drafter,
                    self.target,
                    tokenizer=self.tokenizer,
                    logits_processor=self.processor,
                    gamma=self.gamma,
                    max_gen_len=self.gen_len,
                    eos_tokens_id=self.end_tokens,
                    debug=self.debug,
                    use_cache=self.cache,
                )
                tp = len(out_ids) / (time.time() - t0)
                new_tok = len(out_ids) - len(prefix_ids)

                print(colored(f"Accepted: {new_tok}  |  AccRate: {acc_rate:.3f}", "green"))
                print(colored(f"Throughput: {tp:.1f} tok/s", "green"))
                # print(colored(self.tokenizer.decode(out_ids[len(prefix_ids):], skip_special_tokens=True), "green"))

                # ---- keep best ---------------------------------------
                # if new_tok > best_new:
                #     best_ids, best_name, best_new, best_tp = out_ids, dname, new_tok, tp
                score = (new_tok, tp)          # 先比接受 token，多則勝；再比 tp
                if score > (best_new, best_tp):
                    best_ids, best_name = out_ids, dname
                    best_new, best_tp = new_tok, tp

            # ---- 最終輸出 --------------------------------------------
            print(colored("\n=== Best drafter result ===", "magenta"))
            print(colored(f"Chosen drafter: {best_name}", "magenta"))
            print(colored(f"New tokens: {best_new} | Throughput: {best_tp:.1f} tok/s", "magenta"))
            # print(colored(self.tokenizer.decode(best_ids[len(prefix_ids):], skip_special_tokens=True), "magenta"))

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
            # print(colored(self.tokenizer.decode(out_ids[len(prefix_ids):], skip_special_tokens=True), "blue"))

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
                # print(colored(self.tokenizer.decode(out_ids[len(prefix_ids):], skip_special_tokens=True), "cyan"))

        print("\n" + "=" * 48 + "\n")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qwen multi-drafter speculative runner")

    # ---- 基本參數 -----------------------------------------------------
    parser.add_argument("--prompt", type=str, default="給我一些資本主義的創新點子")
    parser.add_argument("--device", type=str, default="cuda:1")

    # ---- drafter 清單 -------------------------------------------------
    parser.add_argument(
        "--drafters",
        type=str,
        default=os.getenv("DRAFTER_MODELS", "Qwen/Qwen3-4B,Qwen/Qwen3-1.7B"),
        help="逗號分隔的 drafter 型號 (亦可用環境變數 DRAFTER_MODELS)",
    )

    # ---- 其他旗標 -----------------------------------------------------
    parser.add_argument("--measure-prefill", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--chat", action="store_false", help="停用 chat template")

    # ---- 生成超參 -----------------------------------------------------
    parser.add_argument("--gen-len", type=int, default=50)
    parser.add_argument("--gamma", type=int, default=8)

    # ---- Processor 參數 ---------------------------------------------
    parser.add_argument(
        "--processor-name",
        type=str,
        default="greedy",
        choices=["greedy", "multinomial", "topk", "nucleus", "topknucleus"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=1.0)

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

    # ---- 設定 logits processor ---------------------------------------
    proc_cfg = runner.processors[args.processor_name]
    kwargs = {}
    if "temperature" in proc_cfg["building_args"]:
        kwargs["temperature"] = args.temperature
    if "top_k" in proc_cfg["building_args"]:
        kwargs["top_k"] = args.top_k
    if "top_p" in proc_cfg["building_args"]:
        kwargs["top_p"] = args.top_p
    runner.processor = proc_cfg["processor"](**kwargs)
    print(colored(f"Processor: {args.processor_name} {kwargs}", "blue"))

    # ---- GO! ----------------------------------------------------------
    runner._infer(args.prompt)