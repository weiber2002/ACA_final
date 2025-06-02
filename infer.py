import argparse
import random
import numpy as np
import torch
from local_sampling import autoregressive_generate, speculative_generate
from local_sampling import speculative_generate_cascade
from utils.logits_processor import GreedyProcessor, MultinomialProcessor, TopKProcessor, NucleusProcessor, TopKNucleusProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
import os
from termcolor import colored

cascade_total = 0
cascade_acc = 0.0   
total = 0
acc = 0.0
target_total = 0

class InferenceCLI:

    def __init__(self, device: str = "cuda:0", cs: bool = False):
        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.gamma = 1
        self.gen_len = 100
        self.switch_threshold = 0.6
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = False
        self.target_gen = False
        self.cs = False  # enable cascade mode

        self.chat = True  # If using a chat instructed model, set to True

        self.processors = {
            "greedy": {"processor": GreedyProcessor, "building_args": {"temperature": float}},
            "multinomial": {"processor": MultinomialProcessor, "building_args": {"temperature": float}},
            "topk": {"processor": TopKProcessor, "building_args": {"temperature": float, "top_k": int}},
        }
        self.selected_processor = {"name": "greedy", "processor": GreedyProcessor, "args": {"temperature": 1.0}}
        self.processor = GreedyProcessor()

        self._load_models()

    def _load_models(self):
        # Target model
        # target_model = "Qwen/Qwen3-32B-FP8"
        target_model = "Qwen/Qwen3-14B-FP8"
        # target_model = "Qwen/Qwen3-8B-FP8"
        # target_model = "Qwen/Qwen3-4B-FP8"
        # target_model = "Qwen/Qwen3-1.7B-FP8"
        # target_model = "Qwen/Qwen3-0.6B-FP8"
        # target_model = "meta-llama/Llama-3.2-3B-Instruct"

        # Drafter model
        # drafter_model = "Qwen/Qwen3-32B-FP8"
        # drafter_model = "Qwen/Qwen3-14B-FP8"
        # drafter_model = "Qwen/Qwen3-8B-FP8"
        # drafter_model = "Qwen/Qwen3-4B-FP8"
        # drafter_model = "Qwen/Qwen3-1.7B-FP8"
        drafter_model = "Qwen/Qwen3-0.6B-FP8"
        # drafter_model = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Cascade Drafter model
        # cascade_draft_model = "Qwen/Qwen3-32B-FP8"
        # cascade_draft_model = "Qwen/Qwen3-14B-FP8"
        # cascade_draft_model = "Qwen/Qwen3-8B-FP8"
        # cascade_draft_model = "Qwen/Qwen3-4B-FP8"
        # cascade_draft_model = "Qwen/Qwen3-1.7B-FP8"
        cascade_draft_model = "Qwen/Qwen3-0.6B-FP8"

        print(colored("Target model:",  on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Cascade Drafter model:", on_color="on_yellow"), cascade_draft_model)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            device_map=str(self.device),
            trust_remote_code=True,
        )
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            device_map=str(self.device),
            trust_remote_code=True,
        )
        self.cascade_drafter = AutoModelForCausalLM.from_pretrained(
            cascade_draft_model,
            device_map=str(self.device),
            trust_remote_code=True,
        )

        self.target.eval()
        self.drafter.eval()
        self.cascade_drafter.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            trust_remote_code=True,
        )

        self.end_tokens = [self.tokenizer.eos_token_id]

    def _perform_command(self, command: str):
        args = command.split()
        cmd = args[0]
        if cmd == "/quit":
            print(colored("Goodbye!", on_color="on_red"))
            exit(0)
        if cmd == "/debug":
            self.debug = not self.debug
            print(colored(f"Debug mode: {self.debug}", on_color="on_blue"))
            return
        if cmd == "/cache":
            self.cache = not self.cache
            print(colored(f"Cache: {self.cache}", on_color="on_blue"))
            if self.cache:
                print(colored(
                    "Warning: cache feature unstable across models.",
                    "red"
                ))
            return
        if cmd == "/length":
            if len(args) < 2:
                print(colored("Usage: /length <value>", "red"))
                return
            self.gen_len = int(args[1])
            print(colored(f"Generation length: {self.gen_len}", "on_blue"))
            return
        if cmd == "/gamma":
            if len(args) < 2:
                print(colored("Usage: /gamma <value>", "red"))
                return
            self.gamma = int(args[1])
            print(colored(f"Gamma: {self.gamma}", "on_blue"))
            return
        if cmd == "/processor":
            if len(args) < 2:
                print(colored("Usage: /processor <name> [args]", "red"))
                return
            pname = args[1]
            if pname not in self.processors:
                print(colored("Invalid processor", "red"))
                return
            proc_info = self.processors[pname]
            proc_args = {n: t(v) for (n,t), v in zip(proc_info["building_args"].items(), args[2:])}
            self.selected_processor = {"name": pname, "processor": proc_info["processor"], "args": proc_args}
            self.processor = proc_info["processor"](**proc_args)
            print(colored(f"Processor set: {pname} {proc_args}", "blue"))
            return
        print(colored("Unknown command", "red"))
        self._help()

    def _help(self):
        print(colored("Commands:", on_color="on_blue"))
        print("/quit, /debug, /cache, /length <v>, /gamma <v>, /processor <name> [args]")

    def _infer(self, prefix: str):
        if self.chat:
            prefix = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prefix}
            ], add_generation_prompt=True, tokenize=False)

        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()

        if self.cs:
            self.target.to(self.device)
            self.drafter.to(self.device)
            self.cascade_drafter.to(self.device)
            torch.cuda.empty_cache()
            self._set_seed(42)
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, accept_rate = speculative_generate_cascade(
                tokenized,
                drafter0=self.drafter,
                drafter1=self.cascade_drafter,
                target=self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                switch_threshold = self.switch_threshold,
                debug=self.debug,
                use_cache=self.cache,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Cascade Speculative ==========" , "magenta"))
            # print(colored("Out:", "magenta"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "magenta"))
            print(colored(f"Throughput: {len(output_ids) / (end_time - start_time):.1f} tokens/s", "magenta"))
            print(colored("========== Cascade Speculative ==========" , "magenta"))
            global cascade_total
            global cascade_acc
            cascade_total += len(output_ids) / (end_time - start_time)
            cascade_acc += accept_rate

        # (原 speculative / target / drafter 模式略，保持不變)
        # 1. Speculative: load both models
        if self.spec:
            self.target.to(self.device)
            self.drafter.to(self.device)
            torch.cuda.empty_cache()
            self._set_seed(42)
            torch.cuda.synchronize()
            spec_start_time = time.time()
            output_ids, accept_rate = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )
            torch.cuda.synchronize()
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative ==========" , "green"))
            # print(colored("Out:", "green"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            print(colored(f"Throughput: {len(output_ids) / (spec_end_time - spec_start_time):.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========" , "green"))
            global total
            global acc
            total += len(output_ids) / (spec_end_time - spec_start_time)
            acc += accept_rate
        # 2. Target-only
        if self.target_gen:
            self.drafter.to('cpu')  # unload drafter
            torch.cuda.empty_cache()
            self.target.to(self.device)
            self._set_seed(42)
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "blue"))
            # print(colored("Out:", "blue"), output)
            print(colored(f"Throughput: {len(output_ids) / (end_time - start_time):.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))
            global target_total
            target_total += len(output_ids) / (end_time - start_time)
            
        # 3. Drafter-only
        if self.dr:
            self.target.to('cpu')  # unload target
            torch.cuda.empty_cache()
            self.drafter.to(self.device)
            self._set_seed(42)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Drafter AR ==========" , "cyan"))
            # print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {len(output_ids) / (end_time - start_time):.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========" , "cyan"))


    def _run(self):
        while True:
            cmd = input("> ")
            if cmd.startswith("/"):
                self._perform_command(cmd)
            else:
                self._infer(cmd)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for inference")
    parser.add_argument("--cs", action="store_true", help="Use cascade speculative decoding")
    args = parser.parse_args()

    prompt = ["hello", "nice to meet you", "introduct Taiwan",
             "where are my shoes", "oops, could you find my dog",
             "what's the answer of 32*24", "what's the next alphabet of c", 
             "what's the next number of 1, 2, 3, 4, 5", "what's the next word of I love you",
             "what's my name", "what's the meaning of life",
             "what's the weather today", "what's the capital of France"]

    cli = InferenceCLI(device=args.device, cs=args.cs)
    for i in prompt:
        cli._infer(i)

    print("target_model_name", cli.target.name_or_path)
    print("drafter_model_name", cli.drafter.name_or_path)
    if cli.cs:
        average_cascade_throughput = cascade_total / len(prompt)
        print(colored(f"Cascade average throughput: {average_cascade_throughput:.1f} tokens/s", "magenta"))
    if cli.spec:
        average_throughput = total / len(prompt)
        print(colored(f"Average throughput: {average_throughput:.1f} tokens/s", "yellow"))

        acceptance_rate = acc / len(prompt)
        print(colored(f"Average acceptance rate: {acceptance_rate:.3f}", "yellow"))
    if cli.target_gen:
        target_average_throughput = target_total / len(prompt)
        print(colored(f"Target model average throughput: {target_average_throughput:.1f} tokens/s", "yellow"))
