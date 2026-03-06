import argparse
import logging
import os
from pathlib import Path

import mlx.core as mx

from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load

from mylm import DEFAULT_MODEL, DEFAULT_TEMP, DEFAULT_TOP_K, DEFAULT_TOP_P
from mylm.sleep import sleep as train_sleep

logger = logging.getLogger(__name__)
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 8192
SYSTEM_FILE = Path(__file__).parent / "SYSTEM.md"
MEMORY_PATH = Path.home() / ".cache" / "mylm" / "memory"
LORA_LAYERS = 16
LORA_CONFIG = {"rank": 16, "scale": 16.0, "dropout": 0.0}


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="PRNG seed"
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Set the maximum key-value cache size",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    return parser


def main():
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s: %(message)s")

    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load(args.model)

    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)

    with open(SYSTEM_FILE) as f:
        system_prompt = f.read()

    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    history = []
    fresh_cache = True
    logger.info(f"Starting chat session with {args.model}.")

    adapter_file = f"{MEMORY_PATH}/adapters.safetensors"
    has_lora = False
    if os.path.exists(adapter_file):
        linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=LORA_CONFIG)
        model.load_weights(adapter_file, strict=False)
        has_lora = True
        logger.info("Loaded memory adapter from previous session.")

    print("Commands: /quit to exit, /sleep to convert short term memory to long term, /help for help")

    while True:
        query = input(">> ")
        if query == "/quit":
            break
        if query == "/help":
            print("Commands: /quit to exit, /sleep to train on history, /help for help")
            continue
        if query == "/sleep":
            logger.info("Converting short-term memory to long-term...")
            train_sleep(
                model=args.model,
                history=history,
                memory_path=MEMORY_PATH,
                system_prompt=system_prompt,
            )
            history = []
            if not has_lora:
                linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=LORA_CONFIG)
                has_lora = True
            model.load_weights(f"{MEMORY_PATH}/adapters.safetensors", strict=False)
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            fresh_cache = True
            logger.info("Adapter loaded. Prompt cache reset.")
            continue

        if fresh_cache:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": query}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            fresh_cache = False
        else:
            # Only feed the new user turn tokens; the prompt cache
            # already has KV state for all prior turns.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                add_generation_prompt=True,
                enable_thinking=False,
            )

        response_text = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            print(response.text, flush=True, end="")
            response_text += response.text
        print()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
