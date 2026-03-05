import json
import logging
import os
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as opt
from mlx.nn.utils import tree_flatten

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import CacheDataset
from mylm.trainer import ChatDataset, TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load

from mylm import DEFAULT_MODEL, DEFAULT_TEMP, DEFAULT_TOP_K, DEFAULT_TOP_P

logger = logging.getLogger(__name__)

def sleep(
    model=DEFAULT_MODEL,
    history=None,
    memory_path="memory",
    learning_rate=1e-5,
    seed=0,
    system_prompt=None,
):
    mx.random.seed(seed)

    if history is None:
        history = []

    model_name = model
    logger.info(f"Loading {model_name}")
    model, tokenizer = load(model_name)

    # Stage 1: Generate Q&A pairs from the history
    logger.info("Generating Q&A pairs from history...")
    qa_prompt = (
        "Generate question and answer pairs about the preceding conversation."
        " Make sure the questions are asked from the perspective of the user"
        " and the answers are given from the perspective of the assistant."
        " For example a question answer pair might be:\n\n"
        "User: What is my name?\nAssistant: Your name is <name>\n\n"
        "Use the format:\n\nUser: <question>\nAssistant: <answer>\n\n"
        "with a blank line between each pair.\n"
        "Only ask questions which have an answer in the conversation, otherwise stop.\n"
        "Create Q&A pairs based on the user's messages, not the assistant's "
        "messages. The purpose is to ask questions about information provided by the user. "
        "For example if the user asks about something an appropriate question "
        "could be 'What are some topics we discussed earlier?'\n"
    )
    messages = history + [{"role": "user", "content": qa_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False
    )
    sampler = make_sampler(temp=DEFAULT_TEMP, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K)
    qa_text = generate(model, tokenizer, prompt=prompt, max_tokens=16384, verbose=True, sampler=sampler)

    # Strip reasoning tags from reasoning models
    if tokenizer.has_thinking and tokenizer.think_end in qa_text:
        qa_text = qa_text.split(tokenizer.think_end, 1)[1]

    # Parse Q&A pairs
    new_data = []
    question = None
    for line in qa_text.strip().splitlines():
        line = line.strip()
        if line.startswith("User: "):
            question = line[len("User: "):].strip()
        elif line.startswith("Assistant: ") and question:
            answer = line[len("Assistant: "):].strip()
            if answer:
                new_data.append(
                    {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    }
                )
            question = None
    logger.info(f"Parsed {len(new_data)} new Q&A pairs")

    # Load existing Q&A data and append new pairs
    os.makedirs(memory_path, exist_ok=True)
    qa_file = f"{memory_path}/qa.jsonl"
    data = []
    if os.path.exists(qa_file):
        with open(qa_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} existing Q&A pairs from {qa_file}")
    data.extend(new_data)

    # Save all Q&A pairs back to disk
    with open(qa_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(data)} total Q&A pairs to {qa_file}")

    # Prepend system prompt to each example for training
    if system_prompt:
        for item in data:
            if item["messages"][0]["role"] != "system":
                item["messages"].insert(
                    0, {"role": "system", "content": system_prompt}
                )

    # Stage 2: Train on all Q&A pairs
    model.freeze()
    lora_config = {"rank": 16, "scale": 16.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=16, config=lora_config)

    num_train = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    num_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    logger.info(f"Trainable params: {num_train:,} / {num_total:,} ({100 * num_train / num_total:.2f}%)")

    dataset = CacheDataset(ChatDataset(data, tokenizer, mask_prompt=True))

    training_args = TrainingArgs(
        batch_size=1,
        max_seq_length=2048,
        steps_per_report=1,
        adapter_file=f"{memory_path}/adapters.safetensors",
    )

    optimizer = opt.Adam(learning_rate=learning_rate)
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        args=training_args,
    )
    logger.info(f"Adapter saved to {training_args.adapter_file}")

    # Write adapter_config.json so the adapter works with mlx_lm.generate
    adapter_config = {
        "fine_tune_type": "lora",
        "model": model_name,
        "num_layers": 16,
        "lora_parameters": lora_config,
    }
    config_path = f"{memory_path}/adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s: %(message)s")
    synthetic_history = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi, what is your name?"},
        {"role": "user", "content": "My name is Awni?"},
        {"role": "assistant", "content": "Hi Awni, what a lovely name."},
        {"role": "user", "content": "What is a neural network?"},
        {"role": "assistant", "content": "A neural network is a type of machine learning model inspired by the structure and function of the human brain."},
    ]
    system_file = Path(__file__).parent / "SYSTEM.md"
    with open(system_file) as f:
        system_prompt = f.read()
    sleep(history=synthetic_history, system_prompt=system_prompt)
