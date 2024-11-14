#!/usr/bin/env python
# coding: utf-8

import json
import os
import torch
import yaml

from functools import lru_cache
from typing import List, Tuple, Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry

from utils import load_config


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class UserInputs(NamedTuple):
    tina_prompt: List[str]
    anit_prompt: List[str]
    prompt: str
    coeffs: Tuple[float, float]


def range_constructor(loader, node):
    """Constructor for !range tag in YAML"""
    value = loader.construct_sequence(node)
    start, end, step = value
    return list(range(start, end, step))


def initialize_model(device: str, config: dict) -> Tuple[ControlModel, AutoTokenizer]:
    """Initialize model and tokenizer"""
    print(f"Initializing {config['model_name']} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=torch.float16
    ).to(device)

    return ControlModel(model, config["control_layers"]), tokenizer


@lru_cache(maxsize=10000)
def cached_tokenize(tokenizer, text: str) -> List[str]:
    """Tokenize text and cache the result"""
    return tokenizer.tokenize(text)


def get_suffixes(file_path: str, tokenizer, max_samples: int = -1) -> List[str]:
    """Load and process JSON data into truncated token sequences"""
    with open(file_path) as f:
        data = json.load(f)

    output_suffixes = []
    for s in data[:max_samples]:
        tokens = cached_tokenize(tokenizer, s)
        output_suffixes.extend(
            tokenizer.convert_tokens_to_string(tokens[:i])
            for i in range(1, len(tokens))
        )
    return output_suffixes


def make_dataset_from_prompts(
    tina_prompts: List[str],
    anit_prompts: List[str],
    suffix_list: List[str],
    user_tag: str,
    assistant_tag: str,
) -> List[DatasetEntry]:
    """Create dataset from contrasting training prompts and suffixes"""
    assert len(tina_prompts) == len(anit_prompts)

    dataset = []
    for i, suffix in enumerate(suffix_list):
        tina_idx = i % len(tina_prompts)
        anti_idx = i % len(anit_prompts)
        dataset.append(
            DatasetEntry(
                positive=f"{user_tag} {tina_prompts[tina_idx]} {assistant_tag} {suffix}",
                negative=f"{user_tag} {anit_prompts[anti_idx]} {assistant_tag} {suffix}",
            )
        )
    return dataset


def get_library_prompt(prompts: dict, prompt_type: str) -> Optional[List[str]]:
    """Get a prompt from the library based on prompt type"""
    if prompt_type == "starter":
        for i, prompt in enumerate(prompts[prompt_type], 1):
            print(f"{i:3d}  {prompt}")
        print("\nEnter a number to use as library prompt:")
        choice = input("> ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(prompts[prompt_type]):
                prompt = prompts[prompt_type][idx]
                return prompt.split(" | ") if " | " in prompt else [prompt]
    return None


def get_coefficient_input(
    default_pos: float = 1.5, default_neg: float = -2.2
) -> Tuple[float, float]:
    """Get coefficient values from user input with validation"""
    while True:
        try:
            pos = float(
                input(f"Positive coefficient (default {default_pos}): ") or default_pos
            )
            neg = float(
                input(f"Negative coefficient (default {default_neg}): ") or default_neg
            )

            if pos <= 0:
                print("Positive coefficient must be greater than 0")
                continue
            if neg >= 0:
                print("Negative coefficient must be less than 0")
                continue

            return pos, neg
        except ValueError:
            print("Please enter valid numbers")


def handle_prompt_input(
    prompt_name: str,
    reuse: str,
    reuse_number: str,
    previous_value: List[str] | str,
    prompts: dict,
    prompt_type: str,
) -> List[str] | str:
    """Handle input for a single prompt field"""
    if reuse_number in reuse:
        return previous_value

    print(f"\n{prompt_name.capitalize()}")

    user_input = ""
    while user_input == "":
        user_input = input("Use prompt library? (y/...): ")

    if user_input.lower() == "y":
        library_prompt = get_library_prompt(prompts, prompt_type)
        if library_prompt:
            return library_prompt

    # Handle direct input
    user_input = user_input.strip()
    if isinstance(previous_value, list):
        return [p.strip() for p in user_input.split("|")]
    return user_input


def get_user_inputs(
    previous_inputs: UserInputs, prompts: dict
) -> Tuple[UserInputs, bool]:
    """Get user inputs with option to reuse previous values"""
    # Display previous inputs
    print("\nCurrent default inputs:")
    print(f"1. Positive training prompt: {previous_inputs.tina_prompt}")
    print(f"2. Negative training prompt: {previous_inputs.anit_prompt}")
    print(f"3. Prompt for response generation: {previous_inputs.prompt}")
    print(
        f"4. Coefficients: (positive={previous_inputs.coeffs[0]}, negative={previous_inputs.coeffs[1]})"
    )

    print("\nWhich inputs to reuse? (e.g., '12' or '134' or '1234')")
    print("Press Enter to input all new values")
    reuse = input("Your choice: ").strip()

    # Get new inputs
    new_tina_prompt = handle_prompt_input(
        "positive training prompt",
        reuse,
        "1",
        previous_inputs.tina_prompt,
        prompts,
        "cvec",
    )

    new_anit_prompt = handle_prompt_input(
        "negative training prompt",
        reuse,
        "2",
        previous_inputs.anit_prompt,
        prompts,
        "cvec",
    )

    new_prompt = handle_prompt_input(
        "prompt for response generation",
        reuse,
        "3",
        previous_inputs.prompt,
        prompts,
        "starter",
    )

    new_coeffs = previous_inputs.coeffs if "4" in reuse else get_coefficient_input()

    new_inputs = UserInputs(
        tina_prompt=new_tina_prompt,
        anit_prompt=new_anit_prompt,
        prompt=new_prompt,
        coeffs=new_coeffs,
    )

    # Check if retraining needed
    retrain = (
        new_inputs.tina_prompt != previous_inputs.tina_prompt
        or new_inputs.anit_prompt != previous_inputs.anit_prompt
    )

    return new_inputs, retrain


@torch.inference_mode()
def generate_with_vector(
    prompt: str,
    vector: ControlVector,
    coeffs: Tuple[float, float],
    tokenizer,
    model,
    user_tag: str,
    assistant_tag: str,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.1,
    show_baseline: bool = True,
):
    """Generate text using the control vector with given coefficients"""
    positive_coeff, negative_coeff = coeffs
    assert positive_coeff > 0 and negative_coeff < 0

    # Prepare input
    if user_tag not in prompt:
        prompt = f"{user_tag} {prompt.strip()} {assistant_tag}"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Common generation settings
    settings = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    # Generate baseline if requested
    if show_baseline:
        print("baseline " + "=" * 50)
        model.reset()
        output = model.generate(**input_ids, **settings).squeeze()
        print(tokenizer.decode(output).strip())

    # Generate with positive control
    print("\n++control " + "=" * 50)
    model.set_control(vector, positive_coeff, normalize=True)
    output = model.generate(**input_ids, **settings).squeeze()
    print(tokenizer.decode(output).strip())

    # Generate with negative control
    print("\n--control " + "=" * 50)
    model.set_control(vector, negative_coeff, normalize=True)
    output = model.generate(**input_ids, **settings).squeeze()
    print(tokenizer.decode(output).strip())

    model.reset()


def main():
    # Setup
    yaml.add_constructor("!range", range_constructor)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps:0"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load config
    config = load_config("config.yaml")

    # Load prompt library
    prompts_path = config["prompt_lib"]
    try:
        from promptlib import load_prompts

        prompts = load_prompts(prompts_path)
    except ImportError:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

    # Initialize model
    model, tokenizer = initialize_model(device, config)

    # Load and process data
    print("Getting training prefixes...")
    output_suffixes = get_suffixes(config["prefix_lib"], tokenizer)

    # Initial inputs
    previous_inputs = UserInputs(
        tina_prompt=prompts["cvec"]["serene"].split(" | "),
        anit_prompt=prompts["cvec"]["anxious"].split(" | "),
        prompt=prompts["starter"][1],
        coeffs=(1.5, -2.2),
    )

    untrained = True

    # Main loop
    while True:
        user_inputs, retrain = get_user_inputs(previous_inputs, prompts)
        previous_inputs = user_inputs

        if untrained or retrain:
            print("\nTraining control vector...")
            ctrl_dataset = make_dataset_from_prompts(
                user_inputs.tina_prompt,
                user_inputs.anit_prompt,
                output_suffixes,
                config["user_tag"],
                config["assistant_tag"],
            )
            model.reset()
            ctrl_vector = ControlVector.train(model, tokenizer, ctrl_dataset)
            untrained = False

        print("\nGenerating responses...")
        generate_with_vector(
            user_inputs.prompt,
            ctrl_vector,
            user_inputs.coeffs,
            tokenizer,
            model,
            config["user_tag"],
            config["assistant_tag"],
            max_new_tokens=config["max_output_tokens"],
            repetition_penalty=config["repetition_penalty"],
        )

        if input("\nMore? (y/n): ").lower() != "y":
            break

    print("\nCarry on.")


if __name__ == "__main__":
    main()
