import argparse
from cache_config import TRANSFORMERS_CACHE_PATH, apply_cache_config

apply_cache_config()

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import transformers
transformers.logging.set_verbosity_info()
import os

def initialize_ts_encoder(model):
    """
    Applies Xavier normal initialization to the ts_encoder part of the model.
    """
    print("Initializing ts_encoder weights with Xavier Normal...")
    for name, param in model.ts_encoder.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_normal_(param)
            print(f"  Initialized {name} with Xavier normal.")
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
            print(f"  Initialized {name} to zeros.")
    print("ts_encoder initialization complete.")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize ts_encoder weights for a model")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model repo id used to infer local path (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to model (e.g., base_model/Qwen3-8B). Overrides --repo_id.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Define the path to the model that needs to be initialized.
    model_path = args.model_path or f"base_model/{args.repo_id.split('/')[-1]}"
    
    # Check if the model path exists
    if not os.path.isdir(model_path):
        print(f"Error: Model path '{model_path}' not found or is not a directory.")
        print("Please ensure you have downloaded the model and placed it in the correct directory.")
        return

    print(f"Loading model from: {model_path}")
    
    # Load the model using AutoModelForCausalLM.
    # trust_remote_code=True is necessary to load the custom .py files.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=TRANSFORMERS_CACHE_PATH,
        device_map="auto" # Use "cpu" if you don't have a GPU or run into memory issues
    )

    # Apply the custom initialization
    model = initialize_ts_encoder(model)

    # Define the output path. Here we save it back to the same directory.
    # You can change this to a new directory if you want to keep the original.
    output_path = model_path
    print(f"\nSaving initialized model to: {output_path}")
    
    model.save_pretrained(output_path)
    
    # The processor doesn't need changes, but it's good practice to save it alongside the model.
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=TRANSFORMERS_CACHE_PATH,
        )
        processor.save_pretrained(output_path)
        print("Processor saved successfully.")
    except Exception as e:
        print(f"Could not load or save the processor. This might be okay if you handle tokenization separately. Error: {e}")


    print("\nModel initialization process finished successfully!")
    print(f"The initialized model is saved at: {output_path}")

if __name__ == "__main__":
    main()
