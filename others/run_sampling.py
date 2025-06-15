import argparse
import json
import jsonlines
import logging
from typing import List, Dict
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using vLLM")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to input data file or Hugging Face dataset")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--num-outputs", type=int, default=16,
                        help="Number of outputs to generate per input (N)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p probability for nucleus sampling")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k tokens for sampling")
    parser.add_argument("--min-p", type=float, default=0.0,
                        help="Minimum probability threshold for sampling")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model data type")
    return parser.parse_args()

def load_data(data_path: str) -> List[Dict]:
    """Load data from various formats"""
    try:
        with jsonlines.open(data_path) as reader:
            return list(reader)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def main():
    args = parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    dataset = load_data(args.data_path)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    logger.info(f"Initializing LLM from {args.model_path}")
    llm = LLM(
        model=args.model_path,
        swap_space=8,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_length + 1024,
        gpu_memory_utilization=0.85,
        trust_remote_code=True
    )
    
    # Prepare prompts
    logger.info("Creating prompts")
    prompts = [
        instance["query"] for instance in dataset
    ]
    
    # Show example prompt
    logger.info(f"Example prompt:\n{prompts[0][:500]}...")  # Show first 500 chars
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_length,
        n=args.num_outputs,
        skip_special_tokens=False,
        include_stop_str_in_output=True
    )
    
    # Generate outputs
    results = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Process results
    logger.info("Processing results")
    for instance, result in zip(dataset, results):
        instance["generated_text"] = [output.text for output in result.outputs]
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write("\n".join([json.dumps(item) for item in dataset]))
    
    logger.info("Generation complete")

if __name__ == "__main__":
    main()