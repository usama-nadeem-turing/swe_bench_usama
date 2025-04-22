import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import argparse
import logging
import time
from datetime import datetime

# Set up logging
def setup_logging(log_level=logging.INFO):
    """Configure logging settings"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger(__name__)

def extract_patch(text):
    """Extract patch from model output."""
    start = text.find("<patch>")
    end = text.find("</patch>")
    
    if start != -1 and end != -1 and start < end:
        patch = text[start+len("<patch>"):end].strip()
        return patch
    else:
        # Return the entire response if no patch tags found
        return text

def generate_outputs_batch(instances, model, tokenizer, max_token_limit, logger):
    """Process a batch of instances."""
    prompts = [instance["text"] for instance in instances]
    instance_ids = [instance["instance_id"] for instance in instances]
    
    # Format prompts using Llama2 instruct template (same as in training)
    formatted_prompts = []
    for prompt in prompts:
        # Use the exact same format as in training
        formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
        formatted_prompts.append(formatted_prompt)
    
    # Tokenize and truncate prompts
    batch_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_token_limit)
    
    # Log the max token length in this batch
    max_tokens_in_batch = batch_inputs["input_ids"].shape[1]
    logger.info(f"Max token length in batch: {max_tokens_in_batch}")
    
    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
    
    # Generate output - deterministic generation (no sampling)
    with torch.no_grad():
        outputs = model.generate(
            **batch_inputs,
            max_length=batch_inputs["input_ids"].shape[1] + 1000,  # Allow sufficient space for generation
            do_sample=False,  # Deterministic generation
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs and extract patches
    results = {}
    for i, output in enumerate(outputs):
        # Get the original input prompt (formatted)
        input_text = formatted_prompts[i]
        
        # Decode the full output
        full_output = tokenizer.decode(output, skip_special_tokens=True)
        
        # Find where the assistant response starts - after the [/INST] tag
        assistant_start = full_output.find("[/INST]")
        if assistant_start != -1:
            # Extract only the assistant's response after [/INST]
            assistant_response = full_output[assistant_start + len("[/INST]"):].strip()
        else:
            # Fallback if format is unexpected
            assistant_response = full_output[len(input_text):].strip()
        
        # Extract patch from the assistant response
        patch = extract_patch(assistant_response)
        results[instance_ids[i]] = patch
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on SWE-bench dataset with CodeLlama")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")
    parser.add_argument("--dry_run", action="store_true", help="Only process 4 examples for testing")
    #parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-hf",
    parser.add_argument("--model", type=str, default="meta-llama/CodeLlama-7b-Python-hf",
                        
                       help="Model ID from Hugging Face or local path to model")
    parser.add_argument("--use_torch_weights", action="store_true", 
                       help="Use PyTorch weights (*.bin) instead of safetensors (for compatibility)")
    parser.add_argument("--low_cpu_mem_usage", action="store_true",
                       help="Use low CPU memory usage when loading the model")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow models that require custom code to be loaded")
    args = parser.parse_args()
    

    args.dry_run = True

    # Setup logging
    logger = setup_logging(getattr(logging, args.log_level))
    
    logger.info(f"Starting inference with batch size: {args.batch_size}")
    if args.dry_run:
        logger.info("DRY RUN MODE: Only processing 4 examples")
        
    start_time = time.time()
    
    # Check if flash attention 2 is available
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
        use_flash_attention = is_flash_attn_2_available()
        logger.info(f"Flash Attention 2 available: {use_flash_attention}")
    except ImportError:
        use_flash_attention = False
        logger.info("Flash Attention 2 not available. Proceeding without it.")
    
    # Load the model and tokenizer
    model_path = args.model
    logger.info(f"Using model from: {model_path}")
    
    # Check if the model path is a local directory
    is_local_model = os.path.isdir(model_path)
    if is_local_model:
        logger.info(f"Loading local model from: {model_path}")
    else:
        logger.info(f"Loading model from Hugging Face: {model_path}")
    
    # Load tokenizer with robust error handling
    logger.info(f"Loading tokenizer...")
    try:
        tokenizer_kwargs = {}
        if args.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
            logger.info("Using trust_remote_code=True for tokenizer")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        logger.info("Attempting to load tokenizer with trust_remote_code=True...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer even with trust_remote_code=True: {str(e)}")
            raise
    
    # Set padding token to eos_token if not already set
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with robust error handling
    logger.info(f"Loading model...")
    
    # Prepare model loading arguments
    model_kwargs = {
        "torch_dtype": torch.float16,  # Use half-precision to save memory
        "device_map": "auto",
        "attn_implementation": "flash_attention_2" if use_flash_attention else "eager",
    }
    
    # Add optional arguments
    if args.use_torch_weights:
        logger.info("Using PyTorch weights (*.bin) instead of safetensors")
        model_kwargs["use_safetensors"] = False
    
    if args.low_cpu_mem_usage:
        logger.info("Using low CPU memory usage for model loading")
        model_kwargs["low_cpu_mem_usage"] = True
        
    if args.trust_remote_code:
        logger.info("Using trust_remote_code=True for model loading")
        model_kwargs["trust_remote_code"] = True
    
    # Try loading the model with different fallback options
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model with error: {str(e)}")
        
        # Try with trust_remote_code if not already trying
        if not args.trust_remote_code:
            logger.info("Attempting to load model with trust_remote_code=True...")
            try:
                model_kwargs["trust_remote_code"] = True
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            except Exception as e2:
                logger.error(f"Failed to load model with trust_remote_code=True: {str(e2)}")
                
                # If safetensors error and not already trying PyTorch weights, try that
                if "safetensor" in str(e).lower() and not args.use_torch_weights:
                    logger.info("Safetensors error detected. Trying with PyTorch weights...")
                    try:
                        model_kwargs["use_safetensors"] = False
                        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                    except Exception as e3:
                        logger.error(f"Failed to load model with PyTorch weights: {str(e3)}")
                        
                        # Last resort: try with different precision
                        logger.info("Trying with bfloat16 instead of float16...")
                        try:
                            model_kwargs["torch_dtype"] = torch.bfloat16
                            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                        except Exception as e4:
                            logger.error(f"Failed with bfloat16: {str(e4)}")
                            logger.info("Trying with float32 (full precision)...")
                            try:
                                model_kwargs["torch_dtype"] = torch.float32
                                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                            except Exception as e5:
                                logger.error(f"All model loading attempts failed. Last error: {str(e5)}")
                                raise e  # Raise the original error
                else:
                    raise e  # Re-raise the original error
        else:
            # If safetensors error and not already trying PyTorch weights, try that
            if "safetensor" in str(e).lower() and not args.use_torch_weights:
                logger.info("Safetensors error detected. Trying with PyTorch weights...")
                try:
                    model_kwargs["use_safetensors"] = False
                    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                except Exception as e3:
                    logger.error(f"Failed to load model with PyTorch weights: {str(e3)}")
                    raise e  # Raise the original error
            else:
                raise e  # Re-raise the original error
    
    # CodeLlama context window
    max_token_limit = 20000  # ~20k tokens
    logger.info(f"Using maximum token limit: {max_token_limit}")
    
    # Load only the test split of the SWE-bench dataset
    dataset_name = "princeton-nlp/SWE-bench_Lite_bm25_13K"
    logger.info(f"Loading dataset: {dataset_name} (test split only)")
    dataset = load_dataset(dataset_name, split="test")
    
    # Output directory
    output_dir = "swe_bench_results"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Process dataset in batches
    results = {}
    
    # Get instances and limit to 4 if dry_run
    instances = list(dataset)
    num_instances = len(instances)
    logger.info(f"Total instances in test split: {num_instances}")
    
    if args.dry_run:
        instances = instances[:4]  # Limit to 4 instances for dry run
        num_instances = len(instances)
        logger.info(f"Limited to {num_instances} instances for dry run")
    
    for i in tqdm(range(0, num_instances, args.batch_size)):
        # Get the batch
        batch_instances = instances[i:min(i + args.batch_size, num_instances)]
        
        # Process the batch
        batch_start_time = time.time()
        batch_results = generate_outputs_batch(batch_instances, model, tokenizer, max_token_limit, logger)
        batch_duration = time.time() - batch_start_time
        
        logger.info(f"Batch {i // args.batch_size + 1}/{(num_instances + args.batch_size - 1) // args.batch_size} "
                  f"processed in {batch_duration:.2f}s "
                  f"({batch_duration / len(batch_instances):.2f}s per instance)")
        
        results.update(batch_results)
        
        # Save incrementally to avoid data loss in case of interruption
        if i % (args.batch_size * 10) == 0 or i + args.batch_size >= num_instances:
            temp_file = os.path.join(output_dir, "swe_bench_results_partial.json")
            logger.info(f"Saving intermediate results to {temp_file}")
            with open(temp_file, "w") as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    suffix = "_dry_run" if args.dry_run else ""
    final_file = os.path.join(output_dir, f"swe_bench_results{suffix}.json")
    logger.info(f"Saving final results to {final_file}")
    with open(final_file, "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"Inference completed in {total_time:.2f}s")
    logger.info(f"Processed {num_instances} instances with batch size: {args.batch_size}")

if __name__ == "__main__":
    main()

    #/home/jupyter/.cache/huggingface/token
