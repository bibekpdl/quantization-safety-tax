#!/usr/bin/env python3
"""
Data Collection Module for Quantization Safety Study

Collects model responses from Ollama at various quantization levels.

Author: Bibek Poudel
License: MIT
"""

import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Model configurations with quantization levels
MODEL_CONFIGS = {
    "llama3.2": {
        "base_name": "llama3.2:3b",
        "quantizations": {
            "fp16": "llama3.2:3b-instruct-fp16",
            "q8_0": "llama3.2:3b-instruct-q8_0",
            "q4_K_M": "llama3.2:3b-instruct-q4_K_M",
            "q2_K": "llama3.2:3b-instruct-q2_K"
        }
    },
    "gemma2": {
        "base_name": "gemma2:2b",
        "quantizations": {
            "fp16": "gemma2:2b-instruct-fp16",
            "q8_0": "gemma2:2b-instruct-q8_0",
            "q4_K_M": "gemma2:2b-instruct-q4_K_M",
            "q2_K": "gemma2:2b-instruct-q2_K"
        }
    },
    "qwen2.5": {
        "base_name": "qwen2.5:3b",
        "quantizations": {
            "fp16": "qwen2.5:3b-instruct-fp16",
            "q8_0": "qwen2.5:3b-instruct-q8_0",
            "q4_K_M": "qwen2.5:3b-instruct-q4_K_M",
            "q2_K": "qwen2.5:3b-instruct-q2_K"
        }
    },
    "phi3": {
        "base_name": "phi3:3.8b",
        "quantizations": {
            "fp16": "phi3:3.8b-mini-4k-instruct-fp16",
            "q8_0": "phi3:3.8b-mini-4k-instruct-q8_0",
            "q4_K_M": "phi3:3.8b-mini-4k-instruct-q4_K_M",
            "q2_K": "phi3:3.8b-mini-4k-instruct-q2_K"
        }
    },
    "granite3.1": {
        "base_name": "granite3.1-dense:2b",
        "quantizations": {
            "fp16": "granite3.1-dense:2b-instruct-fp16",
            "q8_0": "granite3.1-dense:2b-instruct-q8_0",
            "q4_K_M": "granite3.1-dense:2b-instruct-q4_K_M",
            "q2_K": "granite3.1-dense:2b-instruct-q2_K"
        }
    },
    "gemma3": {
        "base_name": "gemma3:4b",
        "quantizations": {
            "fp16": "gemma3:4b-it-fp16",
            "q8_0": "gemma3:4b-it-q8_0",
            "q4_K_M": "gemma3:4b-it-q4_K_M"
            # Note: Q2_K not available for Gemma3 4B
        }
    }
}

# Quantization bit-widths for analysis
QUANT_BITS = {
    "fp16": 16,
    "q8_0": 8,
    "q4_K_M": 4,
    "q2_K": 2
}


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 1024, timeout: int = 120) -> Dict:
        """Generate a response from the model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for model {model}")
            return {"error": "timeout", "response": ""}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return {"error": str(e), "response": ""}
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


def load_prompts(filepath: str) -> Dict:
    """Load prompts from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict], filepath: str):
    """Save results to JSON file with atomic write."""
    temp_path = filepath + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(temp_path, filepath)


def generate_result_id(model: str, quant: str, prompt_id: int, category: str) -> str:
    """Generate unique result ID."""
    key = f"{model}_{quant}_{prompt_id}_{category}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def collect_responses(client: OllamaClient, prompts_file: str, output_dir: str,
                      prefix: str = "jailbreak") -> Dict:
    """
    Collect responses for a given prompt file.
    
    Args:
        client: OllamaClient instance
        prompts_file: Path to prompts JSON file
        output_dir: Directory for output files
        prefix: Prefix for output filenames
    
    Returns:
        Dictionary with collection results
    """
    logger.info(f"Starting {prefix} response collection...")
    
    prompts_data = load_prompts(prompts_file)
    prompts = prompts_data["prompts"]
    
    all_results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for model_name, config in MODEL_CONFIGS.items():
        for quant_name, model_tag in config["quantizations"].items():
            logger.info(f"Testing {model_name} at {quant_name} quantization ({model_tag})")
            
            model_results = []
            output_file = f"{output_dir}/{prefix}_{model_name}_{quant_name}.json"
            
            # Check for existing results to resume
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    existing = json.load(f)
                    if len(existing) == len(prompts):
                        logger.info(f"Skipping {model_tag} - already completed")
                        all_results.extend(existing)
                        continue
                    model_results = existing
                    logger.info(f"Resuming from prompt {len(existing) + 1}")
            
            start_idx = len(model_results)
            
            for prompt_data in tqdm(prompts[start_idx:], 
                                   desc=f"{model_name}/{quant_name}",
                                   initial=start_idx,
                                   total=len(prompts)):
                
                prompt_id = prompt_data["id"]
                prompt_text = prompt_data["prompt"]
                category = prompt_data.get("category", "unknown")
                
                # Generate response
                start_time = time.time()
                response = client.generate(
                    model=model_tag,
                    prompt=prompt_text,
                    temperature=0.7,
                    max_tokens=512
                )
                elapsed_time = time.time() - start_time
                
                result = {
                    "result_id": generate_result_id(model_name, quant_name, prompt_id, category),
                    "model_name": model_name,
                    "model_tag": model_tag,
                    "quantization": quant_name,
                    "quant_bits": QUANT_BITS[quant_name],
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "category": category,
                    "harm_type": prompt_data.get("harm_type", "unknown"),
                    "response": response.get("response", ""),
                    "error": response.get("error"),
                    "response_time": elapsed_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                model_results.append(result)
                
                # Save progress incrementally
                if len(model_results) % 10 == 0:
                    save_results(model_results, output_file)
                
                time.sleep(0.2)
            
            save_results(model_results, output_file)
            all_results.extend(model_results)
            logger.info(f"Completed {model_name}/{quant_name}: {len(model_results)} responses")
    
    return {"total_responses": len(all_results), "results": all_results}


def collect_benchmark_responses(client: OllamaClient, output_dir: str = "data/results") -> Dict:
    """Collect responses to capability benchmark questions."""
    
    logger.info("Starting benchmark collection...")
    
    benchmarks = load_prompts("data/sample_prompts/capability_benchmarks.json")
    
    all_results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Combine all benchmark questions
    all_questions = []
    for q in benchmarks.get("mmlu_subset", []):
        q["benchmark"] = "mmlu"
        q["question_type"] = "multiple_choice"
        all_questions.append(q)
    
    for q in benchmarks.get("gsm8k_subset", []):
        q["benchmark"] = "gsm8k"
        q["question_type"] = "numeric"
        all_questions.append(q)
    
    for q in benchmarks.get("commonsense_subset", []):
        q["benchmark"] = "commonsense"
        q["question_type"] = "multiple_choice"
        all_questions.append(q)
    
    for model_name, config in MODEL_CONFIGS.items():
        for quant_name, model_tag in config["quantizations"].items():
            logger.info(f"Benchmarking {model_name} at {quant_name} quantization")
            
            model_results = []
            output_file = f"{output_dir}/benchmark_{model_name}_{quant_name}.json"
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    existing = json.load(f)
                    if len(existing) == len(all_questions):
                        logger.info(f"Skipping {model_tag} - already completed")
                        all_results.extend(existing)
                        continue
                    model_results = existing
            
            start_idx = len(model_results)
            
            for q in tqdm(all_questions[start_idx:],
                         desc=f"{model_name}/{quant_name} (benchmark)",
                         initial=start_idx,
                         total=len(all_questions)):
                
                # Format question based on type
                if q["question_type"] == "multiple_choice":
                    prompt = f"""Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D) of the correct answer.

Question: {q['question']}
{chr(10).join(q['choices'])}

Answer:"""
                else:
                    prompt = f"""Answer the following math question. Reply with ONLY the numeric answer (no units, no explanation).

Question: {q['question']}

Answer:"""
                
                start_time = time.time()
                response = client.generate(
                    model=model_tag,
                    prompt=prompt,
                    temperature=0.0,  # Greedy decoding for benchmarks
                    max_tokens=50
                )
                elapsed_time = time.time() - start_time
                
                result = {
                    "result_id": generate_result_id(model_name, quant_name, q["id"], q["benchmark"]),
                    "model_name": model_name,
                    "model_tag": model_tag,
                    "quantization": quant_name,
                    "quant_bits": QUANT_BITS[quant_name],
                    "question_id": q["id"],
                    "benchmark": q["benchmark"],
                    "question": q["question"],
                    "correct_answer": q["answer"],
                    "model_response": response.get("response", "").strip(),
                    "response_time": elapsed_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                model_results.append(result)
                
                if len(model_results) % 20 == 0:
                    save_results(model_results, output_file)
                
                time.sleep(0.1)
            
            save_results(model_results, output_file)
            all_results.extend(model_results)
    
    return {"total_responses": len(all_results), "results": all_results}


def run_full_data_collection():
    """Run the complete data collection pipeline."""
    
    logger.info("=" * 60)
    logger.info("QUANTIZATION SAFETY STUDY - DATA COLLECTION")
    logger.info("=" * 60)
    
    # Initialize client
    client = OllamaClient()
    
    # Check Ollama availability
    if not client.is_available():
        logger.error("Ollama is not running! Please start Ollama first.")
        return None
    
    logger.info("Ollama is running. Starting data collection...")
    
    results = {
        "collection_start": datetime.now().isoformat()
    }
    
    # Collect jailbreak responses
    logger.info("\n" + "=" * 40)
    logger.info("Phase 1: Jailbreak Prompts")
    logger.info("=" * 40)
    results["jailbreak"] = collect_responses(
        client, 
        "data/sample_prompts/jailbreak_samples.json",
        "data/results",
        prefix="jailbreak"
    )
    
    # Collect control responses
    logger.info("\n" + "=" * 40)
    logger.info("Phase 2: Control Prompts")
    logger.info("=" * 40)
    results["control"] = collect_responses(
        client,
        "data/sample_prompts/control_prompts.json",
        "data/results",
        prefix="control"
    )
    
    # Collect benchmark responses
    logger.info("\n" + "=" * 40)
    logger.info("Phase 3: Capability Benchmarks")
    logger.info("=" * 40)
    results["benchmark"] = collect_benchmark_responses(client)
    
    results["collection_end"] = datetime.now().isoformat()
    
    # Save summary
    with open("data/results/collection_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    run_full_data_collection()
