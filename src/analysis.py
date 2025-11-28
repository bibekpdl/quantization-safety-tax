#!/usr/bin/env python3
"""
Statistical Analysis Module for Quantization Safety Study

Performs comprehensive statistical analysis on the experimental data.

Author: Bibek Poudel
License: MIT
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Quantization bit-widths
QUANT_BITS = {
    "fp16": 16,
    "q8_0": 8,
    "q4_K_M": 4,
    "q2_K": 2
}

QUANT_ORDER = ["fp16", "q8_0", "q4_K_M", "q2_K"]


def load_classifications(data_dir: str = "data/results") -> Dict[str, pd.DataFrame]:
    """Load all classification data into DataFrames."""
    
    dataframes = {}
    
    # Load jailbreak classifications
    jailbreak_file = f"{data_dir}/jailbreak_classifications.json"
    if os.path.exists(jailbreak_file):
        with open(jailbreak_file, 'r') as f:
            data = json.load(f)
        dataframes["jailbreak"] = pd.DataFrame(data)
        logger.info(f"Loaded {len(dataframes['jailbreak'])} jailbreak classifications")
    
    # Load control classifications
    control_file = f"{data_dir}/control_classifications.json"
    if os.path.exists(control_file):
        with open(control_file, 'r') as f:
            data = json.load(f)
        dataframes["control"] = pd.DataFrame(data)
        logger.info(f"Loaded {len(dataframes['control'])} control classifications")
    
    # Load benchmark classifications
    benchmark_file = f"{data_dir}/benchmark_classifications.json"
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r') as f:
            data = json.load(f)
        dataframes["benchmark"] = pd.DataFrame(data)
        logger.info(f"Loaded {len(dataframes['benchmark'])} benchmark classifications")
    
    return dataframes


def calculate_jsr_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Jailbreak Success Rate for each model/quantization config."""
    
    grouped = df.groupby(['model_name', 'quantization']).agg({
        'classification': lambda x: (x == 'jailbreak_success').sum(),
        'prompt_id': 'count'
    }).reset_index()
    
    grouped.columns = ['model_name', 'quantization', 'jailbreak_successes', 'total']
    grouped['jsr'] = grouped['jailbreak_successes'] / grouped['total']
    grouped['quant_bits'] = grouped['quantization'].map(QUANT_BITS)
    grouped['quant_order'] = grouped['quantization'].apply(
        lambda x: QUANT_ORDER.index(x) if x in QUANT_ORDER else 99
    )
    grouped = grouped.sort_values(['model_name', 'quant_order'])
    
    return grouped


def calculate_jsr_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate JSR breakdown by jailbreak category."""
    
    grouped = df.groupby(['category', 'quantization']).agg({
        'classification': lambda x: (x == 'jailbreak_success').sum(),
        'prompt_id': 'count'
    }).reset_index()
    
    grouped.columns = ['category', 'quantization', 'jailbreak_successes', 'total']
    grouped['jsr'] = grouped['jailbreak_successes'] / grouped['total']
    grouped['quant_bits'] = grouped['quantization'].map(QUANT_BITS)
    
    return grouped


def calculate_frr_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate False Refusal Rate for each model/quantization config."""
    
    grouped = df.groupby(['model_name', 'quantization']).agg({
        'classification': lambda x: (x == 'false_refusal').sum(),
        'prompt_id': 'count'
    }).reset_index()
    
    grouped.columns = ['model_name', 'quantization', 'false_refusals', 'total']
    grouped['frr'] = grouped['false_refusals'] / grouped['total']
    grouped['quant_bits'] = grouped['quantization'].map(QUANT_BITS)
    
    return grouped


def calculate_benchmark_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate benchmark accuracy for each model/quantization config."""
    
    grouped = df.groupby(['model_name', 'quantization']).agg({
        'classification': lambda x: (x == 'correct').sum(),
        'question_id': 'count'
    }).reset_index()
    
    grouped.columns = ['model_name', 'quantization', 'correct', 'total']
    grouped['accuracy'] = grouped['correct'] / grouped['total']
    grouped['quant_bits'] = grouped['quantization'].map(QUANT_BITS)
    
    return grouped


def generate_summary_table(jsr_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table of JSR by model and quantization."""
    
    pivot = jsr_df.pivot(
        index='model_name',
        columns='quantization',
        values='jsr'
    )[['fp16', 'q8_0', 'q4_K_M', 'q2_K']]
    
    # Convert to percentages
    pivot = pivot * 100
    
    return pivot


def run_analysis(data_dir: str = "data/results", output_dir: str = "data/results"):
    """Run complete analysis pipeline."""
    
    logger.info("Starting analysis...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    dataframes = load_classifications(data_dir)
    
    if not dataframes:
        logger.error("No classification data found!")
        return None
    
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "metrics": {}
    }
    
    # Analyze jailbreak data
    if "jailbreak" in dataframes:
        df = dataframes["jailbreak"]
        
        jsr_by_config = calculate_jsr_by_config(df)
        results["metrics"]["jsr_by_config"] = jsr_by_config.to_dict('records')
        
        jsr_by_category = calculate_jsr_by_category(df)
        results["metrics"]["jsr_by_category"] = jsr_by_category.to_dict('records')
        
        # Summary table
        summary_table = generate_summary_table(jsr_by_config)
        summary_table.to_csv(f"{output_dir}/jsr_summary_table.csv")
        
        logger.info("JSR Summary Table:")
        print(summary_table.round(1))
    
    # Analyze control data
    if "control" in dataframes:
        df = dataframes["control"]
        
        frr_by_config = calculate_frr_by_config(df)
        results["metrics"]["frr_by_config"] = frr_by_config.to_dict('records')
    
    # Analyze benchmark data
    if "benchmark" in dataframes:
        df = dataframes["benchmark"]
        
        benchmark_accuracy = calculate_benchmark_accuracy(df)
        results["metrics"]["benchmark_accuracy"] = benchmark_accuracy.to_dict('records')
    
    # Save results
    with open(f"{output_dir}/analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}/analysis_results.json")
    
    return results


if __name__ == "__main__":
    run_analysis()
