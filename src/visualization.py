#!/usr/bin/env python3
"""
Visualization Module for Quantization Safety Study

Generates publication-quality figures for the paper.

Author: Bibek Poudel
License: MIT
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Color scheme
COLORS = {
    'llama3.2': '#1f77b4',   # Blue
    'gemma2': '#ff7f0e',     # Orange
    'gemma3': '#9467bd',     # Purple
    'qwen2.5': '#2ca02c',    # Green
    'phi3': '#d62728',       # Red
    'granite3.1': '#8c564b', # Brown
    'primary': '#1f77b4',
    'danger': '#d62728'
}

MODEL_DISPLAY_NAMES = {
    'llama3.2': 'Llama 3.2 3B',
    'gemma2': 'Gemma 2 2B',
    'gemma3': 'Gemma 3 4B',
    'qwen2.5': 'Qwen 2.5 3B',
    'phi3': 'Phi-3 3.8B',
    'granite3.1': 'Granite 3.1 2B'
}

QUANT_LABELS = {
    'fp16': 'FP16\n(16-bit)',
    'q8_0': 'Q8\n(8-bit)',
    'q4_K_M': 'Q4\n(4-bit)',
    'q2_K': 'Q2\n(2-bit)'
}

QUANT_ORDER = ['fp16', 'q8_0', 'q4_K_M', 'q2_K']


def load_analysis_results(filepath: str = "data/results/analysis_results.json") -> Dict:
    """Load analysis results."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def figure1_jsr_vs_quantization(results: Dict, output_dir: str = "figures"):
    """
    Figure 1: Main result - JSR vs quantization level for all models.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    jsr_data = pd.DataFrame(results["metrics"]["jsr_by_config"])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, model in enumerate(sorted(jsr_data['model_name'].unique())):
        model_data = jsr_data[jsr_data['model_name'] == model]
        model_data = model_data.sort_values('quant_bits', ascending=False)
        
        x_positions = [QUANT_ORDER.index(q) for q in model_data['quantization'] if q in QUANT_ORDER]
        y_values = [model_data[model_data['quantization'] == QUANT_ORDER[x]]['jsr'].values[0] * 100 
                    for x in x_positions]
        
        ax.plot(x_positions, y_values, 
                marker=markers[idx % len(markers)], markersize=8, linewidth=2,
                color=COLORS.get(model, f'C{idx}'),
                label=MODEL_DISPLAY_NAMES.get(model, model))
    
    ax.set_xticks(range(len(QUANT_ORDER)))
    ax.set_xticklabels([QUANT_LABELS[q] for q in QUANT_ORDER])
    ax.set_xlabel('Quantization Level', fontweight='bold')
    ax.set_ylabel('Jailbreak Success Rate (%)', fontweight='bold')
    ax.set_title('Jailbreak Success Rate vs Quantization Level', fontweight='bold', pad=15)
    
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    
    # Shaded risk regions
    ax.axhspan(0, 10, alpha=0.1, color='green')
    ax.axhspan(10, 25, alpha=0.1, color='yellow')
    ax.axhspan(25, 100, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_jsr_vs_quantization.png", dpi=300)
    plt.savefig(f"{output_dir}/figure1_jsr_vs_quantization.pdf")
    plt.close()
    
    logger.info(f"Saved Figure 1 to {output_dir}")


def figure2_heatmap(results: Dict, output_dir: str = "figures"):
    """
    Figure 2: Heatmap of JSR by model and quantization.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    jsr_data = pd.DataFrame(results["metrics"]["jsr_by_config"])
    
    # Create pivot table
    pivot = jsr_data.pivot(index='model_name', columns='quantization', values='jsr')
    
    # Reorder columns and rows
    quant_cols = [q for q in QUANT_ORDER if q in pivot.columns]
    pivot = pivot[quant_cols]
    pivot = pivot * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                vmin=0, vmax=50, cbar_kws={'label': 'JSR (%)'}, ax=ax)
    
    ax.set_xlabel('Quantization Level', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Jailbreak Success Rate Heatmap', fontweight='bold', pad=15)
    
    # Update y-tick labels
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in pivot.index], rotation=0)
    ax.set_xticklabels(['FP16', 'Q8', 'Q4', 'Q2'][:len(quant_cols)])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_heatmap.png", dpi=300)
    plt.savefig(f"{output_dir}/figure2_heatmap.pdf")
    plt.close()
    
    logger.info(f"Saved Figure 2 to {output_dir}")


def figure3_safety_vs_capability(results: Dict, output_dir: str = "figures"):
    """
    Figure 3: Safety degradation vs capability degradation comparison.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    jsr_data = pd.DataFrame(results["metrics"]["jsr_by_config"])
    
    if "benchmark_accuracy" not in results["metrics"]:
        logger.warning("No benchmark data available for Figure 3")
        return
    
    bench_data = pd.DataFrame(results["metrics"]["benchmark_accuracy"])
    
    models = sorted(jsr_data['model_name'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, model in enumerate(models[:6]):
        ax = axes[idx]
        
        model_jsr = jsr_data[jsr_data['model_name'] == model].set_index('quantization')
        model_bench = bench_data[bench_data['model_name'] == model].set_index('quantization')
        
        quants = [q for q in QUANT_ORDER if q in model_jsr.index]
        x = range(len(quants))
        
        jsr_vals = [model_jsr.loc[q, 'jsr'] * 100 for q in quants]
        acc_vals = [model_bench.loc[q, 'accuracy'] * 100 if q in model_bench.index else 0 for q in quants]
        
        ax.plot(x, jsr_vals, 'o-', color=COLORS['danger'], linewidth=2, markersize=8, label='JSR')
        ax.set_ylabel('JSR (%)', color=COLORS['danger'])
        ax.tick_params(axis='y', labelcolor=COLORS['danger'])
        ax.set_ylim(0, max(50, max(jsr_vals) * 1.2))
        
        ax2 = ax.twinx()
        ax2.plot(x, acc_vals, 's--', color=COLORS['primary'], linewidth=2, markersize=8, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', color=COLORS['primary'])
        ax2.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax2.set_ylim(0, 105)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['FP16', 'Q8', 'Q4', 'Q2'][:len(quants)])
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model), fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Safety vs Capability Degradation by Model', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_safety_vs_capability.png", dpi=300)
    plt.savefig(f"{output_dir}/figure3_safety_vs_capability.pdf")
    plt.close()
    
    logger.info(f"Saved Figure 3 to {output_dir}")


def generate_all_figures(data_dir: str = "data/results", output_dir: str = "figures"):
    """Generate all figures."""
    
    logger.info("Generating figures...")
    
    results = load_analysis_results(f"{data_dir}/analysis_results.json")
    
    if not results:
        logger.error("No analysis results found!")
        return
    
    figure1_jsr_vs_quantization(results, output_dir)
    figure2_heatmap(results, output_dir)
    figure3_safety_vs_capability(results, output_dir)
    
    logger.info("All figures generated successfully")


if __name__ == "__main__":
    generate_all_figures()
