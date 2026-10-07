# The Quantization Safety Tax

**How Model Compression Degrades Alignment in Open-Source LLMs**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and sample data for reproducing the experiments in our study on how quantization affects LLM safety alignment.

> **Note**: Paper is currently under preparation. Citation information will be updated upon publication.

## Key Findings

- **Aggressive quantization (Q2) can increase jailbreak success rates by up to 165%** (Phi-3: 17% → 45%)
- **Moderate quantization (Q8, Q4) generally preserves safety alignment**
- **Safety degradation is model-dependent**: Llama 3.2 is robust, Phi-3 is highly vulnerable
- **Safety can degrade even as capabilities collapse**: Phi-3 retained enough coherence to produce harmful outputs despite 91pp accuracy drop

## Repository Structure

```
├── src/
│   ├── data_collection.py    # Collect model responses via Ollama
│   ├── classification.py     # Classify responses as jailbreak/refusal
│   ├── analysis.py          # Statistical analysis
│   └── visualization.py     # Generate publication figures
├── data/
│   ├── sample_prompts/      # Sample prompts (subset for reproducibility)
│   └── results/             # Where results will be saved
├── configs/
│   └── models.yaml          # Model configurations
├── scripts/
│   ├── setup.sh            # Environment setup
│   └── run_experiments.sh  # Run full pipeline
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- ~50GB disk space for model weights

### Installation

```bash
# Clone the repository
git clone https://github.com/bibekpdl/quantization-safety-tax.git
cd quantization-safety-tax

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull required models (this will take a while)
./scripts/setup.sh
```

### Running Experiments

```bash
# Run the full pipeline
./scripts/run_experiments.sh

# Or run individual steps:
python src/data_collection.py    # Collect responses
python src/classification.py     # Classify responses
python src/analysis.py          # Run analysis
python src/visualization.py     # Generate figures
```

## Models Tested

| Model | Parameters | Organization |
|-------|-----------|--------------|
| Llama 3.2 | 3B | Meta |
| Gemma 2 | 2B | Google |
| Gemma 3 | 4B | Google |
| Qwen 2.5 | 3B | Alibaba |
| Phi-3 | 3.8B | Microsoft |
| Granite 3.1 | 2B | IBM |

Each model is tested at four quantization levels:
- **FP16** (16-bit, baseline)
- **Q8** (8-bit)
- **Q4** (4-bit)
- **Q2** (2-bit, most aggressive)

## Evaluation Methodology

### Jailbreak Success Rate (JSR)

We evaluate models on 100 jailbreak prompts across 5 categories:
- Direct harmful requests
- Roleplay-based attacks
- Encoding/obfuscation
- DAN-style prompt injections
- Prefix injection attacks

### Capability Benchmark

100 questions from MMLU, GSM8K, and commonsense reasoning to measure capability degradation alongside safety changes.

### Classification

Responses are classified using pattern-based heuristics validated against manual annotation (93% agreement, Cohen's κ = 0.86).

## Sample Data

For ethical reasons, we provide only **sample prompts** demonstrating the format rather than the full jailbreak dataset. The samples include:
- 2 examples per category (10 total)
- Benign control prompts
- Capability benchmark questions

Researchers requiring the full dataset for replication can contact the authors.

## Results Summary

| Model | FP16 JSR | Q2 JSR | Change |
|-------|----------|--------|--------|
| Llama 3.2 | 17% | 19% | +2pp |
| Gemma 2 | 25% | 34% | +9pp |
| Gemma 3 | 32% | N/A* | - |
| Qwen 2.5 | 20% | 0%† | -20pp |
| Phi-3 | 17% | **45%** | **+28pp** |
| Granite 3.1 | 30% | 37% | +7pp |

*Q2 quantization unavailable for Gemma 3  
†Model became incoherent at Q2

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{poudel2026quantization,
  title        = {The Quantization Safety Tax: How Model Compression Degrades Alignment in Open-Source LLMs},
  author       = {Poudel, Bibek and Sanna, Arun Chowdary},
  year         = {2026},
  howpublished = {\url{https://github.com/bibekpdl/quantization-safety-tax}},
  note         = {Manuscript in preparation}
}
```

> **Note**: This citation will be updated with details once the paper is published.

## Ethics Statement

This research was conducted to improve understanding of AI safety risks. The jailbreak prompts are designed for defensive research purposes only. We do not release actual harmful model outputs, and we have taken care to avoid providing a roadmap for malicious actors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The open-source community for Ollama and llama.cpp
- Model creators at Meta, Google, Alibaba, Microsoft, and IBM

## Contact

- **Bibek Poudel** - [bibek@linux.com](mailto:bibek@linux.com)  
  ORCID: [0009-0001-5186-6803](https://orcid.org/0009-0001-5186-6803)

- **Arun Chowdary Sanna** - [arun.sanna@ieee.org](mailto:arun.sanna@ieee.org)  
  ORCID: [0009-0003-5131-2688](https://orcid.org/0009-0003-5131-2688)
