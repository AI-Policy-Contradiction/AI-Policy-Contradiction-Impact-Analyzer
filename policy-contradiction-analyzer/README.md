# AI Policy Contradiction & Impact Analyzer

## Research Project

Explainable Detection of Contradictions and Inconsistencies in Regulatory Documents using Transformer-Based NLP.

## Research Pipeline

1. Dataset Understanding
2. Preprocessing
3. TF-IDF + Logistic Regression baseline
4. Transformer model
5. Evaluation
6. Explainability / Evidence Extraction
7. Error Analysis
8. Optional LLM Explanation Layer
9. Policy Impact Analyzer
10. Final Application

## Project Structure

```text
policy-contradiction-analyzer/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_transformer_experiments.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── explain.py
│   └── impact_analysis.py
├── app/
│   └── app.py
├── results/
├── requirements.txt
└── README.md
```

## Current Status

Project structure created.

Next phase: inspect the supplied dataset before implementing the research pipeline.
