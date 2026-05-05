# A Multi-Aspect Linguistic Feature Fusion Network for Mental Health Text Classification

Graduation research project at Hanoi University of Science and Technology.

## Overview

This project classifies mental health conditions from Reddit posts using a multi-aspect feature fusion approach that combines:

- **Semantic features** — MentalRoBERTa embeddings
- **Lexical features** — MTLD, word rates, pronouns, punctuation markers
- **Syntactic features** — Complexity, POS ratios, readability
- **Affective features** — GoEmotions scores, VAD scores, sentiment arc
- **Structural features** — Sentence coherence, topic drift, coherence breaks, tense distribution

Features are combined with **late concatenation fusion** by default, with a gated fusion baseline available for comparison.

## Dataset

Reddit posts from the Murarka et al. (2021) dataset covering 6 classes:
ADHD, Anxiety, Bipolar, Depression, PTSD, None (control).

## Project Structure

\`\`\`
scripts/
├── config.py                     # Central config
├── data/                         # Data preprocessing & EDA
├── features/                     # Feature extraction (5 groups)
│   ├── semantic/
│   ├── lexical/
│   ├── syntactic/
│   ├── affective/
│   └── structural/
├── analysis/                     # Statistical heatmaps
├── models/                       # Fusion network + baselines
└── evaluation/                   # Metrics & ablation
\`\`\`

## Setup

\`\`\`bash
# Clone and enter
git clone https://github.com/YOUR_USERNAME/mental-health-fusion.git
cd mental-health-fusion

# Create virtual environment
python -m venv amh_venv
amh_venv\Scripts\activate   # Windows
# source amh_venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
\`\`\`

## Requirements

See `requirements.txt`. Main dependencies:
- PyTorch
- transformers, datasets, evaluate
- spaCy, NLTK
- pandas, numpy, scikit-learn
- matplotlib, seaborn
