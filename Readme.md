# A Multi-Aspect Linguistic Feature Fusion Network for Mental Health Text Classification

Graduation research project at Hanoi University of Science and Technology.

## Overview

This project classifies mental health conditions from Reddit posts using a multi-aspect feature fusion approach that combines:

- **Semantic features** — MentalRoBERTa embeddings, psycholinguistic ratios, lexical diversity
- **Affective features** — GoEmotions scores, VAD scores, sentiment arc
- **Structural features** — Speech graph cohesion, discourse coherence, tense distribution
- **Stylistic features** — POS ratios, readability, hedging markers

Features are combined via a **gated fusion network** trained with **GradNorm** loss balancing.

## Dataset

Reddit posts from the Murarka et al. (2021) dataset covering 6 classes:
ADHD, Anxiety, Bipolar, Depression, PTSD, None (control).

## Project Structure

\`\`\`
scripts/
├── config.py                     # Central config
├── data/                         # Data preprocessing & EDA
├── features/                     # Feature extraction (4 groups)
│   ├── semantic/
│   ├── affective/
│   ├── structural/
│   └── stylistic/
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

## Author

Duong Thi Huyen — huyen.dt224317@sis.hust.edu.vn
Supervised by Prof. Do Thi Ngoc Diep