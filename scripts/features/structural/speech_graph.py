# scripts/features/structural/speech_graph.py

import logging
import pandas as pd
import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    STRUCTURAL_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# NLTK SETUP
# =============================================================================

try:
    sent_tokenize("Test.")
    word_tokenize("Test.")
except LookupError:
    logger.info("Downloading NLTK tokenizers...")
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)


# =============================================================================
# CORE GRAPH CONSTRUCTION
# =============================================================================

def build_sentence_graph(sentence: str) -> nx.DiGraph:
    """
    Build a directed weighted word transition graph from a single sentence.

    Input:
        sentence — a single sentence string

    Output:
        networkx.DiGraph where:
            - nodes = unique lowercase words
            - edges = directed from word_i to word_{i+1}
            - edge weights = count of how many times that transition occurs

    Why lowercase?
    "I" at sentence start and "i" inside should count as the same node.
    Lowercasing is standard in this feature family.

    Why directed graph?
    Word order matters. "I love you" and "you love I" have different
    meanings — a directed graph preserves this by pointing edges in the
    direction of word flow.

    Why weighted edges?
    In a sentence like "really really really happy", the transition
    "really → really" happens multiple times. Edge weight captures this
    repetition strength, which is key for measuring rumination patterns.
    """
    # Tokenize and lowercase. NLTK's word_tokenize handles contractions
    # and punctuation better than simple split() — "don't" becomes ["do", "n't"]
    # rather than being kept as one token, which is the standard linguistic
    # treatment.
    words = [w.lower() for w in word_tokenize(sentence) if w.isalpha()]
    # w.isalpha() filters out punctuation and numbers — we want only
    # actual word transitions, not "I , think" being treated as 3 tokens.

    # Initialize directed graph
    G = nx.DiGraph()

    # Build edges between consecutive words
    for i in range(len(words) - 1):
        source = words[i]
        target = words[i + 1]

        if G.has_edge(source, target):
            # Edge already exists — increment its weight
            G[source][target]["weight"] += 1
        else:
            # New edge — add with initial weight 1
            G.add_edge(source, target, weight=1)

    return G


# =============================================================================
# PER-SENTENCE METRICS
# =============================================================================

def compute_graph_metrics(G: nx.DiGraph, num_raw_transitions: int) -> dict:
    """
    Compute 5 graph-level statistics for one sentence graph.

    Input:
        G                   — the directed word transition graph
        num_raw_transitions — total word-to-word transitions BEFORE
                              deduplication (i.e., sentence length - 1)

    Output:
        dict with 5 keys:
        - nodes    : number of unique words
        - edges    : number of unique directed transitions
        - density  : graph density in [0, 1]
        - loops    : number of repeated transitions (recurrence)
        - weight   : average edge weight (transition strength)

    Why pass num_raw_transitions instead of computing from G?
    Raw transitions = sentence_length - 1, which equals number of edges
    BEFORE counting repeats as separate. But G.number_of_edges() only
    counts UNIQUE edges. The difference is the loop count.
    We need both to compute recurrence correctly.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Graph density = edges / (nodes * (nodes - 1))
    # For a directed graph, max possible edges = n * (n - 1)
    # Density of 1 means every word transitions to every other word.
    # Density of 0 means no repeated structure.
    # networkx computes this correctly; we just check for edge case.
    density = nx.density(G) if num_nodes > 1 else 0.0

    # Recurrence loops = how many word transitions were repeated.
    # If sentence has 10 transitions total but only 7 unique transitions,
    # then 3 transitions were repeats = 3 recurrence loops.
    loops = num_raw_transitions - num_edges

    # Average edge weight = average strength of word transitions.
    # A value > 1 means at least some transitions are repeated.
    if num_edges > 0:
        total_weight = sum(data["weight"] for _, _, data in G.edges(data=True))
        avg_weight   = total_weight / num_edges
    else:
        avg_weight = 0.0

    return {
        "nodes"  : num_nodes,
        "edges"  : num_edges,
        "density": density,
        "loops"  : loops,
        "weight" : avg_weight,
    }


# =============================================================================
# POST-LEVEL AGGREGATION — MEAN AND STD
# =============================================================================

def compute_graph_features(text: str) -> dict:
    """
    Compute mean and std of graph metrics across all sentences in a post.

    Input:
        text — a single merged post string

    Output:
        dict with 10 keys:
        - avg_nodes, std_nodes
        - avg_edges, std_edges
        - avg_density, std_density
        - avg_loops, std_loops
        - avg_weight, std_weight

    Why both mean and std?
    Mean captures typical sentence structure.
    Std captures within-post consistency.

    Examples:
        - Depression: low std (consistently repetitive across sentences)
        - ADHD:       high std (mix of structured and fragmented sentences)
        - Bipolar:    high std (rapid sentence-level changes)
        - Control:    moderate std (normal natural variation)

    Baseline used only means — this upgrade adds the std dimension
    which is a key methodological contribution of your graduation work.
    """
    # Default zero vector for edge cases
    empty_result = {
        "avg_nodes"  : 0.0, "std_nodes"  : 0.0,
        "avg_edges"  : 0.0, "std_edges"  : 0.0,
        "avg_density": 0.0, "std_density": 0.0,
        "avg_loops"  : 0.0, "std_loops"  : 0.0,
        "avg_weight" : 0.0, "std_weight" : 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    # Split post into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception:
        return empty_result

    # Collect per-sentence metric dictionaries
    per_sentence_metrics = []

    for sentence in sentences:
        # Count raw word transitions (before deduplication) for loop calculation
        words = [w.lower() for w in word_tokenize(sentence) if w.isalpha()]

        # Need at least 2 words to have any transitions
        if len(words) < 2:
            continue

        num_raw_transitions = len(words) - 1

        # Build graph and compute its metrics
        G       = build_sentence_graph(sentence)
        metrics = compute_graph_metrics(G, num_raw_transitions)

        per_sentence_metrics.append(metrics)

    # Edge case: no valid sentences (all were too short)
    if len(per_sentence_metrics) == 0:
        return empty_result

    # Convert list of dicts to DataFrame for vectorized mean/std
    # Shape: (num_sentences, 5) — rows are sentences, columns are metrics
    df_metrics = pd.DataFrame(per_sentence_metrics)

    # Compute mean and std column-wise
    # .mean() returns a Series with one value per column
    # .std() with ddof=0 computes population std (not sample std) — we want
    # the actual variability in this specific post, not an estimator for
    # some hypothetical population.
    mean_vals = df_metrics.mean()
    std_vals  = df_metrics.std(ddof=0)

    return {
        "avg_nodes"  : round(float(mean_vals["nodes"]),   6),
        "std_nodes"  : round(float(std_vals["nodes"]),    6),
        "avg_edges"  : round(float(mean_vals["edges"]),   6),
        "std_edges"  : round(float(std_vals["edges"]),    6),
        "avg_density": round(float(mean_vals["density"]), 6),
        "std_density": round(float(std_vals["density"]),  6),
        "avg_loops"  : round(float(mean_vals["loops"]),   6),
        "std_loops"  : round(float(std_vals["loops"]),    6),
        "avg_weight" : round(float(mean_vals["weight"]),  6),
        "std_weight" : round(float(std_vals["weight"]),   6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract graph features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/structural/speech_graph_{split}.csv
        Shape: (num_samples, 10)

    Runtime:
        NetworkX graph construction and metric computation is pure Python
        and scales linearly with total word count across all posts.
        Expect ~5-10 minutes per split for 13k posts on CPU.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Graph features ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_graph_features(text))
    )

    save_dir  = STRUCTURAL_FEATURES_DIR / "speech_graph"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"speech_graph_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_speech_graph_pipeline() -> None:
    """
    Extract and save speech graph features for all three splits.
    Output: 10-dimensional feature vector per sample
            [avg/std × nodes, edges, density, loops, weight].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting speech graph feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Speech graph extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_speech_graph_pipeline()