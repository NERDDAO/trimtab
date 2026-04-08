"""Build grammars from text corpora via n-gram extraction and clustering."""

import logging
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from trimtab.grammar import Grammar
from trimtab.embedder import Embedder

logger = logging.getLogger(__name__)

# Common English stopwords to filter
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "it", "its", "this", "that", "these", "those", "and", "but", "or",
    "not", "no", "so", "if", "then", "than", "too", "very", "just",
}


@dataclass
class NGram:
    """An extracted n-gram with frequency count."""
    text: str
    count: int
    n: int  # 2, 3, or 4


def extract_ngrams(
    texts: list[str],
    min_n: int = 2,
    max_n: int = 4,
    min_count: int = 2,
) -> list[NGram]:
    """Extract n-grams from a list of texts.

    Args:
        texts: Input text corpus.
        min_n: Minimum n-gram size.
        max_n: Maximum n-gram size.
        min_count: Minimum frequency to keep.

    Returns:
        List of NGram objects sorted by count descending.
    """
    counters: dict[int, Counter] = {n: Counter() for n in range(min_n, max_n + 1)}

    for text in texts:
        # Tokenize: lowercase, split on non-alphanumeric
        words = re.findall(r'\b[a-z]+\b', text.lower())

        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i:i + n])
                # Skip if all stopwords
                if all(w in _STOPWORDS for w in gram):
                    continue
                counters[n][gram] += 1

    results: list[NGram] = []
    for n, counter in counters.items():
        for gram, count in counter.items():
            if count >= min_count:
                results.append(NGram(text=" ".join(gram), count=count, n=n))

    results.sort(key=lambda x: x.count, reverse=True)
    return results


def cluster_ngrams(
    ngrams: list[NGram],
    embedder: Embedder,
    min_cluster_size: int = 3,
) -> dict[int, list[NGram]]:
    """Cluster n-grams by embedding similarity.

    Args:
        ngrams: N-grams to cluster.
        embedder: Embedding model.
        min_cluster_size: Minimum cluster size for HDBSCAN.

    Returns:
        Dict mapping cluster_id -> list of NGrams. Cluster -1 is noise.
    """
    if len(ngrams) < min_cluster_size:
        return {0: ngrams}

    texts = [ng.text for ng in ngrams]
    vectors = embedder.embed(texts)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors)

    clusters: dict[int, list[NGram]] = {}
    for ng, label in zip(ngrams, labels):
        clusters.setdefault(int(label), []).append(ng)

    return clusters


def clusters_to_grammar(
    clusters: dict[int, list[NGram]],
    rule_names: dict[int, str] | None = None,
) -> Grammar:
    """Convert clusters into a Grammar.

    Args:
        clusters: Cluster ID -> n-grams mapping.
        rule_names: Optional mapping of cluster ID -> rule name.
                    If not provided, rules are named "cluster_0", "cluster_1", etc.

    Returns:
        Grammar with one rule per cluster, plus an origin rule referencing all.
    """
    rules: dict[str, list[str]] = {}
    rule_refs: list[str] = []

    for cluster_id, ngrams in clusters.items():
        if cluster_id == -1:  # skip noise cluster
            continue

        name = (rule_names or {}).get(cluster_id, f"cluster_{cluster_id}")
        rules[name] = [ng.text for ng in ngrams]
        rule_refs.append(f"#{name}#")

    if rule_refs:
        rules["origin"] = rule_refs

    return Grammar(rules=rules)


def build_grammar(
    texts: list[str],
    embedder: Embedder,
    min_count: int = 2,
    min_cluster_size: int = 3,
    rule_names: dict[int, str] | None = None,
) -> Grammar:
    """Full pipeline: extract n-grams, cluster, build grammar.

    Args:
        texts: Input corpus.
        embedder: Embedding model.
        min_count: Minimum n-gram frequency.
        min_cluster_size: Minimum HDBSCAN cluster size.
        rule_names: Optional cluster-to-rule-name mapping.

    Returns:
        Grammar built from the corpus.
    """
    ngrams = extract_ngrams(texts, min_count=min_count)
    logger.info("Extracted %d n-grams", len(ngrams))

    if not ngrams:
        return Grammar(rules={"origin": ["[empty corpus]"]})

    clusters = cluster_ngrams(ngrams, embedder, min_cluster_size=min_cluster_size)
    logger.info("Found %d clusters", len([k for k in clusters if k != -1]))

    grammar = clusters_to_grammar(clusters, rule_names=rule_names)
    return grammar
