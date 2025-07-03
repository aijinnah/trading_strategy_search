import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Any, Dict

import numpy as np

from utils import setup_logger
from vars import LOG_FILE

logger = setup_logger("Logger1", LOG_FILE)


class Island:
    def __init__(self, initial_strategy_code):
        self.strategy_codes = [
            {
                "current": initial_strategy_code,
                "fallback": initial_strategy_code,
                "is_broken": False,
                "window_scores": defaultdict(dict),
                "trade_log": "",
            }
        ]


@dataclass
class StrategyResult:
    perf_metrics: Dict[str, Any]
    trade_log: str
    weaknesses: str


class StrategyCache:
    """
    Thread/process-safe cache for strategy evaluation results.
    """

    def __init__(self):
        self.manager = Manager()
        self.cache = self.manager.dict()

    @staticmethod
    def hash_strategy(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()

    def get_result(self, strategy_code: str, window_ix: int):
        strategy_hash = self.hash_strategy(strategy_code)
        return self.cache.get((strategy_hash, window_ix))

    def cache_result(self, strategy_code: str, window_ix: int, result: StrategyResult):
        strategy_hash = self.hash_strategy(strategy_code)
        self.cache[(strategy_hash, window_ix)] = result


def get_island_score(island, agg_type="mean"):
    """
    Aggregate island's strategies' window scores (average or max).
    """
    scores = []
    for strat_dict in island.strategy_codes:
        valid_scores = [
            ws["score"] if not np.isneginf(ws["score"]) else 0
            for ws in strat_dict["window_scores"].values()
        ]
        if valid_scores:
            scores.append(np.mean(valid_scores))
    if not scores:
        return float("-inf")

    if agg_type == "mean":
        return np.mean(scores)
    elif agg_type == "max":
        return max(scores)
    else:
        return float("-inf")


def create_clusters_by_signature(island):
    cluster_signatures = defaultdict(list)
    for strat_ix, strat_dict in enumerate(island.strategy_codes):
        window_scores = []
        for window_ix, score in strat_dict["window_scores"].items():
            window_scores.append(score["score"])
        signature = np.mean(window_scores)
        cluster_signatures[signature].append(strat_ix)
    return cluster_signatures


def sample_weighted_cluster(cluster_signatures, n_parents=1, temperature=0.5):
    """
    Implements Boltzmann tournament selection to sample N parents from the population.

    Args:
        cluster_signatures: Dictionary of fitness scores to items
        n_parents: Number of parents to select (default: 2)
        temperature: Temperature parameter for softmax (default: 0.5)
                    Lower values make selection more greedy

    Returns:
        List of tuples (score, item) for selected parents
    """
    # Extract valid scores and items
    scores = np.array([k for k, _ in cluster_signatures.items() if not np.isneginf(k)])
    items = [v for k, v in cluster_signatures.items() if not np.isneginf(k)]

    if len(scores) == 0:
        return []

    # Compute softmax probabilities
    # Shift scores to prevent overflow and apply temperature scaling
    shifted_scores = scores - np.max(scores)  # Subtract max for numerical stability
    exp_scores = np.exp(shifted_scores / temperature)
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample n_parents with replacement using softmax probabilities
    selected_indices = np.random.choice(
        len(scores), size=min(n_parents, len(scores)), p=probabilities, replace=True
    )

    # Return selected parents as (score, item) tuples
    selected_parents = [(scores[idx], items[idx]) for idx in selected_indices]
    return selected_parents


def sample_strategies_for_prompt(island, generation, k_mod=1):
    """
    Simple random or cluster-based sampling of strategies for LLM prompt building.
    """
    if len(island.strategy_codes) <= k_mod:
        return island.strategy_codes

    cluster_signatures = create_clusters_by_signature(island)

    for ix, strat_dict in enumerate(island.strategy_codes):
        means_score = np.mean(
            ([ws["score"] for ws in strat_dict["window_scores"].values()])
        )
        logger.info(f"Strategy {ix} Score: {means_score}")

    logger.info(f"Number of Clusters: {len(cluster_signatures)}")

    sampled_strategies_with_scores = []
    c = 0

    while (
        len({s[0]["current"] for s in sampled_strategies_with_scores}) < k_mod
        and c < 20
    ):
        c += 1
        valid_clusters = [
            k for k, v in cluster_signatures.items() if not np.isneginf(k)
        ]
        if not valid_clusters:
            continue
        if len(valid_clusters) > 1:
            sampled_cluster_signature = sample_weighted_cluster(
                cluster_signatures, temperature=0.5
            )
            if sampled_cluster_signature[0] is None:
                continue
        else:
            sampled_cluster_signature = random.choice(
                [(k, v) for k, v in cluster_signatures.items() if not np.isneginf(k)]
            )
        if len(valid_clusters) == 1 or (c == 10):
            k_mod = 1

        sampled_programs_ix = random.choice(sampled_cluster_signature[1])
        strat_obj = island.strategy_codes[sampled_programs_ix]
        sampled_strategies_with_scores.append((strat_obj, sampled_cluster_signature[0]))

    # Sort by score ascending
    sampled_strategies_sorted = sorted(
        sampled_strategies_with_scores, key=lambda x: x[1], reverse=False
    )

    sampled_strategies = [x[0] for x in sampled_strategies_sorted]
    return sampled_strategies
