import os
import pickle
import random
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from eval import (
    evaluate_island,
    final_evaluation,
    log_top_performers,
    plot_strategy_performance,
    reconstruct_data_dict_from_multi,
)
from island import Island, StrategyCache, get_island_score
from llm import generate_new_strategy_code, generate_new_strategy_code_openai
from prepare_data import fetch_and_store_data_in_bulk, load_all_data, split_data_rolling
from stocks import msci_stocks
from utils import setup_logger
from vars import BEST_PERFORMERS_LOG_FILE, LOG_FILE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
load_dotenv(".env")

T_0 = 0.1
N = 30000


logger = setup_logger("Logger1", LOG_FILE)
best_performers_logger = setup_logger("BestPerformersLogger", BEST_PERFORMERS_LOG_FILE)


BASE_STRATEGY_CODE = generate_new_strategy_code_openai(
    model="gemini-2.0-flash-001", provider="google"
)


def evolutionary_loop(
    data_dict,
    generations=20,
    num_islands=5,
    k=2,
    discard_interval=5,
    m=2,
    train_years=3,
    test_months=6,
    max_strategies=20,
    n_new_strategies=1,
):
    """
    Main evolutionary loop, with concurrency at the Island level.
    """
    combined_data = pd.concat(
        {
            sym: df[["open", "high", "low", "close", "volume"]]
            for sym, df in data_dict.items()
        },
        axis=1,
    )
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    # Clean up symbol names
    combined_data.columns = pd.MultiIndex.from_tuples(
        [(symbol.strip(), field) for symbol, field in combined_data.columns],
        names=[None, "Price"],
    )

    # Rolling windows
    windows = split_data_rolling(combined_data, train_years, test_months)

    # Prepare islands
    def generate_new_island():
        time.sleep(1)
        return Island(
            generate_new_strategy_code_openai(
                model="gemini-2.0-flash-001", provider="google"
            )
        )

    islands = []

    with ThreadPoolExecutor(max_workers=num_islands) as executor:
        futures = [executor.submit(generate_new_island) for _ in range(num_islands)]

        for future in as_completed(futures):
            try:
                result = future.result()
                islands.append(result)
                logger.debug(f"Generated new island: {result}")
            except Exception as e:
                logger.error(f"Error generating island: {e}")

    # Shared cache
    strategy_cache = StrategyCache()
    cache_dict = strategy_cache.cache

    performance_history = []

    for generation in range(1, generations + 1):
        logger.info(f"\n=== Generation {generation} ===")

        # Evolve each Island in parallel
        with ProcessPoolExecutor(
            max_workers=min(num_islands, os.cpu_count())
        ) as executor:
            future_to_island_ix = {}
            for island_idx, island in enumerate(islands):
                fut = executor.submit(
                    evaluate_island,
                    island_idx,
                    island,
                    windows,
                    cache_dict,
                    generation,
                    k,
                    max_strategies,
                    n_new_strategies,
                )
                future_to_island_ix[fut] = island_idx

            # Collect results
            for fut in as_completed(future_to_island_ix):
                island_idx = future_to_island_ix[fut]
                try:
                    updated_island, idx_returned = fut.result()
                except Exception as e:
                    print(f"error in island eval")
                    print(traceback.format_exc())
                islands[idx_returned] = updated_island

        # Log top performers
        log_top_performers(
            islands, generation, log_interval=1, top_n=5, logger=best_performers_logger
        )

        # Track performance
        for island_idx, island in enumerate(islands):
            island_score = get_island_score(island, agg_type="mean")
            performance_history.append(
                {
                    "generation": generation,
                    "island": island_idx,
                    "score": island_score,
                    "num_strategies": len(island.strategy_codes),
                }
            )

        # Periodic culling
        if generation % discard_interval == 0 and generation != 0:
            logger.info(f"\n--- Culling & Seeding at Generation {generation} ---")
            # Sort islands by best composite
            islands_sorted = sorted(
                islands, key=lambda x: get_island_score(x), reverse=True
            )
            num_to_discard = m
            worst_islands = islands_sorted[-num_to_discard:]
            surviving_islands = islands_sorted[:-num_to_discard]

            # Best strategies
            best_strategies = []
            for isl in surviving_islands:
                if isl.strategy_codes:
                    best_strat = max(
                        isl.strategy_codes,
                        key=lambda sc: np.mean(
                            [ws["score"] for ws in sc["window_scores"].values()]
                        ),
                    )
                    best_strategies.append(best_strat["current"])

            # Reseed worst
            for island in worst_islands:
                if best_strategies and random.random() < 0.5:
                    seed_strategy = random.choice(best_strategies)
                else:
                    seed_strategy = generate_new_strategy_code_openai(
                        model="gemini-2.0-flash-001", provider="google"
                    )
                island.strategy_codes = [
                    {
                        "current": seed_strategy,
                        "fallback": seed_strategy,
                        "is_broken": False,
                        "window_scores": defaultdict(dict),
                        "trade_log": "",
                    }
                ]

            # Reassemble
            islands = surviving_islands + worst_islands

        # Dump progress
        with open("dump_08_02_25_perf_history.pkl", "wb") as f:
            pickle.dump(performance_history, f)

        with open("dump_08_02_25_strategies.pkl", "wb") as f:
            pickle.dump([isl.strategy_codes for isl in islands], f)

    return islands, windows, performance_history


def main():
    """
    Example usage with islands, generations, culling, etc.
    """
    random.seed(634310)
    symbols = random.sample(msci_stocks, 50)

    # 1) Fetch and store data
    fetch_and_store_data_in_bulk(symbols)

    # 2) Load data
    data_dict = load_all_data(symbols)

    for symbol, df in data_dict.items():
        if isinstance(df.columns, pd.MultiIndex) and "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")

    # 3) Run evolutionary loop
    islands, windows, perf_hist = evolutionary_loop(
        data_dict,
        generations=300,
        num_islands=7,
        k=2,
        discard_interval=5,
        m=2,
        train_years=6,
        test_months=20,
        max_strategies=99999999,
        n_new_strategies=3,
    )

    # 4) Pick best Island & Strategy
    best_island = max(islands, key=lambda x: get_island_score(x))
    best_strategy_dict = max(
        best_island.strategy_codes,
        key=lambda sc: np.mean([ws["score"] for ws in sc["window_scores"].values()]),
    )
    best_strategy_code = best_strategy_dict["current"]

    # 5) Final evaluation on the last window
    final_train, final_test = windows[-1]
    perf_train, perf_test, log_train, log_test = final_evaluation(
        best_strategy_code, final_train, final_test
    )

    logger.info("\nTraining Results (last window):")
    logger.info(log_train)
    logger.info("\nTesting Results (last window):")
    logger.info(log_test)

    # Optionally, plot
    test_dict = reconstruct_data_dict_from_multi(final_test)
    plot_strategy_performance(best_strategy_code, test_dict)

    logger.info("\n=== DONE ===")


if __name__ == "__main__":
    main()

# for isl in islands:
#     print(max([np.mean([ws["score"]/10 for ws in sd["window_scores"].values()]) for sd in isl.strategy_codes]))

# for ph in performance_history:
#     gen = ph["generation"]
#     print(f"gen: {gen}")
#     print(max([np.mean([ws["score"]/10 for ws in sd["window_scores"].values()]) for sd in ph['strategy_codes']]))
#     print("")

# codes = defaultdict(int)


#     evolutionary_loop(
#         data_dict,
#         generations=50,
#         num_islands=5,  # Number of separate islands
#         k=2,  # Number of strategies to sample from an island
#         discard_interval=5,  # Perform culling every 5 generations
#         m=4,  # Number of islands to discard each interval
#         train_years=2,
#         test_months=6,
#         max_strategies=20,
#     )
# ) -> best score was 0.467 sharpe ratio on 86 symbols, this is with passing two strats to the prompt to ge the new one


# test_strategy_window = islands[2].strategy_codes[0]

# generate_orders_func = compile_strategy_code(test_strategy_window["current"])
# test_dict = reconstruct_data_dict_from_multi(test_data)
# perf_metrics, trade_log, weaknesses = evaluate_strategy(
#     generate_orders_func, test_dict
# )
# plot_strategy_performance(test_strategy_window["current"], test_dict)

# subset_data_dict = {
#     k: v
#     for k, v in data_dict.items()
#     if k
#     in ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "INTC"]
# }


# we dont need to launch the process pool for cached values, waste of time.
# try test 7 with just k=1 and see if it works

# write 1 a generate orders func and 2 an obs func

# this is is using john jane feedback with the inspection_information thingy
# BEST STRATEGY AFTER 120 GENERATIONS
# Rank 1:
# Island: 7
# Strategy Hash: e49434e8e21d4b3e907d255f06d63518abdb7e4c2ca47e711d449b83544a2b13
# Average Score: 0.3422

# Performance Summary:
# ----------------------------------------
# Combined 42 symbols
# Total Trades: 16481
# Win Rate: 82.95%
# Net P&L: $76243.28
# Returns: 18.15%
# Avg Sharpe Ratio: 0.40
# Sortino Ratio: 1.17
# Calmar Ratio: 0.10
# CAGR: 2.82%
# Profit Factor: 1.52
# Max Drawdown: 28.05%
# Avg Benchmark Returns: 374.90%
# Weaknesses:
# Low risk-adjusted returns (Sharpe < 1).
# High maximum drawdown.  - Symbol: AIR:
# ----------------------------------------

# Rank 2:
# Island: 0
# Strategy Hash: 8de2cafd4e5ee947d846037d805fcca4b89efd35082c11b71ae6ec013a15b86b
# Average Score: 0.3388

# Performance Summary:
# ----------------------------------------
# Combined 42 symbols
# Total Trades: 201
# Win Rate: 58.71%
# Net P&L: $4387.53
# Returns: 1.04%
# Avg Sharpe Ratio: 0.14
# Sortino Ratio: 0.18
# Calmar Ratio: 0.01
# CAGR: 0.17%
# Profit Factor: 1.25
# Max Drawdown: 11.80%
# Avg Benchmark Returns: 374.90%
# Weaknesses:
# Low risk-adjusted returns (Sharpe < 1).  - Symbol: AIR:
#       long_trades_successful: 0
# ----------------------------------------

# Rank 3:
# Island: 3
# Strategy Hash: 4e4926c19addb3e9904c60c5c82be8ac4545f1cbca6f44db59956f6a38224b8b
# Average Score: 0.3354

# Performance Summary:
# ----------------------------------------
# Combined 42 symbols
# Total Trades: 20818
# Win Rate: 78.02%
# Net P&L: $149969.11
# Returns: 35.71%
# Avg Sharpe Ratio: 0.54
# Sortino Ratio: 1.30
# Calmar Ratio: 0.15
# CAGR: 5.22%
# Profit Factor: 1.78
# Max Drawdown: 34.69%
# Avg Benchmark Returns: 374.90%
# Weaknesses:
# Low risk-adjusted returns (Sharpe < 1).
# High maximum drawdown.  - Symbol: AIR:
# ----------------------------------------

# Rank 4:
# Island: 1
# Strategy Hash: 6ab1b6450ecbed8120bcd859f44a237e8a2f079ea6728fea80501fafef18cd17
# Average Score: 0.3209

# Performance Summary:
# ----------------------------------------
# Combined 42 symbols
# Total Trades: 287
# Win Rate: 55.75%
# Net P&L: $2227.69
# Returns: 0.53%
# Avg Sharpe Ratio: 0.05
# Sortino Ratio: 0.09
# Calmar Ratio: 0.01
# CAGR: 0.09%
# Profit Factor: 1.09
# Max Drawdown: 13.21%
# Avg Benchmark Returns: 374.90%
# Weaknesses:
# Low risk-adjusted returns (Sharpe < 1).  - Symbol: AIR:
#       indicator_analysis:
# ----------------------------------------

# Rank 5:
# Island: 2
# Strategy Hash: c1a9cbd94d3370a4abd1885a721c872490663649fb7649f218c82ec744ca8fb3
# Average Score: 0.3209

# Performance Summary:
# ----------------------------------------
# Combined 42 symbols
# Total Trades: 287
# Win Rate: 55.75%
# Net P&L: $2227.69
# Returns: 0.53%
# Avg Sharpe Ratio: 0.05
# Sortino Ratio: 0.09
# Calmar Ratio: 0.01
# CAGR: 0.09%
# Profit Factor: 1.09
# Max Drawdown: 13.21%
# Avg Benchmark Returns: 374.90%
# Weaknesses:
# Low risk-adjusted returns (Sharpe < 1).  - Symbol: AIR:
#       indicator_analysis:
# ----------------------------------------

# __________________________________________________
# add mutation types, not just basic? more variety?
# allow model to gen a strategy for selecting stocks?
