import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np
import pandas as pd
import vectorbt as vbt

from island import Island, StrategyResult, sample_strategies_for_prompt
from llm import generate_new_strategy_code, generate_new_strategy_code_openai
from utils import setup_logger
from vars import LOG_FILE

logger = setup_logger("Logger1", LOG_FILE)


def identify_weaknesses(metrics):
    """
    Check performance metrics and return descriptive weaknesses
    as a string (or empty string if none).
    """
    weaknesses = []
    if metrics["sharpe_ratio"] < 1:
        weaknesses.append("Low risk-adjusted returns (Sharpe < 1).")
    if metrics["max_drawdown"] > 20:
        weaknesses.append("High maximum drawdown.")
    if metrics["win_rate"] < 50:
        weaknesses.append("Low win rate.")
    if metrics["total_trades"] < 10:
        weaknesses.append("Insufficient number of trades.")
    if metrics["returns_pct"] < 0:
        weaknesses.append("Negative returns.")
    return "\n".join(weaknesses)


def compile_strategy_code(strategy_code):
    """
    Compile the user-generated code once, so we don't exec() repeatedly.
    Must define a function named 'generate_orders'.
    """
    namespace = {}
    try:
        exec(strategy_code, namespace)
    except Exception as e:
        raise ValueError(f"Error executing strategy code: {str(e)}")

    if "generate_orders" not in namespace:
        raise ValueError(
            "Strategy code must define a function named 'generate_orders'."
        )

    return namespace["generate_orders"]


def compute_composite_score(perf_metrics):
    sharpe_ratio = perf_metrics.get("sharpe_ratio", 0.0)
    max_dd = perf_metrics.get("max_drawdown", 0.0)
    cagr = perf_metrics.get("cagr", 0.0)

    # Handle invalid inputs
    if any([np.isnan(x) for x in [sharpe_ratio, max_dd, cagr]]):
        return float("-inf")

    # Sharpe ratio: bounded [-2, 2] and scaled to [-1, 1]
    sharpe_factor = np.clip(sharpe_ratio, -2, 2) / 2

    # Max drawdown: sigmoid transition centered at 40% DD
    # Result ranges from ~0 (bad) to 1 (good)
    max_dd_factor = 1 / (1 + np.exp((max_dd - 40) / 10))

    # CAGR: Clip at 50% and scale to [0, 1]
    cagr_factor = np.clip(cagr, 0, 50) / 50

    # Equal weighting of all factors
    score = (sharpe_factor + max_dd_factor + cagr_factor) / 3

    return score


def evaluate_strategy(generate_orders_func, data_dict, symbol_list=None):
    """
    Evaluate strategy across multiple symbols in one shot.
    Returns (perf_metrics, trade_log, weaknesses).
    """
    if symbol_list is None:
        symbol_list = list(data_dict.keys())

    # 1) Combine data
    big_data = pd.concat(
        {
            sym: data_dict[sym][["open", "high", "low", "close", "volume"]]
            for sym in symbol_list
        },
        axis=1,
    )

    # 2) Generate positions
    desired_positions = []
    close_frames = []

    # You could collect optional inspection info here if the strategy returns it
    combined_inspection_data = {}

    for sym in symbol_list:
        sym_data = big_data[sym]
        position_output = generate_orders_func(sym_data)

        if isinstance(position_output, tuple):
            position_series, inspection_data = position_output
            combined_inspection_data[sym] = inspection_data
        else:
            position_series = position_output
            combined_inspection_data[sym] = {}

        if not isinstance(position_series, pd.Series):
            raise ValueError(
                f"generate_orders did not return a Series for symbol={sym}. "
                f"Got: {type(position_series)}"
            )

        position_sym = position_series.rename(sym)
        desired_positions.append(position_sym)
        close_frames.append(sym_data["close"].rename(sym))

    desired_positions_df = pd.concat(desired_positions, axis=1).sort_index()
    closes_df = pd.concat(close_frames, axis=1).sort_index()

    desired_positions_df, closes_df = desired_positions_df.align(
        closes_df, join="inner", axis=0
    )

    # 3) Build Portfolio
    portfolio = vbt.Portfolio.from_orders(
        close=closes_df,
        size=desired_positions_df,
        size_type="targetpercent",
        direction="longonly",
        fees=0.001,
        slippage=0.0005,
        init_cash=10000,
        freq="1D",
    )

    # 4) Summarize
    final_values = portfolio.value()
    final_values_at_end = final_values.iloc[-1]
    total_initial = 10000 * len(symbol_list)
    total_final = final_values_at_end.sum()
    net_profit = total_final - total_initial
    returns_pct = (net_profit / total_initial) * 100

    # Gather stats
    all_sharpes = []
    all_dd = []
    all_trades = 0
    all_won = 0
    all_lost = 0
    all_pnl_net = 0.0
    all_profits = 0.0
    all_losses = 0.0
    all_benchmark_returns = []

    # Sortino
    daily_returns = portfolio.returns()
    combined_daily_returns = daily_returns.mean(axis=1)
    negative_returns = combined_daily_returns[combined_daily_returns < 0]
    daily_mean_return = combined_daily_returns.mean()
    daily_down_stdev = negative_returns.std()
    annualized_return = daily_mean_return * 252
    annualized_down_stdev = daily_down_stdev * np.sqrt(252)
    if annualized_down_stdev == 0:
        sortino_ratio = np.inf
    else:
        sortino_ratio = annualized_return / annualized_down_stdev

    for i, sym in enumerate(symbol_list):
        col_stats = portfolio.stats(column=i)
        trades_col = portfolio.trades.select_one(sym)
        trades = trades_col.records_readable

        total_trades_sym = len(trades)
        won_trades = (trades["PnL"] > 0).sum()
        lost_trades = (trades["PnL"] < 0).sum()
        pnl_net = trades["PnL"].sum()

        sr = col_stats.get("Sharpe Ratio", 0.0)
        md = col_stats.get("Max Drawdown [%]", 0.0)
        bench = col_stats.get("Benchmark Return [%]", 0.0)

        pos_pnl = trades.loc[trades["PnL"] > 0, "PnL"].sum()
        neg_pnl = trades.loc[trades["PnL"] < 0, "PnL"].sum()

        all_sharpes.append(sr)
        all_dd.append(md)
        all_trades += total_trades_sym
        all_won += won_trades
        all_lost += lost_trades
        all_pnl_net += pnl_net
        all_profits += pos_pnl
        all_losses += neg_pnl
        all_benchmark_returns.append(bench)

    avg_sharpe = (
        np.mean([x for x in all_sharpes if not np.isnan(x)])
        if len(all_sharpes) > 0
        else 0.0
    )
    max_dd = np.max(all_dd) if len(all_dd) > 0 else 0.0
    win_rate = (all_won / all_trades * 100) if all_trades > 0 else 0.0

    # CAGR
    start_date = final_values.index[0]
    end_date = final_values.index[-1]
    total_days = (end_date - start_date).days
    years = total_days / 365.25 if total_days > 0 else 1.0
    if years > 0:
        cagr = (total_final / total_initial) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Calmar ratio
    calmar_ratio = (cagr / (max_dd / 100.0)) if max_dd != 0 else np.inf

    # Profit Factor
    if all_losses == 0:
        profit_factor = np.inf
    else:
        profit_factor = abs(all_profits / all_losses)

    average_benchmark_returns = np.mean(all_benchmark_returns)

    performance_metrics = {
        "symbol": "AllSymbols",
        "net_profit": net_profit,
        "returns_pct": returns_pct,
        "sharpe_ratio": avg_sharpe,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "cagr": cagr * 100,  # in %
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": all_trades,
        "won_trades": all_won,
        "lost_trades": all_lost,
        "pnl_net": all_pnl_net,
        "profit_factor": profit_factor,
        "benchmark_returns": average_benchmark_returns,
    }

    # Composite Score
    score = compute_composite_score(performance_metrics)
    performance_metrics["composite_score"] = score

    # Summaries as strings
    trade_log = (
        f"Combined {len(symbol_list)} symbols\n"
        f"Total Trades: {all_trades}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Net P&L: ${all_pnl_net:.2f}\n"
        f"Returns: {returns_pct:.2f}%\n"
        f"Avg Sharpe Ratio: {avg_sharpe:.2f}\n"
        f"Sortino Ratio: {sortino_ratio:.2f}\n"
        f"Calmar Ratio: {calmar_ratio:.2f}\n"
        f"CAGR: {cagr * 100:.2f}%\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Avg Benchmark Returns: {average_benchmark_returns:.2f}%\n"
    )

    weaknesses = identify_weaknesses(performance_metrics)
    if weaknesses:
        trade_log += "Weaknesses:\n" + weaknesses

    for symbol, inspection_data in combined_inspection_data.items():
        trade_log += f"  - Symbol: {symbol}:\n"
        for key, value in inspection_data.items():
            # Handle different types of values dynamically
            if isinstance(value, (list, tuple)):
                trade_log += f"      {key}: {', '.join(map(str, value))}\n"
            elif isinstance(value, dict):
                trade_log += f"      {key}:\n"
                for sub_key, sub_value in value.items():
                    trade_log += f"          {sub_key}: {sub_value}\n"
            else:
                trade_log += f"      {key}: {value}\n"

    if len(trade_log) > 50000:
        logger.info("truncating trade log...")
        trade_log = f"{trade_log[:25000]}\n[Output Too Large... Truncating]..."

    return performance_metrics, trade_log, weaknesses


def reconstruct_data_dict_from_multi(multi_df):
    """
    Helper to reconstruct {symbol: df} from a multi-index DataFrame
    """
    data_dict = {}
    for sym in multi_df.columns.levels[0]:
        data_sym = multi_df[sym]
        data_dict[sym] = data_sym
    return data_dict


def evaluate_strategy_in_process(
    strat_code: str,
    train_data,
    window_ix: int,
    strat_ix: int,
    cache_dict: Dict,
):
    """
    Evaluate a single strategy on a single window, returning the metrics.
    """
    strategy_hash = hashlib.sha256(strat_code.encode()).hexdigest()
    cache_key = (strategy_hash, window_ix)
    cached_result = cache_dict.get(cache_key)
    if cached_result is not None:
        return (
            cached_result.perf_metrics,
            cached_result.trade_log,
            cached_result.weaknesses,
            strat_code,
            window_ix,
            strat_ix,
        )

    try:
        generate_orders_func = compile_strategy_code(strat_code)

        # Evaluate on train data
        train_dict = reconstruct_data_dict_from_multi(train_data)
        perf_metrics, trade_log, weaknesses = evaluate_strategy(
            generate_orders_func, train_dict
        )

        if pd.isna(perf_metrics["composite_score"]):
            perf_metrics["composite_score"] = float("-inf")

        result = StrategyResult(perf_metrics, trade_log, weaknesses)
        cache_dict[cache_key] = result

        return perf_metrics, trade_log, weaknesses, strat_code, window_ix, strat_ix

    except Exception as exc:
        logger.info(f"[Process Error] {exc}")
        fallback_perf = {"composite_score": float("-inf")}
        return fallback_perf, "", "", strat_code, window_ix, strat_ix


def evaluate_island(
    island_idx: int,
    island: Island,
    windows: list,
    cache_dict: Dict,
    generation: int,
    k: int,
    max_strategies: int = 9999999,
    n_new_strategies: int = 1,
):
    """
    1) Sample strategies => build LLM prompt => generate new code
    2) Add new candidate strategy
    3) Evaluate all island strategies across all windows
    4) Cull if oversize or broken
    """
    logger.info(f"\n--- Evaluating Island {island_idx} ---")

    # 1) Sample strategies
    sampled_strategies = sample_strategies_for_prompt(island, generation, k_mod=k)

    # 2) Generate new code (multithreaded)
    def generate_code():
        # time.sleep(10)
        return generate_new_strategy_code(sampled_strategies, provider="google")

    new_strategy_codes = []

    with ThreadPoolExecutor(max_workers=n_new_strategies) as executor:
        futures = [executor.submit(generate_code) for _ in range(n_new_strategies)]

        for future in as_completed(futures):
            try:
                result = future.result()
                new_strategy_codes.append(result)
                logger.debug(f"Generated new island: {result}")
            except Exception as e:
                logger.error(f"Error generating island: {e}")

    for new_strategy_code in new_strategy_codes:
        island.strategy_codes.append(
            {
                "current": new_strategy_code,
                "fallback": new_strategy_code,
                "is_broken": False,
                "window_scores": defaultdict(dict),
                "trade_log": "",
            }
        )

    # 3) Evaluate all strategies
    for window_ix, (train_data, test_data) in enumerate(windows):
        for strat_ix, strat_dict in enumerate(island.strategy_codes):
            perf_metrics, trade_log, weaknesses, returned_code, w_ix, s_ix = (
                evaluate_strategy_in_process(
                    strat_dict["current"], train_data, window_ix, strat_ix, cache_dict
                )
            )

            island.strategy_codes[strat_ix]["window_scores"][window_ix] = {
                "score": perf_metrics.get("composite_score", float("-inf")),
                "trade_log": trade_log,
            }

            if perf_metrics["composite_score"] == float("-inf"):
                # Mark broken
                island.strategy_codes[strat_ix]["is_broken"] = True
                # revert to fallback
                island.strategy_codes[strat_ix]["current"] = island.strategy_codes[
                    strat_ix
                ]["fallback"]

            island.strategy_codes[strat_ix]["trade_log"] = trade_log

    # 4) Cull
    island.strategy_codes = [sc for sc in island.strategy_codes if not sc["is_broken"]]
    if len(island.strategy_codes) > max_strategies:
        island.strategy_codes = sorted(
            island.strategy_codes,
            key=lambda sc: np.mean(
                [ws["score"] for ws in sc["window_scores"].values()]
            ),
            reverse=True,
        )[:max_strategies]

    return island, island_idx


def log_top_performers(
    islands, generation, log_interval=5, top_n=5, logger=logging.getLogger()
):
    """
    Logs the performance of the top N strategies across all islands every log_interval generations.
    """
    if generation % log_interval != 0:
        return

    all_strategies = []
    seen = set()

    for island_idx, island in enumerate(islands):
        for strat_idx, strategy in enumerate(island.strategy_codes):
            window_scores = [
                score["score"]
                for score in strategy["window_scores"].values()
                if not np.isneginf(score["score"])
            ]
            if window_scores:
                avg_score = np.mean(window_scores)
                code_hash = hashlib.sha256(strategy["current"].encode()).hexdigest()

                if code_hash in seen:
                    continue

                all_strategies.append(
                    {
                        "island_idx": island_idx,
                        "strategy_idx": strat_idx,
                        "avg_score": avg_score,
                        "code_hash": code_hash,
                        "trade_log": strategy["trade_log"],
                    }
                )
                seen.add(code_hash)

    top_strategies = sorted(all_strategies, key=lambda x: x["avg_score"], reverse=True)[
        :top_n
    ]

    logger.info(f"\n{'='*80}")
    logger.info(f"Top {top_n} Strategies at Generation {generation}")
    logger.info(f"{'='*80}")

    for rank, strat in enumerate(top_strategies, 1):
        logger.info(f"\nRank {rank}:")
        logger.info(f"Island: {strat['island_idx']}")
        logger.info(f"Strategy Hash: {strat['code_hash']}")
        logger.info(f"Average Score: {strat['avg_score']:.4f}")
        logger.info(f"\nPerformance Summary:")
        logger.info("-" * 40)
        # Only print first ~15 lines to keep log manageable
        summary = strat["trade_log"].split("\n")[0:15]
        for line in summary:
            logger.info(line)
        logger.info("-" * 40)

    logger.info("\n" + "_" * 50)


def final_evaluation(strategy_code, train_data, test_data):
    """
    Evaluate final strategy on training & testing sets, returning metrics in memory.
    """
    logger.info("\n=== FINAL EVALUATION ===")
    generate_orders_func = compile_strategy_code(strategy_code)

    train_dict = reconstruct_data_dict_from_multi(train_data)
    perf_train, log_train, _ = evaluate_strategy(generate_orders_func, train_dict)

    test_dict = reconstruct_data_dict_from_multi(test_data)
    perf_test, log_test, _ = evaluate_strategy(generate_orders_func, test_dict)

    return perf_train, perf_test, log_train, log_test


def plot_strategy_performance(strategy_code, data_dict):
    """
    Simple performance chart per symbol using vectorbt.
    """
    generate_orders_func = compile_strategy_code(strategy_code)
    for symbol, df in data_dict.items():
        position_output = generate_orders_func(df)
        if isinstance(position_output, tuple):
            positions_series, _ = position_output
        else:
            positions_series = position_output

        positions_series = positions_series.fillna(0)
        portfolio = vbt.Portfolio.from_orders(
            close=df["close"].fillna(method="ffill").fillna(0),
            size=positions_series,
            size_type="targetpercent",
            direction="longonly",
            fees=0.001,
            slippage=0.0005,
            init_cash=10000.0,
            freq="1D",
        )
        fig = portfolio.plot()
        fig.update_layout(title=f"Performance on {symbol}")
        fig.show()
