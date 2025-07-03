GENERATE_NEW_STRATEGY_PROMPT = """
    Create a generate_orders function that generates orders and and inspection_dict for a trading strategy. i.e. positions_series, inspection_dict)
    The function should take a DataFrame 'data' with columns 'open', 'high', 'low', 'close', 'volume'.
    The function should return a Series of positions (1 for long, 0 for neutral, -1 for short).

    The inspection_dict must essentially read out the story of the strategy, containing metrics, information, insights about how the strategy performed such that when it is provided as feedback, someone would be able to take it and adjust the strategy. We do not want surface level insights, but detail about what we could use to improve the strategy.
    The printed output of this inspection_dict must be under 5000 characters total, keep it concise. 
    Do not place these into the inspection_dict, as they are already calculated:
    Total Trades, Win Rate, Net P&L, Returns, Avg Sharpe Ratio, Sortino Ratio, Calmar Ratio, CAGR, Profit Factor, Max Drawdown, Avg Benchmark Returns.

    Do not place these into the inspection code, as they are already calculated:
    Total Trades, Win Rate, Net P&L, Returns, Avg Sharpe Ratio, Sortino Ratio, Calmar Ratio, CAGR, Profit Factor, Max Drawdown, Avg Benchmark Returns.

    Do not cheat by creating a strategy with any possible look-ahead bias, i.e. training on the input data.

    Example data:
    data.head()
              open  high   low  close  volume
    timestamp
    2020-01-01  100  101   99   100    10000
    ...

    keep the name as generate_orders. place this within ```python and ``` code fences.
    You must not cheat by creating a strategy with any possible look-ahead bias, i.e. training on the input data.
    Start with your thoughts and step by step thinking, before getting to the final python code. 

    Your generate_orders will be used as follows:
        Generate positions
        desired_positions = []
        close_frames = []

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

    """

GENERATE_STRATEGY_PROMPT = """<Task>
Create a new generate_orders function that returns (positions_series, inspection_dict).
The inspection_dict must essentially read out the story of the strategy, containing metrics, information, insights about how the strategy performed such that when it is provided as feedback, someone would be able to take it and adjust the strategy. We do not want surface level insights, but detail about what we could use to improve the strategy.
The printed output of this inspection_dict must be under 5000 characters total, keep it concise. 
Do not place these into the inspection_dict, as they are already calculated:
Total Trades, Win Rate, Net P&L, Returns, Avg Sharpe Ratio, Sortino Ratio, Calmar Ratio, CAGR, Profit Factor, Max Drawdown, Avg Benchmark Returns.

positions_series is a pd.Series of positions  (1=long, 0=flat, -1=short).

Do not cheat by creating a strategy with any possible look-ahead bias, i.e. training on the input data.

Here are two existing generate_orders and their inspection_dicts outputs:
{sampled_strategies_str}
</Task>

<CRITICAL THINKING INSTRUCTIONS>
In this process, I want you to play two roles to simulate critical thinking.
Let's name them Jane and John.
Jane is a critic who analyzes generate_orders and inspection_dicts.
Jane is careful to notice both strengths and shortcomings of the previous generate_orders, and to point out possible ways the generate_orders can be combined to make the new plan better.
Wherever possible, Jane makes concrete suggestions about how to fix the flaws she finds.
After Jane's analysis, John comes up with the actual improved plan.
John thinks carefully about Jane's analysis and looks for opportunities to make a dramatically better plan, while keeping in mind all of the constraints.
John's final answer is contained in triple backticks i.e. ```python and ```, and contains a function named generate_orders.
</CRITICAL THINKING INSTRUCTIONS>
"""
