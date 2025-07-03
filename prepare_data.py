import os
import pickle

import pandas as pd
import yfinance as yf

from utils import setup_logger
from vars import LOG_FILE

logger = setup_logger("Logger1", LOG_FILE)


def prepare_data(symbol):
    """
    Prepare data for 'symbol' from local pickle file (must exist).
    """
    pkl_file = f"yahoo/{symbol}_data.pkl"
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        # Ensure data is sorted by date
        data = data.sort_index()
        return data
    else:
        raise FileNotFoundError(f"No data file found for {symbol}")


def fetch_historical_data(symbol, start_date="2000-01-01", end_date=None):
    """
    Fetches daily historical data for 'symbol' from Yahoo Finance.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data.index.name = "timestamp"
        data = data[["open", "high", "low", "close", "volume"]]
        return data
    except Exception as e:
        raise Exception(f"Failed to fetch data for {symbol}: {e}")


def store_data(symbol, data, engine):
    """
    Store DataFrame 'data' in an SQL table named after 'symbol'.
    """
    data.to_sql(symbol, engine, if_exists="replace")


def get_data(symbol, engine):
    """
    Read data from an SQL table named after 'symbol'.
    """
    query = f"SELECT * FROM '{symbol}'"
    data = pd.read_sql(query, engine, index_col="timestamp", parse_dates=["timestamp"])
    data.sort_index(inplace=True)
    return data


def fetch_and_store_data_in_bulk(symbols):
    """
    Fetch data for each symbol from Yahoo Finance (or read from
    local pickle if present), and store in SQLite DB.
    """
    engine = create_engine("sqlite:///historical_data.db")
    for symbol in symbols:
        pkl_file = f"yahoo/{symbol}_data.pkl"
        if os.path.exists(pkl_file):
            logger.info(f"Loading {symbol} data from pickle file.")
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        else:
            logger.info(f"Fetching {symbol} data from Yahoo Finance.")
            data = fetch_historical_data(symbol)
            if not isinstance(data, pd.DataFrame):
                logger.info(f"Failed to fetch data for {symbol}. Skipping.")
                continue
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)
        store_data(symbol, data, engine)


def get_all_data(symbols):
    """
    Return a dict of DataFrame for each symbol from the local SQL database
    """
    engine = create_engine("sqlite:///historical_data.db")
    data_dict = {}
    for symbol in symbols:
        df = get_data(symbol, engine)
        data_dict[symbol] = df
    return data_dict


def split_data_rolling(
    df,
    train_years=3,
    test_months=6,
    min_symbols=5,
    data_quality_threshold=0.9,
    min_coverage=0.8,
):
    """
    Split data into rolling train-test windows, ensuring data quality.
    Returns list of (train_data, test_data) tuples with consistent column structure.

    Args:
        df (pd.DataFrame): Multi-index DataFrame with symbols and OHLCV data
        train_years (int): Number of years for training window
        test_months (int): Number of months for testing window
        min_symbols (int): Minimum number of symbols required per window
        data_quality_threshold (float): Minimum ratio of non-NA data required (0.0-1.0)
        min_coverage (float): Minimum data coverage required for a symbol across entire dataset (0.0-1.0)

    Returns:
        list: List of (train_data, test_data) tuples with consistent column structure
    """
    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "DataFrame must have MultiIndex columns with (symbol, field) structure"
        )

    # Basic health checks
    logger.info("\nPerforming Health Checks:")
    logger.info("-" * 80)

    # Check column structure
    expected_fields = {"open", "high", "low", "close", "volume"}
    actual_fields = set(df.columns.levels[1])
    if not expected_fields.issubset(actual_fields):
        missing_fields = expected_fields - actual_fields
        raise ValueError(f"Missing required price fields: {missing_fields}")
    logger.info("✓ Column structure validated")

    # Check for negative prices
    price_fields = ["open", "high", "low", "close"]
    has_negative = False
    negative_counts = {}

    for symbol in df.columns.levels[0]:
        for field in price_fields:
            neg_count = (df[symbol][field] < 0).sum()
            if neg_count > 0:
                has_negative = True
                negative_counts[f"{symbol}-{field}"] = neg_count

    if has_negative:
        logger.info("⚠ Warning: Found negative prices:")
        for field, count in negative_counts.items():
            logger.info(f"  - {field}: {count} instances")
    else:
        logger.info("✓ No negative prices found")

    # Check for zero volumes
    zero_volume_counts = {}
    for symbol in df.columns.levels[0]:
        zero_count = (df[symbol]["volume"] == 0).sum()
        if zero_count > 0:
            zero_volume_counts[symbol] = zero_count

    if zero_volume_counts:
        logger.info("ℹ Zero volume days found:")
        for symbol, count in zero_volume_counts.items():
            logger.info(f"  - {symbol}: {count} days ({(count/len(df)*100):.1f}%)")
    else:
        logger.info("✓ No zero volume days found")

    # Check data continuity
    logger.info("\nChecking Data Continuity:")
    logger.info("-" * 80)
    expected_days = pd.date_range(start=df.index[0], end=df.index[-1], freq="B")
    missing_days = set(expected_days) - set(df.index)
    if missing_days:
        logger.info(f"ℹ Found {len(missing_days)} missing business days")
        logger.info(f"  First few missing days: {sorted(list(missing_days))[:3]}")
    else:
        logger.info("✓ No missing business days")

    # Print dataset date range
    logger.info("\nDataset Coverage:")
    logger.info("-" * 80)
    logger.info(f"Dataset starts: {df.index[0].strftime('%Y-%m-%d')}")
    logger.info(f"Dataset ends:   {df.index[-1].strftime('%Y-%m-%d')}")
    logger.info(f"Total trading days: {len(df)}")

    # Calculate and display data availability
    logger.info("\nData Availability by Symbol:")
    logger.info("-" * 80)
    symbols_to_exclude = []
    availability_stats = {}

    for symbol in df.columns.levels[0]:
        data_points = df[symbol]["close"].notna().sum()
        availability = data_points / len(df) * 100
        availability_stats[symbol] = availability

        if availability < min_coverage * 100:
            status = "EXCLUDED"
            symbols_to_exclude.append(symbol)
        else:
            status = "Included"

        logger.info(f"{symbol:6} : {availability:.1f}% ({data_points} days) - {status}")

    # Filter out symbols with poor coverage
    if symbols_to_exclude:
        logger.info(
            f"\nExcluding {len(symbols_to_exclude)} symbols due to insufficient coverage (<{min_coverage*100}%):"
        )
        logger.info(", ".join(symbols_to_exclude))
        included_symbols = [
            sym for sym in df.columns.levels[0] if sym not in symbols_to_exclude
        ]
        df = df[included_symbols]
        df.columns = df.columns.remove_unused_levels()

    # Get final list of symbols
    symbols = df.columns.levels[0]
    logger.info(f"\nProceeding with {len(symbols)} symbols")

    windows = []

    # Convert years/months to business days (approximate)
    train_days = train_years * 252  # ~252 trading days per year
    test_days = test_months * 21  # ~21 trading days per month
    window_size = train_days + test_days

    logger.info("\nWindow Periods:")
    logger.info("-" * 100)
    logger.info(
        f"{'Window':^8} | {'Training Period':^32} | {'Testing Period':^32} | {'Symbols':^10}"
    )
    logger.info("-" * 100)

    # Generate windows
    start_idx = 0
    window_count = 0
    skipped_windows = 0

    while (start_idx + window_size) <= len(df):
        train_end = start_idx + train_days
        test_end = train_end + test_days

        # Extract window data
        train_window = df.iloc[start_idx:train_end].copy()
        test_window = df.iloc[train_end:test_end].copy()

        # Check data quality for each symbol
        valid_symbols = []
        for symbol in symbols:
            train_data_quality = train_window[symbol]["close"].notna().sum() / len(
                train_window
            )
            test_data_quality = test_window[symbol]["close"].notna().sum() / len(
                test_window
            )

            if (
                train_data_quality >= data_quality_threshold
                and test_data_quality >= data_quality_threshold
            ):
                valid_symbols.append(symbol)

        # Only create windows if we have enough valid symbols
        if len(valid_symbols) >= min_symbols:
            window_count += 1

            # Print window periods
            train_start_date = train_window.index[0].strftime("%Y-%m-%d")
            train_end_date = train_window.index[-1].strftime("%Y-%m-%d")
            test_start_date = test_window.index[0].strftime("%Y-%m-%d")
            test_end_date = test_window.index[-1].strftime("%Y-%m-%d")

            logger.info(
                f"{window_count:^8} | {train_start_date} to {train_end_date} | {test_start_date} to {test_end_date} | {len(valid_symbols):^10}"
            )

            # Keep only valid symbols
            valid_train = train_window[valid_symbols]
            valid_test = test_window[valid_symbols]

            # Adjust column structure
            valid_train.columns = pd.MultiIndex.from_tuples(
                [
                    (sym, price_col)
                    for sym in valid_symbols
                    for price_col in ["open", "high", "low", "close", "volume"]
                ]
            )
            valid_test.columns = pd.MultiIndex.from_tuples(
                [
                    (sym, price_col)
                    for sym in valid_symbols
                    for price_col in ["open", "high", "low", "close", "volume"]
                ]
            )

            # Verify we still have data
            if not valid_train.empty and not valid_test.empty:
                windows.append((valid_train, valid_test))
        else:
            skipped_windows += 1

        # Roll forward by test_days
        start_idx += test_days

    logger.info("-" * 100)
    logger.info(f"\nSummary:")
    logger.info(f"- Created {len(windows)} valid windows")
    logger.info(f"- Skipped {skipped_windows} windows due to insufficient data")
    logger.info(
        f"- Each training period: {train_years} years ({train_days} trading days)"
    )
    logger.info(
        f"- Each testing period: {test_months} months ({test_days} trading days)"
    )
    logger.info(f"- Data quality threshold: {data_quality_threshold*100}%")
    logger.info(f"- Minimum symbols required: {min_symbols}")

    if windows:
        logger.info(f"\nFirst Window Statistics:")
        logger.info(
            f"- Window sizes - Train: {windows[0][0].shape}, Test: {windows[0][1].shape}"
        )
        logger.info(
            f"- Number of valid symbols: {len(windows[0][0].columns.levels[0])}"
        )
        logger.info("\nValid symbols in first window:")
        logger.info(", ".join(sorted(windows[0][0].columns.levels[0].tolist())))

    # Final validation
    if not windows:
        raise ValueError("No valid windows could be created with the given criteria")

    return windows


def load_all_data(symbols):
    """
    Load data from local pickles, filtering by date range if desired.
    """
    data_dict = {}
    for symbol in symbols:
        try:
            df = prepare_data(symbol)
            df = df.loc[df.index >= "2016-01-01"]
            df = df.loc[df.index <= "2024-07-06"]
            if df.empty:
                logger.info(f"Skipping {symbol}, no data in date range.")
                continue
            data_dict[symbol] = df
        except Exception as e:
            logger.info(f"Error loading data for {symbol}: {str(e)}")
    return data_dict
