import pandas as pd

def load_transaction_data(filepath: str = "data/fraudTrain.csv", sample_size: int = None) -> pd.DataFrame:
    """
    Load and preprocess the transaction data.

    Args:
        filepath (str): Path to the CSV file.
        sample_size (int, optional): Number of rows to sample (for faster training).

    Returns:
        pd.DataFrame: Loaded and cleaned DataFrame.
    """
    df = pd.read_csv(filepath)

    # Optional sampling (for dev/testing)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Convert datetime column
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Standardize column names
    df.columns = df.columns.str.lower()

    return df
