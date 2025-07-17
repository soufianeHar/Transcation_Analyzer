def is_suspicious(row):
    """
    Rule-based fraud detection on a single transaction row.
    Returns True if suspicious, else False.
    """
    # Rule 1: Large amount
    if row["amt"] > 3000:
        return True

    # Rule 2: Risky category
    if row["category"] in ["shopping_net", "misc_net", "grocery_pos"]:
        return True

    # Rule 3: Late night transactions with large amount
    if 2 <= row["hour"] <= 5 and row["amt"] > 1000:
        return True

    return False


def flag_suspicious_transactions(df):
    """
    Apply rule-based detection to the entire DataFrame.
    Returns a copy with a new column 'is_suspicious'.
    """
    df_copy = df.copy()
    df_copy["is_suspicious"] = df_copy.apply(is_suspicious, axis=1)
    return df_copy
