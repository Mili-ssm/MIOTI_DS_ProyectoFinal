import pandas as pd


def filter_outliers(df: pd.DataFrame, column: list[str], percentile: float = 0.95) -> pd.DataFrame:
    min_percentil = (1 - percentile) / 2
    max_percentil = percentile + min_percentil

    final_df = df

    for col in column:
        min_value = df[col].quantile(min_percentil)
        max_value = df[col].quantile(max_percentil)
        final_df = final_df[(final_df[col] > min_value) & (final_df[col] < max_value)]

    return final_df