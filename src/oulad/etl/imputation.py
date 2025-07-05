import pandas as pd

NULL_IMPUTATIONS = {
    "student_infos": {
        "imd_band": ("fill", "Unknown")
    },
    "student_assessments": {
        "score": ("fill", 0)
    }
}

def apply_null_imputations(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if table_name not in NULL_IMPUTATIONS:
        return df

    strategies = NULL_IMPUTATIONS[table_name]

    for col, (strategy, value) in strategies.items():
        if col not in df.columns:
            continue

        if strategy == "fill":
            df[col] = df[col].fillna(value)
        elif strategy == "fill_mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "fill_median":
            df[col] = df[col].fillna(df[col].median())
        else:
            raise ValueError(f"Estrategia de imputación no reconocida: {strategy} para {col}")

    return df
def apply_format_to_imd_band_field(df: pd.DataFrame) -> pd.DataFrame:
    df["imd_band"] =  df["imd_band"].replace("10-20", "10-20%")
    return df

