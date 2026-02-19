import pandas as pd


# Maps must match the encoding you used in notebooks
BSMT_EXPOSURE_MAP = {"No": 0, "Mn": 1, "Av": 2, "Gd": 3}
BSMT_FIN_MAP = {"Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5}
GARAGE_FINISH_MAP = {"Unf": 0, "RFn": 1, "Fin": 2}
KITCHEN_QUAL_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def _safe_map(series: pd.Series, mapping: dict, default_value: int) -> pd.Series:
    """
    Map categorical strings to numbers.
    Unknown values -> default_value.
    """
    mapped = series.map(mapping)
    return mapped.fillna(default_value)


def build_feature_frame(
    user_values: dict,
    feature_columns: pd.Index,
    train_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a 1-row dataframe with EXACT feature columns used during training.

    Strategy:
    - Start with median defaults (more realistic than zeros)
    - Overwrite only the fields the user provided
    - Ensure no NaNs
    """
    defaults = train_features_df.median(numeric_only=True)
    row = {}

    for col in feature_columns:
        row[col] = float(defaults.get(col, 0))

    for key, value in user_values.items():
        if key in row:
            row[key] = value

    df = pd.DataFrame([row], columns=feature_columns)
    df = df.fillna(0)
    return df


def preprocess_inherited(
    df: pd.DataFrame,
    feature_columns: pd.Index,
    train_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare inherited_houses.csv to match training features:
    - Encode text columns
    - Fill missing numeric values using training medians
    - Reindex to exact feature_columns order
    """
    df = df.copy()

    # Encode if present and is text
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = _safe_map(df["BsmtExposure"], BSMT_EXPOSURE_MAP, 0)

    if "BsmtFinType1" in df.columns:
        df["BsmtFinType1"] = _safe_map(df["BsmtFinType1"], BSMT_FIN_MAP, 0)

    if "GarageFinish" in df.columns:
        df["GarageFinish"] = _safe_map(df["GarageFinish"], GARAGE_FINISH_MAP, 0)

    if "KitchenQual" in df.columns:
        df["KitchenQual"] = _safe_map(df["KitchenQual"], KITCHEN_QUAL_MAP, 3)

    # Fill missing values using training medians
    medians = train_features_df.median(numeric_only=True)
    for col in df.columns:
        if col in medians.index:
            df[col] = df[col].fillna(medians[col])
        else:
            df[col] = df[col].fillna(0)

    # Ensure exact columns and order
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df
