import pandas as pd
from src.logger_config import get_logger

logger = get_logger("pipeline")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully | shape={df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        raise


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info("No missing values found.")
    else:
        logger.warning(f"Missing values detected:\n{missing}")

    return missing


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Starting missing value imputation")

    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
    for col in categorical_cols:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            logger.info(f"Filled '{col}' nulls with mode = {mode_value}")

    if "LoanAmount" in df.columns:
        median_value = df["LoanAmount"].median()
        df["LoanAmount"] = df["LoanAmount"].fillna(median_value)
        logger.info(f"Filled 'LoanAmount' nulls with median = {median_value}")

    if "Loan_Amount_Term" in df.columns:
        mode_value = df["Loan_Amount_Term"].mode()[0]
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(mode_value)
        logger.info(f"Filled 'Loan_Amount_Term' nulls with mode = {mode_value}")

    if "Credit_History" in df.columns:
        mode_value = df["Credit_History"].mode()[0]
        df["Credit_History"] = df["Credit_History"].fillna(mode_value)
        logger.info(f"Filled 'Credit_History' nulls with mode = {mode_value}")

    logger.debug("Missing value imputation complete")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    before_shape = df.shape

    cols_to_encode = ["Gender", "Married", "Dependents", "Education",
                       "Self_Employed", "Property_Area"]
    cols_to_encode = [c for c in cols_to_encode if c in df.columns]

    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    logger.info(f"Encoded categorical columns {cols_to_encode} | shape changed {before_shape} -> {df.shape}")
    return df


def run_pipeline(filepath: str) -> pd.DataFrame:
    logger.info("===== Pipeline started =====")
    df = load_data(filepath)
    check_missing_values(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    logger.info("===== Pipeline finished successfully =====")
    return df