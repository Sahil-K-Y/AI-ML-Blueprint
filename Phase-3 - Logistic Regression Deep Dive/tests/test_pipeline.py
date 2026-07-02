import os
import pytest
import pandas as pd
import numpy as np
from src.pipeline import load_data, check_missing_values, handle_missing_values, encode_categorical, run_pipeline

# 1. Fixture for raw mock data containing missing values
@pytest.fixture
def mock_raw_data():
    data = {
        "Gender": ["Male", "Female", None, "Male"],
        "Married": ["Yes", "No", "Yes", None],
        "Dependents": ["0", "1", "2", None],
        "Self_Employed": ["No", "Yes", None, "No"],
        "LoanAmount": [100.0, None, 150.0, 200.0],
        "Loan_Amount_Term": [360.0, 360.0, None, 180.0],
        "Credit_History": [1.0, 0.0, 1.0, None],
        "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate"],
        "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
        "Income": [5000, 4000, 6000, 8000]
    }
    return pd.DataFrame(data)

# 2. Test for missing value detection logic
def test_check_missing_values(mock_raw_data):
    missing_series = check_missing_values(mock_raw_data)
    assert "Gender" in missing_series
    assert "Married" in missing_series
    assert "Dependents" in missing_series
    assert "Self_Employed" in missing_series
    assert "LoanAmount" in missing_series
    assert "Loan_Amount_Term" in missing_series
    assert "Credit_History" in missing_series
    assert missing_series["Gender"] == 1

# 3. Test that imputation works and leaves no missing values
def test_handle_missing_values(mock_raw_data):
    df_imputed = handle_missing_values(mock_raw_data.copy())
    
    # Assert no nulls remain in the columns we impute
    assert df_imputed["Gender"].isnull().sum() == 0
    assert df_imputed["Married"].isnull().sum() == 0
    assert df_imputed["Dependents"].isnull().sum() == 0
    assert df_imputed["Self_Employed"].isnull().sum() == 0
    assert df_imputed["LoanAmount"].isnull().sum() == 0
    assert df_imputed["Loan_Amount_Term"].isnull().sum() == 0
    assert df_imputed["Credit_History"].isnull().sum() == 0
    
    # Assert values are imputed using mode/median
    assert df_imputed.loc[2, "Gender"] == "Male"  # Mode
    assert df_imputed.loc[1, "LoanAmount"] == 150.0  # Median of 100, 150, 200 is 150

# 4. Test that categorical encoding behaves correctly
def test_encode_categorical(mock_raw_data):
    df_imputed = handle_missing_values(mock_raw_data.copy())
    df_encoded = encode_categorical(df_imputed)
    
    # Check that original columns are removed and new dummy columns exist
    assert "Gender" not in df_encoded.columns
    # With drop_first=True, 'Gender_Male' should represent gender
    assert "Gender_Male" in df_encoded.columns
    assert df_encoded["Gender_Male"].dtype in [bool, np.uint8, int]

# 5. Test file loading exception
def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
