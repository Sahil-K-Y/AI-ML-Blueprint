import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

PIPELINE_PATH = 'titanic_pipeline.joblib'
DATA_PATH = 'train.csv'

@pytest.fixture
def model():
    """Fixture to load the saved model pipeline."""
    assert os.path.exists(PIPELINE_PATH), f'{PIPELINE_PATH} does not exist. Run Day 31 notebook first.'
    return joblib.load(PIPELINE_PATH)

@pytest.fixture
def data():
    """Fixture to load the Titanic dataset."""
    assert os.path.exists(DATA_PATH), f'{DATA_PATH} not found.'
    return pd.read_csv(DATA_PATH)

def test_model_loading(model):
    """Test that the model object is loaded correctly and is an sklearn Pipeline."""
    assert model is not None
    assert isinstance(model, Pipeline)

def test_pipeline_imputation_and_transform(model, data):
    """Test that the preprocessor successfully fills NaN values and scales data."""
    # Take rows with missing ages
    test_df = data[data['Age'].isnull()].head(5)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X_test = test_df[features]
    
    # Run through preprocessing step of pipeline
    X_transformed = model.named_steps['preprocessor'].transform(X_test)
    
    # Ensure there are no NaNs in the output array
    assert not np.isnan(X_transformed).any()
    # Check that output columns are scaled/encoded correctly
    assert X_transformed.shape[1] > len(features)  # Due to one-hot encoding

def test_predictions_shape_and_type(model, data):
    """Test that predictions return correct datatypes and array size."""
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X_test = data[features].head(10)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Assert output shapes
    assert len(predictions) == 10
    assert probabilities.shape == (10, 2)
    # Assert values inside probability are bounds [0, 1]
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
