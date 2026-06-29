# Day 27: Mini Regression App Planning

## 1. Project Specifications
The goal is to turn the trained California housing regression model into an interactive local web service using Streamlit.

### User Interface Design
- **Sidebar**:
  - Model specs (Lasso Regression, alpha=0.001)
  - Train set metrics (R² score)
- **Main Panel**:
  - Grid of input elements:
    - Median Income (numerical input)
    - House Age (numerical input)
    - Average Rooms (slider)
    - Average Bedrooms (slider)
    - Population (slider)
    - Average Occupancy (slider)
    - Location Coordinate inputs (Latitude and Longitude sliders/number boxes)
  - **Predict Button**: Triggers inference pipeline.
  - **Metric card**: Displays prediction value formatted in USD.

### Backend Pipeline Flow
1. User provides inputs.
2. Inputs are aggregated into a single NumPy array of shape (1, 8).
3. The pre-trained `model.joblib` pipeline scales the inputs using the standard scaler fit parameters, and executes the Lasso regression model.
4. Prediction value is returned and output format conversions are applied ($val \times 100k$ USD).
