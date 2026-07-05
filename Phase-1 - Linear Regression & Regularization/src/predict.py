import joblib
import numpy as np
import sys
import os

def predict(features):
    # Locate model.joblib relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "artifacts", "model.joblib")
    model = joblib.load(model_path)
    pred = model.predict(np.array([features]))
    return pred[0]

if __name__ == "__main__":
    if len(sys.argv) == 9:
        features = [float(x) for x in sys.argv[1:]]
        price = predict(features)
        print(f"Predicted Price: ${price * 100000:,.2f}")
    else:
        # Default test prediction (approx. California median value)
        test_features = [3.0, 20.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]
        price = predict(test_features)
        print(f"Running default prediction test...")
        print(f"Features: {test_features}")
        print(f"Predicted Price: ${price * 100000:,.2f}")
