import joblib
import numpy as np


def load_model(model_path, scaler_path="saved_models/scaler.pkl"):
    """Load ML model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, input_array):
    """
    Predict class or probability for a given input array.
    input_array → single sample or batch (list or np array)
    """
    arr = np.array(input_array).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = model.predict_proba(arr_scaled)[0][1]   # Probability of class 1
    pred = int(prob >= 0.5)
    return pred, prob


if __name__ == "__main__":
    model_path = "saved_models/stacking_classifier.pkl"

    sample_input = [0.34, -1.2, 3.4, ...]   # ADD ALL FEATURE VALUES HERE

    model, scaler = load_model(model_path)
    pred, prob = predict(model, scaler, sample_input)

    print("Prediction:", pred)
    print("Probability:", prob)
