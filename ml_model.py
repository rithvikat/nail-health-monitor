import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# User-friendly class names mapping (NOT claiming medical diagnosis)
CLASS_MAPPING = {
    0: "Acral Lentiginous Melanoma Screening Finding",
    1: "Blue Finger Screening Finding",
    2: "Clubbing Screening Finding",
    3: "Healthy Nail",
    4: "Onychogryphosis Screening Finding",
    5: "Possible Onychomycosis Screening Finding",
    6: "Nail Pitting Screening Finding",
    7: "Psoriasis Nail Screening Finding"
}

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1280)]

# Global model references (loaded ONCE at application startup)
feature_extractor = None
svm_pipeline = None


def load_ml_models():
    """
    Load MobileNetV2 feature extractor and PCA+SVM pipeline into memory once.
    """
    global feature_extractor, svm_pipeline

    mobilenet_path = os.path.join("models", "finetuned_mobilenetv2.keras")
    svm_path = os.path.join("models", "nail_pca_svm_model.pkl")

    if not os.path.exists(mobilenet_path):
        raise FileNotFoundError(f"MobileNetV2 model not found at {mobilenet_path}")

    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"PCA+SVM pipeline not found at {svm_path}")

    print("Loading MobileNetV2 feature extractor...")
    full_model = load_model(mobilenet_path)
    feature_extractor = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("feature_layer").output
    )
    print("MobileNetV2 feature extractor ready.")

    print("Loading PCA+SVM pipeline...")
    with open(svm_path, "rb") as f:
        svm_pipeline = pickle.load(f)
    print("PCA+SVM classifier ready.")


def predict_nail(image_path: str) -> dict:
    """
    Extract 1280-dimensional features using fine-tuned MobileNetV2,
    then predict class and confidence using PCA+SVM pipeline.

    Returns:
        {
            "predicted_class": int,
            "predicted_label": str,
            "confidence": float,
            "probabilities": list[float]
        }
    """
    global feature_extractor, svm_pipeline

    if feature_extractor is None or svm_pipeline is None:
        load_ml_models()

    # 1. Load and preprocess image exactly as trained
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_batch = np.expand_dims(image_array, axis=0)

    # 2. Extract 1280-dimensional deep features
    features = feature_extractor.predict(image_batch, verbose=0)

    # 3. Format as DataFrame with feature names to match training format
    features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)

    # 4. Predict class and probabilities via PCA + SVM pipeline
    probabilities = svm_pipeline.predict_proba(features_df)[0]

    # Calculate top predictions dynamically from all classes
    class_probs = []
    for idx, prob in enumerate(probabilities):
        cls_id = int(idx)
        cls_label = CLASS_MAPPING.get(cls_id, f"Class {cls_id} Finding")
        class_probs.append({
            "class_id": cls_id,
            "label": cls_label,
            "confidence": float(prob),
            "percentage": round(float(prob) * 100, 1)
        })
    class_probs.sort(key=lambda x: x["confidence"], reverse=True)
    top_2 = class_probs[:2]

    # The TOP-1 prediction matches the primary AI Screening Finding
    top_1 = top_2[0]
    pred_class = top_1["class_id"]
    label = top_1["label"]
    confidence = top_1["confidence"]

    # Debug prediction output showing all 8 class probabilities (Requirement 10)
    raw_class_names = [
        "Acral_Lentiginous_Melanoma",
        "Blue_Finger",
        "Clubbing",
        "Healthy",
        "Onychogryphosis",
        "Onychomycosis",
        "Pitting",
        "Psoriasis"
    ]
    print("\n" + "=" * 45)
    print(f"DEBUG PREDICTION OUTPUT ({os.path.basename(image_path)})")
    print("=" * 45)
    for name, prob in zip(raw_class_names, probabilities):
        print(f"{name}: {prob * 100:.1f}%")
    print()
    print(f"Top prediction: {label}")
    print(f"Top confidence: {confidence * 100:.1f}%")
    print("=" * 45 + "\n")

    return {
        "predicted_class": pred_class,
        "predicted_label": label,
        "confidence": confidence,
        "probabilities": [float(p) for p in probabilities],
        "top_2": top_2
    }

