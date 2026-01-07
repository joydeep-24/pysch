# vision_analyzer.py
from deepface import DeepFace
import cv2
import numpy as np

# Fixed emotion order expected by fusion model
VISION_EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def analyze_facial_expression(image_np):
    """
    Human-readable facial emotion analysis.
    Returns dominant emotion and raw emotion scores.
    """
    try:
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            img_path=rgb_img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        if isinstance(result, list) and len(result) > 0:
            return {
                "dominant_emotion": result[0].get('dominant_emotion', "unknown"),
                "emotion_scores": result[0].get('emotion', {})
            }

    except Exception as e:
        return {"dominant_emotion": "error", "error": str(e)}

    return {"dominant_emotion": "unknown"}


# -------------------------------------------------------------------------
# NEW: Fusion-safe vector output
# -------------------------------------------------------------------------
def analyze_facial_expression_vector(image_np):
    """
    Returns a fixed-size 7D facial emotion probability vector
    in the order expected by the fusion model.

    Output shape: (7,)
    """
    try:
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            img_path=rgb_img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        if isinstance(result, list) and len(result) > 0:
            scores = result[0].get('emotion', {})

            vector = np.zeros(len(VISION_EMOTION_ORDER), dtype=np.float32)

            for i, emotion in enumerate(VISION_EMOTION_ORDER):
                # DeepFace gives percentages (0–100)
                vector[i] = scores.get(emotion, 0.0) / 100.0

            return vector

    except Exception:
        pass

    # If face not detected or error occurs
    return np.zeros(len(VISION_EMOTION_ORDER), dtype=np.float32)
