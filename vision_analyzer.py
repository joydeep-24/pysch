# vision_analyzer.py
from deepface import DeepFace
import cv2

def analyze_facial_expression(image_np):
    """
    Analyzes a single image frame for facial expressions.
    Ensures correct color space and handles failures gracefully.
    """
    try:
        # Convert BGR (OpenCV default) to RGB for DeepFace
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
        # Log the error for debugging
        return {"dominant_emotion": "error", "error": str(e)}

    return {"dominant_emotion": "unknown"}
