# vision_analyzer.py
from deepface import DeepFace

def analyze_facial_expression(image_np):
    """
    Analyzes a single image frame for facial expressions.
    """
    try:
        result = DeepFace.analyze(
            img_path=image_np,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        if isinstance(result, list) and len(result) > 0:
            return {
                "dominant_emotion": result[0]['dominant_emotion'],
                "emotion_scores": result[0]['emotion']
            }
    except Exception:
        pass # Ignore errors if no face is detected or other issues occur
    
    return {"dominant_emotion": "unknown"}