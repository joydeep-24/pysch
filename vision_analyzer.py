# vision_analyzer.py
from deepface import DeepFace

def analyze_facial_expression(image_np):
    """
    Analyze a single frame for facial expressions using DeepFace.
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
        elif isinstance(result, dict):
            return {
                "dominant_emotion": result['dominant_emotion'],
                "emotion_scores": result['emotion']
            }
    except Exception as e:
        print(f"⚠️ DeepFace error: {e}")

    return {"dominant_emotion": "unknown", "emotion_scores": {}}
