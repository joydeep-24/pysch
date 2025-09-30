import cv2
from deepface import DeepFace
import sys

def test_deepface(image_path=None):
    try:
        if image_path:
            print(f"🔹 Loading image from: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                print("❌ Failed to load image. Check the path.")
                return
        else:
            print("🔹 Capturing frame from webcam...")
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            cap.release()
            if not ret:
                print("❌ Failed to capture from webcam. Camera not accessible.")
                return

        print("🔹 Running DeepFace...")
        analysis = DeepFace.analyze(
            img_path=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=False
        )

        print("\n✅ DeepFace Result:")
        print(analysis)

    except Exception as e:
        print("\n❌ DeepFace Error:", str(e))


if __name__ == "__main__":
    # Usage: python test_vision.py [optional_image_path]
    if len(sys.argv) > 1:
        test_deepface(sys.argv[1])
    else:
        test_deepface()
