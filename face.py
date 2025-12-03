import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Emotion → Persona Emoji Mapping
personas = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "fear": "😨",
    "neutral": "🙂",
    "disgust": "🤢"
}

def put_emoji(frame, emoji, x, y, size=80):
    # Draw emoji text on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    cv2.putText(frame, emoji, (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(min_detection_confidence=0.6) as face_detector:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe detection
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        emotion_text = "Detecting..."

        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)

                # Face Bounding Box
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                w_box = int(box.width * w)
                h_box = int(box.height * h)

                # Crop Face
                face_crop = frame[y:y+h_box, x:x+w_box]

                if face_crop.size != 0:
                    try:
                        # DeepFace Emotion Detection
                        analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                        emotion = analysis['dominant_emotion']
                        emotion_text = emotion.upper()

                        # Choose persona emoji
                        emoji = personas.get(emotion, "🙂")

                        # Overlay persona above face
                        put_emoji(frame, emoji, x, y - 20)

                    except:
                        emotion_text = "Error"

        # Show text on screen
        cv2.putText(frame, f"Emotion: {emotion_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow("AI Face Emotion Persona Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
