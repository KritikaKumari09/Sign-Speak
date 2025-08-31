# main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from collections import Counter
import json
import tempfile

# ----------------------------
# Setup
# ----------------------------
app = FastAPI()

# Pretrained labels (same as in notebook)
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Load Mediapipe Holistic
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model("model.h5")
model.load_weights("model_weights.h5")


# ----------------------------
# API Endpoints
# ----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a video file, processes frames with Mediapipe,
    runs LSTM model, and returns predictions.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    sequence, predictions, results_array = [], [], []
    threshold = 0.5

    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prediction_data = {
                    "probabilities": res.tolist(),
                    "predicted_label": actions[np.argmax(res)].item()
                }
                results_array.append(prediction_data)
                predictions.append(np.argmax(res))

    cap.release()

    # Majority vote
    all_labels = [p["predicted_label"] for p in results_array]
    unique_outputs = list(set(all_labels))
    final_output = Counter(all_labels).most_common(1)[0][0] if all_labels else "No Prediction"

    # Build response
    response = {
        "all_predictions": results_array,
        "unique_outputs": unique_outputs,
        "final_output": final_output
    }

    return response


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
