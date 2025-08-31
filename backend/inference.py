# inference.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Global state
sequence = []
sentence = []
predictions = []
threshold = 0.5

# --- MODEL LOADING ---
model = tf.keras.models.load_model('model.h5')
model.load_weights('model_weights.h5')
print("Model loaded successfully!")

# --- Mediapipe helpers ---
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# --- Main prediction function ---
def run_inference(base64_string: str):
    global sequence, sentence, predictions

    # Strip prefix if present
    if "data:image" in base64_string:
        _, base64_string = base64_string.split(",", 1)

    # Decode base64 to image
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise ValueError("Invalid or empty image data.")

    # Run Mediapipe holistic
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)

    # Append to sequence
    sequence.append(keypoints)
    sequence = sequence[-30:]

    # If enough frames, predict
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))

        if np.unique(predictions[-10:])[0] == np.argmax(res):
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        return {
            "prediction": actions[np.argmax(res)],
            "status": "predicted",
            "sentence": sentence
        }
    else:
        return {
            "prediction": None,
            "status": "collecting_data",
            "sentence": sentence
        }
