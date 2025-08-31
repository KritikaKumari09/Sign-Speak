import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import deque
import json
import uvicorn
from contextlib import asynccontextmanager
import base64
import time

# --- MODEL INFERENCE FUNCTIONS FROM JUPYTER NOTEBOOK ---

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou'])

def mediapipe_detection(image, model):
    """
    Processes a single video frame for hand and pose landmarks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extracts landmark coordinates and flattens them into a single array.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# --- FASTAPI SETUP ---

# This stores the model in memory once on startup
class AppState:
    model: tf.keras.models.Model = None

app = FastAPI()
app_state = AppState()

# Global variables to store the sequence of frames and the sentence
sequence = []
sentence = []
predictions = []
threshold = 0.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup and clean up on shutdown.
    """
    print("Loading model...")
    app_state.model = tf.keras.models.load_model('model.h5')
    app_state.model.load_weights('model_weights.h5')
    print("Model loaded successfully!")
    yield
    print("Shutting down...")
    app_state.model = None

# Create the FastAPI app with the lifespan event
app = FastAPI(lifespan=lifespan)

# Pydantic model for the incoming data from the React frontend
class VideoFrame(BaseModel):
    frame_data: str

# --- API ENDPOINT ---

@app.post("/predict")
async def predict(video_frame: VideoFrame):
    """
    Receives a video frame, makes a prediction, and returns the result.
    """
    try:
        # Check for and strip the prefix from the Base64 string
        base64_string = video_frame.frame_data
        if "data:image" in base64_string:
            header, base64_string = base64_string.split(",", 1)

        # Decode the base64 string to an image
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or empty image data received.")
        
        # Initialize Mediapipe model for detection
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Process the frame to get keypoints
            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

        # Append new keypoints to the sequence and keep it at max 30
        global sequence, sentence, predictions
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Make prediction only if the sequence is full
        if len(sequence) == 30:
            res = app_state.model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            # Sentence building logic from the notebook
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            
            if len(sentence) > 5:
                sentence = sentence[-5:]
            
            return {"prediction": actions[np.argmax(res)], "status": "predicted", "sentence": sentence}
        else:
            return {"prediction": None, "status": "collecting_data", "sentence": sentence}

    except Exception as e:
        return {"error": str(e)}

# --- RUN THE SERVER ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
