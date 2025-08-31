from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import os
from dotenv import load_dotenv
from inference import run_inference

load_dotenv()

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]

print(allowed_origins)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function that processes one frame
def process_frame(frame_json: dict) -> str:
    try:
        frame_data = frame_json["frame_data"]
        image_bytes = base64.b64decode(frame_data)
        print(f"Received frame with {len(image_bytes)} bytes")
        # Your ML / processing logic here
        return "Processed single frame"
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


# ✅ Modified API: accepts array of base64 strings instead of UploadFile
@app.post("/process-frames/")
async def process_frames_endpoint(frames: List[str]):
    results = []
    print(frames)
    for frame_str in frames:
        frame_json = {"frame_data": frame_str}  # wrap into JSON
        result = run_inference(frame_json)
        print(result)
        results.append(result)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
