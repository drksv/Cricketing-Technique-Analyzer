from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import tempfile
from cricket_pose_utils import PoseAnalyzer

app = FastAPI()
analyzer = PoseAnalyzer()

@app.get("/")
def home():
    return {"message": "Cricket Analyzer API Running"}

@app.post("/analyze")
async def analyze_video(
    user_video: UploadFile = File(...),
    ideal_url: str = Form(...)
):
    try:
        # save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await user_video.read())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return JSONResponse({"error": "Could not read video"}, status_code=400)

        all_angles = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # process 1 every 5 frames for performance
            if frame_count % 5 == 0:
                result = analyzer.analyze_frame(frame)
                if result["status"] == "ok":
                    all_angles.append(result["elbow_angle"])

        cap.release()

        if not all_angles:
            return JSONResponse({"error": "No pose detected"}, status_code=400)

        avg_angle = sum(all_angles) / len(all_angles)

        return {
            "ideal_reference_used": ideal_url,
            "average_elbow_angle": round(avg_angle, 2),
            "total_frames_analyzed": len(all_angles),
            "status": "success"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
