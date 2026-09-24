from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from PIL import Image
import io
import os
import pickle
from pathlib import Path

app = FastAPI(title="Face Identification API")

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_CACHE = "encodings_cache.pkl"
TOLERANCE = 0.5

known_encodings = []
known_names = []


def load_known_faces():
    global known_encodings, known_names

    if os.path.exists(ENCODINGS_CACHE):
        with open(ENCODINGS_CACHE, "rb") as f:
            data = pickle.load(f)
            known_encodings = data["encodings"]
            known_names = data["names"]
        print(f"Loaded {len(known_names)} faces from cache.")
        return

    known_encodings = []
    known_names = []

    Path(KNOWN_FACES_DIR).mkdir(exist_ok=True)

    for img_file in os.listdir(KNOWN_FACES_DIR):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = os.path.splitext(img_file)[0]
        img_path = os.path.join(KNOWN_FACES_DIR, img_file)

        # FIX: convert to RGB to handle PNG/RGBA/CMYK
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)

        encodings = face_recognition.face_encodings(img)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"Loaded: {name}")
        else:
            print(f"WARNING: No face found in {img_file}, skipping.")

    with open(ENCODINGS_CACHE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"Total known faces loaded: {len(known_names)}")


@app.on_event("startup")
def startup_event():
    load_known_faces()


@app.post("/reload-faces")
def reload_faces():
    if os.path.exists(ENCODINGS_CACHE):
        os.remove(ENCODINGS_CACHE)
    load_known_faces()
    return {"message": f"Reloaded {len(known_names)} faces.", "names": known_names}


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()

    # FIX: convert uploaded image to RGB too
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(pil_img)

    face_locations = face_recognition.face_locations(img, model="hog")
    if not face_locations:
        return JSONResponse(content={
            "match_found": False,
            "message": "No face detected in the uploaded image.",
            "results": []
        })

    face_encs = face_recognition.face_encodings(img, face_locations)

    results = []
    for i, enc in enumerate(face_encs):
        result = {"face_index": i + 1, "match_found": False, "name": "Unknown", "confidence": None}

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = round((1 - best_dist) * 100, 2)

            if best_dist <= TOLERANCE:
                result["match_found"] = True
                result["name"] = known_names[best_idx]
                result["confidence"] = confidence
                result["distance"] = round(best_dist, 4)
            else:
                result["name"] = "Unknown"
                result["confidence"] = confidence
                result["distance"] = round(best_dist, 4)
        else:
            result["message"] = "No reference faces loaded. Add images to known_faces/ and call /reload-faces."

        results.append(result)

    match_found = any(r["match_found"] for r in results)
    return JSONResponse(content={
        "faces_detected": len(face_locations),
        "match_found": match_found,
        "results": results
    })


@app.get("/known-faces")
def list_known_faces():
    return {"count": len(known_names), "names": known_names}


@app.get("/health")
def health():
    return {"status": "ok", "known_faces": len(known_names)}