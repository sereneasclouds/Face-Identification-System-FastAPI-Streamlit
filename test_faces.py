from PIL import Image
import numpy as np
import face_recognition
import os

KNOWN_FACES_DIR = "known_faces"

for img_file in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, img_file)
    print(f"\nTesting: {img_file}")
    
    pil_img = Image.open(img_path).convert("RGB")
    img = np.array(pil_img, dtype=np.uint8)
    
    print(f"  Shape: {img.shape}, Dtype: {img.dtype}")
    
    locs = face_recognition.face_locations(img, number_of_times_to_upsample=2)
    print(f"  Faces found: {len(locs)}")
    
    if locs:
        encs = face_recognition.face_encodings(img, locs)
        print(f"   Success — {len(encs)} encoding(s) generated")
    else:
        print(f"   No face detected — use a clearer frontal photo")