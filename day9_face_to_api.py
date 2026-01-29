import cv2
import os
import time
import requests
import numpy as np
from deepface import DeepFace

API_URL = "http://127.0.0.1:5000/api/attendance/mark"

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def get_embedding(face_img):
    emb = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        enforce_detection=False
    )
    vec = np.array(emb[0]["embedding"])
    return vec / np.linalg.norm(vec)

# -----------------------------
# LOAD STUDENT FACE DATABASE
# -----------------------------
def load_students(student_dir="students"):
    db = {}

    for sid in os.listdir(student_dir):
        path = os.path.join(student_dir, sid)
        if not os.path.isdir(path):
            continue

        embeddings = []
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
            embeddings.append(get_embedding(face))

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            db[sid] = avg / np.linalg.norm(avg)

    print(f"✅ Loaded {len(db)} students")
    return db

students_db = load_students()

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
marked = set()

print("🎥 Day 9 Face → API Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emb = get_embedding(face)

        best_id = None
        min_dist = 999

        for sid, db_emb in students_db.items():
            dist = np.linalg.norm(db_emb - emb)
            if dist < min_dist:
                min_dist = dist
                best_id = sid

        if min_dist < 1.2 and best_id not in marked:
            # CALL ERP API
            response = requests.post(API_URL, json={
                "student_id": best_id
            })

            print(f"📡 API Response: {response.json()}")
            marked.add(best_id)

        label = best_id if min_dist < 1.2 else "UNKNOWN"
        color = (0,255,0) if min_dist < 1.2 else (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Day 9 - Face → API", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
