import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# SAFE EMBEDDING FUNCTION
# -----------------------------
def get_embedding(face_img):
    try:
        emb = DeepFace.represent(
            img_path=face_img,
            model_name="ArcFace",
            enforce_detection=False
        )
        vec = np.array(emb[0]["embedding"])
        return vec / np.linalg.norm(vec)
    except Exception:
        return None

# -----------------------------
# LOAD STUDENTS DATABASE
# -----------------------------
def load_students(student_dir="students"):
    students_db = {}

    for student_id in os.listdir(student_dir):
        path = os.path.join(student_dir, student_id)
        if not os.path.isdir(path):
            continue

        embeddings = []

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]

            emb = get_embedding(face)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            students_db[student_id] = avg / np.linalg.norm(avg)

    if not students_db:
        raise RuntimeError("❌ No students loaded")

    print(f"✅ Loaded {len(students_db)} students")
    return students_db

students_db = load_students()

# -----------------------------
# TRACKING & ATTENDANCE
# -----------------------------
tracked_faces = {}
attendance_log = {}

TRACK_TIMEOUT = 3
track_id_counter = 0

# -----------------------------
# CAMERA START
# -----------------------------
cap = cv2.VideoCapture(0)
print("🎥 Attendance system started (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        h_img, w_img = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_img - x), min(h, h_img - y)

        face = frame[y:y+h, x:x+w]
        emb = get_embedding(face)
        if emb is None:
            continue

        best_match = None
        min_dist = float("inf")

        for sid, db_emb in students_db.items():
            dist = np.linalg.norm(db_emb - emb)
            if dist < min_dist:
                min_dist = dist
                best_match = sid

        if min_dist >= 1.2:
            continue

        # Do not re-mark attendance
        if best_match in attendance_log:
            label = f"{best_match} ✔"
        else:
            label = best_match

        matched_track = None
        for tid, data in tracked_faces.items():
            if data["student_id"] == best_match:
                matched_track = tid
                break

        if matched_track is None:
            track_id_counter += 1
            tracked_faces[track_id_counter] = {
                "student_id": best_match,
                "last_seen": current_time,
                "marked": False
            }
            matched_track = track_id_counter

        tracked_faces[matched_track]["last_seen"] = current_time

        if (
            not tracked_faces[matched_track]["marked"]
            and best_match not in attendance_log
        ):
            attendance_log[best_match] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tracked_faces[matched_track]["marked"] = True
            print(f"✅ Attendance marked: {best_match}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    tracked_faces = {
        tid: data
        for tid, data in tracked_faces.items()
        if current_time - data["last_seen"] < TRACK_TIMEOUT
    }

    cv2.imshow("Day 5 - Attendance Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n📋 FINAL ATTENDANCE LOG:")
for sid, ts in attendance_log.items():
    print(f"{sid} -> {ts}")
