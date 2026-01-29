
import cv2
import os
import csv
import numpy as np
from deepface import DeepFace
from datetime import datetime

# ------------------------------------
# FORCE RTSP TCP (VERY IMPORTANT)
# ------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ------------------------------------
# FACE DETECTOR
# ------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------------
# EMBEDDING FUNCTION
# ------------------------------------
def get_embedding(face_img):
    emb = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        enforce_detection=False
    )
    vec = np.array(emb[0]["embedding"])
    return vec / np.linalg.norm(vec)

# ------------------------------------
# LOAD STUDENT MASTER
# ------------------------------------
def load_student_master(file="student_master.csv"):
    students = {}
    with open(file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            students[row["student_id"]] = row
    print(f"✅ Loaded {len(students)} students from master file")
    return students

student_master = load_student_master()

# ------------------------------------
# LOAD FACE DATABASE
# ------------------------------------
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

    print(f"✅ Loaded {len(db)} face profiles")
    return db

students_db = load_students()

# ------------------------------------
# ATTENDANCE FILE
# ------------------------------------
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "student_id", "name",
            "class", "section", "roll_no",
            "time", "status"
        ])

# ------------------------------------
# MARK ATTENDANCE
# ------------------------------------
def mark_attendance(student):
    today = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["student_id"] == student["student_id"] and row["date"] == today:
                return False

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            today,
            student["student_id"],
            student["name"],
            student["class"],
            student["section"],
            student["roll_no"],
            datetime.now().strftime("%H:%M:%S"),
            "PRESENT"
        ])
    return True

# ------------------------------------
# CAMERA START (CP PLUS CCTV)
# ------------------------------------
RTSP_URL = (
    "rtsp://admin:Yashkhatri123@192.168.29.253:554/"
    "cam/realmonitor?channel=1&subtype=1"
)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("🎥 Attendance System Started (press 'q' to quit)")

marked_today = set()

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠ Frame skipped")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emb = get_embedding(face)

        best_id = None
        min_dist = float("inf")

        for sid, db_emb in students_db.items():
            dist = np.linalg.norm(db_emb - emb)
            if dist < min_dist:
                min_dist = dist
                best_id = sid

        if min_dist < 1.2 and best_id in student_master:
            student = student_master[best_id]

            if best_id not in marked_today:
                if mark_attendance(student):
                    marked_today.add(best_id)
                    print(f"✅ Attendance marked: {student['name']}")

            label = f"{student['name']} ✓"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("CP Plus Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Attendance system stopped")


# import cv2
# import os
# import csv
# import numpy as np
# from deepface import DeepFace
# from datetime import datetime

# # FORCE RTSP TCP
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# # ---------------- FACE DETECTOR ----------------
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# def get_embedding(face_img):
#     emb = DeepFace.represent(
#         img_path=face_img,
#         model_name="ArcFace",
#         enforce_detection=False
#     )
#     vec = np.array(emb[0]["embedding"])
#     return vec / np.linalg.norm(vec)

# # ---------------- LOAD STUDENTS ----------------
# def load_student_master(file="student_master.csv"):
#     students = {}
#     with open(file, newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             students[row["student_id"]] = row
#     return students

# student_master = load_student_master()

# def load_students(student_dir="students"):
#     db = {}
#     for sid in os.listdir(student_dir):
#         path = os.path.join(student_dir, sid)
#         if not os.path.isdir(path):
#             continue
#         embs = []
#         for img_name in os.listdir(path):
#             img = cv2.imread(os.path.join(path, img_name))
#             if img is None:
#                 continue
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#             if len(faces) == 0:
#                 continue
#             x,y,w,h = faces[0]
#             embs.append(get_embedding(img[y:y+h, x:x+w]))
#         if embs:
#             avg = np.mean(embs, axis=0)
#             db[sid] = avg / np.linalg.norm(avg)
#     return db

# students_db = load_students()

# # ---------------- ATTENDANCE ----------------
# ATTENDANCE_FILE = "attendance.csv"

# def mark_attendance(student):
#     today = datetime.now().strftime("%Y-%m-%d")
#     with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             if row["student_id"] == student["student_id"] and row["date"] == today:
#                 return False
#     with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
#         csv.writer(f).writerow([
#             today,
#             student["student_id"],
#             student["name"],
#             student["class"],
#             student["section"],
#             student["roll_no"],
#             datetime.now().strftime("%H:%M:%S"),
#             "PRESENT"
#         ])
#     return True

# # ---------------- CAMERAS ----------------
# IP_WEBCAM_URL = "http://192.168.29.73:8080/video"
# RTSP_URL = "rtsp://admin:Yashkhatri123@192.168.29.253:554/cam/realmonitor?channel=1&subtype=1"

# cap_ip = cv2.VideoCapture(IP_WEBCAM_URL)
# cap_cctv = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# marked_today = set()

# def process_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         face = frame[y:y+h, x:x+w]
#         emb = get_embedding(face)

#         best_id, min_dist = None, float("inf")
#         for sid, db_emb in students_db.items():
#             d = np.linalg.norm(db_emb - emb)
#             if d < min_dist:
#                 min_dist, best_id = d, sid

#         if min_dist < 1.2 and best_id in student_master:
#             student = student_master[best_id]
#             if best_id not in marked_today:
#                 mark_attendance(student)
#                 marked_today.add(best_id)
#             label, color = student["name"], (0,255,0)
#         else:
#             label, color = "UNKNOWN", (0,0,255)

#         cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
#         cv2.putText(frame,label,(x,y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
#     return frame

# # ---------------- MAIN LOOP ----------------
# while True:
#     r1, f1 = cap_ip.read()
#     r2, f2 = cap_cctv.read()

#     if r1:
#         cv2.imshow("IP Webcam", process_frame(f1))
#     if r2:
#         cv2.imshow("CCTV Camera", process_frame(f2))

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap_ip.release()
# cap_cctv.release()
# cv2.destroyAllWindows()
