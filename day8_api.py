from flask import Flask, jsonify, request
from flask_cors import CORS
import csv
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

STUDENT_MASTER = "student_master.csv"
ATTENDANCE_FILE = "attendance.csv"

# --------------------------------
# ROOT ROUTE (FIXES 404)
# --------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "Smart Attendance API is running",
        "endpoints": [
            "/api/students",
            "/api/attendance",
            "/api/attendance/mark"
        ]
    })

# --------------------------------
# Load Student Master
# --------------------------------
def load_students():
    students = {}
    if not os.path.exists(STUDENT_MASTER):
        return students

    with open(STUDENT_MASTER, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            students[row["student_id"]] = row
    return students

students = load_students()

# --------------------------------
# API 1: Get All Students
# --------------------------------
@app.route("/api/students", methods=["GET"])
def get_students():
    return jsonify(list(students.values()))

# --------------------------------
# API 2: Get Attendance Records
# --------------------------------
@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    records = []
    if not os.path.exists(ATTENDANCE_FILE):
        return jsonify(records)

    with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return jsonify(records)

# --------------------------------
# API 3: Mark Attendance (ERP CALL)
# --------------------------------
@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    data = request.json
    if not data or "student_id" not in data:
        return jsonify({"error": "student_id required"}), 400

    student_id = data["student_id"]

    if student_id not in students:
        return jsonify({"error": "Student not found"}), 404

    today = datetime.now().strftime("%Y-%m-%d")

    # Prevent duplicate attendance
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["student_id"] == student_id and row["date"] == today:
                    return jsonify({"message": "Already marked today"})

    student = students[student_id]

    file_exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "date", "student_id", "name",
                "class", "section", "roll_no",
                "time", "status"
            ])

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

    return jsonify({"message": "Attendance marked successfully"})

# --------------------------------
# START SERVER
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
