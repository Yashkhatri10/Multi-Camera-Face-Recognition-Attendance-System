# Multi-Camera-Face-Recognition-Attendance-System
This project is a production-level face recognition attendance system designed for schools and colleges. It supports multiple cameras (CCTV + IP Webcam) and automatically marks student attendance using real-time face recognition.
🚀 Features

✅ Real-time face detection and recognition

✅ Supports RTSP CCTV cameras and IP Webcam apps

✅ Multi-camera & multi-classroom support

✅ ArcFace embeddings for high accuracy

✅ Automatic attendance logging (CSV)

✅ Prevents duplicate attendance per day

✅ Scalable architecture

🛠 Tech Stack

Language: Python 3.10

Libraries: OpenCV, DeepFace, NumPy

Models: ArcFace

Camera Protocols: RTSP (TCP), HTTP

Storage: CSV files

📷 Supported Cameras

✔ CCTV Cameras (RTSP)

✔ IP Webcam Mobile App (HTTP)

⚙ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YashKhatri10/attendance-system.git
cd attendance-system

2️⃣ Create Virtual Environment (Python 3.10)
python -m venv myenv
myenv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🎓 Adding a New Student

Create a folder inside students/ with student ID

Add 3–5 clear face images

Add student details to student_master.csv

▶ Running the System
python main.py


Press q to stop the system.

👨‍💻 Author

Yash Khatri

AI / ML Developer | Computer Vision | Python

⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub
