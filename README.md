# VisiAttend

AI-based face recognition attendance system using OpenCV and DeepFace.

## Features

- Enroll new students from webcam snapshots
- Detect and recognize faces live from camera feed
- Mark attendance to a date-wise CSV file

## Project Structure

```text
VisiAttend/
	backend/
		attendance_scanner.py
		enroll_student.py
		test_cam.py
		requirements.txt
		dataset/
	frontend/
```

## Prerequisites

- Python 3.10 or 3.11 (recommended)
- Webcam
- Git

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd VisiAttend
```

## 2. Create and Activate Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Windows (CMD)

```bat
python -m venv venv
venv\Scripts\activate.bat
```

## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

## 4. Enroll Students

Run enrollment script and enter student name when prompted:

```bash
python backend/enroll_student.py
```

This creates images in `backend/dataset/<student_name>/`.

## 5. Test Camera (Optional)

```bash
python backend/test_cam.py
```

## 6. Start Attendance Scanner

```bash
python backend/attendance_scanner.py
```

- Press `q` to quit.
- Attendance is saved as `backend/attendance_YYYY-MM-DD.csv`.

## GitHub Push Checklist

Before push:

1. Confirm `.gitignore` is present.
2. Do not push `venv/`.
3. Do not push `backend/dataset/` face images.
4. Do not push `backend/attendance_*.csv` logs.

Then push:

```bash
git add .
git commit -m "Initial VisiAttend project setup"
git push origin main
```

## Notes

- On first run, some model files may download automatically.
- If your webcam is not detected, close other camera apps and retry.
