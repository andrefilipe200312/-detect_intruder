from __future__ import annotations

from pathlib import Path

# Tipos de ficheiro aceites pelo programa.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
REFERENCE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Nomes e pastas usados por defeito.
WINDOW_NAME = "ScreenGuard Prototype"
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_DRAWINGS_DIR = Path("intruder_drawings")

# Valores usados para suavizar o desenho com o nariz.
NOSE_SMOOTHING_ALPHA = 0.32
NOSE_DEADZONE = 0.004
NOSE_MAX_JUMP = 0.22
NOSE_MISSING_FRAME_GRACE = 5

# Modelos oficiais usados pelo OpenCV.
MODEL_SPECS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}
