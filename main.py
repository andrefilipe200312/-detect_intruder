from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
REFERENCE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WINDOW_NAME = "ScreenGuard Prototype"
DEFAULT_MODELS_DIR = Path("models")
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


@dataclass
class IdentityProfile:
    label: str
    source: str = "session"
    sample_count: int = 0
    descriptor_sum: np.ndarray | None = None

    def add_descriptor(self, descriptor: np.ndarray) -> None:
        normalized = descriptor.astype(np.float32)
        if self.descriptor_sum is None:
            self.descriptor_sum = normalized.copy()
        else:
            self.descriptor_sum += normalized
        self.sample_count += 1

    def merge_from(self, other: "IdentityProfile") -> None:
        if other.descriptor_sum is None or other.sample_count == 0:
            return
        if self.descriptor_sum is None:
            self.descriptor_sum = other.descriptor_sum.astype(np.float32).copy()
        else:
            self.descriptor_sum += other.descriptor_sum
        self.sample_count += other.sample_count
        if other.source == "reference":
            self.source = "reference"

    def signature(self) -> np.ndarray | None:
        if self.descriptor_sum is None or self.sample_count == 0:
            return None
        centroid = (self.descriptor_sum / float(self.sample_count)).astype(np.float32)
        return normalize_vector(centroid)


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]
    confidence: float
    descriptor: np.ndarray | None
    label: str = "Rosto"
    status: str = "pending"
    similarity: float = 0.0


@dataclass
class GuardState:
    calibration_started_at: float
    calibration_profiles: list[IdentityProfile] = field(default_factory=list)
    authorized_profiles: list[IdentityProfile] = field(default_factory=list)
    alert_active: bool = False
    intruder_streak: int = 0
    trigger_reason: str = ""
    next_temporary_label: int = 1
    last_authorized_labels: list[str] = field(default_factory=list)
    last_intruder_labels: list[str] = field(default_factory=list)


@dataclass
class FaceEmbeddingEngine:
    detector: object
    recognizer: object
    detector_input_size: tuple[int, int] = (320, 320)

    def set_input_size(self, frame_width: int, frame_height: int) -> None:
        input_size = (max(1, frame_width), max(1, frame_height))
        if input_size != self.detector_input_size:
            self.detector.setInputSize(input_size)
            self.detector_input_size = input_size


@dataclass
class AudioPlayer:
    video_path: Path
    pygame: object | None = None
    extracted_audio_path: Path | None = None
    ready: bool = False

    def initialize(self) -> None:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as error:
            print(
                f"Aviso: audio desativado porque falta uma dependencia ({error}).",
                file=sys.stderr,
            )
            return

        audio_path = extract_audio_track(self.video_path, Path(get_ffmpeg_exe()))
        if audio_path is None:
            return

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(audio_path))
        except Exception as error:
            print(f"Aviso: nao consegui iniciar o audio do video ({error}).", file=sys.stderr)
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            safe_unlink(audio_path)
            return

        self.pygame = pygame
        self.extracted_audio_path = audio_path
        self.ready = True

    def play_from_start(self) -> None:
        if not self.ready or self.pygame is None:
            return
        self.pygame.mixer.music.stop()
        self.pygame.mixer.music.play()

    def pause(self) -> None:
        if not self.ready or self.pygame is None:
            return
        if self.pygame.mixer.music.get_busy():
            self.pygame.mixer.music.pause()

    def close(self) -> None:
        if self.pygame is not None:
            try:
                self.pygame.mixer.music.stop()
                unload = getattr(self.pygame.mixer.music, "unload", None)
                if callable(unload):
                    unload()
            finally:
                if self.pygame.mixer.get_init():
                    self.pygame.mixer.quit()
        safe_unlink(self.extracted_audio_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduz um video e vigia a webcam. Em vez de apenas contar rostos, "
            "aprende quem esta autorizado durante a calibracao inicial e bloqueia "
            "o ecra quando aparece outra pessoa."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Caminho para o video. Se omitires, o script tenta usar o primeiro video da pasta atual.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice da webcam a usar no OpenCV.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=2.0,
        help="Segundos iniciais usados para aprender quem esta autorizado.",
    )
    parser.add_argument(
        "--trigger-frames",
        type=int,
        default=8,
        help="Numero de frames consecutivos com uma pessoa nao autorizada antes do alerta.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.65,
        help="Confianca minima para aceitar uma detecao do YuNet.",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.01,
        help="Area minima relativa do rosto para ser considerada.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.08,
        help="Margem relativa ignorada nas bordas para reduzir falsos positivos.",
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=0.363,
        help="Similaridade cosine minima do SFace para considerar que dois rostos sao da mesma pessoa.",
    )
    parser.add_argument(
        "--min-profile-observations",
        type=int,
        default=3,
        help="Numero minimo de observacoes de um rosto para esse perfil ficar autorizado.",
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        help=(
            "Pasta opcional com fotos de referencia organizadas por pessoa "
            "(ex.: referencias\\Ana\\foto1.jpg)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Pasta onde os modelos ONNX do YuNet e SFace sao guardados.",
    )
    parser.add_argument(
        "--hostile-image",
        type=Path,
        help="Imagem opcional para usar como ecra hostil.",
    )
    parser.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Desativa o modo fullscreen.",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Nao reinicia o video automaticamente quando chega ao fim.",
    )
    return parser.parse_args()


def resolve_video_path(video_arg: Path | None) -> Path:
    if video_arg is not None:
        video_path = video_arg.expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video nao encontrado: {video_path}")
        return video_path

    candidates = sorted(
        path
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            "Nao encontrei nenhum video na pasta atual. "
            "Passa o caminho do video na linha de comandos."
        )
    return candidates[0].resolve()


def configure_window(fullscreen: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )


def safe_fps(video_capture: cv2.VideoCapture) -> int:
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return int(round(fps))
    return 30


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def extract_audio_track(video_path: Path, ffmpeg_executable: Path) -> Path | None:
    temp_file = tempfile.NamedTemporaryFile(
        prefix="screenguard_audio_",
        suffix=".wav",
        delete=False,
    )
    audio_path = Path(temp_file.name)
    temp_file.close()

    command = [
        str(ffmpeg_executable),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    stderr_text = result.stderr.lower()
    if (
        "does not contain any stream" in stderr_text
        or "output file #0 does not contain any stream" in stderr_text
        or "stream map" in stderr_text
    ):
        print("Aviso: o video nao tem uma faixa de audio utilizavel.", file=sys.stderr)
    else:
        print("Aviso: falha a extrair o audio do video; continuo apenas com imagem.", file=sys.stderr)
    safe_unlink(audio_path)
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model_file(model_dir: Path, filename: str) -> Path:
    spec = MODEL_SPECS[filename]
    destination = model_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    model_dir.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    safe_unlink(temp_destination)

    print(f"A descarregar modelo facial {filename}...", file=sys.stderr)
    try:
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            with temp_destination.open("wb") as output_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
    except (urllib.error.URLError, OSError) as error:
        safe_unlink(temp_destination)
        raise RuntimeError(
            "Nao consegui descarregar os modelos faciais oficiais. "
            f"Podes tentar novamente ou colocar manualmente o ficheiro em {destination}. "
            f"Erro original: {error}"
        ) from error

    downloaded_hash = file_sha256(temp_destination)
    if downloaded_hash.lower() != spec["sha256"].lower():
        safe_unlink(temp_destination)
        raise RuntimeError(
            f"O modelo {filename} foi descarregado mas a verificacao SHA256 falhou."
        )

    temp_destination.replace(destination)
    return destination


def create_face_embedding_engine(
    models_dir: Path,
    min_detection_confidence: float,
) -> FaceEmbeddingEngine:
    resolved_models_dir = models_dir.expanduser().resolve()
    detector_path = ensure_model_file(resolved_models_dir, "face_detection_yunet_2023mar.onnx")
    recognizer_path = ensure_model_file(resolved_models_dir, "face_recognition_sface_2021dec.onnx")

    if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
        raise RuntimeError(
            "A tua instalacao do OpenCV nao expoe FaceDetectorYN/FaceRecognizerSF. "
            "Instala uma versao recente de opencv-python."
        )

    try:
        detector = cv2.FaceDetectorYN_create(
            str(detector_path),
            "",
            (320, 320),
            min_detection_confidence,
            0.3,
            5000,
        )
        recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")
    except cv2.error as error:
        raise RuntimeError(f"Nao consegui iniciar os modelos faciais ONNX: {error}") from error

    return FaceEmbeddingEngine(detector=detector, recognizer=recognizer)


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame: np.ndarray, top_left: tuple[int, int], size: tuple[int, int]) -> None:
    x, y = top_left
    width, height = size
    frame_height, frame_width = frame.shape[:2]
    width = max(1, min(width, frame_width - x - 1))
    height = max(1, min(height, frame_height - y - 1))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)


def draw_tag(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    frame_height, frame_width = frame.shape[:2]
    x = max(6, min(origin[0], frame_width - text_width - 12))
    y = max(text_height + 8, min(origin[1], frame_height - baseline - 8))
    top_left = (x - 4, y - text_height - 6)
    bottom_right = (x + text_width + 4, y + baseline + 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        text,
        (x, y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def unique_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


def format_label_list(labels: list[str], fallback: str, max_chars: int = 52) -> str:
    text = ", ".join(unique_labels(labels))
    if not text:
        return fallback
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2]) * max(0, box[3])


def profile_labels(profiles: list[IdentityProfile]) -> list[str]:
    return [profile.label for profile in profiles]


def next_temporary_label(state: GuardState) -> str:
    label = f"Pessoa {state.next_temporary_label}"
    state.next_temporary_label += 1
    return label


def status_color(status: str) -> tuple[int, int, int]:
    if status == "authorized":
        return (90, 220, 90)
    if status == "calibrating":
        return (120, 220, 255)
    if status == "unauthorized":
        return (0, 165, 255)
    if status == "unknown":
        return (0, 70, 255)
    return (180, 180, 180)


def extract_face_embedding(
    engine: FaceEmbeddingEngine,
    webcam_frame: np.ndarray,
    detection: np.ndarray,
) -> np.ndarray | None:
    try:
        aligned_face = engine.recognizer.alignCrop(webcam_frame, detection)
        descriptor = engine.recognizer.feature(aligned_face)
    except cv2.error:
        return None

    if descriptor is None:
        return None
    return normalize_vector(np.asarray(descriptor, dtype=np.float32).reshape(-1))


def detect_faces(
    engine: FaceEmbeddingEngine,
    webcam_frame: np.ndarray,
    min_face_area_ratio: float,
    edge_margin: float,
) -> list[DetectedFace]:
    faces: list[DetectedFace] = []
    frame_height, frame_width = webcam_frame.shape[:2]
    engine.set_input_size(frame_width, frame_height)

    try:
        _, detections = engine.detector.detect(webcam_frame)
    except cv2.error:
        return faces

    if detections is None or len(detections) == 0:
        return faces

    frame_area = float(max(frame_width * frame_height, 1))
    for detection in np.asarray(detections):
        x = float(detection[0])
        y = float(detection[1])
        w = float(detection[2])
        h = float(detection[3])
        area_ratio = (max(0.0, w) * max(0.0, h)) / frame_area
        center_x = (x + (w / 2.0)) / float(frame_width)
        center_y = (y + (h / 2.0)) / float(frame_height)

        if area_ratio < min_face_area_ratio:
            continue
        if not (edge_margin <= center_x <= 1.0 - edge_margin):
            continue
        if not (edge_margin <= center_y <= 1.0 - edge_margin):
            continue

        x1 = int(clamp(round(x), 0, max(frame_width - 1, 0)))
        y1 = int(clamp(round(y), 0, max(frame_height - 1, 0)))
        x2 = int(clamp(round(x + w), x1 + 1, frame_width))
        y2 = int(clamp(round(y + h), y1 + 1, frame_height))
        pixel_box = (x1, y1, x2 - x1, y2 - y1)

        descriptor = extract_face_embedding(engine, webcam_frame, detection)
        confidence = float(detection[-1]) if detection.size else 0.0
        faces.append(
            DetectedFace(
                box=pixel_box,
                confidence=confidence,
                descriptor=descriptor,
            )
        )

    faces.sort(key=lambda face: face.box[0])
    return faces


def best_profile_match(
    descriptor: np.ndarray | None,
    profiles: list[IdentityProfile],
    excluded_profile_ids: set[int] | None = None,
) -> tuple[IdentityProfile | None, float]:
    if descriptor is None:
        return None, -1.0

    excluded = excluded_profile_ids or set()
    best_profile: IdentityProfile | None = None
    best_score = -1.0

    for profile in profiles:
        if id(profile) in excluded:
            continue
        signature = profile.signature()
        if signature is None:
            continue
        score = float(np.dot(descriptor, signature))
        if score > best_score:
            best_profile = profile
            best_score = score

    return best_profile, best_score


def learn_calibration_profiles(
    state: GuardState,
    faces: list[DetectedFace],
    reference_profiles: list[IdentityProfile],
    recognition_threshold: float,
) -> None:
    used_profile_ids: set[int] = set()
    visible_labels: list[str] = []

    for face in faces:
        if face.descriptor is None:
            face.label = "Rosto"
            face.status = "pending"
            continue

        matched_profile, matched_score = best_profile_match(
            face.descriptor,
            state.calibration_profiles,
            excluded_profile_ids=used_profile_ids,
        )
        reference_profile, reference_score = best_profile_match(face.descriptor, reference_profiles)

        if matched_profile is None or matched_score < recognition_threshold:
            if reference_profile is not None and reference_score >= recognition_threshold:
                matched_profile = IdentityProfile(label=reference_profile.label, source="reference")
            else:
                matched_profile = IdentityProfile(label=next_temporary_label(state))
            state.calibration_profiles.append(matched_profile)
        elif reference_profile is not None and reference_score >= recognition_threshold:
            matched_profile.label = reference_profile.label
            matched_profile.source = "reference"

        matched_profile.add_descriptor(face.descriptor)
        used_profile_ids.add(id(matched_profile))

        face.label = matched_profile.label
        face.status = "calibrating"
        face.similarity = max(matched_score, reference_score)
        visible_labels.append(face.label)

    state.last_authorized_labels = unique_labels(visible_labels)
    state.last_intruder_labels = []


def finalize_authorized_profiles(
    calibration_profiles: list[IdentityProfile],
    min_profile_observations: int,
) -> list[IdentityProfile]:
    eligible_profiles = [
        profile for profile in calibration_profiles if profile.sample_count >= min_profile_observations
    ]
    if not eligible_profiles:
        eligible_profiles = [profile for profile in calibration_profiles if profile.sample_count > 0]

    merged_by_label: dict[str, IdentityProfile] = {}
    for profile in eligible_profiles:
        if profile.label not in merged_by_label:
            merged_by_label[profile.label] = IdentityProfile(label=profile.label, source=profile.source)
        merged_by_label[profile.label].merge_from(profile)

    return sorted(merged_by_label.values(), key=lambda profile: profile.label.lower())


def classify_faces(
    faces: list[DetectedFace],
    authorized_profiles: list[IdentityProfile],
    reference_profiles: list[IdentityProfile],
    recognition_threshold: float,
) -> tuple[list[str], list[str]]:
    authorized_labels: list[str] = []
    intruder_labels: list[str] = []
    used_authorized_ids: set[int] = set()

    for face in faces:
        if face.descriptor is None:
            face.label = "Desconhecido"
            face.status = "unknown"
            face.similarity = 0.0
            intruder_labels.append(face.label)
            continue

        authorized_profile, authorized_score = best_profile_match(
            face.descriptor,
            authorized_profiles,
            excluded_profile_ids=used_authorized_ids,
        )

        if authorized_profile is not None and authorized_score >= recognition_threshold:
            face.label = authorized_profile.label
            face.status = "authorized"
            face.similarity = authorized_score
            authorized_labels.append(face.label)
            used_authorized_ids.add(id(authorized_profile))
            continue

        reference_profile, reference_score = best_profile_match(face.descriptor, reference_profiles)
        if reference_profile is not None and reference_score >= recognition_threshold:
            face.label = reference_profile.label
            face.status = "unauthorized"
            face.similarity = reference_score
        else:
            face.label = "Desconhecido"
            face.status = "unknown"
            face.similarity = max(authorized_score, reference_score)

        intruder_labels.append(face.label)

    return unique_labels(authorized_labels), unique_labels(intruder_labels)


def annotate_webcam_preview(
    webcam_frame: np.ndarray,
    faces: list[DetectedFace],
    calibrating: bool,
    learned_profiles_count: int,
) -> np.ndarray:
    preview = webcam_frame.copy()
    for face in faces:
        x, y, w, h = face.box
        color = status_color(face.status)
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        draw_tag(preview, face.label, (x, max(24, y - 6)), color)

    draw_panel(preview, (10, 10), (250, 68))
    put_text(preview, f"Pessoas: {len(faces)}", (22, 38), scale=0.58)
    if calibrating:
        put_text(preview, f"Calibracao | perfis: {learned_profiles_count}", (22, 62), scale=0.52, color=(120, 220, 255))
    else:
        intruder_count = sum(face.status in {"unauthorized", "unknown"} for face in faces)
        if intruder_count:
            put_text(preview, f"Intrusas: {intruder_count}", (22, 62), scale=0.52, color=(0, 165, 255))
        else:
            put_text(preview, "Todas autorizadas", (22, 62), scale=0.52, color=(110, 235, 140))
    return preview


def overlay_webcam_preview(base_frame: np.ndarray, preview_frame: np.ndarray) -> None:
    frame_height, frame_width = base_frame.shape[:2]
    target_width = max(120, min(int(frame_width * 0.24), frame_width - 40))
    scale = target_width / preview_frame.shape[1]
    target_height = int(preview_frame.shape[0] * scale)
    resized_preview = cv2.resize(preview_frame, (target_width, target_height))

    padding = 20
    x1 = max(10, frame_width - target_width - padding)
    y1 = padding
    x2 = x1 + target_width
    y2 = y1 + target_height

    if y2 + 30 >= frame_height:
        y1 = max(10, frame_height - target_height - 40)
        y2 = y1 + target_height

    overlay = base_frame.copy()
    cv2.rectangle(overlay, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, base_frame, 0.4, 0, base_frame)
    base_frame[y1:y2, x1:x2] = resized_preview
    cv2.rectangle(base_frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (200, 200, 200), 1)
    put_text(base_frame, "Webcam", (x1, y2 + 28), scale=0.6)


def calibration_progress(state: GuardState, calibration_seconds: float, now: float) -> float:
    elapsed = now - state.calibration_started_at
    if calibration_seconds <= 0:
        return 1.0
    return clamp(elapsed / calibration_seconds, 0.0, 1.0)


def render_playback_frame(
    video_frame: np.ndarray,
    webcam_preview: np.ndarray,
    faces: list[DetectedFace],
    state: GuardState,
    calibration_seconds: float,
    trigger_frames: int,
    now: float,
) -> np.ndarray:
    frame = video_frame.copy()
    panel_width = max(330, min(760, frame.shape[1] - 40))
    panel_height = 228
    draw_panel(frame, (20, 20), (panel_width, panel_height))

    if not state.authorized_profiles:
        progress = calibration_progress(state, calibration_seconds, now)
        put_text(frame, "A calibrar identidades...", (40, 60), scale=0.9, color=(255, 220, 120))
        put_text(frame, f"Pessoas visiveis: {len(faces)}", (40, 95), scale=0.7)
        put_text(frame, f"Perfis aprendidos: {len(state.calibration_profiles)}", (40, 126), scale=0.7)
        if progress < 1.0:
            put_text(frame, f"Tempo alvo: {calibration_seconds:.1f}s", (40, 157), scale=0.7)
            bar_x = 40
            bar_y = 176
            bar_width = panel_width - 80
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 18), (80, 80, 80), 1)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_width * progress), bar_y + 18),
                (70, 190, 255),
                -1,
            )
        else:
            put_text(
                frame,
                "Aguardando um rosto estavel para autorizar.",
                (40, 157),
                scale=0.66,
                color=(0, 165, 255),
            )
    else:
        current_labels = unique_labels([face.label for face in faces if face.status != "pending"])
        put_text(frame, "Vigilancia de identidade ativa", (40, 60), scale=0.9, color=(110, 235, 140))
        put_text(
            frame,
            f"Autorizadas: {format_label_list(profile_labels(state.authorized_profiles), 'nenhuma')}",
            (40, 95),
            scale=0.65,
        )
        put_text(
            frame,
            f"No frame: {format_label_list(current_labels, 'ninguem')}",
            (40, 126),
            scale=0.65,
        )
        put_text(
            frame,
            f"Disparo apos {trigger_frames} frames com intruso",
            (40, 157),
            scale=0.62,
        )
        if state.last_intruder_labels:
            put_text(
                frame,
                f"Intruso em observacao: {format_label_list(state.last_intruder_labels, 'desconhecido')}",
                (40, 188),
                scale=0.62,
                color=(0, 165, 255),
            )

    put_text(frame, "Q sair | R recalibrar", (20, frame.shape[0] - 20), scale=0.65)
    overlay_webcam_preview(frame, webcam_preview)
    return frame


def generate_hostile_background(width: int, height: int) -> np.ndarray:
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = (10, 10, 35)

    gradient = np.linspace(0, 150, width, dtype=np.uint8)
    background[:, :, 2] = np.maximum(background[:, :, 2], gradient)

    for offset in range(-height, width, 45):
        start_point = (offset, 0)
        end_point = (offset + height, height)
        cv2.line(background, start_point, end_point, (0, 0, 0), 18)

    return background


def build_hostile_frame(
    size: tuple[int, int],
    intruder_labels: list[str],
    authorized_labels: list[str],
    trigger_reason: str,
    webcam_preview: np.ndarray,
    custom_image: np.ndarray | None,
) -> np.ndarray:
    width, height = size
    if custom_image is not None:
        frame = cv2.resize(custom_image, (width, height))
    else:
        frame = generate_hostile_background(width, height)

    draw_panel(frame, (40, 40), (min(780, width - 80), min(320, height - 80)))
    put_text(frame, "PRIVACIDADE BLOQUEADA", (70, 110), scale=1.25, color=(40, 70, 255), thickness=3)
    put_text(frame, "Foi detetada uma pessoa nao autorizada junto ao ecra.", (70, 160), scale=0.78)
    put_text(
        frame,
        f"Detetada: {format_label_list(intruder_labels, 'desconhecida')}",
        (70, 200),
        scale=0.76,
        color=(120, 220, 255),
    )
    if authorized_labels:
        put_text(
            frame,
            f"Autorizadas: {format_label_list(authorized_labels, 'ninguem')}",
            (70, 238),
            scale=0.72,
        )
    if trigger_reason:
        put_text(frame, trigger_reason, (70, 274), scale=0.6, color=(255, 220, 120))
    put_text(frame, "Afasta-te e pressiona R para recalibrar.", (70, 308), scale=0.76, color=(120, 220, 255))
    put_text(frame, "Q sair", (70, 342), scale=0.72)
    overlay_webcam_preview(frame, webcam_preview)
    return frame


def load_hostile_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Imagem hostil nao encontrada: {resolved}")
    image = cv2.imread(str(resolved))
    if image is None:
        raise RuntimeError(f"Nao consegui carregar a imagem hostil: {resolved}")
    return image


def load_known_face_profiles(
    known_faces_dir: Path | None,
    engine: FaceEmbeddingEngine,
) -> list[IdentityProfile]:
    if known_faces_dir is None:
        return []

    resolved = known_faces_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Pasta de referencias nao encontrada: {resolved}")
    if not resolved.is_dir():
        raise RuntimeError(f"O caminho de referencias nao e uma pasta: {resolved}")

    profiles: list[IdentityProfile] = []
    for person_dir in sorted(path for path in resolved.iterdir() if path.is_dir()):
        profile = IdentityProfile(label=person_dir.name, source="reference")
        image_paths = sorted(
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in REFERENCE_IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            faces = detect_faces(
                engine=engine,
                webcam_frame=image,
                min_face_area_ratio=0.0,
                edge_margin=0.0,
            )
            best_face = max(faces, key=lambda face: box_area(face.box), default=None)
            if best_face is None or best_face.descriptor is None:
                continue
            profile.add_descriptor(best_face.descriptor)

        if profile.sample_count > 0:
            profiles.append(profile)

    if not profiles:
        raise RuntimeError(
            "Nao consegui criar referencias de identidade. "
            "Confirma se a pasta tem subpastas por pessoa com fotos onde o rosto esteja visivel."
        )

    return profiles


def reset_state() -> GuardState:
    return GuardState(calibration_started_at=time.monotonic())


def main() -> int:
    args = parse_args()

    try:
        video_path = resolve_video_path(args.video)
        hostile_image = load_hostile_image(args.hostile_image)
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        print(f"Nao consegui abrir o video: {video_path}", file=sys.stderr)
        return 1

    camera_capture = cv2.VideoCapture(args.camera_index)
    if not camera_capture.isOpened():
        print(
            f"Nao consegui abrir a webcam no indice {args.camera_index}.",
            file=sys.stderr,
        )
        video_capture.release()
        return 1

    fps = safe_fps(video_capture)
    frame_delay_ms = max(1, int(1000 / fps))
    display_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    display_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    try:
        engine = create_face_embedding_engine(
            models_dir=args.models_dir,
            min_detection_confidence=args.min_detection_confidence,
        )
        reference_profiles = load_known_face_profiles(args.known_faces_dir, engine)
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        video_capture.release()
        camera_capture.release()
        return 1

    state = reset_state()
    audio_player = AudioPlayer(video_path=video_path)

    configure_window(fullscreen=not args.no_fullscreen)
    audio_player.initialize()
    audio_player.play_from_start()

    try:
        while True:
            webcam_ok, webcam_frame = camera_capture.read()
            if not webcam_ok:
                print("Falha a ler um frame da webcam.", file=sys.stderr)
                return 1

            faces = detect_faces(
                engine=engine,
                webcam_frame=webcam_frame,
                min_face_area_ratio=args.min_face_area_ratio,
                edge_margin=args.edge_margin,
            )
            now = time.monotonic()
            alert_just_triggered = False

            if not state.authorized_profiles:
                learn_calibration_profiles(
                    state=state,
                    faces=faces,
                    reference_profiles=reference_profiles,
                    recognition_threshold=args.recognition_threshold,
                )
                if now - state.calibration_started_at >= args.calibration_seconds:
                    authorized_profiles = finalize_authorized_profiles(
                        state.calibration_profiles,
                        min_profile_observations=args.min_profile_observations,
                    )
                    if authorized_profiles:
                        state.authorized_profiles = authorized_profiles
                        authorized_labels, intruder_labels = classify_faces(
                            faces=faces,
                            authorized_profiles=state.authorized_profiles,
                            reference_profiles=reference_profiles,
                            recognition_threshold=args.recognition_threshold,
                        )
                        state.last_authorized_labels = authorized_labels
                        state.last_intruder_labels = intruder_labels
                        state.intruder_streak = 0
            else:
                authorized_labels, intruder_labels = classify_faces(
                    faces=faces,
                    authorized_profiles=state.authorized_profiles,
                    reference_profiles=reference_profiles,
                    recognition_threshold=args.recognition_threshold,
                )
                state.last_authorized_labels = authorized_labels
                state.last_intruder_labels = intruder_labels

                if not state.alert_active:
                    if intruder_labels:
                        state.intruder_streak += 1
                        if state.intruder_streak >= args.trigger_frames:
                            state.alert_active = True
                            alert_just_triggered = True
                            state.trigger_reason = (
                                "Pessoa nao autorizada: "
                                f"{format_label_list(intruder_labels, 'desconhecida', max_chars=34)}"
                            )
                    else:
                        state.intruder_streak = 0
                        state.trigger_reason = ""

            webcam_preview = annotate_webcam_preview(
                webcam_frame=webcam_frame,
                faces=faces,
                calibrating=not state.authorized_profiles,
                learned_profiles_count=len(state.calibration_profiles),
            )

            if alert_just_triggered:
                audio_player.pause()

            if state.alert_active:
                frame_to_show = build_hostile_frame(
                    size=(display_width, display_height),
                    intruder_labels=state.last_intruder_labels,
                    authorized_labels=profile_labels(state.authorized_profiles),
                    trigger_reason=state.trigger_reason,
                    webcam_preview=webcam_preview,
                    custom_image=hostile_image,
                )
            else:
                video_ok, video_frame = video_capture.read()
                if not video_ok:
                    if args.no_loop:
                        break
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    audio_player.play_from_start()
                    continue
                display_height, display_width = video_frame.shape[:2]
                frame_to_show = render_playback_frame(
                    video_frame=video_frame,
                    webcam_preview=webcam_preview,
                    faces=faces,
                    state=state,
                    calibration_seconds=args.calibration_seconds,
                    trigger_frames=args.trigger_frames,
                    now=now,
                )

            cv2.imshow(WINDOW_NAME, frame_to_show)
            key = cv2.waitKey(30 if state.alert_active else frame_delay_ms) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = reset_state()
                audio_player.play_from_start()
    finally:
        audio_player.close()
        video_capture.release()
        camera_capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
