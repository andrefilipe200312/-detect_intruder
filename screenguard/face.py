from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .constants import MODEL_SPECS
from .media import safe_unlink
from .models import DetectedFace, FaceEmbeddingEngine, GuardState, IdentityProfile
from .utils import box_area, clamp, normalize_vector, unique_labels


# Calcula o hash do ficheiro para confirmar se esta correto.
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# Garante que o modelo existe e tem o hash esperado.
def ensure_model_file(model_dir: Path, filename: str) -> Path:
    spec = MODEL_SPECS[filename]
    destination = model_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        existing_hash = file_sha256(destination)
        if existing_hash.lower() == spec["sha256"].lower():
            return destination
        safe_unlink(destination)

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


# Copia o modelo para uma pasta simples se o caminho tiver caracteres dificeis.
def opencv_safe_model_path(path: Path) -> Path:
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        pass

    safe_dir = Path(tempfile.gettempdir()) / "screenguard_models"
    safe_dir.mkdir(parents=True, exist_ok=True)
    safe_path = safe_dir / path.name

    if not safe_path.exists() or safe_path.stat().st_size != path.stat().st_size:
        shutil.copy2(path, safe_path)

    return safe_path


# Cria os objetos do OpenCV que detetam e reconhecem rostos.
def create_face_embedding_engine(
    models_dir: Path,
    min_detection_confidence: float,
) -> FaceEmbeddingEngine:
    resolved_models_dir = models_dir.expanduser().resolve()
    detector_path = opencv_safe_model_path(
        ensure_model_file(resolved_models_dir, "face_detection_yunet_2023mar.onnx")
    )
    recognizer_path = opencv_safe_model_path(
        ensure_model_file(resolved_models_dir, "face_recognition_sface_2021dec.onnx")
    )

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



# Perfis e classificacao dos rostos.

# Devolve apenas os nomes dos perfis.
def profile_labels(profiles: list[IdentityProfile]) -> list[str]:
    return [profile.label for profile in profiles]


# Cria um nome temporario quando nao ha referencia conhecida.
def next_temporary_label(state: GuardState) -> str:
    label = f"Pessoa {state.next_temporary_label}"
    state.next_temporary_label += 1
    return label


# Escolhe a cor usada para desenhar cada tipo de rosto.
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


# Extrai o vetor facial a partir de uma detecao.
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


# Deteta rostos num frame da webcam.
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
        nose_tip = None
        if detection.size >= 10:
            nose_tip = (
                float(clamp(float(detection[8]), 0.0, float(max(frame_width - 1, 0)))),
                float(clamp(float(detection[9]), 0.0, float(max(frame_height - 1, 0)))),
            )

        descriptor = extract_face_embedding(engine, webcam_frame, detection)
        confidence = float(detection[-1]) if detection.size else 0.0
        faces.append(
            DetectedFace(
                box=pixel_box,
                confidence=confidence,
                descriptor=descriptor,
                nose_tip=nose_tip,
            )
        )

    faces.sort(key=lambda face: face.box[0])
    return faces


# Procura o perfil mais parecido com o rosto recebido.
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


# Durante a calibracao, aprende os rostos visiveis.
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


# Escolhe que perfis aprendidos ficam autorizados.
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


# Classifica cada rosto como autorizado ou intruso.
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

