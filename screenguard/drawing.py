from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .constants import NOSE_DEADZONE, NOSE_MAX_JUMP, NOSE_MISSING_FRAME_GRACE, NOSE_SMOOTHING_ALPHA
from .media import write_image
from .models import DetectedFace, GuardState
from .utils import box_area, clamp


# Desenho controlado pelo nariz.

# Procura a posicao do nariz do intruso principal.
def intruder_nose_position(
    faces: list[DetectedFace],
    webcam_shape: tuple[int, ...],
) -> tuple[float, float] | None:
    intruder_faces = [
        face
        for face in faces
        if face.status in {"unauthorized", "unknown"} and face.nose_tip is not None
    ]
    if not intruder_faces:
        return None

    face = max(intruder_faces, key=lambda detected_face: box_area(detected_face.box))
    frame_height, frame_width = webcam_shape[:2]
    if frame_width <= 0 or frame_height <= 0 or face.nose_tip is None:
        return None

    nose_x, nose_y = face.nose_tip
    normalized_x = 1.0 - clamp(nose_x / float(frame_width), 0.0, 1.0)
    normalized_y = clamp(nose_y / float(frame_height), 0.0, 1.0)
    return normalized_x, normalized_y


# Calcula a distancia entre dois pontos.
def distance_between_points(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return float((dx * dx + dy * dy) ** 0.5)


# Suaviza a posicao do nariz para reduzir tremores.
def stabilize_nose_position(
    state: GuardState,
    raw_position: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if raw_position is None:
        state.nose_missing_frames += 1
        if (
            state.smoothed_nose_position is not None
            and state.nose_missing_frames <= NOSE_MISSING_FRAME_GRACE
        ):
            return state.smoothed_nose_position
        state.smoothed_nose_position = None
        state.last_nose_draw_position = None
        return None

    raw_position = (
        clamp(raw_position[0], 0.0, 1.0),
        clamp(raw_position[1], 0.0, 1.0),
    )
    previous_position = state.smoothed_nose_position
    state.nose_missing_frames = 0

    if previous_position is None:
        state.smoothed_nose_position = raw_position
        return raw_position

    jump_distance = distance_between_points(previous_position, raw_position)
    if jump_distance > NOSE_MAX_JUMP:
        state.last_nose_draw_position = None
        state.smoothed_nose_position = raw_position
        return raw_position

    if jump_distance < NOSE_DEADZONE:
        return previous_position

    alpha = NOSE_SMOOTHING_ALPHA
    smoothed_position = (
        previous_position[0] + (raw_position[0] - previous_position[0]) * alpha,
        previous_position[1] + (raw_position[1] - previous_position[1]) * alpha,
    )
    state.smoothed_nose_position = smoothed_position
    return smoothed_position


# Atualiza a tela onde o intruso desenha com o nariz.
def update_nose_drawing(
    state: GuardState,
    nose_position: tuple[float, float] | None,
    size: tuple[int, int],
) -> None:
    width, height = size
    if width <= 0 or height <= 0:
        return

    if state.nose_drawing_canvas is None or state.nose_drawing_canvas.shape[:2] != (height, width):
        state.nose_drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        state.last_nose_draw_position = None

    if nose_position is None:
        state.last_nose_draw_position = None
        return

    x = int(clamp(round(nose_position[0] * (width - 1)), 0, width - 1))
    y = int(clamp(round(nose_position[1] * (height - 1)), 0, height - 1))
    current_position = (x, y)

    if state.last_nose_draw_position is not None:
        cv2.line(
            state.nose_drawing_canvas,
            state.last_nose_draw_position,
            current_position,
            (80, 245, 255),
            max(4, int(round(min(width, height) * 0.008))),
            cv2.LINE_AA,
        )
    cv2.circle(state.nose_drawing_canvas, current_position, 5, (255, 255, 255), -1, cv2.LINE_AA)
    state.last_nose_draw_position = current_position


# Mistura o desenho por cima do frame mostrado.
def overlay_nose_drawing(frame: np.ndarray, drawing_canvas: np.ndarray | None) -> None:
    if drawing_canvas is None:
        return
    if drawing_canvas.shape[:2] != frame.shape[:2]:
        return

    mask = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(mask) == 0:
        return
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = (mask.astype(np.float32) / 255.0 * 0.92)[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - alpha) + drawing_canvas.astype(np.float32) * alpha
    np.copyto(frame, blended.astype(np.uint8))


# Limpa o desenho atual.
def clear_nose_drawing(state: GuardState) -> None:
    if state.nose_drawing_canvas is not None:
        state.nose_drawing_canvas.fill(0)
    state.last_nose_draw_position = None


# Guarda o desenho numa imagem PNG transparente.
def save_nose_drawing(state: GuardState, drawings_dir: Path) -> Path | None:
    if state.nose_drawing_canvas is None:
        return None

    mask = cv2.cvtColor(state.nose_drawing_canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(mask) == 0:
        return None

    resolved_dir = drawings_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = resolved_dir / f"desenho_intruso_{timestamp}.png"

    transparent_drawing = cv2.cvtColor(state.nose_drawing_canvas, cv2.COLOR_BGR2BGRA)
    transparent_drawing[:, :, 3] = mask
    if not write_image(output_path, transparent_drawing):
        raise RuntimeError(f"Nao consegui gravar o desenho em: {output_path}")
    return output_path


