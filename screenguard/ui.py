from __future__ import annotations

import cv2
import numpy as np

from .drawing import overlay_nose_drawing
from .face import profile_labels, status_color
from .models import DetectedFace, GuardState
from .utils import clamp, format_label_list, unique_labels


# Desenho de texto e paineis.

# Escreve texto com contorno para ficar legivel.
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


# Calcula o espaco que um texto ocupa.
def text_size(text: str, scale: float, thickness: int = 2) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    return width, height + baseline


# Reduz o texto ate caber na largura disponivel.
def fitted_text_scale(
    text: str,
    max_width: int,
    preferred_scale: float,
    min_scale: float = 0.42,
    thickness: int = 2,
) -> float:
    if max_width <= 0:
        return min_scale

    scale = preferred_scale
    while scale > min_scale and text_size(text, scale, thickness)[0] > max_width:
        scale -= 0.04
    return max(min_scale, scale)


# Divide uma frase em varias linhas.
def wrap_text_lines(
    text: str,
    max_width: int,
    scale: float,
    thickness: int = 2,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_size(candidate, scale, thickness)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


# Desenha texto em varias linhas.
def put_wrapped_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    line_gap: int = 8,
) -> int:
    x, y = origin
    line_height = text_size("Ag", scale, thickness)[1] + line_gap
    for line in wrap_text_lines(text, max_width, scale, thickness):
        put_text(frame, line, (x, y), scale=scale, color=color, thickness=thickness)
        y += line_height
    return y


# Desenha um painel escuro por cima do frame.
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


# Desenha uma etiqueta pequena junto a um rosto.
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


# Interface mostrada no ecra.

# Cria a pre-visualizacao da webcam com caixas nos rostos.
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
        if face.nose_tip is not None:
            cv2.circle(
                preview,
                (int(round(face.nose_tip[0])), int(round(face.nose_tip[1]))),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )
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


# Coloca a webcam pequena por cima do video.
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


# Calcula o progresso da calibracao inicial.
def calibration_progress(state: GuardState, calibration_seconds: float, now: float) -> float:
    elapsed = now - state.calibration_started_at
    if calibration_seconds <= 0:
        return 1.0
    return clamp(elapsed / calibration_seconds, 0.0, 1.0)


# Monta o frame normal durante a reproducao do video.
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
    panel_width = max(300, min(430, frame.shape[1] - 40))
    panel_height = 166
    draw_panel(frame, (20, 20), (panel_width, panel_height))

    if not state.authorized_profiles:
        progress = calibration_progress(state, calibration_seconds, now)
        put_text(frame, "A calibrar...", (36, 52), scale=0.56, color=(255, 220, 120))
        put_text(frame, f"Pessoas: {len(faces)}", (36, 80), scale=0.46)
        put_text(frame, f"Perfis: {len(state.calibration_profiles)}", (36, 105), scale=0.46)
        if progress < 1.0:
            put_text(frame, f"{calibration_seconds:.1f}s", (36, 130), scale=0.44)
            bar_x = 82
            bar_y = 119
            bar_width = panel_width - 80
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), (80, 80, 80), 1)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + int(bar_width * progress), bar_y + 12),
                (70, 190, 255),
                -1,
            )
        else:
            put_text(
                frame,
                "A aguardar rosto estavel.",
                (36, 130),
                scale=0.44,
                color=(0, 165, 255),
            )
    else:
        current_labels = unique_labels([face.label for face in faces if face.status != "pending"])
        authorized_text = format_label_list(profile_labels(state.authorized_profiles), "nenhuma", max_chars=26)
        current_text = format_label_list(current_labels, "ninguem", max_chars=28)

        put_text(frame, "Vigilancia ativa", (36, 52), scale=0.56, color=(110, 235, 140))
        put_text(frame, f"Aut.: {authorized_text}", (36, 80), scale=0.44)
        put_text(frame, f"Agora: {current_text}", (36, 105), scale=0.44)
        put_text(frame, f"Intruso: {state.intruder_streak}/{trigger_frames}", (36, 130), scale=0.44)
        if state.last_intruder_labels:
            put_text(
                frame,
                f"Alerta: {format_label_list(state.last_intruder_labels, 'desconhecido', max_chars=22)}",
                (36, 155),
                scale=0.42,
                color=(0, 165, 255),
            )

    put_text(frame, "Q sair | R recalibrar", (20, frame.shape[0] - 20), scale=0.65)
    overlay_webcam_preview(frame, webcam_preview)
    return frame


# Cria o fundo usado quando nao ha imagem personalizada.
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


# Monta o ecra de alerta quando aparece um intruso.
def build_hostile_frame(
    size: tuple[int, int],
    intruder_labels: list[str],
    authorized_labels: list[str],
    trigger_reason: str,
    webcam_preview: np.ndarray,
    custom_image: np.ndarray | None,
    nose_drawing_canvas: np.ndarray | None,
) -> np.ndarray:
    width, height = size
    if custom_image is not None:
        frame = cv2.resize(custom_image, (width, height))
    else:
        frame = generate_hostile_background(width, height)

    overlay_nose_drawing(frame, nose_drawing_canvas)

    margin = max(24, int(round(min(width, height) * 0.045)))
    panel_x = margin
    panel_y = margin
    panel_width = max(420, min(width - (margin * 2), int(round(width * 0.76))))
    panel_height = max(330, min(height - (margin * 2), 430))
    content_x = panel_x + 28
    content_y = panel_y + 64
    content_width = max(220, panel_width - 56)

    draw_panel(frame, (panel_x, panel_y), (panel_width, panel_height))

    title = "PRIVACIDADE BLOQUEADA"
    title_scale = fitted_text_scale(title, content_width, preferred_scale=0.9, min_scale=0.52, thickness=3)
    put_text(
        frame,
        title,
        (content_x, content_y),
        scale=title_scale,
        color=(40, 70, 255),
        thickness=3,
    )
    content_y += text_size(title, title_scale, 3)[1] + 24

    content_y = put_wrapped_text(
        frame,
        "Foi detetada uma pessoa nao autorizada junto ao ecra.",
        (content_x, content_y),
        content_width,
        scale=0.54,
    )
    content_y += 8
    content_y = put_wrapped_text(
        frame,
        f"Detetada: {format_label_list(intruder_labels, 'desconhecida')}",
        (content_x, content_y),
        content_width,
        scale=0.58,
        color=(120, 220, 255),
    )
    content_y += 6
    if authorized_labels:
        content_y = put_wrapped_text(
            frame,
            f"Autorizadas: {format_label_list(authorized_labels, 'ninguem')}",
            (content_x, content_y),
            content_width,
            scale=0.54,
        )
        content_y += 6
    if trigger_reason:
        content_y = put_wrapped_text(
            frame,
            trigger_reason,
            (content_x, content_y),
            content_width,
            scale=0.5,
            color=(255, 220, 120),
        )
        content_y += 8
    content_y = put_wrapped_text(
        frame,
        "Afasta-te e pressiona R para recalibrar.",
        (content_x, content_y),
        content_width,
        scale=0.54,
        color=(120, 220, 255),
    )
    put_wrapped_text(
        frame,
        "Move o nariz para desenhar | S gravar | C limpar | Q sair",
        (content_x, content_y + 8),
        content_width,
        scale=0.48,
    )
    overlay_webcam_preview(frame, webcam_preview)
    return frame


