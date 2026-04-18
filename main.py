from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
WINDOW_NAME = "ScreenGuard Prototype"


@dataclass
class GuardState:
    calibration_started_at: float
    calibration_counts: list[int] = field(default_factory=list)
    baseline_faces: int | None = None
    alert_active: bool = False
    extra_face_streak: int = 0
    trigger_reason: str = ""


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
            "Reproduz um video e vigia a webcam. Se aparecerem mais rostos do que "
            "o baseline calculado nos primeiros segundos, o video e substituido "
            "por um ecra hostil."
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
        help="Segundos iniciais usados para medir o numero base de rostos.",
    )
    parser.add_argument(
        "--trigger-frames",
        type=int,
        default=8,
        help="Numero de frames consecutivos com rostos extra antes de disparar o alerta.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.65,
        help="Confianca minima para aceitar uma deteccao do MediaPipe.",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.01,
        help="Area minima relativa do rosto para ser contado.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.08,
        help="Margem relativa ignorada nas bordas para reduzir falsos positivos.",
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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def detect_faces(
    detector: mp.solutions.face_detection.FaceDetection,
    webcam_frame: np.ndarray,
    min_face_area_ratio: float,
    edge_margin: float,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    rgb_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)
    boxes: list[tuple[int, int, int, int]] = []

    if not results.detections:
        return 0, boxes

    frame_height, frame_width = webcam_frame.shape[:2]
    for detection in results.detections:
        relative_box = detection.location_data.relative_bounding_box
        x = clamp(relative_box.xmin, 0.0, 1.0)
        y = clamp(relative_box.ymin, 0.0, 1.0)
        w = clamp(relative_box.width, 0.0, 1.0)
        h = clamp(relative_box.height, 0.0, 1.0)
        area_ratio = w * h
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        if area_ratio < min_face_area_ratio:
            continue
        if not (edge_margin <= center_x <= 1.0 - edge_margin):
            continue
        if not (edge_margin <= center_y <= 1.0 - edge_margin):
            continue

        pixel_box = (
            int(x * frame_width),
            int(y * frame_height),
            int(w * frame_width),
            int(h * frame_height),
        )
        boxes.append(pixel_box)

    return len(boxes), boxes


def annotate_webcam_preview(
    webcam_frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    face_count: int,
) -> np.ndarray:
    preview = webcam_frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (90, 220, 90), 2)
    draw_panel(preview, (10, 10), (165, 46))
    put_text(preview, f"Rostos: {face_count}", (22, 40), scale=0.6)
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
    face_count: int,
    state: GuardState,
    calibration_seconds: float,
    trigger_frames: int,
    now: float,
) -> np.ndarray:
    frame = video_frame.copy()
    panel_width = max(260, min(470, frame.shape[1] - 40))
    panel_height = 205
    draw_panel(frame, (20, 20), (panel_width, panel_height))

    if state.baseline_faces is None:
        progress = calibration_progress(state, calibration_seconds, now)
        put_text(frame, "A calibrar vigilancia...", (40, 60), scale=0.9, color=(255, 220, 120))
        put_text(frame, f"Rostos atuais: {face_count}", (40, 95), scale=0.7)
        put_text(
            frame,
            f"Tempo alvo: {calibration_seconds:.1f}s",
            (40, 126),
            scale=0.7,
        )
        bar_x = 40
        bar_y = 145
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
        put_text(frame, "Vigilancia ativa", (40, 60), scale=0.9, color=(110, 235, 140))
        put_text(frame, f"Baseline: {state.baseline_faces} rosto(s)", (40, 95), scale=0.7)
        put_text(frame, f"Rostos atuais: {face_count}", (40, 126), scale=0.7)
        put_text(
            frame,
            f"Disparo apos {trigger_frames} frames seguidos com excesso",
            (40, 157),
            scale=0.6,
        )
        if face_count > state.baseline_faces:
            put_text(
                frame,
                "Rosto extra em observacao...",
                (40, 188),
                scale=0.7,
                color=(0, 165, 255),
            )

    put_text(frame, "Q sair | R reiniciar", (20, frame.shape[0] - 20), scale=0.65)
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
    face_count: int,
    baseline_faces: int | None,
    webcam_preview: np.ndarray,
    custom_image: np.ndarray | None,
) -> np.ndarray:
    width, height = size
    if custom_image is not None:
        frame = cv2.resize(custom_image, (width, height))
    else:
        frame = generate_hostile_background(width, height)

    draw_panel(frame, (40, 40), (min(680, width - 80), min(250, height - 80)))
    put_text(frame, "PRIVACIDADE BLOQUEADA", (70, 110), scale=1.25, color=(40, 70, 255), thickness=3)
    put_text(frame, "Foi detetado um rosto extra junto ao ecra.", (70, 160), scale=0.82)
    if baseline_faces is not None:
        put_text(frame, f"Baseline: {baseline_faces} | Agora: {face_count}", (70, 200), scale=0.8)
    put_text(frame, "Afasta-te e pressiona R para recalibrar.", (70, 240), scale=0.8, color=(120, 220, 255))
    put_text(frame, "Q sair", (70, 280), scale=0.75)
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

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=args.min_detection_confidence,
    )
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

            face_count, face_boxes = detect_faces(
                detector=detector,
                webcam_frame=webcam_frame,
                min_face_area_ratio=args.min_face_area_ratio,
                edge_margin=args.edge_margin,
            )
            webcam_preview = annotate_webcam_preview(webcam_frame, face_boxes, face_count)
            now = time.monotonic()
            alert_just_triggered = False

            if state.baseline_faces is None:
                state.calibration_counts.append(face_count)
                if now - state.calibration_started_at >= args.calibration_seconds:
                    if state.calibration_counts:
                        state.baseline_faces = int(
                            statistics.median_high(state.calibration_counts)
                        )
                    else:
                        state.baseline_faces = face_count
                    state.extra_face_streak = 0
            elif not state.alert_active:
                if face_count > state.baseline_faces:
                    state.extra_face_streak += 1
                    if state.extra_face_streak >= args.trigger_frames:
                        state.alert_active = True
                        alert_just_triggered = True
                        state.trigger_reason = (
                            f"Detetados {face_count} rostos para baseline {state.baseline_faces}."
                        )
                else:
                    state.extra_face_streak = 0

            if alert_just_triggered:
                audio_player.pause()

            if state.alert_active:
                frame_to_show = build_hostile_frame(
                    size=(display_width, display_height),
                    face_count=face_count,
                    baseline_faces=state.baseline_faces,
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
                    face_count=face_count,
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
        detector.close()
        video_capture.release()
        camera_capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
