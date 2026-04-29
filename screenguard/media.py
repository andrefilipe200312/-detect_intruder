from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import WINDOW_NAME


# Controla o audio extraido do video.
@dataclass
class AudioPlayer:
    video_path: Path
    pygame: object | None = None
    extracted_audio_path: Path | None = None
    ready: bool = False

    # Prepara o pygame e carrega a faixa de audio temporaria.
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

    # Reproduz o audio desde o inicio.
    def play_from_start(self) -> None:
        if not self.ready or self.pygame is None:
            return
        self.pygame.mixer.music.stop()
        self.pygame.mixer.music.play()

    # Pausa o audio quando ha alerta.
    def pause(self) -> None:
        if not self.ready or self.pygame is None:
            return
        if self.pygame.mixer.music.get_busy():
            self.pygame.mixer.music.pause()

    # Fecha o mixer e apaga o ficheiro temporario.
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


# Janela, video e compatibilidade.

# Cria a janela principal do OpenCV.
def configure_window(fullscreen: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )


# Tenta descobrir o tamanho do ecra principal no Windows.
def primary_screen_size() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None

    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None
    return width, height


# Le o tamanho original do video.
def video_capture_size(video_capture: cv2.VideoCapture) -> tuple[int, int]:
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    return width, height


# Decide o tamanho final usado para mostrar a imagem.
def display_target_size(video_capture: cv2.VideoCapture, fullscreen: bool) -> tuple[int, int]:
    if fullscreen:
        screen_size = primary_screen_size()
        if screen_size is not None:
            return screen_size
    return video_capture_size(video_capture)


# Redimensiona o frame para caber na janela.
def resize_for_display(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame

    interpolation = cv2.INTER_AREA
    if width > frame.shape[1] or height > frame.shape[0]:
        interpolation = cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interpolation)


# Usa um FPS seguro quando o video nao informa bem esse valor.
def safe_fps(video_capture: cv2.VideoCapture) -> int:
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return int(round(fps))
    return 30


# Apaga um ficheiro se existir, ignorando falhas pequenas.
def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# Abre uma imagem de forma mais segura em caminhos com acentos.
def read_image(path: Path) -> np.ndarray | None:
    # Forma mais segura de abrir imagens em caminhos com acentos.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Grava uma imagem evitando alguns problemas de caminhos no Windows.
def write_image(path: Path, image: np.ndarray) -> bool:
    # Evita problemas do OpenCV com alguns caminhos no Windows.
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


# Abre o ficheiro de video.
def open_video_capture(path: Path) -> cv2.VideoCapture:
    return cv2.VideoCapture(str(path))


# Lista backends de webcam adequados a cada sistema.
def camera_backend_candidates() -> list[int]:
    # Cada sistema costuma gostar mais de um backend diferente.
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
    return [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]


# Tenta abrir a webcam com varios backends.
def open_camera_capture(camera_index: int) -> cv2.VideoCapture:
    for backend in camera_backend_candidates():
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(camera_index)


# Fecha todos os captures recebidos.
def release_captures(*captures: cv2.VideoCapture) -> None:
    for capture in captures:
        if capture is not None:
            capture.release()


# Audio e modelos faciais.

# Extrai uma faixa wav temporaria do video.
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


