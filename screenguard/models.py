from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import normalize_vector
# Dados usados pelo programa.

# Guarda os dados medios de uma pessoa reconhecida.
@dataclass
class IdentityProfile:
    label: str
    source: str = "session"
    sample_count: int = 0
    descriptor_sum: np.ndarray | None = None

    # Adiciona uma nova amostra do rosto a este perfil.
    def add_descriptor(self, descriptor: np.ndarray) -> None:
        normalized = descriptor.astype(np.float32)
        if self.descriptor_sum is None:
            self.descriptor_sum = normalized.copy()
        else:
            self.descriptor_sum += normalized
        self.sample_count += 1

    # Junta outro perfil com este, mantendo as amostras acumuladas.
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

    # Devolve a assinatura media usada na comparacao facial.
    def signature(self) -> np.ndarray | None:
        if self.descriptor_sum is None or self.sample_count == 0:
            return None
        centroid = (self.descriptor_sum / float(self.sample_count)).astype(np.float32)
        return normalize_vector(centroid)


# Representa um rosto encontrado num frame da webcam.
@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]
    confidence: float
    descriptor: np.ndarray | None
    nose_tip: tuple[float, float] | None = None
    label: str = "Rosto"
    status: str = "pending"
    similarity: float = 0.0


# Guarda o estado atual da vigilancia.
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
    nose_drawing_canvas: np.ndarray | None = None
    last_nose_draw_position: tuple[int, int] | None = None
    smoothed_nose_position: tuple[float, float] | None = None
    nose_missing_frames: int = 0


# Junta o detetor e o reconhecedor facial do OpenCV.
@dataclass
class FaceEmbeddingEngine:
    detector: object
    recognizer: object
    detector_input_size: tuple[int, int] = (320, 320)

    # Atualiza o tamanho de entrada do detetor conforme o frame.
    def set_input_size(self, frame_width: int, frame_height: int) -> None:
        input_size = (max(1, frame_width), max(1, frame_height))
        if input_size != self.detector_input_size:
            self.detector.setInputSize(input_size)
            self.detector_input_size = input_size


