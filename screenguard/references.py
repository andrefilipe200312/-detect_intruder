from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import REFERENCE_IMAGE_EXTENSIONS
from .face import detect_faces
from .media import read_image
from .models import FaceEmbeddingEngine, IdentityProfile
from .utils import box_area


# Imagens e referencias externas.

# Carrega a imagem personalizada do ecra de alerta.
def load_hostile_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Imagem hostil nao encontrada: {resolved}")
    image = read_image(resolved)
    if image is None:
        raise RuntimeError(f"Nao consegui carregar a imagem hostil: {resolved}")
    return image


# Le fotos de referencia e cria perfis com nome.
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
            image = read_image(image_path)
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

