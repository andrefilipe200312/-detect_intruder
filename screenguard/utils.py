from __future__ import annotations

import numpy as np


# Utilitarios de reconhecimento.

# Garante que um valor fica dentro dos limites indicados.
def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# Normaliza um vetor para poder comparar rostos por similaridade.
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


# Remove repetidos sem trocar a ordem original.
def unique_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


# Junta nomes numa frase curta para caber no ecra.
def format_label_list(labels: list[str], fallback: str, max_chars: int = 52) -> str:
    text = ", ".join(unique_labels(labels))
    if not text:
        return fallback
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


# Calcula a area de uma caixa de rosto.
def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2]) * max(0, box[3])
