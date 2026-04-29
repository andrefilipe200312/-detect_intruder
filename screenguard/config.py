from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_DRAWINGS_DIR, DEFAULT_MODELS_DIR, VIDEO_EXTENSIONS
# Argumentos e caminhos.

# Le as opcoes passadas pela linha de comandos.
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
        "--drawings-dir",
        type=Path,
        default=DEFAULT_DRAWINGS_DIR,
        help="Pasta onde os desenhos do intruso sao guardados ao carregar S.",
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


# Escolhe o video indicado ou procura um video na pasta atual.
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

