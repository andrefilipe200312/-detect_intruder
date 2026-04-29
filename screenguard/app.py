from __future__ import annotations

import sys
import time

import cv2

from .config import parse_args, resolve_video_path
from .constants import WINDOW_NAME
from .drawing import clear_nose_drawing, intruder_nose_position, save_nose_drawing, stabilize_nose_position, update_nose_drawing
from .face import create_face_embedding_engine, detect_faces, profile_labels
from .guard import reset_guard_state, update_guard_state
from .media import AudioPlayer, configure_window, display_target_size, open_camera_capture, open_video_capture, release_captures, resize_for_display, safe_fps
from .references import load_hostile_image, load_known_face_profiles
from .ui import annotate_webcam_preview, build_hostile_frame, render_playback_frame


# Ciclo principal do programa.

# Coordena video, webcam, audio, reconhecimento e interface.
def main() -> int:
    args = parse_args()

    try:
        # Resolve os ficheiros indicados antes de abrir a webcam.
        video_path = resolve_video_path(args.video)
        hostile_image = load_hostile_image(args.hostile_image)
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    # Abre o video principal.
    video_capture = open_video_capture(video_path)
    if not video_capture.isOpened():
        print(f"Nao consegui abrir o video: {video_path}", file=sys.stderr)
        return 1

    # Abre a webcam usada para detetar os rostos.
    camera_capture = open_camera_capture(args.camera_index)
    if not camera_capture.isOpened():
        print(
            f"Nao consegui abrir a webcam no indice {args.camera_index}.",
            file=sys.stderr,
        )
        release_captures(video_capture)
        return 1

    # Prepara tamanhos e tempo entre frames.
    fps = safe_fps(video_capture)
    frame_delay_ms = max(1, int(1000 / fps))
    fullscreen = not args.no_fullscreen
    display_width, display_height = display_target_size(video_capture, fullscreen)

    try:
        # Carrega os modelos faciais e as referencias opcionais.
        engine = create_face_embedding_engine(
            models_dir=args.models_dir,
            min_detection_confidence=args.min_detection_confidence,
        )
        reference_profiles = load_known_face_profiles(args.known_faces_dir, engine)
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        release_captures(video_capture, camera_capture)
        return 1

    # Estado inicial da vigilancia.
    state = reset_guard_state()
    audio_player = AudioPlayer(video_path=video_path)

    # Janela e audio arrancam antes do ciclo principal.
    configure_window(fullscreen=fullscreen)
    audio_player.initialize()
    audio_player.play_from_start()

    try:
        while True:
            # A webcam e lida em todos os ciclos.
            webcam_ok, webcam_frame = camera_capture.read()
            if not webcam_ok:
                print("Falha a ler um frame da webcam.", file=sys.stderr)
                return 1

            # Procura rostos no frame atual da webcam.
            faces = detect_faces(
                engine=engine,
                webcam_frame=webcam_frame,
                min_face_area_ratio=args.min_face_area_ratio,
                edge_margin=args.edge_margin,
            )
            now = time.monotonic()
            alert_just_triggered = update_guard_state(
                state=state,
                faces=faces,
                reference_profiles=reference_profiles,
                recognition_threshold=args.recognition_threshold,
                min_profile_observations=args.min_profile_observations,
                calibration_seconds=args.calibration_seconds,
                trigger_frames=args.trigger_frames,
                now=now,
            )

            # Desenha a miniatura da webcam com informacao dos rostos.
            webcam_preview = annotate_webcam_preview(
                webcam_frame=webcam_frame,
                faces=faces,
                calibrating=not state.authorized_profiles,
                learned_profiles_count=len(state.calibration_profiles),
            )

            if alert_just_triggered:
                # Quando o alerta dispara, o audio para.
                audio_player.pause()

            if state.alert_active:
                # No alerta, o nariz passa a controlar o desenho.
                nose_position = stabilize_nose_position(
                    state,
                    intruder_nose_position(faces, webcam_frame.shape),
                )
                update_nose_drawing(
                    state=state,
                    nose_position=nose_position,
                    size=(display_width, display_height),
                )
                frame_to_show = build_hostile_frame(
                    size=(display_width, display_height),
                    intruder_labels=state.last_intruder_labels,
                    authorized_labels=profile_labels(state.authorized_profiles),
                    trigger_reason=state.trigger_reason,
                    webcam_preview=webcam_preview,
                    custom_image=hostile_image,
                    nose_drawing_canvas=state.nose_drawing_canvas,
                )
            else:
                # Fora do alerta, o video continua a avancar.
                video_ok, video_frame = video_capture.read()
                if not video_ok:
                    if args.no_loop:
                        break
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    audio_player.play_from_start()
                    continue
                if not fullscreen:
                    display_height, display_width = video_frame.shape[:2]
                video_frame = resize_for_display(video_frame, (display_width, display_height))
                frame_to_show = render_playback_frame(
                    video_frame=video_frame,
                    webcam_preview=webcam_preview,
                    faces=faces,
                    state=state,
                    calibration_seconds=args.calibration_seconds,
                    trigger_frames=args.trigger_frames,
                    now=now,
                )

            # Mostra o frame final e le o teclado.
            cv2.imshow(WINDOW_NAME, frame_to_show)
            key = cv2.waitKey(30 if state.alert_active else frame_delay_ms) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                # Recalibra e volta ao inicio do video.
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = reset_guard_state()
                audio_player.play_from_start()
            if key in (ord("c"), ord("C")):
                clear_nose_drawing(state)
            if key in (ord("s"), ord("S")):
                # Guarda o desenho feito no ecra de alerta.
                try:
                    saved_path = save_nose_drawing(state, args.drawings_dir)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                else:
                    if saved_path is None:
                        print("Ainda nao ha desenho para gravar.", file=sys.stderr)
                    else:
                        print(f"Desenho do intruso gravado em: {saved_path}")
    finally:
        # Fecha sempre os recursos, mesmo quando ha erro.
        audio_player.close()
        release_captures(video_capture, camera_capture)
        cv2.destroyAllWindows()

    return 0
