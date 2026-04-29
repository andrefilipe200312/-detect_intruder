from __future__ import annotations

import time

from .face import classify_faces, finalize_authorized_profiles, learn_calibration_profiles
from .models import DetectedFace, GuardState, IdentityProfile
from .utils import format_label_list


# Estado e regras principais da vigilancia.

# Cria um estado novo para iniciar ou recalibrar.
def reset_guard_state() -> GuardState:
    return GuardState(calibration_started_at=time.monotonic())


# Atualiza a calibracao inicial e devolve True quando esta acaba.
def update_calibration(
    state: GuardState,
    faces: list[DetectedFace],
    reference_profiles: list[IdentityProfile],
    recognition_threshold: float,
    min_profile_observations: int,
    calibration_seconds: float,
    now: float,
) -> bool:
    learn_calibration_profiles(
        state=state,
        faces=faces,
        reference_profiles=reference_profiles,
        recognition_threshold=recognition_threshold,
    )

    if now - state.calibration_started_at < calibration_seconds:
        return False

    authorized_profiles = finalize_authorized_profiles(
        state.calibration_profiles,
        min_profile_observations=min_profile_observations,
    )
    if not authorized_profiles:
        return False

    state.authorized_profiles = authorized_profiles
    authorized_labels, intruder_labels = classify_faces(
        faces=faces,
        authorized_profiles=state.authorized_profiles,
        reference_profiles=reference_profiles,
        recognition_threshold=recognition_threshold,
    )
    state.last_authorized_labels = authorized_labels
    state.last_intruder_labels = intruder_labels
    state.intruder_streak = 0
    return True


# Atualiza a vigilancia depois de haver perfis autorizados.
def update_surveillance(
    state: GuardState,
    faces: list[DetectedFace],
    reference_profiles: list[IdentityProfile],
    recognition_threshold: float,
    trigger_frames: int,
) -> bool:
    authorized_labels, intruder_labels = classify_faces(
        faces=faces,
        authorized_profiles=state.authorized_profiles,
        reference_profiles=reference_profiles,
        recognition_threshold=recognition_threshold,
    )
    state.last_authorized_labels = authorized_labels
    state.last_intruder_labels = intruder_labels

    if state.alert_active:
        return False

    if not intruder_labels:
        state.intruder_streak = 0
        state.trigger_reason = ""
        return False

    state.intruder_streak += 1
    if state.intruder_streak < trigger_frames:
        return False

    state.alert_active = True
    state.trigger_reason = (
        "Pessoa nao autorizada: "
        f"{format_label_list(intruder_labels, 'desconhecida', max_chars=34)}"
    )
    return True


# Executa um passo completo da logica de seguranca.
def update_guard_state(
    state: GuardState,
    faces: list[DetectedFace],
    reference_profiles: list[IdentityProfile],
    recognition_threshold: float,
    min_profile_observations: int,
    calibration_seconds: float,
    trigger_frames: int,
    now: float,
) -> bool:
    if not state.authorized_profiles:
        update_calibration(
            state=state,
            faces=faces,
            reference_profiles=reference_profiles,
            recognition_threshold=recognition_threshold,
            min_profile_observations=min_profile_observations,
            calibration_seconds=calibration_seconds,
            now=now,
        )
        return False

    return update_surveillance(
        state=state,
        faces=faces,
        reference_profiles=reference_profiles,
        recognition_threshold=recognition_threshold,
        trigger_frames=trigger_frames,
    )
