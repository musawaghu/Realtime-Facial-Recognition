"""Hard-hat accuracy: shared pieces for the synthetic test (LFW + drawn hat) and the real test
(photos of consenting people captured with scripts/capture_hardhat.py)."""
import numpy as np

HAT_YELLOW = np.array([245, 192, 25], dtype=np.float32)


def draw_hard_hat(rgb: np.ndarray, landmarks: np.ndarray, brim: float = 0.65) -> np.ndarray:
    """Draw a hard hat on a face, using MTCNN landmarks to place it.

    Geometry is in face units (d) along the eye line, so it scales and tilts with the head.
    d is the eye-to-mouth distance (~1 eye-width when frontal; unlike eye width it doesn't
    shrink when the head turns):
      - brim `brim`*d above the eyes (mid-forehead, where a hard hat brim sits)
      - dome above the brim, brim wider than the head
      - shadow under the brim fading out just below the eyes (hard hats shade the eyes)
    A proxy only: real hats also change how people hold their head and where light falls.
    """
    out = rgb.astype(np.float32).copy()
    left_eye, right_eye = landmarks[0], landmarks[1]
    center = (left_eye + right_eye) / 2
    eye_w = float(np.linalg.norm(right_eye - left_eye))
    d = float(np.linalg.norm((landmarks[3] + landmarks[4]) / 2 - center))
    if eye_w < 1 or d < 1:
        return rgb.copy()
    ux = (right_eye - left_eye) / eye_w             # along the eye line
    vy = np.array([-ux[1], ux[0]])                  # perpendicular, pointing down the face

    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    u = (dx * ux[0] + dy * ux[1]) / d               # face coordinates, in units of d
    v = (dx * vy[0] + dy * vy[1]) / d
    v_brim = -brim

    # shadow: darkest right under the brim, gone by just below the eyes
    shade_band = (v > v_brim) & (v < 0.2) & (np.abs(u) < 1.6)
    darkness = 0.45 + 0.55 * np.clip((v - v_brim) / (0.2 - v_brim), 0, 1)
    out[shade_band] *= darkness[shade_band][:, None]

    # dome (upper half-ellipse) and brim (flat ellipse)
    dome = (v <= v_brim) & ((u / 1.55) ** 2 + ((v - v_brim) / 1.5) ** 2 <= 1)
    brim_mask = (u / 1.95) ** 2 + ((v - v_brim) / 0.17) ** 2 <= 1
    shading = 0.75 + 0.25 * np.clip(-u / 1.55, -1, 1)   # light from the left, so it isn't a flat sticker
    out[dome] = HAT_YELLOW * shading[dome][:, None]
    out[brim_mask] = HAT_YELLOW * 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def real_report(participants: dict[str, dict[str, np.ndarray]], threshold: float,
                enroll_condition: str = "nohat", n_enroll: int = 3) -> dict:
    """Score real captures like the kiosk does.

    participants: {participant_id: {condition: embeddings (n, 512)}}.
    Each person is enrolled with their first `n_enroll` captures of `enroll_condition`
    (like the kiosk's multi-angle enrollment). Every other capture is a check-in attempt, scored
    against their own enrollment (genuine) and everyone else's (impostor).
    """
    enrolled = {p: c[enroll_condition][:n_enroll] for p, c in participants.items()
                if len(c.get(enroll_condition, [])) >= n_enroll + 1}
    report = {}
    conditions = sorted({cond for c in participants.values() for cond in c})
    for cond in conditions:
        genuine, impostor = [], []
        for p, enroll in enrolled.items():
            probes = participants[p].get(cond, np.empty((0, 512)))
            if cond == enroll_condition:
                probes = probes[n_enroll:]          # never test on the enrollment captures
            for e in probes:
                genuine.append(float(np.max(enroll @ e)))
                impostor += [float(np.max(other @ e)) for q, other in enrolled.items() if q != p]
        genuine, impostor = np.array(genuine), np.array(impostor)
        report[cond] = {
            "people": len(enrolled),
            "check_in_attempts": int(len(genuine)),
            "accepted_rate": round(float(np.mean(genuine >= threshold)), 4) if len(genuine) else None,
            "median_similarity": round(float(np.median(genuine)), 3) if len(genuine) else None,
            "impostor_comparisons": int(len(impostor)),
            "false_accepts": int(np.sum(impostor >= threshold)),
        }
    return report
