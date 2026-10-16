import json

import numpy as np

from benchmarks import hardhat, hardhat_eval

# A 200x200 grey "face": eyes at y=90, mouth at y=140 (eye-to-mouth distance d = 50)
LM = np.array([[80, 90], [120, 90], [100, 115], [85, 140], [115, 140]], dtype=np.float32)
FACE = np.full((200, 200, 3), 128, np.uint8)


def test_hat_covers_the_head_and_shades_the_forehead():
    out = hardhat.draw_hard_hat(FACE, LM)
    assert (out[30, 100] == (hardhat.HAT_YELLOW * 0.75).astype(np.uint8)).all()  # dome (brim spans y~49-66)
    assert (out[58, 100] == (hardhat.HAT_YELLOW * 0.85).astype(np.uint8)).all()  # brim
    assert out[70, 100].mean() < 128      # shadow under the brim (brim at 90 - 0.65*50 = 57.5)
    assert out[85, 100].mean() < 128      # still shaded just above the eyes
    assert (out[150:, :] == 128).all()    # mouth and below untouched
    assert (FACE == 128).all()            # input not modified


def test_hat_follows_head_tilt():
    tilted = LM.copy()
    tilted[1, 1] = 110                    # right eye lower: head tilted
    a = hardhat.draw_hard_hat(FACE, LM)
    b = hardhat.draw_hard_hat(FACE, tilted)
    assert not np.array_equal(a, b)


def test_degenerate_landmarks_leave_image_alone():
    assert np.array_equal(hardhat.draw_hard_hat(FACE, np.zeros((5, 2), np.float32)), FACE)


def person(i, n, noise, rng):
    """n noisy embeddings of 'person i' (unit vectors near basis vector i)."""
    v = np.zeros((n, 512), np.float32)
    v[:, i] = 1
    v += noise * rng.standard_normal((n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_real_report_scores_like_the_kiosk():
    rng = np.random.default_rng(0)
    people = {
        "p01": {"nohat": person(1, 6, 0.01, rng), "hat": person(1, 5, 0.06, rng)},   # hat: noisier
        "p02": {"nohat": person(2, 6, 0.01, rng), "hat": person(2, 5, 0.06, rng)},
    }
    r = hardhat.real_report(people, threshold=0.5)
    assert r["nohat"]["check_in_attempts"] == 6       # 3 enrollment captures each are excluded
    assert r["hat"]["check_in_attempts"] == 10
    assert r["nohat"]["accepted_rate"] == 1.0
    assert r["hat"]["median_similarity"] < r["nohat"]["median_similarity"]
    assert r["hat"]["impostor_comparisons"] == 10 and r["hat"]["false_accepts"] == 0


def test_real_report_skips_people_without_enough_nohat_captures():
    rng = np.random.default_rng(1)
    people = {"p01": {"nohat": person(1, 3, 0.01, rng)}, "p02": {"nohat": person(2, 6, 0.01, rng)}}
    assert hardhat.real_report(people, threshold=0.5)["nohat"]["people"] == 1


def test_real_mode_reads_capture_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    for i, pid in ((1, "p01"), (2, "p02")):
        np.savez(tmp_path / f"{pid}.npz", consent_at=np.array("2026-10-15T09:00:00"),
                 nohat__emb=person(i, 6, 0.01, rng), nohat__yaw=np.zeros(6, np.float32), nohat__frames=np.array(8),
                 hat__emb=person(i, 4, 0.05, rng), hat__yaw=np.full(4, 0.4, np.float32), hat__frames=np.array(8))
    monkeypatch.setattr(hardhat_eval, "CAPTURES", tmp_path)
    monkeypatch.setattr(hardhat_eval, "RESULTS", tmp_path / "results")
    hardhat_eval.real()
    out = json.loads((tmp_path / "results" / "hardhat_real.json").read_text())
    assert out["people"] == 2
    assert out["face_detected_rate"] == {"hat": 0.5, "nohat": 0.75}
    assert out["reads_as_frontal_rate"] == {"hat": 0.0, "nohat": 1.0}
    assert out["conditions"]["hat"]["check_in_attempts"] == 8
