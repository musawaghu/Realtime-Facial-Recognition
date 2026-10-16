"""Smoke test with the real MTCNN + FaceNet models on LFW photos (run: pytest -m slow).
Needs benchmarks/data/lfw (downloaded by benchmarks/lfw_eval.py)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.db import Database

LFW = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "lfw"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not LFW.exists(), reason="run benchmarks/lfw_eval.py to download LFW")]


def photo(person: str, n: int) -> np.ndarray:
    return np.asarray(Image.open(LFW / person / f"{person}_{n:04d}.jpg").convert("RGB"))


@pytest.fixture(scope="module")
def engine():
    from app.face_engine import FaceEngine
    return FaceEngine()


def test_enroll_three_photos_then_recognize_fourth(engine, tmp_path):
    db = Database(str(tmp_path / "smoke.db"))
    wid = db.create_worker("Colin Powell", "", "1234")
    for n in (1, 2, 3):
        faces = engine.analyze(photo("Colin_Powell", n))
        assert len(faces) >= 1
        db.add_embedding(wid, faces[0].embedding)
    db.finish_enrollment(wid)
    engine.gallery.load(*db.gallery())

    same = engine.analyze(photo("Colin_Powell", 4))[0]
    assert engine.gallery.match(same.embedding)[0] == wid
    assert all(type(v) is int for v in same.box)   # API returns the box as JSON

    other = engine.analyze(photo("Tony_Blair", 1))[0]
    assert engine.gallery.match(other.embedding)[0] is None


def test_embeddings_are_unit_length_and_batched(engine):
    rgb = photo("Tony_Blair", 2)
    boxes, probs, landmarks = engine.detect(rgb)
    assert boxes.dtype == np.float32 and landmarks.shape[1:] == (5, 2)
    emb = engine.encode(rgb, np.repeat(boxes[:1], 3, axis=0))   # 3 crops in one pass
    assert emb.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(emb[0], emb[2], atol=1e-5)


def test_head_turn_direction_on_a_real_photo(engine):
    """George_Clooney_0002 is turned toward his own left (checked by eye). The kiosk's
    challenge depends on this sign convention, so pin it with the real detector."""
    from app import config, liveness
    rgb = photo("George_Clooney", 2)
    _, _, lm = engine.detect(rgb)
    assert liveness.yaw(lm[0]) > config.LIVENESS_TURN_YAW
    _, _, lm = engine.detect(rgb[:, ::-1].copy())          # mirror image: turned to his right
    assert liveness.yaw(lm[0]) < -config.LIVENESS_TURN_YAW
