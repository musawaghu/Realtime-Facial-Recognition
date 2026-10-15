import numpy as np

from app.face_engine import Gallery


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def basis(i):
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1
    return v


def test_empty_gallery_matches_nobody():
    assert Gallery(threshold=0.5).match(basis(0)) == (None, 0.0)


def test_best_capture_per_worker_wins():
    # Worker 7 has two captures; the probe is close to the SECOND one only.
    g = Gallery(threshold=0.5)
    g.load(np.array([7, 3, 7]), np.stack([basis(0), basis(1), basis(2)]))
    worker, sim = g.match(unit(basis(2) + 0.1 * basis(5)))
    assert worker == 7 and sim > 0.99


def test_unsorted_ids_are_grouped_correctly():
    g = Gallery(threshold=0.5)
    g.load(np.array([2, 1, 2, 1]), np.stack([basis(0), basis(1), basis(2), basis(3)]))
    assert g.match(basis(3))[0] == 1
    assert g.match(basis(0))[0] == 2
    assert len(g) == 2


def test_threshold_boundary():
    g = Gallery(threshold=0.6)
    g.load(np.array([1]), basis(0)[None])
    at = unit([0.6, 0.8] + [0] * 510)        # similarity exactly 0.6 with basis(0)
    below = unit([0.59, 0.807] + [0] * 510)
    assert g.match(at)[0] == 1
    assert g.match(below)[0] is None
