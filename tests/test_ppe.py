import sqlite3

from app import config
from app.db import Database
from app.ppe import Detection, assess

FACE = [200, 150, 280, 250]       # 80 wide, 100 tall; top at y=150, chin at y=250
TALL = 720                        # frame shows the torso
SHORT = 300                       # frame cut off just below the chin


def det(name, box, conf=0.9):
    return Detection(name, conf, box)


def test_helmet_on_head_counts():
    r = assess([det("helmet", [195, 90, 285, 170])], FACE, TALL)
    assert r["hard_hat"] is True


def test_helmet_elsewhere_does_not_count():
    # someone else's helmet in the background, or one held in the hand at chest height
    assert assess([det("helmet", [500, 80, 580, 150])], FACE, TALL)["hard_hat"] is False
    assert assess([det("helmet", [190, 300, 270, 360])], FACE, TALL)["hard_hat"] is False


def test_no_helmet_beats_weaker_helmet():
    dets = [det("helmet", [195, 90, 285, 170], conf=0.5), det("no_helmet", [195, 130, 285, 250], conf=0.8)]
    assert assess(dets, FACE, TALL)["hard_hat"] is False


def test_low_confidence_is_ignored():
    r = assess([det("helmet", [195, 90, 285, 170], conf=config.PPE_CONF - 0.01)], FACE, TALL)
    assert r["hard_hat"] is False


def test_vest_on_torso():
    assert assess([det("vest", [150, 270, 330, 520])], FACE, TALL)["vest"] is True
    assert assess([], FACE, TALL)["vest"] is False
    assert assess([det("vest", [600, 270, 760, 520])], FACE, TALL)["vest"] is False   # someone else's


def test_vest_unknown_when_torso_out_of_frame():
    assert assess([], FACE, SHORT)["vest"] is None


def test_old_database_gets_ppe_columns(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY, worker_id INTEGER, kind TEXT,
                    method TEXT, similarity REAL, ts TEXT)""")
    conn.commit(); conn.close()
    Database(path)
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(events)")}
    assert {"hard_hat", "vest"} <= cols
