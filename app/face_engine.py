"""Face detection (MTCNN) + embeddings (FaceNet) + matching against enrolled workers."""
from dataclasses import dataclass

import numpy as np

from app import config


class Gallery:
    """In-memory matrix of every enrolled embedding, for fast vectorized matching.

    Each worker has several embeddings (different angles). A probe face is compared
    to all of them in one matrix product, and a worker's score is their best match.
    """

    def __init__(self, threshold: float = config.MATCH_THRESHOLD):
        self.threshold = threshold
        self.load(np.empty(0, dtype=np.int64), np.empty((0, 512), dtype=np.float32))

    def load(self, worker_ids: np.ndarray, vectors: np.ndarray):
        order = np.argsort(worker_ids, kind="stable")    # group rows by worker
        self.vectors = vectors[order].astype(np.float32)
        sorted_ids = worker_ids[order]
        self.ids, self.starts = np.unique(sorted_ids, return_index=True)

    def __len__(self):
        return len(self.ids)

    def match(self, embedding: np.ndarray) -> tuple[int | None, float]:
        """Return (worker_id, similarity) for the best worker, or (None, similarity) if below threshold."""
        if len(self.ids) == 0:
            return None, 0.0
        sims = self.vectors @ embedding                       # cosine similarity (unit vectors)
        per_worker = np.maximum.reduceat(sims, self.starts)   # best capture per worker
        best = int(np.argmax(per_worker))
        score = float(per_worker[best])
        return (int(self.ids[best]) if score >= self.threshold else None), score


@dataclass
class DetectedFace:
    box: list[int]               # [x1, y1, x2, y2]
    prob: float
    landmarks: np.ndarray        # 5x2: eyes, nose, mouth corners (for liveness next round)
    embedding: np.ndarray        # 512, unit length


class FaceEngine:
    def __init__(self, device: str | None = None):
        # Imported here so tests and tools that only need Gallery don't load torch
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(image_size=160, margin=config.FACE_MARGIN, keep_all=True,
                              min_face_size=config.MIN_FACE_SIZE, device=self.device)
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.gallery = Gallery()

    def detect(self, rgb: np.ndarray):
        """Find faces. Returns (boxes, probs, landmarks) for faces above the confidence threshold."""
        boxes, probs, landmarks = self.detector.detect(rgb, landmarks=True)
        if boxes is None:
            return np.empty((0, 4)), np.empty(0), np.empty((0, 5, 2))
        # MTCNN hands back object-dtype arrays; make them plain floats
        boxes, probs = boxes.astype(np.float32), probs.astype(np.float32)
        landmarks = landmarks.astype(np.float32)
        keep = probs >= config.CONFIDENCE_THRESHOLD
        return boxes[keep], probs[keep], landmarks[keep]

    def encode(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Crop every face (with margin, standardized) and embed them all in ONE forward pass."""
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        crops = self.detector.extract(rgb, boxes, None)           # (n, 3, 160, 160) tensor
        with self.torch.no_grad():
            emb = self.encoder(crops.to(self.device)).cpu().numpy()
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def analyze(self, rgb: np.ndarray) -> list[DetectedFace]:
        """Detect + embed every face in an RGB frame."""
        boxes, probs, landmarks = self.detect(rgb)
        embeddings = self.encode(rgb, boxes)
        h, w = rgb.shape[:2]
        faces = []
        for box, prob, lm, emb in zip(boxes, probs, landmarks, embeddings):
            x1, y1, x2, y2 = (int(v) for v in box)   # plain ints: the box is sent as JSON
            faces.append(DetectedFace([max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                                      float(prob), lm, emb))
        return faces
