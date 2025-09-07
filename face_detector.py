import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from typing import List, Tuple
class FaceDetector:
    """Handles face detection using MTCNN (Multi-task Cascaded Convolutional Networks)"""
    def __init__(self, device = 'cuda:0', confidence_threshold = 0.9):
        """
        Initialize MTCNN face detector
        Args:
            device: 'cuda' for GPU or 'cpu'
            confidence_threshold: Minimum confidence to consider a detection valid
        """
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize MTCNN for face detection
        # keep_all=True returns all detected faces, not just the most confident one
        self.detector = MTCNN(
            select_largest=False,
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds for each stage
            min_face_size=40  # Minimum face size to detect
        )
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[List[int]], List[float]]:
        """
        Detect faces in frame snd return cropped faces with bounding boxes
        Args:
            frame: Input from frame/webcam (BGR format from OpenCV)
        Returns:
            Tuple containing:
            - List of cropped face images (RGB format, 160x160 for FaceNet)
            - List of bounding boxes [x1, y1, x2, y2]
            - List of confidence scores
        """
        # Convert BGR (OpenCV) to RGB (required by MTCNN)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces - returns bounding boxes and confidence scores
        boxes, confidences = self.detector.detect(rgb_frame)

        face_images = []
        valid_boxes = []
        valid_confidences = []

        if boxes is not None:
            for box, confidence in zip(boxes, confidences):
                if confidence >= self.confidence_threshold:
                    # Extract face region
                    x1, y1, x2, y2 = box.astype(int)

                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(rgb_frame.shape[1], x2)
                    y2 = min(rgb_frame.shape[0], y2)

                    # Crop and resize face to 160x160 (FaceNet input size)
                    face = rgb_frame[y1:y2, x1:x2]
                    if face.size > 0:  # Check if face region is valid
                        face_resized = cv2.resize(face, (160, 160))
                        face_images.append(face_resized)
                        valid_boxes.append([x1, y1, x2, y2])
                        valid_confidences.append(confidence)

        return face_images, valid_boxes, valid_confidences