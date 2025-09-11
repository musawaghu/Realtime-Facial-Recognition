import cv2
import torch
import numpy as np
import sqlite3
import pickle
import argparse
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import List, Tuple


# ============ Configuration ============
class Config:
    """All settings in one place"""
    # Database
    DB_PATH = "faces.db"

    # Model settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIDENCE_THRESHOLD = 0.9  # Face detection confidence
    MATCH_THRESHOLD = 0.5  # Face matching threshold (lower = stricter)

    # Camera settings
    CAMERA_INDEX = 0
    WINDOW_NAME = "Face Recognition"


# ============ Database Manager ============
class SimpleDatabaseManager:
    """Handles SQLite database operations for face storage"""

    def __init__(self, db_path: str = Config.DB_PATH):
        """Initialize SQLite database connection"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create faces table if it doesn't exist"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        self.conn.commit()

    def add_face(self, name: str, embedding: np.ndarray) -> bool:
        """Add or update a person's face embedding"""
        try:
            # Convert numpy array to bytes for storage
            embedding_bytes = pickle.dumps(embedding)

            # Insert or replace existing
            self.cursor.execute('''
                INSERT OR REPLACE INTO faces (name, embedding) 
                VALUES (?, ?)
            ''', (name, embedding_bytes))
            self.conn.commit()
            print(f"✓ Added {name} to database")
            return True
        except Exception as e:
            print(f"✗ Error adding {name}: {e}")
            return False

    def get_all_faces(self) -> List[Tuple[str, np.ndarray]]:
        """Get all faces from database"""
        self.cursor.execute('SELECT name, embedding FROM faces')
        faces = []
        for name, embedding_bytes in self.cursor.fetchall():
            # Convert bytes back to numpy array
            embedding = pickle.loads(embedding_bytes)
            faces.append((name, embedding))
        return faces

    def delete_face(self, name: str) -> bool:
        """Remove a person from database"""
        self.cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def close(self):
        """Close database connection"""
        self.conn.close()


class SimpleFaceRecognition:
    """Main face recognition system - all in one class"""

    def __init__(self):
        """Initialize models and database"""
        print(f"Initializing... (using {Config.DEVICE})")

        # Face detection model (MTCNN)
        self.detector = MTCNN(
            keep_all=True,
            device=Config.DEVICE,
            min_face_size=40
        )

        # Face recognition model (FaceNet)
        self.encoder = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=Config.DEVICE
        ).eval()

        # Database
        self.db = SimpleDatabaseManager()

        # Load known faces
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load all registered faces from database"""
        faces = self.db.get_all_faces()
        self.known_names = [name for name, _ in faces]
        self.known_faces = [embedding for _, embedding in faces]
        print(f"Loaded {len(self.known_names)} faces from database")
        if self.known_names:
            print(f"Known people: {', '.join(self.known_names)}")

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Detect faces in frame and return cropped faces with bounding boxes

        Returns:
            - List of cropped face images (160x160)
            - List of bounding boxes [x1, y1, x2, y2]
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs = self.detector.detect(rgb_frame)

        faces = []
        valid_boxes = []

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob >= Config.CONFIDENCE_THRESHOLD:
                    # Get face coordinates
                    x1, y1, x2, y2 = box.astype(int)

                    # Ensure coordinates are within frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(rgb_frame.shape[1], x2)
                    y2 = min(rgb_frame.shape[0], y2)

                    # Extract and resize face
                    face = rgb_frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (160, 160))
                        faces.append(face_resized)
                        valid_boxes.append([x1, y1, x2, y2])

        return faces, valid_boxes

    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate embedding vector for a face

        Args:
            face_image: RGB face image (160x160)

        Returns:
            512-dimensional embedding vector
        """
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(face_image).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)

        # Generate embedding
        with torch.no_grad():
            embedding = self.encoder(face_tensor)

        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def find_match(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Find best match for a face embedding

        Returns:
            (name, distance) - returns ("Unknown", 1.0) if no match
        """
        if not self.known_faces:
            return "Unknown", 1.0

        # Calculate distances to all known faces
        distances = []
        for known_embedding in self.known_faces:
            # Cosine distance
            distance = 1 - np.dot(face_embedding, known_embedding)
            distances.append(distance)

        # Find the closest match
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        # Check if match is good enough
        if min_distance <= Config.MATCH_THRESHOLD:
            return self.known_names[min_idx], min_distance
        else:
            return "Unknown", min_distance

    def register_from_webcam(self):
        """Interactive registration of new faces using webcam"""
        print("\n=== Face Registration Mode ===")
        print("Position your face in the frame")
        print("Press 'c' to capture (take 3-5 photos)")
        print("Press 's' to save and exit")
        print("Press 'q' to quit without saving")

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        captured_embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces, boxes = self.detect_faces(frame)

            # Draw boxes
            for x1, y1, x2, y2 in boxes:
                color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Show instructions
            cv2.putText(frame, f"Captures: {len(captured_embeddings)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(Config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) == 1:
                # Capture face embedding
                embedding = self.encode_face(faces[0])
                captured_embeddings.append(embedding)
                print(f"Captured {len(captured_embeddings)} photos")

            elif key == ord('s') and captured_embeddings:
                # Save face
                name = input("\nEnter person's name: ").strip()
                if name:
                    # Average all embeddings
                    avg_embedding = np.mean(captured_embeddings, axis=0)
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                    # Save to database
                    if self.db.add_face(name, avg_embedding):
                        print(f"Successfully registered {name}!")
                        self.load_known_faces()  # Reload faces
                break

            elif key == ord('q'):
                print("Registration cancelled")
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_recognition(self):
        """Main recognition loop"""
        print("\n=== Face Recognition Mode ===")
        print("Press 'q' to quit")
        print("Press 'r' to register new face")

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and recognize faces
            faces, boxes = self.detect_faces(frame)

            # Process each face
            for i, (face, box) in enumerate(zip(faces, boxes)):
                # Get embedding and find match
                embedding = self.encode_face(face)
                name, distance = self.find_match(embedding)

                # Draw box and label
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add name label
                label = f"{name} ({1 - distance:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Show stats
            cv2.putText(frame, f"Registered: {len(self.known_names)} | Detected: {len(faces)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(Config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                self.register_from_webcam()
                # Restart recognition
                cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        cap.release()
        cv2.destroyAllWindows()

    def cleanup(self):
        """Clean up resources"""
        self.db.close()


# ============ Main Entry Point ============
def main():
    """Main function to run the face recognition system"""
    parser = argparse.ArgumentParser(description="Simple Face Recognition System")
    parser.add_argument("--register", action="store_true",
                        help="Start in registration mode")
    parser.add_argument("--clear-db", action="store_true",
                        help="Clear all registered faces")
    args = parser.parse_args()

    # Print system info
    print("=" * 50)
    print("Simple Face Recognition System")
    print("=" * 50)
    print(f"Device: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    # Initialize system
    system = SimpleFaceRecognition()

    try:
        if args.clear_db:
            # Clear database
            response = input("Clear all registered faces? (y/n): ")
            if response.lower() == 'y':
                Path(Config.DB_PATH).unlink(missing_ok=True)
                print("Database cleared!")

        elif args.register:
            # Registration mode
            system.register_from_webcam()

        else:
            # Recognition mode
            system.run_recognition()

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        system.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()