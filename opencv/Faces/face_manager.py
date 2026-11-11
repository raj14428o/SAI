import os
import cv2 as cv
import numpy as np

class FaceManager:
    """
    Manages face detection, dynamic training from uploaded folders,
    and real-time face recognition.
    """

    def __init__(self, faces_dir=None):
        if faces_dir is None:
            faces_dir = os.path.dirname(os.path.abspath(__file__))
        self.faces_dir = faces_dir

        self.cascade_path = os.path.join(self.faces_dir, "haar_face.xml")
        self.haar_cascade = cv.CascadeClassifier(self.cascade_path)

        self.model_path = os.path.join(self.faces_dir, "face_trained.yml")
        self.labels_path = os.path.join(self.faces_dir, "labels_names.npy")

        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.people = []

        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            if os.path.exists(self.labels_path):
                self.people = list(np.load(self.labels_path))
            else:
                self.people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
            print(f"[INFO] FaceManager loaded model with {len(self.people)} people: {self.people}")
        else:
            print("[WARN] Trained face model not found. Retraining will be required.")

    def retrain(self, extra_folders=None):
        """
        Scans training folders (including known_people/) and retrains LBPH model.
        """
        search_dirs = [
            os.path.join(self.faces_dir, "..", "Resources", "Faces", "train"),
            os.path.join(self.faces_dir, "..", "..", "known_people")
        ]
        if extra_folders:
            search_dirs.extend(extra_folders)

        people_names = []
        features = []
        labels = []

        for base_dir in search_dirs:
            if not os.path.exists(base_dir):
                continue
            for person in os.listdir(base_dir):
                person_dir = os.path.join(base_dir, person)
                if not os.path.isdir(person_dir):
                    continue
                if person not in people_names:
                    people_names.append(person)
                label_id = people_names.index(person)

                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv.imread(img_path)
                    if img is None:
                        continue
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
                    for (x, y, w, h) in faces:
                        features.append(gray[y:y+h, x:x+w])
                        labels.append(label_id)

        if len(features) == 0:
            return "No face images found to train."

        features = np.array(features, dtype="object")
        labels = np.array(labels)

        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.train(features, labels)
        self.recognizer.save(self.model_path)
        np.save(self.labels_path, np.array(people_names))
        self.people = people_names

        msg = f"[SUCCESS] Retrained on {len(features)} face samples for {len(people_names)} people: {people_names}"
        print(msg)
        return msg

    def identify_faces(self, frame, max_confidence=85.0):
        """
        Detects faces in frame and identifies known people.
        Returns list of dicts: [{'name': ..., 'confidence': ..., 'box': (x,y,w,h)}]
        In LBPH, lower confidence score = closer match.
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        results = []

        for (x, y, w, h) in faces_rect:
            roi = gray[y:y+h, x:x+w]
            label, conf = self.recognizer.predict(roi)

            # In LBPH: 0 is exact match, typically < 75 is reliable
            if conf <= max_confidence and label < len(self.people):
                person_name = self.people[label]
            else:
                person_name = "Unknown"

            results.append({
                "name": person_name,
                "confidence": round(conf, 1),
                "box": (x, y, w, h)
            })

        return results
