import cv2
import numpy as np
from landmark_detection.landmark_detector import LandmarkDetector


def test_landmarks():
    image = cv2.imread('../landmark_detection/Image.png')
    box = [432, 575, 819, 971]

    landmark_detector = LandmarkDetector('../checkpoint/model.pth')
    preprocessed_data = landmark_detector.preprocess_image(image, box)
    landmarks = landmark_detector.predict(preprocessed_data)

    assert np.array(landmarks['landmarks']).shape == (68, 3)
