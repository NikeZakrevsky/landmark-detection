from landmark_detection.landmark_detector import LandmarkDetector
import cv2

if __name__ == "__main__":
    image = cv2.imread('')

    landmark_detector = LandmarkDetector('model.pth')
    preprocessed_image = landmark_detector.preprocess_image('Image.png', [])
    landmarks = landmark_detector.predict(preprocessed_image)
