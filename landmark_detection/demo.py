from landmark_detection.landmark_detector import LandmarkDetector
import cv2

if __name__ == "__main__":
    image = cv2.imread('Image.png')
    box = [432, 575, 819, 971]

    landmark_detector = LandmarkDetector('../checkpoint/model.pth')
    preprocessed_data = landmark_detector.preprocess_image(image, box)
    landmarks = landmark_detector.predict(preprocessed_data)
