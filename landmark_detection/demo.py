import argparse

from landmark_detection.landmark_detector import LandmarkDetector
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', type=str, help='The evaluation image path.')
    parser.add_argument('--face', nargs='+', type=float, help='The coordinate [x1,x2,y1,y2] of a face')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    box = args.face

    landmark_detector = LandmarkDetector('../checkpoint/model.pth')
    preprocessed_data = landmark_detector.preprocess_image(image, box)
    landmarks = landmark_detector.predict(preprocessed_data)
