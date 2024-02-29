import numpy as np
import argparse
from PIL import Image

class PCADetector:

    def __init__(self, image_path, n_components):
        self.image_path = image_path
        self.n_components = n_components
        self.read_image()
        self.prepare_image_data()
        self.perform_pca()

    def read_image(self):
        print("Reading image...")
        im = Image.open(self.image_path)
        self.image = np.asarray(im)
        if len(self.image.shape) == 2:
            self.image = self.image[:, :, np.newaxis]

    def prepare_image_data(self):
        print("Preparing image data...")
        self.flat_image = np.reshape(self.image, (-1, self.image.shape[-1]))

    def perform_pca(self):
        print("Performing PCA...")
        X_normalized = (self.flat_image - np.mean(self.flat_image, axis=0)) / np.std(self.flat_image, axis=0)
        covariance_matrix = np.cov(X_normalized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(-eigenvalues)
        selected_eigenvectors = eigenvectors[:, sorted_indices[:self.n_components]]
        self.principal_components = np.dot(X_normalized, selected_eigenvectors)

    def find_anomalies(self, percentile=95):
        print("Finding anomalies...")
        mean_vector = np.mean(self.principal_components, axis=0)
        distances = np.linalg.norm(self.principal_components - mean_vector, axis=1)
        threshold = np.percentile(distances, percentile)
        anomalies = np.where(distances > threshold)[0]
        anomaly_image = np.zeros(self.flat_image.shape[0])
        anomaly_image[anomalies] = 255
        anomaly_image = np.reshape(anomaly_image, self.image.shape[:2]).astype(np.float32)
        return anomaly_image

    def save_result_image(self, result_array, save_path):
        print("Converting and saving the resultant image...")
        result_image = Image.fromarray(result_array.astype(np.uint8))
        result_image.save(save_path)


def main(args):
    detector = PCADetector(args.input_image, args.n_components)
    anomaly_image = detector.find_anomalies(args.percentile)
    detector.save_result_image(anomaly_image, args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Detector for image analysis.")
    parser.add_argument("-i", "--input_image", required=True, help="Path to read the input image.")
    parser.add_argument("-o", "--output_image", required=True, help="Path to save the output image.")
    parser.add_argument("-n", "--n_components", type=int, required=True, help="Number of principal components to use.")
    parser.add_argument("-p", "--percentile", type=float, default=95, help="Percentile for anomaly detection.")
    args = parser.parse_args()
    main(args)
