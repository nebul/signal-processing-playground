import numpy as np
import argparse
from PIL import Image


class RXDetector:

    def __init__(self, image_path):
        self.image_path = image_path
        self.read_image()
        self.prepare_image_data()
        self.calculate_statistics()

    def read_image(self):
        print("Reading image...")
        im = Image.open(self.image_path)
        self.image = np.asarray(im)
        if len(self.image.shape) == 2:
            self.image = self.image[:, :, np.newaxis]

    def prepare_image_data(self):
        print("Preparing image data...")
        self.flat_image = np.reshape(self.image, (-1, self.image.shape[-1])).T

    def calculate_statistics(self):
        print("Calculating statistics...")
        self.cov_matrix = np.cov(self.flat_image)
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.mean_vector = np.mean(self.flat_image, axis=1)

    def calculate_rx_coefficient(self):
        print("Calculating RX coefficients...")
        centered_data = self.flat_image.T - self.mean_vector
        rx_values = np.einsum('ij,jk,ik->i', centered_data, self.inv_cov_matrix, centered_data)
        return np.reshape(rx_values, self.image.shape[:2]).astype(np.float32)

    def save_result_image(self, result_array, save_path):
        print("Converting and saving the resultant image...")
        result_image = Image.fromarray(result_array.astype(np.uint8))
        result_image.save(save_path)


def main(args):
    detector = RXDetector(args.input_image)
    rx_coefficients = detector.calculate_rx_coefficient()
    detector.save_result_image(rx_coefficients, args.output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RX Detector for image analysis.")
    parser.add_argument("-i", "--input_image", required=True, help="Path to read the input image.")
    parser.add_argument("-o", "--output_image", required=True, help="Path to save the output image.")
    args = parser.parse_args()
    main(args)