import socket
import data_pb2
import struct
import cv2
import numpy as np

def extract_features(image_path):
    """ Dummy feature extraction """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (32, 32))  # Resize for consistency
    img_bytes = cv2.imencode(".jpg", img)[1].tobytes()

    # Feature 1: Integers + Nested List
    feature1_values = [10, 20, 30]
    feature1_nested = [1, 2, 3, 4]

    # Feature 2: Floats + Nested List
    feature2_values = [0.1, 0.2, 0.3]
    feature2_nested = [5.5, 6.6, 7.7]

    return img_bytes, feature1_values, feature1_nested, feature2_values, feature2_nested

def send_data(image_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 5005))

    img_bytes, f1_vals, f1_nested, f2_vals, f2_nested = extract_features(image_path)

    # Create Protobuf message
    img_feature_data = data_pb2.ImageFeatureData()
    img_feature_data.image_data = img_bytes

    # Feature 1
    img_feature_data.feature1.values.extend(f1_vals)
    img_feature_data.feature1.nested_list.extend(f1_nested)

    # Feature 2
    img_feature_data.feature2.values.extend(f2_vals)
    img_feature_data.feature2.nested_list.extend(f2_nested)

    # Serialize Protobuf message
    serialized_data = img_feature_data.SerializeToString()

    # Send message length first (4 bytes), then the actual data
    client_socket.sendall(struct.pack(">I", len(serialized_data)))
    client_socket.sendall(serialized_data)

    client_socket.close()
    print("Data sent successfully!")

if __name__ == "__main__":
    send_data("test_image.jpeg")
