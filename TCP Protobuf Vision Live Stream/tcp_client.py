import socket
import data_pb2  # Import generated protobuf module
import struct
import cv2
import numpy as np


# Client Code
def extract_features(image_path):
    """
    Dummy feature extraction function.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error reading image file. Check the path.")
    
    img_resized = cv2.resize(img, (32, 32))  # Resize for consistency
    img_bytes = cv2.imencode(".jpg", img_resized)[1].tobytes()

    # Example features: Some integer and float values
    feature_values = [10.5, 20.2, 30.8]

    return img_bytes, feature_values


def send_data(image_path):
    """
    Function to send image and extracted features via Protobuf over TCP.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 5005))

    try:
        img_bytes, features = extract_features(image_path)

        # Create Protobuf message
        img_feature_data = data_pb2.ImageFeatureData()
        img_feature_data.image_data = img_bytes
        img_feature_data.features.extend(features)

        # Serialize Protobuf message
        serialized_data = img_feature_data.SerializeToString()

        # Send message length first (4 bytes), then the actual data
        client_socket.sendall(struct.pack(">I", len(serialized_data)))
        client_socket.sendall(serialized_data)

        print("Data sent successfully!")

    except Exception as e:
        print(f"Error sending data: {e}")

    finally:
        client_socket.close()


if __name__ == "__main__":
    send_data("test_image.jpeg")
