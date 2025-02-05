import socket
import data_pb2  # Import generated protobuf module
import struct
import cv2
import numpy as np

# Server Code
def receive_data():
    """
    Function to receive Protobuf image data and save it.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 5005))
    server_socket.listen(5)
    
    print("Server is listening on port 5005...")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        try:
            while True:
                # Receive message length (4 bytes)
                data_length = conn.recv(4)
                if not data_length:
                    break

                message_size = struct.unpack(">I", data_length)[0]
                data = b""

                # Receive complete message
                while len(data) < message_size:
                    packet = conn.recv(message_size - len(data))
                    if not packet:
                        break
                    data += packet

                # Deserialize Protobuf message
                img_feature_data = data_pb2.ImageFeatureData()
                img_feature_data.ParseFromString(data)

                # Save received image
                with open("received_image.jpg", "wb") as img_file:
                    img_file.write(img_feature_data.image_data)

                # Print Features
                print("Received Features (float):", list(img_feature_data.features))

        except Exception as e:
            print(f"Error: {e}")

        finally:
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    receive_data()
