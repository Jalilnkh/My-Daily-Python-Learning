import socket
import data_pb2  # Import generated protobuf module
import struct

def receive_data():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 5005))
    server_socket.listen(5)
    
    print("Server is listening on port 5005...")
    
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    while True:
        # Receive the length of the incoming message (4 bytes)
        data_length = conn.recv(4)
        if not data_length:
            break

        message_size = struct.unpack(">I", data_length)[0]
        data = conn.recv(message_size)

        # Deserialize the Protobuf message
        img_feature_data = data_pb2.ImageFeatureData()
        img_feature_data.ParseFromString(data)

        # Save the received image
        with open("received_image.jpg", "wb") as img_file:
            img_file.write(img_feature_data.image_data)

        # Print Feature 1 and Feature 2 details
        print("Feature 1 (Integers):", img_feature_data.feature1.values)
        print("Feature 1 (Nested List):", img_feature_data.feature1.nested_list)

        print("Feature 2 (Floats):", img_feature_data.feature2.values)
        print("Feature 2 (Nested List):", img_feature_data.feature2.nested_list)

    conn.close()

if __name__ == "__main__":
    receive_data()
