import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------- 1. Video Dataset (No Feature Extraction) --------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=35, frame_size=64):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = []
        self.labels = []
        self.num_frames = num_frames
        self.frame_size = frame_size

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                self.video_paths.append(os.path.join(cls_path, video))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._load_video(video_path)
        return torch.tensor(frames, dtype=torch.float32), label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.resize(
                    frame_gray, (self.frame_size, self.frame_size)
                )
                frames.append(frame_resized / 255.0)  # Normalize

        cap.release()

        # Ensure exactly num_frames by duplicating last frame if needed
        while len(frames) < self.num_frames:
            frames.append(
                frames[-1] if frames else np.zeros((self.frame_size, self.frame_size))
            )

        return np.array(frames).reshape(self.num_frames, -1)  # Flatten each frame


# -------------------- 2. FastGRNN Model --------------------
class FastGRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FastGRNN, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.W.out_features).to(x.device)
        for t in range(x.size(1)):  # Iterate through time steps
            x_t = x[:, t, :]  # Get raw frame
            h_tilde = torch.tanh(self.W(x_t) + self.U(h_t))
            h_t = self.alpha * h_tilde + self.beta * h_t
        return self.fc(h_t)


# -------------------- 3. Train FastGRNN --------------------
def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    save_path="fastgrnn_model.pth",
):
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {accuracy:.2f}%"
        )

        # Save model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved: {save_path}")


# -------------------- 4. Load Model for Inference --------------------
def load_model(model, load_path="fastgrnn_model.pth", device="cpu"):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {load_path}")


# -------------------- 5. Inference --------------------
def predict(model, video_path, device, frame_size=64, num_frames=35):
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (frame_size, frame_size))
            frames.append(frame_resized / 255.0)

    cap.release()

    # Ensure exactly num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((frame_size, frame_size)))

    features = (
        torch.tensor(np.array(frames).reshape(num_frames, -1), dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

import time
# -------------------- 6. Run Training and Inference --------------------
if __name__ == "__main__":
    dataset_path = "/home/jalilnkh/.cache/kagglehub/datasets/sharjeelmazhar/human-activity-recognition-video-dataset/versions/1/Human Activity Recognition - Video Dataset/"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset
    train_dataset = VideoDataset(dataset_path, num_frames=35, frame_size=32)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize Model
    input_dim = 32 * 32  # Raw frame size (flattened)
    model = FastGRNN(input_dim=input_dim, hidden_dim=32, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and Save the Model
    train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epochs=10,
        save_path="models/fastgrnn_model_raw_imgs.pth",
    )

    # Load the Model for Inference
    load_model(model, load_path="models/fastgrnn_model_raw_imgs.pth", device=device)

    # Test inference on a new video
    test_video = "/home/jalilnkh/.cache/kagglehub/datasets/sharjeelmazhar/human-activity-recognition-video-dataset/versions/1/Human Activity Recognition - Video Dataset/Walking/Walking (1).mp4"  # Replace with actual test video

    start_time = time.time()
    print("Start: ", (time.time() - start_time) * 1000.0)
    prediction = predict(model, test_video, device, frame_size=32, num_frames=35)
    print("After pred: ", (time.time() - start_time) * 1000.0)

    print(f"🎯 Predicted Class: {train_dataset.classes[prediction]}")
