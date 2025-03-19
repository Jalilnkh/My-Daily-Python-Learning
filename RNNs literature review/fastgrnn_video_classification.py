import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------- 1. Video Dataset with Optical Flow + Pose --------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=30, frame_size=112):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = []
        self.labels = []
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.pose = mp.solutions.pose.Pose()
        self.optical_flow = cv2.optflow.createOptFlow_DualTVL1()

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
        features = self._extract_features(video_path)
        return torch.tensor(features, dtype=torch.float32), label

    def _extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            return np.zeros((self.num_frames, 50))  # Return empty feature set if video cannot be read
        
        prev_frame = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (self.frame_size, self.frame_size))
        frames = []
        
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.frame_size, self.frame_size))
            flow = self.optical_flow.calc(prev_frame, frame_gray, None)  # Compute Optical Flow
            prev_frame = frame_gray

            # Optical Flow Features (2D → Flatten to 1D)
            flow_features = np.mean(flow, axis=(0, 1))  # Average over entire frame
            
            # Extract Pose Landmarks (33 points, each with x, y, visibility)
            pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_features = []
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    pose_features.append(landmark.x)
                    pose_features.append(landmark.y)
            else:
                pose_features = [0] * 66  # If no landmarks detected, fill with zeros
            
            # Combine Optical Flow + Pose (Use first 48 features)
            feature_vector = np.concatenate((flow_features, pose_features[:48]))
            frames.append(feature_vector)

        cap.release()

        # Ensure num_frames by duplicating last frame if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros(50))

        return np.array(frames)

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
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_tilde = torch.tanh(self.W(x_t) + self.U(h_t))
            h_t = self.alpha * h_tilde + self.beta * h_t
        return self.fc(h_t)

# -------------------- 3. Train FastGRNN --------------------
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
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

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {100 * correct / total:.2f}%")

# -------------------- 4. Inference --------------------
def predict(model, video_path, device):
    model.to(device)
    model.eval()
    dataset = VideoDataset(root_dir="")  # Dummy instance to use feature extractor
    features = torch.tensor(dataset._extract_features(video_path), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# -------------------- 5. Run Training and Inference --------------------
if __name__ == "__main__":
    dataset_path = "./dataset"  # Replace with actual path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VideoDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = FastGRNN(input_dim=50, hidden_dim=64, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, device, epochs=5)

    test_video = "./dataset/Walking While Reading Book/Walking While Reading Book (1).mp4"  # Replace with actual test video
    prediction = predict(model, test_video, device)
    print(f"Predicted Class: {train_dataset.classes[prediction]}")
