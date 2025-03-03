import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define a simple Dataset for our sequences.
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# FastGRNN implementation.
class FastGRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        FastGRNN update:
            z_t = σ(W x_t + U h_{t-1} + b_z)
            h_tilde = tanh(W x_t + U h_{t-1} + b_h)
            h_t = (ζ*(1 - z_t) + ν) ⊙ h_tilde + z_t ⊙ h_{t-1}
        Here, ζ and ν are trainable scalars.
        """
        super(FastGRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Shared weight matrices for both gate and candidate
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # Biases for the gate and candidate
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        
        # Trainable scalars for the residual connection.
        # To ensure they lie in [0, 1] one might also use a sigmoid transformation.
        self.zeta = nn.Parameter(torch.Tensor(1))
        self.nu   = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weights uniformly
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        # Initialize hidden state h0 as zeros.
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            # Compute the gate:
            z_t = torch.sigmoid(torch.matmul(x_t, self.W.t()) + torch.matmul(h, self.U.t()) + self.b_z)
            # Compute candidate hidden state:
            h_tilde = torch.tanh(torch.matmul(x_t, self.W.t()) + torch.matmul(h, self.U.t()) + self.b_h)
            # Update hidden state with a weighted residual update.
            h = (self.zeta * (1 - z_t) + self.nu) * h_tilde + z_t * h
        return h

# A simple classifier that uses FastGRNN and a final linear layer.
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.rnn = FastGRNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        h_final = self.rnn(x)  # final hidden state (batch_size, hidden_size)
        out = self.fc(h_final) # (batch_size, num_classes)
        return out


# ---------------------------
# Training Loop
# ---------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)  # (batch_size, 1)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model

# ---------------------------
# Inference Function
# ---------------------------
def inference(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)  # (batch_size, 1)
        # Convert logits to probability using sigmoid
        probs = torch.sigmoid(outputs)
    return probs

if "__main__" == __name__:

    # ---------------------------
    # Prepare the toy dataset
    # ---------------------------
    # x1: sequence with label 0 (6 timesteps, 4 features)
    x1 = torch.tensor([[1, 2.3, 24, -5],
                    [1, 2.3, 34, -5],
                    [1, 2.33, 4, -5],
                    [1, 2.3, 4, -5],
                    [31, 2.3, 4, -5],
                    [41, 2.3, 4, -5]], dtype=torch.float32)
    y1 = torch.tensor(0, dtype=torch.float32)

    # x2: sequence with label 1
    x2 = torch.tensor([[12, 23, 3, 3],
                    [12, 23, 3, 23],
                    [12, 3, 3, 3],
                    [122, 2, 63, 73],
                    [132, 23, 3, 3],
                    [62, 3, 3, 3]], dtype=torch.float32)
    y2 = torch.tensor(1, dtype=torch.float32)

    # Create two sequences repeated 500 times each to simulate 1000 samples.
    seq1 = x1.unsqueeze(0).repeat(500, 1, 1)  # shape: (500, 6, 4)
    seq2 = x2.unsqueeze(0).repeat(500, 1, 1)  # shape: (500, 6, 4)
    sequences = torch.cat([seq1, seq2], dim=0)  # shape: (1000, 6, 4)
    labels = torch.cat([torch.zeros(500), torch.ones(500)], dim=0)  # shape: (1000,)

    # Create dataset and dataloader.
    dataset = SequenceDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------------------------
    # Model, Loss, Optimizer
    # ---------------------------
    input_size = 4      # Number of features per timestep.
    hidden_size = 16    # Hidden dimension (can be tuned).
    num_classes = 1     # For binary classification.

    model = Classifier(input_size, hidden_size, num_classes)

    # For binary classification we use BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs)
    # Save the trained model
    torch.save(model.state_dict(), "models/fastgrnn_model.pth")
    print("Model saved successfully.")


    # Load the trained model and test
    model.load_state_dict(torch.load("models/fastgrnn_model.pth"))
    print("Model loaded successfully.")

    # Example inference on one sample from the dataset.
    sample = sequences[0].unsqueeze(0)  # shape: (1, 6, 4)
    probs = inference(trained_model, sample)
    pred = (probs > 0.5).float()
    print("Prediction:", pred.item(), "Probability:", probs.item())
