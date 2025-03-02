import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the FastRNN Cell
class FastRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Implements a FastRNN cell with a residual connection.
        """
        super(FastRNNCell, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices for hidden state and input transformations
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

        # Trainable residual weights (α and β)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Initialized to small value
        self.beta = nn.Parameter(torch.tensor(0.9))  # Initialized to large value

    def forward(self, x, h_prev):
        """
        Forward pass of the FastRNN cell.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
            h_prev (Tensor): Previous hidden state of shape (batch_size, hidden_size).
        Returns:
            h (Tensor): Updated hidden state.
        """
        h_tilde = torch.tanh(self.W(x) + self.U(h_prev))  # Candidate hidden state
        h = self.alpha * h_tilde + self.beta * h_prev  # Residual connection
        return h

# Define the FastRNN Model
class FastRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Implements a full FastRNN model with multiple timesteps.
        """
        super(FastRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = FastRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through FastRNN for a sequence of inputs.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            out (Tensor): Output logits of shape (batch_size, output_size).
        """
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initialize hidden state
        
        # Process each timestep
        for t in range(seq_length):
            h = self.rnn_cell(x[:, t, :], h)

        out = self.fc(h)  # Final prediction from last hidden state
        return out


# Training the Model
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    Train the FastRNN model.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)  # Ensure correct shape
            y_batch = y_batch.view(-1)  # Ensure target shape matches output

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')


# Testing Function
def test_model(model, X_test):
    """
    Test the trained FastRNN model on new input data.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).view(-1)  # Ensure output is the correct shape
        predictions = torch.sigmoid(outputs) > 0.5
    return predictions


if "__main__" == __name__:

    # Prepare Training Data
    x1 = torch.tensor([[1, 2.3, 24, -5],
                    [1, 2.3, 34, -5],
                    [1, 2.33, 4, -5],
                    [1, 2.3, 4, -5],
                    [31, 2.3, 4, -5],
                    [41, 2.3, 4, -5]], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    y1 = torch.tensor([0], dtype=torch.float32)

    x2 = torch.tensor([[12, 23, 3, 3],
                    [12, 23, 3, 23],
                    [12, 3, 3, 3],
                    [122, 2, 63, 73],
                    [132, 23, 3, 3],
                    [62, 3, 3, 3]], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    y2 = torch.tensor([1], dtype=torch.float32)

    # Combine Data
    X_train = torch.cat([x1, x2], dim=0)
    y_train = torch.cat([y1, y2], dim=0)

    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Model Configuration
    input_size = 4
    hidden_size = 8
    output_size = 1  # Binary classification

    # Initialize Model
    model = FastRNN(input_size, hidden_size, output_size)

    # Loss Function & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Save the trained model
    torch.save(model.state_dict(), "models/fastrnn_model.pth")
    print("Model saved successfully.")


    # Load the trained model and test
    model.load_state_dict(torch.load("models/fastrnn_model.pth"))
    print("Model loaded successfully.")

    # Perform Testing
    predictions = test_model(model, X_train)
    print("Predictions:", predictions.numpy())
