import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the Simple RNN Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        A Simple RNN model for binary classification.
        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Number of hidden units in the RNN cell.
            output_size (int): Number of output classes (1 for binary classification).
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the RNN model.
        """
        _, h_n = self.rnn(x)  # h_n has shape (1, batch_size, hidden_size)
        h_n = h_n.squeeze(0)  # Remove the first dimension -> (batch_size, hidden_size)
        out = self.fc(h_n)  # Pass through fully connected layer -> (batch_size, 1)
        return out.view(-1)  # Flatten to (batch_size,)


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    Train the Simple RNN model.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)  # Now correctly shaped as (batch_size,)

            y_batch = y_batch.view(-1)  # Ensure y_batch matches output shape

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')



# Testing Function
def test_model(model, X_test):
    """
    Test the trained RNN model on new input data.
    Args:
        model (nn.Module): Trained RNN model.
        X_test (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
    Returns:
        Tensor: Predicted labels.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
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
    model = SimpleRNN(input_size, hidden_size, output_size)
    # Loss Function & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Train the model
    train_model(model, train_loader, criterion, optimizer)
    # Save the trained model
    torch.save(model.state_dict(), "models/simple_rnn_model.pth")
    print("Model saved successfully.")
    # Load the trained model and test
    model.load_state_dict(torch.load("simple_rnn_model.pth"))
    print("Model loaded successfully.")

    # Perform Testing
    predictions = test_model(model, X_train)
    print("Predictions:", predictions.numpy())
