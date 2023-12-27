from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import model , loss_function , optimizer
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

X = np.load('facemesh_images.npy')
Y = np.load('labels.npy')
print('X Shape: ' , X.shape)
print('Y Shape: ' , Y.shape)

# Now you can proceed with your train-test split and the rest of your pipeline
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Convert to torch tensors and reshape
x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)  # Reshape to (batch, channels, height, width)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)   # Reshape to (batch, channels, height, width)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def calculate_accuracy(y_pred, y_true):
    predicted = torch.round(y_pred)
    correct = (predicted == y_true).float() 
    accuracy = correct.sum() / len(correct)
    return accuracy

for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(output, target)

        # Print stats after every batch or after a fixed number of batches
        if batch_idx % 10 == 0:  # Adjust the number 10 as per your preference
            print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {running_loss / 10}, Accuracy: {running_accuracy / 10}')
            running_loss = 0.0
            running_accuracy = 0.0

    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_function(output, target).item()
            test_accuracy += calculate_accuracy(output, target)
    
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    print(f'Epoch: {epoch}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')



torch.save(model.state_dict(), 'model.pth')