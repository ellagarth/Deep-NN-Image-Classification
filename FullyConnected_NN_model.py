import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------
# CONFIGURATION VARIABLES FOR FULLY-CONNECTED NEURAL NETWORK (MULTI-LAYER PERCEPTRON)
# -----------------------
# Number of hidden layers (between 1 and 10)
num_hidden_layers = 1

# resize image size (i.e. 10 = 10x10 pixels)
image_size = 28

# Number of neurons/nodes in each layer (input, hidden, and output)
input_neurons = image_size*image_size  # 10x10 image input
hidden_neurons = 10  # Number of neurons in each hidden layer
output_neurons = 4    # 4 output classes (circle, square, triangle, cross)

# Activation functions for hidden layer (choose between 'relu', 'leaky_relu', 'sigmoid', 'tanh')
activation_function = 'leaky_relu'
# output layer is softmax to convert to probabilities for multi-class classification

# Maximum epochs for training
num_epochs = 1000

# Learning rate
learning_rate = 0.00001

# Batch size
batch_size = 1

# Use early stopping (True/False)
use_early_stopping = False

# Patience for early stopping (only used if early stopping is enabled)
patience = 10

# Use learning rate scheduler (True/False)
use_lr_scheduler = False

# Dropout rate (set to 0.0 if no dropout is required)
dropout_rate = 0.0

# set constant random seed to give consistent results
torch.manual_seed(42)
np.random.seed(42)

# -----------------------
# END CONFIGURATION
# -----------------------

# Set device (GPU preferable for paralell processing like deep learning models)
# cuda = GPU (Graphics Proecessing Unit), or CPU (Central Processing Unit) if no GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activation function choices
def get_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation function name: {name}")

# Custom dataset class
class ShapeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Load image as grayscale
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Image transformations (resize image to pixal values set at top of script and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Load the dataset
train_dataset = ShapeDataset(csv_file='/Users/ellagarth/Desktop/Portfolio3/Training_Images/training.csv', 
                             img_dir='/Users/ellagarth/Desktop/Portfolio3/Training_Images/', 
                             transform=transform)

# Split the data into training (80%) and validation (20%) sets (split is reproducable when using random_state=42)
# uses skikit-learn function (i.e. train_test_split)
# train_subset is a dataset, which contains 80% of the data for training
# val_subset is a dataset, which contains 20% of the data for validation
train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

# Use PyTorch DataLoaders with adjustable batch size
# shuffle (the training data should be shuffled before creating batches, ensuring the model doesn't learn any unintended patterns from the order of the data to help improve generalisation)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Dynamic Neural Network class with configurable hidden layers, neurons, and dropout
class NeuralNet(nn.Module):
    def __init__(self, num_hidden_layers, input_neurons, hidden_neurons, output_neurons, dropout_rate, activation_function):
        super(NeuralNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None
        self.activation_function = get_activation_function(activation_function)

        # Input to first hidden layer
        self.hidden_layers.append(nn.Linear(input_neurons, hidden_neurons))
        
        # Add the rest of the hidden layers dynamically
        for i in range(1, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_neurons, hidden_neurons))

        # Output layer
        self.output_layer = nn.Linear(hidden_neurons, output_neurons)

        # Softmax for output layer
        self.softmax = nn.Softmax(dim=1)
    
    # Forward method of the neural network. Defines how the input x passes through the neural network's layers.
    # X is a batch of 10x10 images (or other data) passed through the network.
    def forward(self, x):
        # reshapes the input tensor x without changing the underlying data.
        # x.view reshapes the tensors (like NumPy's reshape function)
        # -1 tells PyTorch to infer the appropriate dimensions based on the number of elements in the tensor (i.e. calculate the size of this dimensions based on the total number of elements and the other dimensions)
        x = x.view(-1, input_neurons)  # Flatten the 10x10 image. Input neurons represents the number of input neurons for the next layer, typically the flattened size of the input data.
        #The purpose of x.view(-1, input_neurons) is to flatten the 10x10 images into a 1D vector of size input_neurons. Flattening is often required before feeding image data into a fully connected layer in a neural network because fully connected layers expect 1D vectors as inputs, not 2D images.
        
        # Pass through each hidden layer
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
            if self.dropout:
                x = self.dropout(x)

        # Pass through output layer
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

# Initialise the model, loss function, and optimiser
model = NeuralNet(num_hidden_layers=num_hidden_layers, input_neurons=input_neurons, hidden_neurons=hidden_neurons,
                  output_neurons=output_neurons, dropout_rate=dropout_rate, activation_function=activation_function).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
if use_lr_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0

# Track training loss for visualisation
loss_history = []

# Training loop with loss printed every 100 epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # Save the loss history for plotting
    loss_history.append(running_loss / len(train_loader))

    # Learning rate scheduler step
    if use_lr_scheduler:
        scheduler.step(val_loss)

    # Early stopping
    if use_early_stopping:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Plot the training loss over time
plt.figure()
plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Load the test dataset
test_dataset = ShapeDataset(csv_file='/Users/ellagarth/Desktop/Portfolio3/Test_Images/test.csv', 
                            img_dir='/Users/ellagarth/Desktop/Portfolio3/Test_Images', 
                            transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the network on test data
model.eval()
all_labels = []
all_preds = []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Print the accuracy
print(f'Accuracy on test images: {100 * correct / total:.2f}%')

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Circle', 'Square', 'Triangle', 'Cross']))

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Circle', 'Square', 'Triangle', 'Cross'], yticklabels=['Circle', 'Square', 'Triangle', 'Cross'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
