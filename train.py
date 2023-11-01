import os
import torch
import torch.nn as nn
from torchvision import transforms
from models.simple_cnn import SimpleCNN
from loss import categorical_loss
from train_iter import train_iteration
from load_data import load_custom_dataset  
import argparse

# Define the main function
def main(source_path, dest_path):
    # Hyperparameters
    num_epochs = 8
    batch_size = 32
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print whether the model is running on GPU or CPU
    if device == 'cuda':
        print('Training on GPU')
    else:
        print('Training on CPU')

    # Load your custom dataset using data_loader.py
    train_loader, val_loader = load_custom_dataset(data_root=source_path, batch_size=batch_size, num_workers=4)

    # Initialize the model
    model = SimpleCNN().to(device)

    # Loss and optimizer
    # criterion = categorical_loss(outputs, targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss_history = []
    train_accuracy_history = []

    for epoch in range(num_epochs):
        train_losses, train_accuracies = train_iteration(model, train_loader, optimizer, categorical_loss, device, num_epochs)

        avg_loss = train_losses[-1]  # Get the last training loss from the list
        avg_accuracy = train_accuracies[-1]  # Get the last training accuracy from the list

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

        train_loss_history.extend(train_losses)
        train_accuracy_history.extend(train_accuracies)

    print('Training complete.')

    # Save the trained model
    os.makedirs(dest_path, exist_ok=True)
    model_path = os.path.join(dest_path, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)

    # Save the training loss and accuracy plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    plot_path = os.path.join(dest_path, 'training_plot.png')
    plt.savefig(plot_path)

    print(f'Model weights saved in {model_path}')
    print(f'Training plot saved in {plot_path}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Training Script")
    
    # Add arguments 
    # parser.add_argument()
    # parser.add_argument()
    parser.add_argument("--source", type=str, required=True, help="Path to the source dataset directory")
    parser.add_argument("--dest", type=str, required=True, help="Path to the destination directory to save the model and results")


    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the source and destination paths
    source_path = args.source
    dest_path = args.dest

    main(source_path, dest_path)
