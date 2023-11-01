import torch

def train_iteration(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    losses = []  
    accuracies = []  

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = correct_predictions / total_samples

        # Append the loss and accuracy for this epoch to the lists
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

    return losses, accuracies  # Return lists of losses and accuracies for each epoch
