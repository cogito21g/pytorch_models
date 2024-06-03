import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from src.data_loader import load_data
import os

def train_model(model_fn, epochs=10, batch_size=32, learning_rate=0.001, save_path='./results'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_data(batch_size)

    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

    # Save model weights
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, f'{model_fn.__name__}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot and save the training loss graph
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    loss_plot_path = os.path.join(save_path, f'{model_fn.__name__}_training_loss.png')
    plt.savefig(loss_plot_path)
    plt.show()
    print(f"Training loss plot saved to {loss_plot_path}")

    return model