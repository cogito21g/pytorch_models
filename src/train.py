import torch
import torch.optim as optim
import torch.nn as nn
from models.model import get_resnet_model
from src.data_loader import load_data

def train_model(epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    trainloader, testloader = load_data(batch_size)

    model = get_resnet_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader)}")

    print("Finished Training")
    return model