import torch
from src.data_loader import load_data
from models.model import get_resnet_model

def evaluate_model(model, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    _, testloader = load_data(batch_size)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test images: {100 * correct / total}%")