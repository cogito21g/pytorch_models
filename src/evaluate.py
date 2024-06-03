import torch
from src.data_loader import load_data
import os

def evaluate_model(model, model_fn, batch_size=32, save_path='./results'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {accuracy}%")

    # Save the evaluation result
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_path = os.path.join(save_path, f'{model_fn.__name__}_evaluation_result.txt')
    with open(result_path, 'w') as f:
        f.write(f"Accuracy of the network on the test images: {accuracy}%\n")
    print(f"Evaluation result saved to {result_path}")