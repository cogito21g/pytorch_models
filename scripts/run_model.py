import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.evaluate import evaluate_model
from models.model import get_resnet_model
from models.densenet import get_densenet_model

if __name__ == "__main__":
    save_path = './results'
    
    # Train and evaluate ResNet model
    resnet_model, training_time, data_processing_time = train_model(get_resnet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, resnet_evaluation_time, result_path = evaluate_model(resnet_model, get_resnet_model, batch_size=32, save_path=save_path)
    with open(result_path, 'w') as f:
        f.write(f"Accuracy of the network on the test images: {accuracy}%\n")
        f.write(f"Data processing time: {data_processing_time:.2f} seconds\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation time: {resnet_evaluation_time:.2f} seconds\n")
    print(f"Evaluation result saved to {result_path}")

    
    # Train and evaluate Densenet model
    densenet_model, training_time, data_processing_time = train_model(get_densenet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, densenet_evaluation_time, result_path = evaluate_model(densenet_model, get_densenet_model, batch_size=32, save_path=save_path)
    with open(result_path, 'w') as f:
        f.write(f"Accuracy of the network on the test images: {accuracy}%\n")
        f.write(f"Data processing time: {data_processing_time:.2f} seconds\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation time: {resnet_evaluation_time:.2f} seconds\n")
    print(f"Evaluation result saved to {result_path}")