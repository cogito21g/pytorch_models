import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    save_path = './results'
    model = train_model(epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    evaluate_model(model, batch_size=32, save_path=save_path)