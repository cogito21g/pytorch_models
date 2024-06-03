import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.evaluate import evaluate_model
from models.model import get_resnet_model
from models.densenet import get_densenet_model
from models.alexnet import get_alexnet_model
from models.vggnet import get_vggnet_model
from utils.utils import logging_result

if __name__ == "__main__":
    save_path = './results'
    
    # Train and evaluate ResNet model
    resnet_model, resnet_training_time, resnet_data_processing_time = train_model(get_resnet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, resnet_evaluation_time, result_path = evaluate_model(resnet_model, get_resnet_model, batch_size=32, save_path=save_path)
    logging_result(result_path, accuracy, resnet_data_processing_time, resnet_training_time, resnet_evaluation_time)

    # Train and evaluate Densenet model
    densenet_model, densenet_training_time, densenet_data_processing_time = train_model(get_densenet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, densenet_evaluation_time, result_path = evaluate_model(densenet_model, get_densenet_model, batch_size=32, save_path=save_path)
    logging_result(result_path, accuracy, densenet_data_processing_time, densenet_training_time, densenet_evaluation_time)

    # Train and evaluate AlexNet model
    alexnet_model, alexnet_training_time, alexnet_data_processing_time = train_model(get_alexnet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, alexnet_evaluation_time, result_path = evaluate_model(alexnet_model, get_alexnet_model, batch_size=32, save_path=save_path)
    logging_result(result_path, accuracy, alexnet_data_processing_time, alexnet_training_time, alexnet_evaluation_time)

    # Train and evaluate VGGNet model
    vggnet_model, vggnet_training_time, vggnet_data_processing_time = train_model(get_vggnet_model, epochs=10, batch_size=32, learning_rate=0.001, save_path=save_path)
    accuracy, vggnet_evaluation_time, result_path = evaluate_model(vggnet_model, get_vggnet_model, batch_size=32, save_path=save_path)
    logging_result(result_path, accuracy, vggnet_data_processing_time, vggnet_training_time, vggnet_evaluation_time)
