import time

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def logging_result(result_path, accuracy, data_processing_time, training_time, evaluation_time):
    with open(result_path, 'w') as f:
        f.write(f"Accuracy of the network on the test images: {accuracy}%\n")
        f.write(f"Data processing time: {data_processing_time:.2f} seconds\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation time: {evaluation_time:.2f} seconds\n")
    print(f"Evaluation result saved to {result_path}")