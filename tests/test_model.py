import unittest
import torch
from models.model import get_resnet_model

class TestResNetModel(unittest.TestCase):
    def test_model_output(self):
        model = get_resnet_model()
        input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 이미지 크기
        output = model(input_tensor)
        self.assertEqual(output.shape[1], 10)  # CIFAR-10은 10개의 클래스를 가짐

if __name__ == '__main__':
    unittest.main()