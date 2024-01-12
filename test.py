from Network import CNNnetwork
import torch

testCNN = CNNnetwork((4, 84, 84), 4, [64], False, True)

print(testCNN)

test_data = torch.rand((3, 4, 84, 84))

y = testCNN(test_data)