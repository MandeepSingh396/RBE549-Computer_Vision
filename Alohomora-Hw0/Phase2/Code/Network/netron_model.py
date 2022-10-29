from Network import CIFAR10Model, ResNet,  DenseNet, transition_block, conv2d_block, VGGNet
import torch

# from Network.Network import DenseNet

# model = CIFAR10Model(3,10)
model = DenseNet()
# model = VGGNet()
# model = ResNet9(3,10)
torch.save(model, 'model_Densenet.pt')
print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
total_params = sum(param.numel() for param in model.parameters())
print(total_params)
# count=0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
#         count+=1
# print(count)
# import netron

# netron.start('model_LeNet.pt')