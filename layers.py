#Custom L1 Distance layer module:
#Needed to load the custom model->
#Import Dependencies:
import torch
from torch import nn

#Custom L1 distance Layer from model:
#Custom L1Distance Calculator Class: (Specially For Saimese Neural Network)
class L1Dist(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)

