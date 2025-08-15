#Base Saimese Model:
import torch
from torch import nn
from layers import L1Dist

class ModelSaimese(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Block=nn.Sequential(
            nn.Conv2d(in_channels=3,
                     out_channels=64,
                     kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4),
            nn.ReLU(inplace=True),
        )

        self.Classifier=nn.Sequential(
            nn.Linear(in_features=256*6*6,
                     out_features=4096),
            nn.Sigmoid()
        )
        
    def forward_once(self,x):
        x=self.Conv_Block(x)
        x=x.view(x.size(0),-1) #nn.Flatten()
        x=self.Classifier(x)
        return x

    def forward(self,input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)
        return output1,output2
    
#Combining Base Model and L1Dist Class:
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = ModelSaimese()
        self.l1_distance = L1Dist()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4096,
                      out_features=1), 
            nn.Sigmoid()
        )
    
    def forward(self, input1, input2):
        output1 = self.base_model.forward_once(input1)
        output2 = self.base_model.forward_once(input2)
        l1_dist = self.l1_distance(output1, output2)
        out= self.classifier(l1_dist)
        return out