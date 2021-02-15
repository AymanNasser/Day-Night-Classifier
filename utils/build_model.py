import torch
import torch.nn as nn
from torchvision import models, transforms
import io
from PIL import Image


def transform_image(image):
    transform_pipeline = transforms.Compose([

                      transforms.Resize((224,224)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])      

    image = Image.open(io.BytesIO(image))
    return transform_pipeline(image).unsqueeze(0)                  
      
      
class DayNightClassifier(nn.Module):
    def __init__(self, num_classes = 1):
        super(DayNightClassifier, self).__init__()
        self.res_net = models.resnet101(pretrained=True)
        num_ftrs = self.res_net.fc.in_features
        
        self.res_net.fc = nn.Linear(num_ftrs, 1024)
        self.final_layers = nn.Sequential(nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(256, num_classes),
                                  nn.Sigmoid())
        
    def forward(self, X):
        out = self.res_net(X)
        return self.final_layers(out).squeeze(-1)